########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################

import argparse
from datetime import datetime
import json
import os
import copy
import sys
import time
import warnings
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR
from src.args import ArgumentParser
from src.build_model import build_model
from src import utils
from src.prepare_data import prepare_data
from src.utils import save_ckpt_every_epoch
from src.utils import load_ckpt
from src.utils import print_log


from torchmetrics import JaccardIndex as IoU


def parse_args():
    parser = ArgumentParser(
        description="Open-World Semantic Segmentation (Training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_common_args()
    args = parser.parse_args()
    # The provided learning rate refers to the default batch size of 8.
    # When using different batch sizes we need to adjust the learning rate
    # accordingly:
    if args.batch_size != 8:
        args.lr = args.lr * args.batch_size / 8
        print(
            f"Notice: adapting learning rate to {args.lr} because provided "
            f"batch size differs from default batch size of 8."
        )

    return args


def train_main():
    args = parse_args()

    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(
        args.results_dir, args.dataset, f"{args.id}", f"{training_starttime}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "confusion_matrices"), exist_ok=True)

    with open(os.path.join(ckpt_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(ckpt_dir, "argsv.txt"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    # data preparation ---------------------------------------------------------
    data_loaders = prepare_data(args, ckpt_dir)

    train_loader, valid_loader, _ = data_loaders

    n_classes_without_void = train_loader.dataset.n_classes_without_void
    if args.class_weighting != "None":
        class_weighting = train_loader.dataset.compute_class_weights(
            weight_mode=args.class_weighting, c=args.c_for_logarithmic_weighting
        )
    else:
        class_weighting = np.ones(n_classes_without_void)
    # model building -----------------------------------------------------------
    model, device = build_model(args, n_classes=n_classes_without_void)
    if args.freeze > 0:
        print("Freeze everything but the output layer(s).")
        for name, param in model.named_parameters():
            if "out" not in name:
                param.requires_grad = False

    # loss, optimizer, learning rate scheduler, csvlogger  ----------

    # loss functions
    loss_function_train = utils.CrossEntropyLoss2d(
        weight=class_weighting, device=device
    )
    loss_objectosphere = utils.ObjectosphereLoss()
    loss_mav = utils.OWLoss(n_classes=n_classes_without_void)
    loss_contrastive = utils.ContrastiveLoss(n_classes=n_classes_without_void)

    pixel_sum_valid_data = valid_loader.dataset.compute_class_weights(
        weight_mode="linear"
    )
    pixel_sum_valid_data_weighted = np.sum(pixel_sum_valid_data * class_weighting)
    loss_function_valid = utils.CrossEntropyLoss2dForValidData(
        weight=class_weighting,
        weighted_pixel_sum=pixel_sum_valid_data_weighted,
        device=device,
    )

    train_loss = [loss_function_train, loss_objectosphere, loss_mav, loss_contrastive]
    val_loss = [loss_function_valid, loss_objectosphere, loss_mav, loss_contrastive]
    if not args.obj:
        train_loss[1] = None
        val_loss[1] = None
    if not args.mav:
        train_loss[2] = None
        val_loss[2] = None
    if not args.closs:
        train_loss[3] = None
        val_loss[3] = None

    optimizer = get_optimizer(args, model)

    # in this script lr_scheduler.step() is only called once per epoch
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=[i["lr"] for i in optimizer.param_groups],
        total_steps=args.epochs,
        div_factor=25,
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=1e4,
    )

    # load checkpoint if parameter last_ckpt is provided
    if args.last_ckpt:
        ckpt_path = args.last_ckpt
        epoch_last_ckpt, best_miou, best_miou_epoch, mav_dict, std_dict = load_ckpt(
            model, optimizer, ckpt_path, device
        )
        start_epoch = epoch_last_ckpt + 1
    else:
        start_epoch = 0
        best_miou = 0
        best_miou_epoch = 0

    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights))

    writer = SummaryWriter("runs/" + ckpt_dir.split(args.dataset)[-1])

    # start training -----------------------------------------------------------
    for epoch in range(int(start_epoch), args.epochs):
        # unfreeze
        if args.freeze == epoch and args.finetune is None:
            for param in model.parameters():
                param.requires_grad = True

        mean, var = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            train_loss=train_loss,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            debug_mode=args.debug,
            writer=writer,
        )

        miou = validate(
            model=model,
            valid_loader=valid_loader,
            device=device,
            val_loss=val_loss,
            epoch=epoch,
            debug_mode=args.debug,
            writer=writer,
            classes=args.num_classes,
        )

        writer.flush()

        # save weights
        if not args.overfit:
            # save / overwrite latest weights (useful for resuming training)
            save_ckpt_every_epoch(
                ckpt_dir, model, optimizer, epoch, best_miou, best_miou_epoch, mean, var
            )
            if (epoch + 1) % 20 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_dir, "epoch_" + str(epoch) + ".pth"),
                )
            if miou > best_miou:
                best_miou = miou
                best_miou_epoch = epoch
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_dir, "best_miou.pth"),
                )

    # save mavs to a pickle
    with open("mavs.pickle", "wb") as h1:
        pickle.dump(mean, h1, protocol=pickle.HIGHEST_PROTOCOL)
    with open("vars.pickle", "wb") as h2:
        pickle.dump(var, h2, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training completed ")


def train_one_epoch(
    model,
    train_loader,
    device,
    optimizer,
    train_loss,
    epoch,
    lr_scheduler,
    writer,
    debug_mode=False,
):
    lr_scheduler.step(epoch)
    samples_of_epoch = 0

    # set model to train mode
    model.train()

    loss_function_train, loss_obj, loss_mav, loss_contrastive = train_loss

    # summed loss of all resolutions
    total_loss_list = []
    total_sem_loss = []
    total_obj_loss = []
    total_ows_loss = []
    total_con_loss = []

    mavs = None
    if epoch and loss_contrastive is not None:
        mavs = loss_mav.read()
    for i, sample in enumerate(train_loader):
        start_time_for_one_step = time.time()

        # load the data and send them to gpu
        image = sample["image"].to(device)
        batch_size = image.data.shape[0]

        label_ss = sample["label"].clone().cuda()
        label_ss[label_ss == 255] = 0
        target_scales = label_ss

        # this is more efficient than optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        # forward pass
        pred_scales, ow_res = model(image)
        cw_target = target_scales.clone()
        # cw_target[cw_target > 16] = 255
        losses = loss_function_train(pred_scales, cw_target)
        loss_segmentation = sum(losses)
        loss_objectosphere = torch.tensor(0.0)
        loss_ows = torch.tensor(0.0)
        loss_con = torch.tensor(0.0)
        total_loss = 0.9 * loss_segmentation
        label = sample["label"].long().cuda() - 1
        label[label < 0] = 255

        if loss_obj is not None:
            label_ow = label.clone().cuda().to(torch.uint8)
            loss_objectosphere = loss_obj(ow_res, label_ow)
            total_loss += 0.5 * loss_objectosphere
        if loss_mav is not None:
            loss_ows = loss_mav(pred_scales, label, is_train=True)
            total_loss += 0.1 * loss_ows
        if loss_contrastive is not None:
            loss_con = loss_contrastive(mavs, ow_res, label, epoch)
            total_loss += 0.5 * loss_con

        total_loss.backward()
        optimizer.step()

        # append loss values to the lists. Later we can calculate the
        # mean training loss of this epoch
        total_loss = total_loss.cpu().detach().numpy()
        loss_segmentation = loss_segmentation.cpu().detach().numpy()
        loss_objectosphere = loss_objectosphere.cpu().detach().numpy()
        loss_ows = loss_ows.cpu().detach().numpy()
        loss_con = loss_con.cpu().detach().numpy()

        total_loss_list.append(total_loss)
        total_sem_loss.append(loss_segmentation)
        total_obj_loss.append(loss_objectosphere)
        total_ows_loss.append(loss_ows)
        total_con_loss.append(loss_con)

        if np.isnan(total_loss):
            import ipdb;ipdb.set_trace()  # fmt: skip
            raise ValueError("Loss is None")

        # print log
        samples_of_epoch += batch_size
        time_inter = time.time() - start_time_for_one_step

        learning_rates = lr_scheduler.get_lr()

        print_log(
            epoch,
            samples_of_epoch,
            batch_size,
            len(train_loader.dataset),
            total_loss,
            time_inter,
            learning_rates,
        )

        if debug_mode:
            # only one batch while debugging
            break

    # fill the logs for csv log file and web logger
    writer.add_scalar("Loss/train", np.mean(total_loss_list), epoch)
    writer.add_scalar("Loss/semantic", np.mean(total_sem_loss), epoch)
    writer.add_scalar("Loss/objectosphere", np.mean(total_obj_loss), epoch)
    writer.add_scalar("Loss/ows", np.mean(total_ows_loss), epoch)
    writer.add_scalar("Loss/contrastive", np.mean(total_con_loss), epoch)

    if loss_mav is not None:
        mean, var = loss_mav.update()
        return mean, var
    else:
        return {}, {}


def validate(
    model,
    valid_loader,
    device,
    val_loss,
    epoch,
    writer,
    loss_function_valid_unweighted=None,
    add_log_key="",
    debug_mode=False,
    classes=19,
):
    valid_split = valid_loader.dataset.split + add_log_key

    # we want to track how long each part of the validation takes
    forward_time = 0
    copy_to_gpu_time = 0

    # set model to eval mode
    model.eval()

    # we want to store miou and ious for each camera
    miou = dict()
    ious = dict()

    loss_function_valid, loss_obj, loss_mav, loss_contrastive = val_loss

    # reset loss (of last validation) to zero
    loss_function_valid.reset_loss()

    if loss_function_valid_unweighted is not None:
        loss_function_valid_unweighted.reset_loss()

    compute_iou = IoU(
        task="multiclass", num_classes=classes, average="none", ignore_index=255
    ).to(device)

    mavs = None
    if loss_contrastive is not None:
        mavs = loss_mav.read()

    total_loss_obj = []
    total_loss_mav = []
    total_loss_con = []
    # validate each camera after another as all images of one camera have
    # the same resolution and can be resized together to the ground truth
    # segmentation size.

    for i, sample in enumerate(tqdm(valid_loader, desc="Valid step")):
        # copy the data to gpu
        image = sample["image"].to(device)

        if not device.type == "cpu":
            torch.cuda.synchronize()

        # forward pass
        with torch.no_grad():
            prediction_ss, prediction_ow = model(image)

            if not device.type == "cpu":
                torch.cuda.synchronize()

            target = sample["label"].long().cuda() - 1
            target[target == -1] = 255
            compute_iou.update(prediction_ss, target.cuda())

            # compute valid loss
            loss_function_valid.add_loss_of_batch(
                prediction_ss, sample["label"].to(device)
            )

            loss_objectosphere = torch.tensor(0)
            loss_ows = torch.tensor(0)
            loss_con = torch.tensor(0)
            if loss_obj is not None:
                target_obj = sample["label"]
                target_obj[target_obj == 16] = 255
                target_obj[target_obj == 17] = 255
                target_obj[target_obj == 18] = 255
                loss_objectosphere = loss_obj(prediction_ow, sample["label"])
            total_loss_obj.append(loss_objectosphere.cpu().detach().numpy())
            if loss_mav is not None:
                loss_ows = loss_mav(prediction_ss, target.cuda(), is_train=False)
            total_loss_mav.append(loss_ows.cpu().detach().numpy())
            if loss_contrastive is not None:
                loss_con = loss_contrastive(mavs, prediction_ow, target, epoch)
            total_loss_con.append(loss_con.cpu().detach().numpy())

            if debug_mode:
                # only one batch while debugging
                break

    ious = compute_iou.compute().detach().cpu()
    miou = ious.mean()

    total_loss = (
        loss_function_valid.compute_whole_loss()
        + np.mean(total_loss_obj)
        + np.mean(total_loss_mav)
        + np.mean(total_loss_con)
    )
    writer.add_scalar("Loss/val", total_loss, epoch)
    writer.add_scalar("Metrics/miou", miou, epoch)
    for i, iou in enumerate(ious):
        writer.add_scalar(
            "Class_metrics/iou_{}".format(i),
            torch.mean(iou),
            epoch,
        )

    return miou


def test_ow(
    model,
    test_loader,
    device,
    val_loss,
    epoch,
    writer,
    classes=19,
    mean=None,
    var=None,
):
    delta = 0.6

    # set model to eval mode
    model.eval()

    compute_iou = IoU(
        task="multiclass", num_classes=2, average="none", ignore_index=255
    ).to(device)

    _, loss_obj, loss_mav, _ = val_loss

    with open("mavs.pickle", "rb") as h1:
        mavs = pickle.load(h1)
    with open("vars.pickle", "rb") as h2:
        vars = pickle.load(h2)

    mavs = torch.vstack(tuple(mavs.values())).cpu()  # 19x19
    new_mavs = None

    for i, sample in enumerate(tqdm(test_loader, desc="Test step")):
        # copy the data to gpu
        image = sample["image"].to(device)
        label = sample["label"].to(device)

        if not device.type == "cpu":
            torch.cuda.synchronize()

        # forward pass
        with torch.no_grad():
            prediction, ow_pred = model(image)

            ows_target = label.long() - 1
            ows_target[ows_target < classes] = 0
            ows_binary_gt = ows_target.bool().long()

            s_cont = contrastive_inference(ow_pred)
            s_sem, similarity = semantic_inference(prediction, mavs, vars)
            s_unk = (s_cont + s_sem) / 2

            ows_binary_pred = (s_unk - delta).relu().bool().int()

            compute_iou.update(ows_binary_pred, ows_binary_gt)

            prediction = prediction.permute(1, 0, 2, 3)
            unk_pixels = prediction[:, :, ows_binary_pred == 0]

            tmp = torch.ones(unk_pixels.shape)
            if new_mavs is not None:
                for i in range(new_mavs.shape[0]):
                    mav = new_mavs[:, i].unsqueeze(1)
                    dist = torch.norm(unk_pixels - mav, dim=0)
                    dist = (dist < 0.5).int()
                    tmp[:, dist == 1] = 0
                    upd = torch.mean(unk_pixels[dist == 1], dim=0)
                    new_mavs[i, :] = (new_mavs[i, :] + upd) / 2
            preds = unk_pixels * tmp
            preds = torch.unique(preds, dim=1)
            if tmp.sum():
                preds = preds[:, 1:]

            clusters = ac(
                n_clusters=None, affinity="euclidean", distance_threshold=0.5
            ).fit(preds.cpu().numpy().T)
            groups = clusters.labels_

            nc = groups.max()
            for c in nc:
                new = preds[:, groups == c]
                new = torch.mean(torch.tensor(new), dim=1)
                if new_mavs is None:
                    new_mavs = new
                else:
                    new_mavs = torch.vstack((new_mavs, new))

    ious = compute_iou.compute().detach().cpu()
    writer.add_scalar("Metrics/OWS/known", ious[0], epoch)
    writer.add_scalar("Metrics/OWS/unknown", ious[1], epoch)


def contrastive_inference(predictions, radius=1.0):
    scores = F.relu(1 - torch.norm(predictions, dim=1) / radius)
    return scores


def semantic_inference(predictions, mavs, var):
    stds = torch.vstack(tuple(var.values())).cpu()  # 19x19
    d_pred = (
        predictions[:, None, ...] - mavs[None, :, :, None, None]
    )  # [8,1,19,h,w] - [1,19,19,1,1]
    d_pred_ = d_pred / (stds[None, :, :, None, None] + 1e-8)
    scores = torch.exp(-torch.einsum("bcfhw,bcfhw->bchw", d_pred_, d_pred) / 2)
    best = scores.max(dim=1)
    return 1 - best[0], best[1]


def get_optimizer(args, model):
    # set different learning rates fo different parts of the model
    # when using default parameters the whole model is trained with the same
    # learning rate
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
    else:
        raise NotImplementedError(
            "Currently only SGD and Adam as optimizers are "
            "supported. Got {}".format(args.optimizer)
        )

    print("Using {} as optimizer".format(args.optimizer))
    print(
        "\n\n=========================================================================\n\n"
    )
    return optimizer


if __name__ == "__main__":
    train_main()
