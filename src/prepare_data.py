########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


from torch.utils.data import DataLoader

from src import preprocessing
from src.datasets import Cityscapes


def prepare_data(args, ckpt_dir=None, with_input_orig=False, split=None):
    train_preprocessor_kwargs = {}
    if args.dataset == "cityscapes":
        Dataset = Cityscapes
        dataset_kwargs = {"n_classes": 19}
        valid_set = "val"
    else:
        raise ValueError(f"Unknown dataset: `{args.dataset}`")
    if args.aug_scale_min != 1 or args.aug_scale_max != 1.4:
        train_preprocessor_kwargs["train_random_rescale"] = (
            args.aug_scale_min,
            args.aug_scale_max,
        )

    if split in ["valid", "val", "test"]:
        valid_set = split

    # train data
    train_data = Dataset(
        data_dir=args.dataset_dir,
        split="train",
        with_input_orig=with_input_orig,
        overfit=args.overfit,
        classes=args.num_classes,
        **dataset_kwargs,
    )

    valid_data = Dataset(
        data_dir=args.dataset_dir,
        split=valid_set,
        with_input_orig=with_input_orig,
        overfit=args.overfit,
        classes=args.num_classes,
        **dataset_kwargs,
    )

    test_data = Dataset(
        data_dir=args.dataset_dir,
        split="test",
        with_input_orig=with_input_orig,
        overfit=args.overfit,
        classes=args.num_classes,
        **dataset_kwargs,
    )

    train_preprocessor = preprocessing.get_preprocessor(
        height=args.height,
        width=args.width,
        phase="train",
        **train_preprocessor_kwargs,
    )

    train_data.preprocessor = train_preprocessor

    # valid data
    valid_preprocessor = preprocessing.get_preprocessor(
        height=args.height,
        width=args.width,
        phase="test",
    )

    if args.valid_full_res:
        valid_preprocessor_full_res = preprocessing.get_preprocessor(
            phase="test",
        )

    valid_data.preprocessor = valid_preprocessor
    test_data.preprocessor = valid_preprocessor

    if args.dataset_dir is None:
        # no path to the actual data was passed -> we cannot create dataloader,
        # return the valid dataset and preprocessor object for inference only
        if args.valid_full_res:
            return valid_data, valid_preprocessor_full_res
        else:
            return valid_data, valid_preprocessor

    if args.overfit:
        args.batch_size = 2
        args.batch_size_valid = 2

        # create the data loaders
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True,
            shuffle=False,
        )

    # create the data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )

    # for validation we can use higher batch size as activations do not
    # need to be saved for the backwards pass
    batch_size_valid = args.batch_size_valid or args.batch_size
    valid_loader = DataLoader(
        valid_data, batch_size=batch_size_valid, num_workers=args.workers, shuffle=False
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size_valid, num_workers=args.workers, shuffle=False
    )

    # count_classes(train_loader, valid_loader)

    return train_loader, valid_loader, test_loader


import torch


def count_classes(train_loader, valid_loader):
    train_count = torch.zeros(20)
    val_count = torch.zeros(20)
    for data in iter(train_loader):
        label = torch.unique(data["label"]).long()
        train_count[label] += 1
    for data in iter(valid_loader):
        label = torch.unique(data["label"]).long()
        val_count[label] += 1
    import ipdb;ipdb.set_trace()  # fmt: skip
