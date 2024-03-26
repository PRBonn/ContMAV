########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import cv2
import matplotlib
import matplotlib.colors
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def get_preprocessor(
    height=None,
    width=None,
    phase="train",
    train_random_rescale=(1.0, 1.4),
):
    assert phase in ["train", "test"]

    if phase == "train":
        transform_list = [
            RandomRescale(train_random_rescale),
            RandomCrop(crop_height=height, crop_width=width),
            RandomHSV((0.9, 1.1), (0.9, 1.1), (25, 25)),
            RandomFlip(),
            ToTensor(),
            Normalize(),
        ]

    else:
        if height is None and width is None:
            transform_list = []
        else:
            transform_list = [Rescale(height=height, width=width)]
        transform_list.extend(
            [
                ToTensor(),
                Normalize(),
            ]
        )
    transform = transforms.Compose(transform_list)
    return transform


class Rescale:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        image = sample["image"]

        image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )

        sample["image"] = image

        if "label" in sample:
            label = sample["label"]
            label = cv2.resize(
                label, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )
            sample["label"] = label

        return sample


class RandomRescale:
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        target_scale = np.random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))

        image = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )
        label = cv2.resize(
            label, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )
        sample["image"] = image
        sample["label"] = label

        return sample


class RandomCrop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.rescale = Rescale(self.crop_height, self.crop_width)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        h = image.shape[0]
        w = image.shape[1]
        if h <= self.crop_height or w <= self.crop_width:
            # simply rescale instead of random crop as image is not large enough
            sample = self.rescale(sample)
        else:
            i = np.random.randint(0, h - self.crop_height)
            j = np.random.randint(0, w - self.crop_width)
            image = image[i : i + self.crop_height, j : j + self.crop_width, :]
            label = label[i : i + self.crop_height, j : j + self.crop_width]
            sample["image"] = image
            sample["label"] = label
        return sample


class RandomHSV:
    def __init__(self, h_range, s_range, v_range):
        assert (
            isinstance(h_range, (list, tuple))
            and isinstance(s_range, (list, tuple))
            and isinstance(v_range, (list, tuple))
        )
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample["image"]
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h = img_hsv[:, :, 0]
        img_s = img_hsv[:, :, 1]
        img_v = img_hsv[:, :, 2]

        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        sample["image"] = img_new

        return sample


class RandomFlip:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()

        sample["image"] = image
        sample["label"] = label

        return sample


class Normalize:
    def __call__(self, sample):
        image = sample["image"]
        image = image / 255
        image = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image)

        sample["image"] = image

        return sample


class ToTensor:
    def __call__(self, sample):
        image = sample["image"]
        image = image.transpose((2, 0, 1))

        sample["image"] = torch.from_numpy(image).float()

        if "label" in sample:
            label = sample["label"]
            sample["label"] = torch.from_numpy(label).float()

        return sample


class MultiScaleLabel:
    def __init__(self, downsampling_rates=None):
        if downsampling_rates is None:
            self.downsampling_rates = [8, 16, 32]
        else:
            self.downsampling_rates = downsampling_rates

    def __call__(self, sample):
        label = sample["label"]

        h, w = label.shape

        sample["label_down"] = dict()

        # Nearest neighbor interpolation
        for rate in self.downsampling_rates:
            label_down = cv2.resize(
                label.numpy(), (w // rate, h // rate), interpolation=cv2.INTER_NEAREST
            )
            sample["label_down"][rate] = torch.from_numpy(label_down)

        return sample
