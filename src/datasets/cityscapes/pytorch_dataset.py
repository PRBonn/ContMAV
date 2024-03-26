########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import os
import glob

import numpy as np
import cv2

from ..dataset_base import DatasetBase
from .cityscapes import CityscapesBase


class Cityscapes(CityscapesBase, DatasetBase):
    def __init__(
        self,
        data_dir=None,
        n_classes=19,
        split="train",
        with_input_orig=False,
        overfit=False,
        classes=19,
    ):
        super(Cityscapes, self).__init__()
        assert split in self.SPLITS
        assert n_classes in self.N_CLASSES
        print(split)
        self._n_classes = classes
        self._split = split
        self._with_input_orig = with_input_orig
        self._cameras = ["camera1"]  # just a dummy camera name
        self.overfit = overfit

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir)
            self._data_dir = data_dir

            # load file lists
            if self.overfit:
                self.images_path = os.path.join(data_dir, "leftImg8bit", "val")
                self.labels_path = os.path.join(data_dir, "gtFine", "val")
            else:
                self.images_path = os.path.join(data_dir, "leftImg8bit", split)
                self.labels_path = os.path.join(data_dir, "gtFine", split)
            # self._files = {
            #     "rgb": _loadtxt(f"{self._split}_rgb.txt"),
            #     "label": _loadtxt(f"{self._split}_labels_{self._n_classes}.txt"),
            # }
            # assert all(len(l) == len(self._files["rgb"]) for l in self._files.values())
            self.images = []
            self.labels = []

            for filename in glob.iglob(self.images_path + "/**/*.*", recursive=True):
                self.images.append(filename)
            for filename in glob.iglob(
                self.labels_path + "/**/*labelTrainIds.png", recursive=True
            ):
                self.labels.append(filename)
            self.images.sort()
            self.labels.sort()

            self._files = {}

        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")
        # class names, class colors, and label directory
        if self._n_classes == 19:
            self._class_names = self.CLASS_NAMES_REDUCED
            self._class_colors = np.array(self.CLASS_COLORS_REDUCED, dtype="uint8")
            self._label_dir = self.LABELS_REDUCED_DIR
        else:
            self._class_names = self.CLASS_NAMES_FULL
            self._class_colors = np.array(self.CLASS_COLORS_FULL, dtype="uint8")
            self._label_dir = self.LABELS_FULL_DIR

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def _load(self, filename):
        # all the other files are pngs
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def load_name(self, idx):
        return self.images[idx]

    def load_image(self, idx):
        return self._load(self.images[idx])

    def load_label(self, idx):
        label = self._load(self.labels[idx]) + 1
        return label

    def __len__(self):
        if self.overfit:
            return 2
        return 40  # len(self.images)
