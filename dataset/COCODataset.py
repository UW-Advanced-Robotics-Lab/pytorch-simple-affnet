import os
from os import listdir
from os.path import splitext
from glob import glob

import copy

import logging

import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

from skimage import io
import skimage.transform
from skimage.util import crop

import torch
from torch.utils import data
from torch.utils.data import Dataset
import torchvision

import utils.vision.transforms as T

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

######################
######################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

from utils import helper_utils

from dataset.utils.UMD import umd_bbox_utils
from dataset.utils.UMD import umd_coco_utils
from dataset.utils.UMD import umd_utils

######################
######################

class COCODataSet(data.Dataset):

    def __init__(self,
                 dataset_dir,
                 split,
                 ###
                 is_train=False,
                 is_eval=False,
                 ):

        from pycocotools.coco import COCO

        self.dataset_dir = dataset_dir
        self.split = split
        ###
        self.is_train = is_train
        self.is_eval = is_eval
        self.transform = self.get_transform()

        ################################
        # FORMAT COCO
        ################################

        ann_file = os.path.join(self.dataset_dir, "annotations/instances_{}.json".format(split))
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]

        self._classes = {k: v["name"] for k, v in self.coco.cats.items()}
        self.classes = tuple(self.coco.cats[k]["name"] for k in sorted(self.coco.cats))
        # resutls' labels convert to annotation labels
        self.ann_labels = {self.classes.index(v): k for k, v in self._classes.items()}

        checked_id_file = os.path.join(self.dataset_dir, "checked_{}.txt".format(split))
        if is_train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def convert_to_xyxy(box):  # box format: (xmin, ymin, w, h)
        new_box = torch.zeros_like(box)
        new_box[:, 0] = box[:, 0]
        new_box[:, 1] = box[:, 1]
        new_box[:, 2] = box[:, 0] + box[:, 2]
        new_box[:, 3] = box[:, 1] + box[:, 3]
        return new_box  # new_box format: (xmin, ymin, xmax, ymax)


    def get_transform(self):
        transforms = []
        transforms.append(T.ToTensor())
        return T.Compose(transforms)

    def __getitem__(self, index):

        img_id = self.ids[index]

        ##################
        ### GET IMAGE
        ##################
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.dataset_dir, "{}".format(self.split), img_info["file_name"]))

        ##################
        ### GET TARGET
        ##################
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                name = self._classes[ann["category_id"]]
                labels.append(self.classes.index(name))
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels)
            masks = torch.tensor(masks)

        # print(len(boxes))
        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)

        ##################
        ### SEND TO TORCH
        ##################

        if self.is_train or self.is_eval:
            image, target = self.transform(image, target)
        else:
            image = np.array(image, dtype=np.uint8)

        return image, target