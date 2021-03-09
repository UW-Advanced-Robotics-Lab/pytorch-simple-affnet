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

import torch
from torch.utils import data
from torch.utils.data import Dataset

######################
######################

# from pathlib import Path
# ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

######################
######################

from dataset.PennFudanDataset import PennFudanDataset
from dataset.utils import pennfudan_utils

from dataset.UMDDataset import BasicDataSet
from dataset.utils import umd_utils

from model.utils.bbox_utils import AnchorGenerator

from utils import helper_utils

######################
######################

def main():

    # use our dataset and defined transformations
    # dataset = PennFudanDataset(config.ROOT_DATASET_PATH)

    dataset = BasicDataSet(
                        ### REAL
                        dataset_dir=config.DATA_DIRECTORY_TARGET_TRAIN,
                        mean=config.IMG_MEAN_TARGET,
                        std=config.IMG_STD_TARGET,
                        resize=config.RESIZE_TARGET,
                        crop_size=config.INPUT_SIZE_TARGET,
                        ### EXTENDING DATASET
                        extend_dataset=False,
                        max_iters=1000,
                        ### IMGAUG
                        apply_imgaug=False)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for i, (image, target) in enumerate(data_loader):

        # print(f'\ngt_label:{target["labels"].size()}, gt_aff_label:{target["aff_labels"].size()}')
        # print(f'gt_label:{target["labels"]}, gt_aff_label:{target["aff_labels"]}')

        #######################
        ### formatting data
        #######################
        image = np.squeeze(np.array(image, dtype=np.uint8))
        target = helper_utils.format_target_data(image, target)

        #######################
        ### images
        #######################
        cv2.imshow('rgb', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        #######################
        ### bbox
        #######################
        bbox_img = helper_utils.draw_bbox_on_img(image=image,
                                                 labels=target['labels'],
                                                 boxes=target['boxes'],
                                                 is_gt=True)
        cv2.imshow('bbox', cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))

        #######################
        ### masks
        #######################
        mask = helper_utils.get_segmentation_masks(image=image,
                                                   labels=target['labels'],
                                                   binary_masks=target['masks'],
                                                   is_gt=True)
        color_mask = umd_utils.colorize_mask(mask)
        cv2.imshow('mask', color_mask)

        gt_color_mask = umd_utils.colorize_mask(target['gt_mask'])
        cv2.imshow('gt', gt_color_mask)

        #######################
        ### Anchors
        #######################
        # anchor_img = helper_utils.visualize_anchors(image=image)
        # cv2.imshow('Anchors', cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB))

        #######################
        #######################
        print()
        cv2.waitKey(0)

if __name__ == "__main__":
    main()

