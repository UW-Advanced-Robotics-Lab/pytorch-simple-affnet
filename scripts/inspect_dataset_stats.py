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

from dataset.COCODataset import COCODataSet

from dataset.UMDDataset import UMDDataSet
from dataset.utils.UMD import umd_utils

from dataset.ElevatorDataset import ElevatorDataSet
from dataset.utils.Elevator import elevator_utils

from model.utils.bbox_utils import AnchorGenerator

from utils import helper_utils

######################
######################

def main():

    ######################
    # COCO
    ######################
    # dataset = COCODataSet(dataset_dir='/data/Akeaveny/Datasets/COCO/',
    #                       split='val2017')
    #
    # np.random.seed(config.RANDOM_SEED)
    # total_idx = np.arange(0, len(dataset), 1)
    # test_idx = np.random.choice(total_idx, size=int(20), replace=False)
    # dataset = torch.utils.data.Subset(dataset, test_idx)

    # ######################
    # # UMD
    # ######################
    # dataset = UMDDataSet(
    #                     ### REAL
    #                     dataset_dir=config.DATA_DIRECTORY_TRAIN,
    #                     mean=config.IMAGE_MEAN,
    #                     std=config.IMAGE_STD,
    #                     resize=config.RESIZE,
    #                     crop_size=config.INPUT_SIZE,
    #                     ### EXTENDING DATASET
    #                     extend_dataset=False,
    #                     max_iters=100,
    #                     ### IMGAUG
    #                     apply_imgaug=False)

    ######################
    # Elevator
    ######################
    dataset = ElevatorDataSet(
                        ### REAL
                        dataset_dir=config.DATA_DIRECTORY_TRAIN,
                        mean=config.IMAGE_MEAN,
                        std=config.IMAGE_STD,
                        resize=config.RESIZE,
                        crop_size=config.INPUT_SIZE,
                        ### EXTENDING DATASET
                        extend_dataset=False,
                        max_iters=100,
                        ### IMGAUG
                        apply_imgaug=False)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for i, (image, target) in enumerate(data_loader):

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
        # helper_utils.print_class_labels(mask)
        # cv2.imshow('mask', mask)

        helper_utils.print_class_labels(mask)
        color_mask = elevator_utils.colorize_obj_mask(mask)
        cv2.imshow('mask_color', cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))

        helper_utils.print_class_labels(target['gt_mask'])
        gt_color_mask = elevator_utils.colorize_obj_mask(target['gt_mask'])
        cv2.imshow('gt_color', cv2.cvtColor(gt_color_mask, cv2.COLOR_BGR2RGB))

        #######################
        ### Anchors
        #######################
        # anchor_img = helper_utils.visualize_anchors(image=image)
        # cv2.imshow('Anchors', cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB))

        #######################
        #######################
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

