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

from dataset.PennFudanDataset import PennFudanDataset
from dataset.utils.PennFudan import pennfudan_utils

from dataset.UMDDataset import BasicDataSet
from dataset.utils.UMD import umd_utils

from model.utils.bbox_utils import AnchorGenerator

from utils import helper_utils

######################
######################

def main():

    # PennFudanDataset
    # dataset = PennFudanDataset(config.ROOT_DATASET_PATH)

    # COCO
    dataset = COCODataSet(dataset_dir='/data/Akeaveny/Datasets/COCO/',
                          split='val2017')
    # UMD
    # dataset = BasicDataSet(
    #                     ### REAL
    #                     dataset_dir=config.DATA_DIRECTORY_TRAIN,
    #                     mean=config.IMG_MEAN,
    #                     std=config.IMG_STD,
    #                     resize=config.RESIZE,
    #                     crop_size=config.INPUT_SIZE,
    #                     ### EXTENDING DATASET
    #                     extend_dataset=False,
    #                     max_iters=1000,
    #                     ### IMGAUG
    #                     apply_imgaug=False)

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for i, (image, target) in enumerate(data_loader):
        print()

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
        helper_utils.print_class_labels(mask)
        cv2.imshow('mask', mask)

        _labels = np.unique(mask)[1:]
        for _label in _labels:
            from dataset.utils.COCO import coco_utils
            print(coco_utils.object_id_to_name(_label))

        # color_mask = umd_utils.colorize_mask(mask)
        # cv2.imshow('mask_color', cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))

        # gt_color_mask = umd_utils.colorize_mask(target['gt_mask'])
        # cv2.imshow('gt_color', cv2.cvtColor(gt_color_mask, cv2.COLOR_BGR2RGB))

        #######################
        ### Anchors
        #######################
        # anchor_img = helper_utils.visualize_anchors(image=image)
        # cv2.imshow('Anchors', cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB))

        #######################
        #######################
        print()
        cv2.waitKey(1000)

if __name__ == "__main__":
    main()

