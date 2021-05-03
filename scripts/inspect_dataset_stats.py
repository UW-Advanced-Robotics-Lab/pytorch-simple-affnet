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

import cfg as config

######################
######################

from dataset.COCODataset import COCODataSet

from dataset.UMDDataset import UMDDataSet
from dataset.utils.UMD import umd_utils

from dataset.ElevatorDataset import ElevatorDataSet
from dataset.utils.Elevator import elevator_utils

from dataset.ARLViconDataset import ARLViconDataSet
from dataset.utils.ARLVicon import arl_vicon_dataset_utils

from dataset.ARLAffPoseDataset import ARLAffPoseDataSet
from dataset.utils.ARLAffPose import affpose_dataset_utils

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

    # ######################
    # # UMD
    # ######################
    # dataset = UMDDataSet(
    #                     ### REAL
    #                     dataset_dir=config.DATA_DIRECTORY_TRAIN,
    #                     mean=config.IMAGE_MEAN,
    #                     std=config.IMAGE_STD,
    #                     resize=config.RESIZE,
    #                     crop_size=config.CROP_SIZE,
    #                     ### EXTENDING DATASET
    #                     extend_dataset=False,
    #                     max_iters=100,
    #                     ### IMGAUG
    #                     apply_imgaug=False)

    ######################
    # Elevator
    ######################
    # dataset = ElevatorDataSet(
    #                     ### REAL
    #                     dataset_dir=config.DATA_DIRECTORY_TRAIN,
    #                     mean=config.IMAGE_MEAN,
    #                     std=config.IMAGE_STD,
    #                     resize=config.RESIZE,
    #                     crop_size=config.CROP_SIZE,
    #                     ### EXTENDING DATASET
    #                     extend_dataset=False,
    #                     max_iters=100,
    #                     ### IMGAUG
    #                     apply_imgaug=False)

    ######################
    # ARL Vicon
    ######################
    dataset = ARLViconDataSet(
                        ### REAL
                        dataset_dir=config.DATA_DIRECTORY_TRAIN,
                        mean=config.IMAGE_MEAN,
                        std=config.IMAGE_STD,
                        resize=config.RESIZE,
                        crop_size=config.CROP_SIZE,
                        ### EXTENDING DATASET
                        extend_dataset=False,
                        max_iters=100,
                        ### IMGAUG
                        apply_imgaug=False)

    ######################
    # ARL Vicon
    ######################
    # dataset = ARLAffPoseDataSet(
    #                     ### REAL
    #                     dataset_dir=config.DATA_DIRECTORY_TRAIN,
    #                     mean=config.IMAGE_MEAN,
    #                     std=config.IMAGE_STD,
    #                     resize=config.RESIZE,
    #                     crop_size=config.CROP_SIZE,
    #                     ### EXTENDING DATASET
    #                     extend_dataset=False,
    #                     max_iters=100,
    #                     ### IMGAUG
    #                     apply_imgaug=False)

    ######################
    ######################

    # np.random.seed(config.RANDOM_SEED)
    # total_idx = np.arange(0, len(dataset), 1)
    # test_idx = np.random.choice(total_idx, size=int(20), replace=False)
    # dataset = torch.utils.data.Subset(dataset, test_idx)

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
                                                 labels=target['obj_labels'],
                                                 boxes=target['boxes'],
                                                 is_gt=True)
        cv2.imshow('bbox', cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))

        #######################
        ### masks
        #######################
        mask = helper_utils.get_segmentation_masks(image=image,
                                                   labels=target['obj_labels'],
                                                   binary_masks=target['masks'],
                                                   is_gt=True)

        helper_utils.print_class_labels(mask)
        color_mask = arl_vicon_dataset_utils.colorize_obj_mask(mask)
        color_mask = cv2.addWeighted(bbox_img, 0.35, color_mask, 0.65, 0)
        cv2.imshow('mask_color', cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))

        helper_utils.print_class_labels(target['gt_mask'])
        gt_color_mask = arl_vicon_dataset_utils.colorize_obj_mask(target['gt_mask'])
        gt_color_mask = cv2.addWeighted(bbox_img, 0.35, gt_color_mask, 0.65, 0)
        cv2.imshow('gt_color', cv2.cvtColor(gt_color_mask, cv2.COLOR_BGR2RGB))

        #######################
        ### Anchors
        #######################
        # anchor_img = helper_utils.visualize_anchors(image=image)
        # cv2.imshow('Anchors', cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB))

        #######################
        #######################
        cv2.waitKey(0)

if __name__ == "__main__":
    main()

