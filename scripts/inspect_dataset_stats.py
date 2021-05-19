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

import sys
sys.path.append('../')

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

    ######################
    # UMD
    ######################
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
    # dataset = ARLViconDataSet(
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
    # ARL AffPose
    ######################
    dataset = ARLAffPoseDataSet(
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
    ######################

    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(dataset), 1)
    test_idx = np.random.choice(total_idx, size=int(1000), replace=False)
    dataset = torch.utils.data.Subset(dataset, test_idx)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    print(f'Selecting {len(data_loader)} immages ..')

    ######################
    # todo (stats):
    ######################
    rgb_mean, rgb_std = 0, 0
    depth_mean, depth_std = 0, 0
    nb_bins = int(2 ** 8)
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)
    count_d = np.zeros(nb_bins)

    for i, (image, target) in enumerate(data_loader):

        #######################
        # formatting data
        #######################
        image = np.squeeze(np.array(image, dtype=np.uint8))
        target = helper_utils.format_target_data(image, target)
        if i % 100 == 0:
            print(f'{i}/{len(data_loader)} ..')

        # #######################
        # # todo (stats): img mean and std
        # #######################
        # # rgb
        # img_stats = image
        # img_stats = img_stats.reshape(3, -1)
        # rgb_mean += np.mean(img_stats, axis=1)
        # rgb_std += np.std(img_stats, axis=1)
        # # depth
        # # img_stats = depths
        # # img_stats = img_stats.reshape(1, -1)
        # # depth_mean += np.mean(img_stats, axis=1)
        # # depth_std += np.std(img_stats, axis=1)
        # #######################
        # # todo (stats): histogram
        # #######################
        # # rgb
        # hist_r = np.histogram(image[0], bins=nb_bins, range=[0, 255])
        # hist_g = np.histogram(image[1], bins=nb_bins, range=[0, 255])
        # hist_b = np.histogram(image[2], bins=nb_bins, range=[0, 255])
        # count_r += hist_r[0]
        # count_g += hist_g[0]
        # count_b += hist_b[0]
        # # depth
        # # hist_d = np.histogram(depths, bins=nb_bins, range=[0, 255])
        # # count_d += hist_d[0]

        #######################
        ### images
        #######################
        # cv2.imshow('rgb', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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
                                                   # labels=target['labels'],
                                                   labels=target['aff_labels'],
                                                   binary_masks=target['masks'],
                                                   is_gt=True)

        print('')
        helper_utils.print_class_labels(mask)
        # color_mask = umd_utils.colorize_aff_mask(mask)
        # color_mask = arl_vicon_dataset_utils.colorize_obj_mask(mask)
        # color_mask = affpose_dataset_utils.colorize_obj_mask(mask)
        color_mask = affpose_dataset_utils.colorize_aff_mask(mask)
        color_mask = cv2.addWeighted(bbox_img, 0.35, color_mask, 0.65, 0)
        cv2.imshow('mask_color', cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))

        #######################
        ### gt mask (i.e. not using polygons)
        #######################

        helper_utils.print_class_labels(target['gt_mask'])
        # gt_color_mask = umd_utils.colorize_aff_mask(target['gt_mask'])
        # gt_color_mask = arl_vicon_dataset_utils.colorize_obj_mask(target['gt_mask'])
        # gt_color_mask = affpose_dataset_utils.colorize_obj_mask(target['gt_mask'])
        gt_color_mask = affpose_dataset_utils.colorize_aff_mask(target['gt_mask'])
        gt_color_mask = cv2.addWeighted(bbox_img, 0.35, gt_color_mask, 0.65, 0)
        cv2.imshow('gt_color', cv2.cvtColor(gt_color_mask, cv2.COLOR_BGR2RGB))

        ### helper_utils.print_class_obj_names(target['obj_labels'])
        ### helper_utils.print_class_aff_names(target['aff_labels'])

        # ######################
        # ## Anchors
        # ######################
        # anchor_img = helper_utils.visualize_anchors(image=image)
        # cv2.imshow('Anchors', cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB))

        ######################
        ######################
        cv2.waitKey(0)

    # #######################
    # # todo (stats):
    # #######################
    # rgb_mean /= i
    # rgb_std /= i
    # print(f'\nRGB: mean:{rgb_mean}\nstd:{rgb_std}')
    # depth_mean /= i
    # depth_std /= i
    # print(f'Depth: mean:{depth_mean}\nstd:{depth_std}')
    # #######################
    # #######################
    # bins = hist_r[1]
    # ### rgb
    # plt.figure(figsize=(12, 6))
    # plt.bar(hist_r[1][:-1], count_r, color='r', label='Red', alpha=0.33)
    # plt.axvline(x=rgb_mean[0], color='r', ls='--')
    # plt.bar(hist_g[1][:-1], count_g, color='g', label='Green', alpha=0.33)
    # plt.axvline(x=rgb_mean[1], color='g', ls='--')
    # plt.bar(hist_b[1][:-1], count_b, color='b', label='Blue', alpha=0.33)
    # plt.axvline(x=rgb_mean[2], color='b', ls='--')
    # plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    # plt.show()
    ### depth
    # plt.figure(figsize=(12, 6))
    # plt.bar(x=hist_d[1][:-1], height=count_d, color='k', label='depth', alpha=0.33)
    # plt.axvline(x=depth_mean, color='k', ls='--')
    # plt.axvline(x=rgb_mean[0], color='r', ls='--')
    # plt.axvline(x=rgb_mean[1], color='g', ls='--')
    # plt.axvline(x=rgb_mean[2], color='b', ls='--')
    # plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    # plt.show()

if __name__ == "__main__":
    main()

