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
import torch.nn.functional as F

######################
######################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

######################
######################

from model.utils import bbox_utils

from dataset.utils import pennfudan_utils
from dataset.utils import umd_utils

######################
# IMG UTILS
######################

def print_depth_info(depth):
    depth = np.array(depth)
    print(f"Depth of type:{depth.dtype} has min:{np.min(depth)} & max:{np.max(depth)}")

def print_class_labels(seg_mask):
    class_ids = np.unique(np.array(seg_mask, dtype=np.uint8))
    print(f"Mask has {len(class_ids)-1} Labels: {class_ids[1:]}")

######################
# FORMAT UTILS
######################

def format_target_data(image, target):
    height, width = image.shape[:2]

    target['labels'] = np.array(target['labels'], dtype=np.int32).flatten()
    target['boxes'] = np.array(target['boxes'], dtype=np.int32).reshape(-1, 4)
    target['aff_labels'] = np.array(target['aff_labels'], dtype=np.int32).flatten()
    target['masks'] = np.array(target['masks'], dtype=np.uint8).reshape(-1, height, width)

    target['gt_mask'] = np.array(target['gt_mask'], dtype=np.uint8).reshape(height, width)

    return target

def format_label(label):
    return np.array(label, dtype=np.int32)

def format_bbox(bbox):
    return np.array(bbox, dtype=np.int32).flatten()

######################
# DATASET UTILS
######################

def numpy_2_torch(numpy_img, mean=config.IMG_MEAN, std=config.IMG_STD,
                  is_rgb=False, is_depth=False):
    torch_img = np.asarray(numpy_img, np.float32)

    if is_rgb:
        torch_img = torch_img[:, :, ::-1]               # change to BGR
        torch_img = (torch_img - np.array(mean[0:-1], dtype=np.float32)) / np.array(std[0:-1], dtype=np.float32)
        torch_img = torch_img.transpose((2, 0, 1))      # images are represented as [C, H, W] in torch

    if is_depth:
        torch_img = torch_img[np.newaxis, :, :]
        mean_ = np.array(mean[-1], dtype=np.float32)
        std_  = np.array(std[-1], dtype=np.float32)
        torch_img = (torch_img - mean_) / std_

    return torch_img

def torch_2_numpy(torch_img, mean=config.IMG_MEAN, std=config.IMG_STD,
                  is_rgb=False, is_depth=False):

        numpy_img = np.squeeze(np.array(torch_img, dtype=np.float32))

        if is_rgb:
            numpy_img = np.transpose(numpy_img, (1, 2, 0))  # images are represented as [C, H W] in torch
            numpy_img = (numpy_img * np.array(std[0:-1], dtype=np.float32)) + np.array(mean[0:-1], dtype=np.float32)
            numpy_img = numpy_img[:, :, ::-1]               # change to BGR

        if is_depth:
            mean_ = np.array(mean[-1], dtype=np.float32)
            std_ = np.array(std[-1], dtype=np.float32)
            numpy_img = (numpy_img * std_) + mean_

        return np.array(numpy_img, dtype=np.uint8)

def cuda_2_numpy(cuda_img, mean=config.IMG_MEAN, is_rgb=False, is_pred=False):
    numpy_img = cuda_img.squeeze().cpu().detach().numpy()

    if is_rgb:
        numpy_img = np.transpose(numpy_img, (1, 2, 0))  # images are represented as [C, H W] in torch
        numpy_img += np.array(mean[0:-1], dtype=np.uint8)
        numpy_img = numpy_img[:, :, ::-1]  # change to BGR

    if is_pred:
        numpy_img = np.transpose(numpy_img, (1, 2, 0))
        numpy_img = np.asarray(np.argmax(numpy_img, axis=2), dtype=np.uint8)

        # probs = F.softmax(cuda_img, dim=1)
        # numpy_img = probs.squeeze().cpu().detach().numpy()
        # numpy_img = [numpy_img[c, :, :] > config.CONFIDENCE_THRESHOLD for c in range(numpy_img.shape[0])]
        # numpy_img = np.asarray(np.argmax(np.asarray(numpy_img), axis=0), dtype=np.uint8)

    return np.array(numpy_img, dtype=np.uint8)

def cuda_img_2_tensorboard(cuda_img, is_depth=False):

    img = cuda_img.cpu().detach().squeeze()
    # now format for to [BS, C, H W] in tensorboard
    if is_depth:
        return np.array(img)[np.newaxis, np.newaxis, :, :]
    else:
        return np.array(img)[np.newaxis, :, :, :]

def cuda_label_2_tensorboard(cuda_label, is_pred=False):

    colour_label = cuda_label.cpu().detach().squeeze()
    colour_label = np.array(colour_label)
    if is_pred:
        colour_label = np.transpose(colour_label, (1, 2, 0))
        colour_label = np.asarray(np.argmax(colour_label, axis=2), dtype=np.uint8)
    colour_label = np.array(colour_label, dtype=np.uint8)
    # colour_label = colorize_mask(colour_label)
    # now format for to [BS, C, H W] in tensorboard
    return np.transpose(colour_label, (2, 0, 1))[np.newaxis, :, :, :]

######################
# ANCHOR UTILS
######################

def visualize_anchors(image):
    anchor_img = image.copy()

    # prelim
    image_shape = image.shape[:2]
    resent_50_features = torch.randn(1, 256, 4, 4, device='cpu')

    # rpn
    rpn_anchor_generator = bbox_utils.AnchorGenerator(config.ANCHOR_SIZES, config.ANCHOR_RATIOS)
    anchors = rpn_anchor_generator(resent_50_features, image_shape)

    for anchor in anchors:
        anchor = np.array(anchor, dtype=np.int32)
        # clip anchors
        height, width = image.shape[:2]
        anchor[0] = np.max([0, anchor[0]])  # x1
        anchor[1] = np.max([0, anchor[1]])  # y1
        anchor[2] = np.min([width, anchor[2]])  # x2
        anchor[3] = np.min([height, anchor[3]])  # y2
        # x1,y1 ------
        # |          |
        # |          |
        # |          |
        # --------x2,y2
        _color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        anchor_img = cv2.rectangle(anchor_img, (anchor[0], anchor[1]), (anchor[2], anchor[3]), _color, 1)
    return anchor_img

######################
# MaskRCNN UTILS
######################

def draw_bbox_on_img(image, labels, boxes, scores=None, is_gt=False):
    bbox_img = image.copy()

    if is_gt:
        for label, bbox in zip(labels, boxes):
            bbox = format_bbox(bbox)
            # x1,y1 ------
            # |          |
            # |          |
            # |          |
            # --------x2,y2
            bbox_img = cv2.rectangle(bbox_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, 1)

            cv2.putText(bbox_img,
                        umd_utils.object_id_to_name(label),
                        # umd_utils.aff_id_to_name(label),
                        (bbox[0], bbox[1] - 5),
                        cv2.FONT_ITALIC,
                        0.4,
                        (255, 255, 255))

    else:
        for idx, score in enumerate(scores):
            if score > config.CONFIDENCE_THRESHOLD:
                bbox = format_bbox(boxes[idx])
                bbox_img = cv2.rectangle(bbox_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, 1)

                label = labels[idx]
                cv2.putText(bbox_img,
                            umd_utils.object_id_to_name(label),
                            # umd_utils.aff_id_to_name(label),
                            (bbox[0], bbox[1] - 5),
                            cv2.FONT_ITALIC,
                            0.4,
                            (255, 255, 255))

    return bbox_img

def get_segmentation_masks(image, labels, binary_masks, scores=None, is_gt=False):

    height, width = image.shape[:2]
    # print(f'height:{height}, width:{width}')

    instance_masks = np.zeros((height, width), dtype=np.uint8)
    instance_mask_one = np.ones((height, width), dtype=np.uint8)

    if len(binary_masks.shape) == 2:
        binary_masks = binary_masks[np.newaxis, :, :]

    if is_gt:
        for idx, label in enumerate(labels):
            binary_mask = binary_masks[idx, :, :]

            instance_mask = instance_mask_one * label
            instance_masks = np.where(binary_mask, instance_mask, instance_masks).astype(np.uint8)

    else:
        for idx, score in enumerate(scores):
            # if score > config.CONFIDENCE_THRESHOLD:
            label = labels[idx]
            binary_mask = np.array(binary_masks[idx, :, :], dtype=np.uint8)
            # print_class_labels(binary_mask)

            instance_mask = instance_mask_one * label
            instance_masks = np.where(binary_mask, instance_mask, instance_masks).astype(np.uint8)

    # print_class_labels(instance_masks)
    return instance_masks

