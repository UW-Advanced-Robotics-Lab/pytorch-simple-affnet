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

import cfg as config

######################
######################

from model.utils import bbox_utils

from dataset.utils.COCO import coco_utils
from dataset.utils.UMD import umd_utils
from dataset.utils.Elevator import elevator_utils
from dataset.utils.ARLVicon import arl_vicon_dataset_utils
from dataset.utils.ARLAffPose import affpose_dataset_utils

######################
# IMG UTILS
######################

def convert_16_bit_depth_to_8_bit(depth):
    depth = np.array(depth, np.uint16)
    depth = depth / np.max(depth) * (2 ** 8 - 1)
    return np.array(depth, np.uint8)

def print_depth_info(depth):
    depth = np.array(depth)
    print(f"Depth of type:{depth.dtype} has min:{np.min(depth)} & max:{np.max(depth)}")

def print_class_labels(seg_mask):
    class_ids = np.unique(np.array(seg_mask, dtype=np.uint8))
    print(f"Mask has {len(class_ids)-1} Labels: {class_ids[1:]}")

######################
######################

def crop(pil_img, crop_size, is_img=False):
    _dtype = np.array(pil_img).dtype
    pil_img = Image.fromarray(pil_img)
    crop_w, crop_h = crop_size
    img_width, img_height = pil_img.size
    left, right = (img_width - crop_w) / 2, (img_width + crop_w) / 2
    top, bottom = (img_height - crop_h) / 2, (img_height + crop_h) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    # pil_img = pil_img.crop((left, top, right, bottom)).resize((crop_w, crop_h))
    pil_img = pil_img.crop((left, top, right, bottom))
    ###
    if is_img:
        img_channels = np.array(pil_img).shape[-1]
        img_channels = 3 if img_channels == 4 else img_channels
        resize_img = np.zeros((crop_h, crop_w, img_channels))
        resize_img[0:(bottom - top), 0:(right - left), :img_channels] = np.array(pil_img)[..., :img_channels]
    else:
        resize_img = np.zeros((crop_h, crop_w))
        resize_img[0:(bottom - top), 0:(right - left)] = np.array(pil_img)

    return np.array(resize_img, dtype=_dtype)

######################
# FORMAT UTILS
######################

def format_target_data(image, target):
    height, width = image.shape[:2]

    target['gt_mask'] = np.array(target['gt_mask'], dtype=np.uint8).reshape(height, width)
    target['labels'] = np.array(target['labels'], dtype=np.int32).flatten()
    target['boxes'] = np.array(target['boxes'], dtype=np.int32).reshape(-1, 4)
    target['masks'] = np.array(target['masks'], dtype=np.uint8).reshape(-1, height, width)

    if 'obj_labels' in target.keys():
        target['obj_labels'] = np.array(target['obj_labels'], dtype=np.int32).flatten()

    if 'aff_labels' in target.keys():
        target['aff_labels'] = np.array(target['aff_labels'], dtype=np.int32).flatten()

    return target

def format_label(label):
    return np.array(label, dtype=np.int32)

def format_bbox(bbox):
    return np.array(bbox, dtype=np.int32).flatten()

######################
# ANCHOR UTILS
######################

def visualize_anchors(image):
    anchor_img = image.copy()

    # prelim
    image_shape = image.shape[:2]
    stride_length = 16
    resent_50_features = torch.randn(1, 256,
                                     int(image_shape[0]/stride_length),
                                     int(image_shape[1]/stride_length),
                                     device='cpu')

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

        # clip anchors
        x1 = height // 2 - np.abs(anchor[0])
        y1 = width // 2 - np.abs(anchor[1])
        x2 = height // 2 + np.abs(anchor[2])
        y2 = width // 2 + np.abs(anchor[3])
        # x1,y1 ------
        # |          |
        # |          |
        # |          |
        # --------x2,y2
        _color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        anchor_img = cv2.rectangle(anchor_img, (x1, y1), (x2, y2), _color, 1)
        print(f'x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}')
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
                        # coco_utils.object_id_to_name(label),
                        # umd_utils.object_id_to_name(label),
                        # umd_utils.aff_id_to_name(label),
                        # elevator_utils.object_id_to_name(label),
                        # arl_vicon_dataset_utils.map_obj_id_to_name(label),
                        affpose_dataset_utils.map_obj_id_to_name(label),
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
                            # coco_utils.object_id_to_name(label),
                            # umd_utils.object_id_to_name(label),
                            # umd_utils.aff_id_to_name(label),
                            # elevator_utils.object_id_to_name(label),
                            # arl_vicon_dataset_utils.map_obj_id_to_name(label),
                            affpose_dataset_utils.map_obj_id_to_name(label),
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
        for idx, label in enumerate(labels):
            label = labels[idx]
            binary_mask = np.array(binary_masks[idx, :, :], dtype=np.uint8)
            # print(f'binary_mask: label:{label}, data:{np.unique(binary_mask, return_counts=True)}')
            # print_class_labels(binary_mask)

            instance_mask = instance_mask_one * label
            instance_masks = np.where(binary_mask, instance_mask, instance_masks).astype(np.uint8)

    # print_class_labels(instance_masks)
    return instance_masks

