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
from torchvision.transforms import functional as F

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

######################
######################

import cfg as config

from utils import helper_utils
from utils import coco_utils
from utils import bbox_utils

######################
######################

class ARLViconDataSet(data.Dataset):

    def __init__(self,
                 dataset_dir,
                 real_or_syn_images='Real',
                 # FOLDER MUST BE CORRECTLY FORMATTED
                 rgb_dir='rgb/',
                 rgb_suffix='',
                 obj_masks_dir='masks_obj/',
                 obj_masks_suffix='_obj_label',
                 obj_part_masks_dir='masks_obj_part/',
                 obj_part_masks_suffix='_obj_part_labels',
                 aff_masks_dir='masks_aff/',
                 aff_masks_suffix='_aff_label',
                 depth_folder='depth/',
                 depth_suffix='_depth',
                 # EXTENDING DATASET
                 extend_dataset=False,
                 max_iters=int(250e3),
                 # TRAIN OR EVAL
                 is_train=False,
                 is_eval=False,
                 # PRE-PROCESSING
                 mean=config.IMAGE_MEAN,
                 std=config.IMAGE_STD,
                 resize=config.RESIZE,
                 crop_size=config.CROP_SIZE,
                 # IMGAUG
                 apply_imgaug=False):

        self.dataset_dir = dataset_dir
        # FOLDER MUST BE CORRECTLY FORMATTED
        self.rgb_dir = self.dataset_dir + rgb_dir
        self.rgb_suffix = rgb_suffix
        self.obj_masks_dir = self.dataset_dir + obj_masks_dir
        self.obj_masks_suffix = obj_masks_suffix
        self.obj_part_masks_dir = self.dataset_dir + obj_part_masks_dir
        self.obj_part_masks_suffix = obj_part_masks_suffix
        self.aff_masks_dir = self.dataset_dir + aff_masks_dir
        self.aff_masks_suffix = aff_masks_suffix
        self.depth_dir = self.dataset_dir + depth_folder
        self.depth_suffix = depth_suffix
        # TRAIN OR EVAL
        self.is_train = is_train
        self.is_eval = is_eval
        self.transform = self.get_transform()
        # PRE-PROCESSING
        self.mean = mean
        self.std = std
        self.RESIZE = resize
        self.CROP_SIZE = crop_size

        ################################
        # EXTENDING DATASET
        ################################
        self.extend_dataset = extend_dataset
        self.max_iters = max_iters

        self.rgb_ids = [splitext(file)[0] for file in listdir(self.rgb_dir) if not file.startswith('.')]
        self.obj_masks_ids = [splitext(file)[0] for file in listdir(self.obj_masks_dir) if not file.startswith('.')]
        self.obj_part_masks_ids = [splitext(file)[0] for file in listdir(self.obj_part_masks_dir) if not file.startswith('.')]
        self.aff_masks_ids = [splitext(file)[0] for file in listdir(self.aff_masks_dir) if not file.startswith('.')]
        self.depth_ids = [splitext(file)[0] for file in listdir(self.depth_dir) if not file.startswith('.')]
        assert(len(self.rgb_ids) == len(self.obj_masks_ids) == len(self.depth_ids))
        print(f'Dataset has {len(self.rgb_ids)} examples .. {dataset_dir}')

        # creating larger dataset
        if self.extend_dataset:
            ids = []
            total_idx = np.arange(0, len(self.rgb_ids), 1)
            for image_idx in range(self.max_iters):
                idx = np.random.choice(total_idx, size=1, replace=False)
                ids.append(self.rgb_ids[int(idx)])
            self.rgb_ids = ids
            print(f'Extended dataset has {len(self.rgb_ids)} examples')

        ################################
        # IMGAUG
        ################################
        self.apply_imgaug = apply_imgaug

        self.affine = iaa.Sequential([
            iaa.Fliplr(0.5),   # horizontally flip 50% of the images
            # iaa.Flipud(0.5), # vertical flip 50% of the images
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
        ], random_order=True)

        self.colour_aug = iaa.Sometimes(0.833, iaa.Sequential([
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))
                          ),
            # Strengthen or weaken the contrast in each image.
            iaa.contrast.LinearContrast((0.75, 1.25)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
        ], random_order=True))  # apply augmenters in random order

        self.depth_aug = iaa.Sometimes(0.833, iaa.Sequential([
            iaa.CoarseDropout(5e-4, size_percent=0.5),
            iaa.SaltAndPepper(5e-4),
        ], random_order=True))  # apply augmenters in random order

    def __len__(self):
        return len(self.rgb_ids)

    def apply_imgaug_to_imgs(self, rgb, mask, depth=None):
        rgb, mask, depth = np.array(rgb), np.array(mask), np.array(depth)

        H, W, C = rgb.shape[0], rgb.shape[1], rgb.shape[2]

        concat_img = np.zeros(shape=(H, W, C + 1))
        concat_img[:, :, :C] = rgb
        concat_img[:, :, -1] = depth[:, :]
        concat_img = np.array(concat_img, dtype=np.uint8)

        segmap = SegmentationMapsOnImage(mask, shape=np.array(rgb).shape)
        aug_concat_img, segmap = self.affine(image=concat_img, segmentation_maps=segmap)
        mask = segmap.get_arr()

        rgb = aug_concat_img[:, :, :C]
        depth = aug_concat_img[:, :, -1]

        rgb = self.colour_aug(image=rgb)
        depth = self.depth_aug(image=depth)

        rgb = np.array(rgb, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)
        depth = np.array(depth, dtype=np.uint8)

        return rgb, mask, depth

    def __getitem__(self, index):

        idx = self.rgb_ids[index]
        img_file = glob(self.rgb_dir + idx + self.rgb_suffix + '.*')
        obj_mask_file = glob(self.obj_masks_dir + idx + self.obj_masks_suffix + '.*')
        obj_part_mask_file = glob(self.obj_part_masks_dir + idx + self.obj_part_masks_suffix + '.*')
        aff_mask_file = glob(self.aff_masks_dir + idx + self.aff_masks_suffix + '.*')

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(obj_mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {obj_mask_file}'
        assert len(obj_part_mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {obj_part_mask_file}'
        assert len(aff_mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {aff_mask_file}'

        image = Image.open(img_file[0]).convert('RGB')
        obj_label = Image.open(obj_mask_file[0])
        obj_part_label = Image.open(obj_part_mask_file[0])
        aff_label = Image.open(aff_mask_file[0])

        ##################
        # TODO: DEPTH
        ##################
        depth_file = glob(self.depth_dir + idx + self.depth_suffix + '.*')
        assert len(depth_file) == 1, f'Either no image or multiple images found for the ID {idx}: {depth_file}'

        depth_16bit = cv2.imread(depth_file[0], -1)
        depth_16bit = np.array(depth_16bit, dtype=np.float16)
        # helper_utils.print_depth_info(depth)

        depth = helper_utils.convert_16_bit_depth_to_8_bit(depth_16bit)
        # helper_utils.print_depth_info(depth)

        ##################
        ##################

        obj_id = np.unique(obj_label)[1:]
        mask_label = np.ma.getmaskarray(np.ma.masked_equal(obj_label, obj_id)).astype(np.uint8)
        mask_depth_16bit = mask_label * depth_16bit.copy()
        mask_depth = mask_label * depth.copy()

        ##################
        # RESIZE & CROP
        ##################

        image = np.array(image, dtype=np.uint8)
        depth = np.array(depth, dtype=np.uint8)
        obj_label = np.array(obj_label, dtype=np.uint8)
        obj_part_label = np.array(obj_part_label, dtype=np.uint8)
        aff_label = np.array(aff_label, dtype=np.uint8)

        image = cv2.resize(image, self.RESIZE, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, self.RESIZE, interpolation=cv2.INTER_CUBIC)
        obj_label = cv2.resize(obj_label, self.RESIZE, interpolation=cv2.INTER_NEAREST)
        obj_part_label = cv2.resize(obj_part_label, self.RESIZE, interpolation=cv2.INTER_NEAREST)
        aff_label = cv2.resize(aff_label, self.RESIZE, interpolation=cv2.INTER_NEAREST)

        image = helper_utils.crop(image, self.CROP_SIZE, is_img=True)
        depth = helper_utils.crop(depth, self.CROP_SIZE)
        obj_label = helper_utils.crop(obj_label, self.CROP_SIZE)
        obj_part_label = helper_utils.crop(obj_part_label, self.CROP_SIZE)
        aff_label = helper_utils.crop(aff_label, self.CROP_SIZE)

        ##################
        # IMGAUG
        ##################

        if self.apply_imgaug:
            image, obj_label, depth = self.apply_imgaug_to_imgs(rgb=image, mask=obj_label, depth=depth) # todo: obj_label or aff_label

        ##################
        ### SEND TO NUMPY
        ##################
        image = np.array(image, dtype=np.uint8)
        gt_mask = np.array(obj_label, dtype=np.uint8) # todo: obj_label or aff_label
        depth = np.array(depth, dtype=np.uint8)

        ##################
        ### MASK
        ##################
        binary_masks, aff_ids = coco_utils.extract_polygon_masks(image_idx=idx, rgb_img=image, label_img=gt_mask)

        ##################
        ### BBOX
        ##################

        H, W = image.shape[0], image.shape[1]

        obj_ids = np.unique(obj_label)[1:]
        obj_boxes = bbox_utils.get_obj_bbox(mask=obj_label, obj_ids=obj_ids, img_width=W, img_height=H)

        obj_part_ids = np.unique(obj_part_label)[1:]
        aff_boxes = bbox_utils.get_obj_bbox(mask=obj_part_label, obj_ids=obj_part_ids, img_width=W, img_height=H)

        ##################
        ### SEND TO TORCH
        ##################

        image_id = torch.tensor([index])

        gt_mask = torch.as_tensor(gt_mask, dtype=torch.uint8)
        masks = torch.as_tensor(binary_masks, dtype=torch.uint8)

        obj_labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        obj_boxes = torch.as_tensor(obj_boxes, dtype=torch.float32)
        aff_labels = torch.as_tensor(aff_ids, dtype=torch.int64)
        aff_boxes = torch.as_tensor(aff_boxes, dtype=torch.float32)

        target = {}
        target["depth_16bit"] = mask_depth_16bit
        target["depth"] = mask_depth
        target["image_id"] = image_id
        target['gt_mask'] = gt_mask
        target["masks"] = masks
        target["labels"] = obj_labels
        target["boxes"] = obj_boxes
        target["aff_labels"] = aff_labels
        target["aff_boxes"] = aff_boxes
        target["obj_labels"] = obj_labels
        target["obj_boxes"] = obj_boxes

        if self.is_train or self.is_eval:
            img, target = self.transform(image, target)
        else:
            img = np.array(image, dtype=np.uint8)

        return img, target

    ################################
    ################################

    def get_transform(self):
        transforms = []
        transforms.append(ToTensor())
        return Compose(transforms)

################################
################################

class ToTensor(object):
        def __call__(self, image, target):
            image = F.to_tensor(image)
            return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target