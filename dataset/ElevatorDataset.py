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

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

from utils import helper_utils

from dataset.utils.Elevator import elevator_utils
from dataset.utils.Elevator import elevator_coco_utils
from dataset.utils.Elevator import elevator_bbox_utils

######################
######################

class ElevatorDataSet(data.Dataset):

    def __init__(self,
                 dataset_dir,
                 ### FOLDER MUST BE CORRECTLY FORMATTED
                 rgb_dir='rgb/',
                 rgb_suffix='',
                 masks_dir='masks/',
                 masks_suffix='_label',
                 depth_dir='depth/',
                 depth_suffix='_depth',
                 ### EXTENDING DATASET
                 extend_dataset=False,
                 max_iters=int(250e3),
                 ###
                 is_train=False,
                 is_eval=False,
                 ### PRE-PROCESSING
                 mean=config.IMAGE_MEAN,
                 std=config.IMAGE_STD,
                 resize=config.RESIZE,
                 crop_size=config.INPUT_SIZE,
                 ### IMGAUG
                 apply_imgaug=False):

        self.dataset_dir = dataset_dir
        ### FOLDER MUST BE CORRECTLY FORMATTED
        self.rgb_dir = self.dataset_dir + rgb_dir
        self.rgb_suffix = rgb_suffix
        self.masks_dir = self.dataset_dir + masks_dir
        self.masks_suffix = masks_suffix
        self.depth_dir = self.dataset_dir + depth_dir
        self.depth_suffix = depth_suffix
        ###
        self.is_train = is_train
        self.is_eval = is_eval
        self.transform = self.get_transform()
        ### pre-processing
        self.mean = mean
        self.std = std
        self.resize = resize
        self.crop_size = crop_size

        ################################
        ### EXTENDING DATASET
        ################################
        self.extend_dataset = extend_dataset
        self.max_iters = max_iters

        self.rgb_ids = [splitext(file)[0] for file in listdir(self.rgb_dir) if not file.startswith('.')]
        self.masks_ids = [splitext(file)[0] for file in listdir(self.masks_dir) if not file.startswith('.')]
        self.depth_ids = [splitext(file)[0] for file in listdir(self.depth_dir) if not file.startswith('.')]
        assert(len(self.rgb_ids) == len(self.masks_ids) == len(self.depth_ids))
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

    def crop(self, pil_img, is_img=False):
        _dtype = np.array(pil_img).dtype
        pil_img = Image.fromarray(pil_img)
        crop_w, crop_h = self.crop_size
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
            resize_img = np.zeros((crop_w, crop_h, img_channels))
            resize_img[0:(bottom - top), 0:(right - left), :img_channels] = np.array(pil_img)[..., :img_channels]
        else:
            resize_img = np.zeros((crop_w, crop_h))
            resize_img[0:(bottom - top), 0:(right - left)] = np.array(pil_img)
        ###
        resize_img = np.array(resize_img, dtype=_dtype)

        return Image.fromarray(resize_img)

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
        mask_file = glob(self.masks_dir + idx + self.masks_suffix + '.*')

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'

        image = Image.open(img_file[0]).convert('RGB')

        label = Image.open(mask_file[0])

        ##################
        ### TODO: DEPTH
        ##################
        depth_file = glob(self.depth_dir + idx + self.depth_suffix + '.*')
        assert len(depth_file) == 1, f'Either no image or multiple images found for the ID {idx}: {depth_file}'

        depth = cv2.imread(depth_file[0], -1)
        depth = np.array(depth, dtype=np.uint16)
        # helper_utils.print_depth_info(depth)

        depth = helper_utils.convert_16_bit_depth_to_8_bit(depth)
        # helper_utils.print_depth_info(depth)

        ##################
        ### RESIZE & CROP
        ##################

        image = np.array(image, dtype=np.uint8)
        label = np.array(label, dtype=np.uint8)
        depth = np.array(depth, dtype=np.uint8)

        image = cv2.resize(image, self.resize, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, self.resize, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, self.resize, interpolation=cv2.INTER_NEAREST)

        image = self.crop(image, is_img=True)
        label = self.crop(label)
        depth = self.crop(depth)

        ##################
        ### IMGAUG
        ##################

        if self.apply_imgaug:
            image, label, depth = self.apply_imgaug_to_imgs(rgb=image, mask=label, depth=depth)

        ##################
        ### SEND TO NUMPY
        ##################
        image = np.array(image, dtype=np.uint8)
        H, W, C = image.shape[0], image.shape[1], image.shape[2]
        gt_mask = np.array(label, dtype=np.uint8)
        depth = np.array(depth, dtype=np.uint8)

        ##################
        ### MASK
        ##################
        binary_masks, obj_id = elevator_coco_utils.extract_polygon_masks(image_idx=idx, rgb_img=image, label_img=label)

        ##################
        ### BBOX
        ##################

        step = 40
        border_list = np.arange(start=0, stop=np.max([W, H]) + step, step=step)
        # drawing bbox = (x1, y1), (x2, y2)
        obj_boxes = elevator_bbox_utils.get_obj_bbox(mask=gt_mask, obj_id=obj_id,
                                                     img_width=W, img_length=H,
                                                     border_list=border_list)

        ##################
        ### SEND TO TORCH
        ##################

        gt_mask = torch.as_tensor(gt_mask, dtype=torch.uint8)

        image_id = torch.tensor([index])
        obj_boxes = torch.as_tensor(obj_boxes, dtype=torch.float32)
        obj_labels = torch.as_tensor((obj_id,), dtype=torch.int64)
        masks = torch.as_tensor(binary_masks, dtype=torch.uint8)

        target = {}
        target['gt_mask'] = gt_mask
        target["labels"] = obj_labels
        target["boxes"] = obj_boxes
        target["masks"] = masks
        target["image_id"] = image_id

        if self.is_train or self.is_eval:
            img, target = self.transform(image, target)
        else:
            img = np.array(image, dtype=np.uint8)

        return img, target

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