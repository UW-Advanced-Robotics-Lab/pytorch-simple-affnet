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

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

######################
######################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

from utils import helper_utils

from dataset.utils import umd_bbox_utils
from dataset.utils import umd_coco_utils
from dataset.utils import umd_utils

######################
######################

class BasicDataSet(data.Dataset):

    def __init__(self,
                 dataset_dir,
                 use_dr_and_pr_images=False,
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
                 ### PRE-PROCESSING
                 mean=config.IMG_MEAN,
                 std=config.IMG_STD,
                 resize=config.RESIZE,
                 crop_size=config.INPUT_SIZE,
                 ### IMGAUG
                 apply_imgaug=False):

        self.dataset_dir = dataset_dir
        self.use_dr_and_pr_images = use_dr_and_pr_images
        ### FOLDER MUST BE CORRECTLY FORMATTED
        self.rgb_dir = self.dataset_dir + rgb_dir
        self.rgb_suffix = rgb_suffix
        self.masks_dir = self.dataset_dir + masks_dir
        self.masks_suffix = masks_suffix
        self.depth_dir = self.dataset_dir + depth_dir
        self.depth_suffix = depth_suffix
        ### pre-processing
        self.mean = mean
        self.std = std
        self.resize = resize
        self.crop_size = crop_size

        if self.use_dr_and_pr_images:
            print(f'Using PR and DR Synthetic Images ..')

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

        # if np.max(depth) > int(2**8-1):
        #     depth = depth / 6500 * (2 ** 8 - 1)

        depth = depth / np.max(depth) * (2 ** 8 - 1)
        depth = np.array(depth, dtype=np.uint8)
        # helper_utils.print_depth_info(depth)

        ##################
        ##################
        # depth_file = glob(self.depth_dir + idx + self.depth_suffix + '.*')
        # depth = skimage.io.imread(depth_file[0])
        # helper_utils.print_depth_info(depth)

        ##################
        ##################

        if self.use_dr_and_pr_images:
            if int(idx) <= config.PR_NUM_IMAGES:
                # print(f'PR image ..')
                self.resize = config.PR_RESIZE
                self.mean = config.PR_IMG_MEAN
                self.std = config.PR_IMG_STD
            else:
                self.resize = config.DR_RESIZE
                self.mean = config.DR_IMG_MEAN
                self.std = config.DR_IMG_STD

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
        label = np.array(label, dtype=np.uint8)
        depth = np.array(depth, dtype=np.uint8)

        # helper_utils.print_depth_info(depth)
        # depth = skimage.color.gray2rgb(depth)
        # print(depth.shape)

        ##################
        ### MASK
        ##################
        binary_masks, aff_IDs = umd_coco_utils.extract_polygon_masks(image_idx=idx, rgb_img=image, label_img=label)

        ##################
        ### BBOX
        ##################
        object_name = idx.split("_")[0]
        object_IDs = umd_utils.object_name_to_id(object_name=object_name)
        boxes = umd_bbox_utils.extract_bboxes(mask=label)

        ##################
        ### SEND TO TORCH
        ##################

        image = helper_utils.numpy_2_torch(image, mean=self.mean, std=self.std, is_rgb=True)
        depth = helper_utils.numpy_2_torch(depth, mean=self.mean,  std=self.std, is_depth=True)
        label = helper_utils.numpy_2_torch(label)

        images = dict(rgb=image, depth=depth, label=label)

        object_IDs = torch.tensor(object_IDs)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        aff_IDs = torch.tensor(aff_IDs)
        binary_masks = torch.tensor(binary_masks)

        target = dict(image_id=idx, object_IDs=object_IDs, boxes=boxes, aff_IDs=aff_IDs, binary_masks=binary_masks)

        return images, target

if __name__ == '__main__':
    dst = BasicDataSet(
                        ### SYN
                        # dataset_dir=config.DATA_DIRECTORY_SOURCE_TRAIN,
                        # use_dr_and_pr_images=True,
                        # mean=config.IMG_MEAN,
                        # std=config.IMG_STD,
                        # resize=config.RESIZE,
                        # crop_size=config.INPUT_SIZE,
                        ### REAL
                        dataset_dir=config.DATA_DIRECTORY_TRAIN,
                        mean=config.IMG_MEAN,
                        std=config.IMG_STD,
                        resize=config.RESIZE,
                        crop_size=config.INPUT_SIZE,
                        ### EXTENDING DATASET
                        extend_dataset=True,
                        max_iters=1000,
                        ### IMGAUG
                        apply_imgaug=False)
    trainloader = data.DataLoader(dst, batch_size=1)
    rgb_mean, rgb_std = 0, 0
    depth_mean, depth_std = 0, 0
    nb_bins = int(2**8)
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)
    count_d = np.zeros(nb_bins)
    print("Dataset has: {}".format(len(trainloader)))
    for i, data in enumerate(trainloader):
        images, target = data
        imgs, depths, labels = images['rgb'], images['depth'], images['label']
        if i % 100 == 0:
            print(f'{i}/{len(trainloader)} ..')
        #######################
        # torch 2 numpy
        #######################
        imgs = helper_utils.torch_2_numpy(imgs, mean=dst.mean, std=dst.std, is_rgb=True)
        depths = helper_utils.torch_2_numpy(depths, mean=dst.mean, std=dst.std, is_depth=True)
        labels = helper_utils.torch_2_numpy(labels)
        ###
        # helper_utils.print_depth_info(imgs)
        # helper_utils.print_class_labels(labels)
        # helper_utils.print_depth_info(dst.depth_16bit)
        # helper_utils.print_depth_info(depths)
        #######################
        ### img mean and std
        #######################
        # rgb
        img_stats = imgs
        img_stats = img_stats.reshape(3, -1)
        rgb_mean += np.mean(img_stats, axis=1)
        rgb_std += np.std(img_stats, axis=1)
        # depth
        img_stats = depths
        img_stats = img_stats.reshape(1, -1)
        depth_mean += np.mean(img_stats, axis=1)
        depth_std += np.std(img_stats, axis=1)
        #######################
        ### histogram
        #######################
        ### RGB
        hist_r = np.histogram(imgs[0], bins=nb_bins, range=[0, 255])
        hist_g = np.histogram(imgs[1], bins=nb_bins, range=[0, 255])
        hist_b = np.histogram(imgs[2], bins=nb_bins, range=[0, 255])
        count_r += hist_r[0]
        count_g += hist_g[0]
        count_b += hist_b[0]
        ### Depth
        hist_d = np.histogram(depths, bins=nb_bins, range=[0, 255])
        count_d += hist_d[0]
        #######################
        ### bbox
        #######################
        boxes = target['boxes']
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        for bbox in boxes:
            # y1, x1, y2, x2
            bbox = np.array(bbox, dtype=np.int32)
            imgs = cv2.rectangle(imgs, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, 2)
        #######################
        ### cv2
        #######################
        cv2.imshow('rgb', cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB))
        cv2.imshow('label', umd_utils.colorize_mask(labels))
        cv2.imshow('depth', np.array(depths, dtype=np.uint8))
        cv2.imshow('heatmap', cv2.applyColorMap(np.array(depths, dtype=np.uint8), cv2.COLORMAP_JET))
        cv2.waitKey(1)
    #######################
    #######################
    rgb_mean /= i
    rgb_std /= i
    print(f'\nRGB: mean:{rgb_mean}\nstd:{rgb_std}')
    depth_mean /= i
    depth_std /= i
    print(f'Depth: mean:{depth_mean}\nstd:{depth_std}')
    #######################
    #######################
    bins = hist_r[1]
    ### rgb
    plt.figure(figsize=(12, 6))
    plt.bar(hist_r[1][:-1], count_r, color='r', label='Red', alpha=0.33)
    plt.axvline(x=rgb_mean[0], color='r', ls='--')
    plt.bar(hist_g[1][:-1], count_g, color='g', label='Green', alpha=0.33)
    plt.axvline(x=rgb_mean[1], color='g', ls='--')
    plt.bar(hist_b[1][:-1], count_b, color='b', label='Blue', alpha=0.33)
    plt.axvline(x=rgb_mean[2], color='b', ls='--')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    ### depth
    plt.figure(figsize=(12, 6))
    plt.bar(x=hist_d[1][:-1], height=count_d, color='k', label='depth', alpha=0.33)
    plt.axvline(x=depth_mean, color='k', ls='--')
    plt.axvline(x=rgb_mean[0], color='r', ls='--')
    plt.axvline(x=rgb_mean[1], color='g', ls='--')
    plt.axvline(x=rgb_mean[2], color='b', ls='--')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.show()