import os

import unittest

import numpy as np
import scipy.io as scio

import cv2
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import torch
from torch.utils import data
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')

import config

from training import train_utils
from dataset.umd import umd_dataset
from dataset.umd import umd_dataset_utils


NUM_IMAGES = 1000


class DatasetStatisticsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(DatasetStatisticsTest, self).__init__(*args, **kwargs)

        # Load ARL AffPose dataset.
        dataset = umd_dataset.UMDDataset(
            dataset_dir=config.UMD_DATA_DIRECTORY_TEST,
            mean=config.UMD_IMAGE_MEAN,
            std=config.UMD_IMAGE_STD,
            resize=config.UMD_RESIZE,
            crop_size=config.UMD_CROP_SIZE,
            apply_imgaug=False,
        )

        # creating subset.
        np.random.seed(0)
        total_idx = np.arange(0, len(dataset), 1)
        test_idx = np.random.choice(total_idx, size=int(NUM_IMAGES), replace=True)
        dataset = torch.utils.data.Subset(dataset, test_idx)

        # create dataloader.
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        print(f'Selecting {len(self.data_loader)} images ..')

    def test_rgbd_distribution(self):
        # init stats.
        rgb_mean, rgb_std = 0, 0
        depth_mean, depth_std = 0, 0
        distance_to_object_mean, distance_to_object_std = 0, 0
        nb_bins = int(2 ** 8)
        count_r = np.zeros(nb_bins)
        count_g = np.zeros(nb_bins)
        count_b = np.zeros(nb_bins)
        count_d = np.zeros(nb_bins)

        # loop over dataset.
        for i, (image, target) in enumerate(self.data_loader):
            if i % 100 == 0:
                print(f'{i}/{len(self.data_loader)} ..')

            # format rgb.
            image = np.squeeze(np.array(image))
            image = np.array(image * (2 ** 8 - 1), dtype=np.uint8)
            image, target = umd_dataset_utils.format_target_data(image, target)

            # get depth.
            depth_8bit = target['depth_8bit']
            depth_16bit = target['depth_16bit']
            # get masked depth.
            aff_mask = np.array(target['aff_mask'], dtype=np.uint8)
            masked_depth_16bit = np.ma.masked_equal(aff_mask, 0).astype(int) * depth_16bit
            # set depth for stats.
            depth = depth_8bit

            # mean and std
            # rgb
            img_stats = image
            img_stats = img_stats.reshape(3, -1)
            rgb_mean += np.mean(img_stats, axis=1)
            rgb_std += np.std(img_stats, axis=1)
            # getting nonzero mean depth.
            img_stats = image
            img_stats = img_stats.reshape(1, -1)
            depth_mean += np.mean(img_stats, axis=1)
            depth_std += np.std(img_stats, axis=1)
            # getting nonzero mean depth.
            masked_depth_16bit_nan = np.where(masked_depth_16bit != 0, masked_depth_16bit, np.nan)
            img_stats = masked_depth_16bit_nan.reshape(1, -1)
            distance_to_object_mean += np.nanmean(img_stats, axis=1)
            distance_to_object_std += np.nanstd(img_stats, axis=1)

            # histogram.
            # rgb
            hist_r = np.histogram(image[0], bins=nb_bins, range=[0, 255])
            hist_g = np.histogram(image[1], bins=nb_bins, range=[0, 255])
            hist_b = np.histogram(image[2], bins=nb_bins, range=[0, 255])
            count_r += hist_r[0]
            count_g += hist_g[0]
            count_b += hist_b[0]
            # depth
            hist_d = np.histogram(depth, bins=nb_bins, range=[0, 255])
            count_d += hist_d[0]

        # get stats.
        rgb_mean /= i
        rgb_std /= i
        print(f'\nRGB: mean:{rgb_mean}\nstd:{rgb_std}')
        depth_mean /= i
        depth_std /= i
        print(f'Depth: mean:{depth_mean}\nstd:{depth_std}')
        distance_to_object_mean /= i
        distance_to_object_std /= i
        print(f'Distance to Object: mean:{distance_to_object_mean}\nstd:{distance_to_object_std}')

        # plot histograms.
        plt.close()
        bins = hist_r[1]
        # rgb
        plt.figure(figsize=(12, 6))
        plt.bar(hist_r[1][:-1], count_r, color='r', label='Red', alpha=0.33)
        plt.axvline(x=rgb_mean[0], color='r', ls='--')
        plt.bar(hist_g[1][:-1], count_g, color='g', label='Green', alpha=0.33)
        plt.axvline(x=rgb_mean[1], color='g', ls='--')
        plt.bar(hist_b[1][:-1], count_b, color='b', label='Blue', alpha=0.33)
        plt.axvline(x=rgb_mean[2], color='b', ls='--')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.show()
        # depth
        plt.figure(figsize=(12, 6))
        plt.bar(x=hist_d[1][:-1], height=count_d, color='k', label='depth', alpha=0.33)
        plt.axvline(x=depth_mean, color='k', ls='--')
        plt.axvline(x=rgb_mean[0], color='r', ls='--')
        plt.axvline(x=rgb_mean[1], color='g', ls='--')
        plt.axvline(x=rgb_mean[2], color='b', ls='--')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.show()

    def test_class_distributions(self):

        num_obj_ids = np.zeros(shape=(config.UMD_NUM_OBJECT_CLASSES))
        num_aff_ids = np.zeros(shape=(config.UMD_NUM_AFF_CLASSES))

        # loop over dataset.
        for i, (image, target) in enumerate(self.data_loader):
            if i % 10 == 0:
                print(f'{i}/{len(self.data_loader)} ..')

            # format data.
            image = np.squeeze(np.array(image)).transpose(1, 2, 0)
            image = np.array(image * (2 ** 8 - 1), dtype=np.uint8).reshape(config.UMD_CROP_SIZE[0], config.UMD_CROP_SIZE[1], -1)
            target = umd_dataset_utils.format_target_data(image, target)

            # obj_ids.
            obj_ids = target['obj_ids']
            for obj_id in obj_ids:
                num_obj_ids[0] += 1
                num_obj_ids[obj_id] += 1

            # aff_ids.
            aff_ids = target['aff_ids']
            for aff_id in aff_ids:
                num_aff_ids[0] += 1
                num_aff_ids[aff_id] += 1

        print(f'num_obj_ids: {num_obj_ids}')
        print(f'num_aff_ids: {num_aff_ids}')

if __name__ == '__main__':
    # unittest.main()

    # run desired test.
    suite = unittest.TestSuite()
    suite.addTest(DatasetStatisticsTest("test_class_distributions"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

