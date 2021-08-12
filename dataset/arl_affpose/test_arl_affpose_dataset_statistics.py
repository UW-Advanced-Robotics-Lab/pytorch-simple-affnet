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
from dataset.arl_affpose import arl_affpose_dataset
from dataset.arl_affpose import arl_affpose_dataset_utils
from dataset.arl_affpose import load_arl_affpose_obj_ply_files

_NUM_IMAGES = 1000

class DatasetStatisticsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(DatasetStatisticsTest, self).__init__(*args, **kwargs)

        # Load ARL AffPose dataset.
        dataset = arl_affpose_dataset.ARLAffPoseDataset(
            dataset_dir=config.DATA_DIRECTORY_TEST,
            mean=config.IMAGE_MEAN,
            std=config.IMAGE_STD,
            resize=config.RESIZE,
            crop_size=config.CROP_SIZE,
            apply_imgaug=False,
        )

        # creating subset.
        np.random.seed(0)
        total_idx = np.arange(0, len(dataset), 1)
        test_idx = np.random.choice(total_idx, size=int(_NUM_IMAGES), replace=True)
        dataset = torch.utils.data.Subset(dataset, test_idx)

        # create dataloader.
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        print(f'Selecting {len(self.data_loader)} images ..')

    def test_occlusion(self):

        # load object ply files.
        self.cld, self.cld_obj_centered, self.cld_obj_part_centered, \
        self.obj_classes, self.obj_part_classes = load_arl_affpose_obj_ply_files.load_obj_ply_files()

        obj_occlusion = np.zeros(shape=(len(self.data_loader), len(self.obj_classes)))
        obj_part_occlusion = np.zeros(shape=(len(self.data_loader), len(self.obj_part_classes)))

        # loop over dataset.
        for i, (image, target) in enumerate(self.data_loader):
            if i % 10 == 0:
                print(f'{i}/{len(self.data_loader)} ..')

            # format rgb.
            image = np.squeeze(np.array(image))
            image, target = arl_affpose_dataset_utils.format_target_data(image, target)
            H, W, C = image.shape[0], image.shape[1], image.shape[2]

            # get image idx.
            image_id = target['image_id'].item()
            image_id = str(1000000 + image_id)[1:]

            # load meta data.
            meta_addr = self.data_loader.dataset.dataset.dataset_dir + 'meta/' + image_id + '_meta.mat'
            meta = scio.loadmat(meta_addr)

            # overlay obj mask on rgb images.
            obj_mask = target['obj_mask']
            obj_part_mask = target['obj_part_mask']
            colour_label = arl_affpose_dataset_utils.colorize_obj_mask(obj_mask)
            colour_label = cv2.addWeighted(image, 0.35, colour_label, 0.65, 0)

            # Img to draw 6-DoF Pose.
            cv2_pose_img = colour_label.copy()

            #######################################
            #######################################

            cam_cx = meta['cam_cx'][0][0]
            cam_cy = meta['cam_cy'][0][0]
            cam_fx = meta['cam_fx'][0][0]
            cam_fy = meta['cam_fy'][0][0]

            cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
            cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

            #######################################
            # OBJECT
            #######################################

            obj_ids = np.array(meta['object_class_ids']).flatten()
            for idx, obj_id in enumerate(obj_ids):
                obj_id = int(obj_id)
                obj_color = arl_affpose_dataset_utils.obj_color_map(obj_id)
                # print(f"\tObject: {obj_id}, {self.obj_classes[int(obj_id) - 1]}")

                obj_meta_idx = str(1000 + obj_id)[1:]
                obj_r = meta['obj_rotation_' + str(obj_meta_idx)]
                obj_t = meta['obj_translation_' + str(obj_meta_idx)]

                obj_r = np.array(obj_r, dtype=np.float64).reshape(3, 3)
                obj_t = np.array(obj_t, dtype=np.float64).reshape(-1, 3)

                cv2_obj_label = np.zeros(shape=(obj_mask.shape), dtype=np.uint8)

                #######################################
                # ITERATE OVER OBJ PARTS
                #######################################

                obj_part_ids = arl_affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
                # print(f'\tobj_part_ids:{obj_part_ids}')
                for obj_part_id in obj_part_ids:
                    obj_part_id = int(obj_part_id)
                    aff_id = arl_affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                    aff_color = arl_affpose_dataset_utils.aff_color_map(aff_id)
                    # print(f"\t\tAff: {aff_id}, {self.obj_part_classes[int(obj_part_id) - 1]}")
                    
                    #######################################
                    # OBJ POSE
                    #######################################

                    # projecting 3D model to 2D image
                    obj_centered = self.cld_obj_centered[obj_part_id]
                    imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, cam_mat, cam_dist)
                    cv2_pose_img = cv2.polylines(cv2_pose_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                    #######################################
                    # OBJ MASK
                    #######################################

                     # Draw obj mask.
                    cv2_drawpoints_img = np.zeros(shape=(obj_mask.shape), dtype=np.uint8)
                    cv2_drawpoints_img = cv2.polylines(cv2_drawpoints_img, np.int32([np.squeeze(imgpts)]), False, (obj_id))
                    cv2_masked = np.ma.getmaskarray(np.ma.masked_equal(cv2_drawpoints_img, obj_id))

                    # get contours.
                    # cv2_masked = np.ma.getmaskarray(np.ma.masked_equal(obj_mask, obj_id)) * cv2_masked
                    res = cv2.findContours(cv2_masked.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    contours = res[-2]  # for cv2 v3 and v4+ compatibility

                    # get obj mask.
                    cv2_obj_label = cv2.drawContours(cv2_obj_label, contours, contourIdx=-1, color=(obj_id), thickness=-1)

                    #######################################
                    # OBJ PART POSE
                    #######################################

                    obj_part_meta_idx = str(1000 + obj_part_id)[1:]
                    obj_part_r = meta['obj_part_rotation_' + str(obj_part_meta_idx)]
                    obj_part_t = meta['obj_part_translation_' + str(obj_part_meta_idx)]

                    # projecting 3D model to 2D image
                    obj_part_centered = self.cld_obj_part_centered[obj_part_id]
                    imgpts, jac = cv2.projectPoints(obj_part_centered * 1e3, obj_part_r, obj_part_t * 1e3, cam_mat, cam_dist)

                    #######################################
                    # OBJ PART MASK
                    #######################################

                     # Draw obj mask.
                    cv2_obj_part_label = np.zeros(shape=(obj_part_mask.shape), dtype=np.uint8)
                    cv2_drawpoints_img = np.zeros(shape=(obj_mask.shape), dtype=np.uint8)
                    cv2_drawpoints_img = cv2.polylines(cv2_drawpoints_img, np.int32([np.squeeze(imgpts)]), False, (obj_part_id))
                    cv2_masked = np.ma.getmaskarray(np.ma.masked_equal(cv2_drawpoints_img, obj_part_id))

                    # get contours.
                    res = cv2.findContours(cv2_masked.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    contours = res[-2]  # for cv2 v3 and v4+ compatibility

                    # get obj mask.
                    cv2_obj_part_label = cv2.drawContours(cv2_obj_part_label, contours, contourIdx=-1, color=(obj_part_id), thickness=-1)

                    #######################################
                    # OBJ PART Occlusion
                    #######################################

                    # actual.
                    masked_obj_part_label = np.ma.getmaskarray(np.ma.masked_equal(obj_part_mask, obj_part_id)).astype(np.uint8)
                    actual_points = np.count_nonzero(masked_obj_part_label)

                    # expected.
                    masked_cv2_obj_part_label = np.ma.getmaskarray(np.ma.masked_equal(cv2_obj_part_label, obj_part_id)).astype(np.uint8)
                    expected_points = np.count_nonzero(masked_cv2_obj_part_label)

                    # calc occlusion.
                    percent_occlusion = actual_points / expected_points
                    obj_part_occlusion[i, obj_part_id - 1] = percent_occlusion

                    # print(f"\t\tObject Part: Id:{obj_part_id}, Name:{self.obj_part_classes[int(obj_part_id) - 1]}, "
                    #       f"% Occlusion: {percent_occlusion * 100:.2f}")

                #######################################
                # OBJ Occlusion
                #######################################

                # actual.
                masked_obj_label = np.ma.getmaskarray(np.ma.masked_equal(obj_mask, obj_id)).astype(np.uint8)
                actual_points = np.count_nonzero(masked_obj_label)

                # expected.
                masked_cv2_obj_label = np.ma.getmaskarray(np.ma.masked_equal(cv2_obj_label, obj_id)).astype(np.uint8)
                expected_points = np.count_nonzero(masked_cv2_obj_label)

                # calc occlusion.
                percent_occlusion = actual_points / expected_points
                obj_occlusion[i, obj_id-1] = percent_occlusion

                # print(f"\tObject: Id:{obj_id}, Name:{self.obj_classes[int(obj_id) - 1]}, "
                #       f"% Occlusion: {percent_occlusion*100:.2f}")

                #######################################
                # Plotting
                #######################################

                # debugging.
                # cv2.imshow('cv2_pose_img', cv2_pose_img)
                # cv2.waitKey(0)

                # debugging.
                # cv2.imshow('label_bbox', masked_obj_label * 100)
                # cv2.imshow('masked_cv2', masked_cv2_obj_label * 100)
                # cv2.waitKey(0)

                # debugging.
                # cv2.imshow('cv2_obj_part_label', cv2_obj_part_label * 100)
                # cv2.waitKey(0)

        print()
        for obj_id in range(len(self.obj_classes)):
            _obj_occlusion = obj_occlusion[:, obj_id]
            # get nonzero values.
            idxs = np.nonzero(_obj_occlusion)[0]
            if len(idxs) != 0:
                masked_obj_occlusion = _obj_occlusion[idxs]
                # standardize to 1.
                masked_obj_occlusion += (1 - np.nanmax(masked_obj_occlusion))
                # get mean and std.
                avg_obj_occlusion = np.mean(masked_obj_occlusion)
                std_obj_occlusion = np.std(masked_obj_occlusion)
                print(f"Occlusion: "
                      f"\tMean: {avg_obj_occlusion * 100:.2f}, "
                      f"\tStd Dev: {std_obj_occlusion * 100:.2f}, "
                      f"\tObject: Id:{obj_id+1}, "
                      f"\tName:{self.obj_classes[int(obj_id)]}, "
                      )

        print()
        for obj_part_id in range(len(self.obj_part_classes)):
            _obj_part_occlusion = obj_part_occlusion[:, obj_part_id]
            # get nonzero values.
            idxs = np.nonzero(_obj_part_occlusion)[0]
            if len(idxs) != 0:
                masked_obj_part_occlusion = _obj_part_occlusion[idxs]
                # standardize to 1.
                masked_obj_part_occlusion += (1 - np.nanmax(masked_obj_part_occlusion))
                # get mean and std.
                avg_obj_part_occlusion = np.mean(masked_obj_part_occlusion)
                std_obj_part_occlusion = np.std(masked_obj_part_occlusion)

                print(f"Occlusion: "
                      f"\tMean: {avg_obj_part_occlusion * 100:.2f}, "
                      f"\tStd Dev: {std_obj_part_occlusion * 100:.2f}, "
                      f"\tObject Part: Id:{obj_part_id+1}, "
                      f"\tName:{self.obj_part_classes[int(obj_part_id)]},"
                      )

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
            if i % 10 == 0:
                print(f'{i}/{len(self.data_loader)} ..')

            # format rgb.
            image = np.squeeze(np.array(image)).transpose(1, 2, 0)
            image = np.array(image * (2 ** 8 - 1), dtype=np.uint8).reshape(config.CROP_SIZE[0], config.CROP_SIZE[1], -1)
            image, target = arl_affpose_dataset_utils.format_target_data(image, target)

            # get depth.
            depth_8bit = target['depth_8bit']
            depth_16bit = target['depth_16bit']
            masked_depth_16bit = target['masked_depth_16bit']
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

if __name__ == '__main__':
    # unittest.main()

    # run desired test.
    suite = unittest.TestSuite()
    suite.addTest(DatasetStatisticsTest("test_occlusion"))
    suite.addTest(DatasetStatisticsTest("test_rgbd_distribution"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

