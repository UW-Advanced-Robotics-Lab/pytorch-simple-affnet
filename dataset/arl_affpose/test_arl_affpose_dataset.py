import unittest

import numpy as np

import cv2

import torch
from torch.utils import data

import sys
sys.path.append('../../')

import config

from dataset.arl_affpose import arl_affpose_dataset
from dataset.arl_affpose import arl_affpose_dataset_utils


class ARLAffPoseDatasetTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ARLAffPoseDatasetTest, self).__init__(*args, **kwargs)

        # Load ARL AffPose dataset.
        dataset = arl_affpose_dataset.ARLAffPoseDataset(
            dataset_dir=config.ARL_DATA_DIRECTORY_TRAIN,
            mean=config.ARL_IMAGE_MEAN,
            std=config.ARL_IMAGE_STD,
            resize=config.ARL_RESIZE,
            crop_size=config.ARL_CROP_SIZE,
            apply_imgaug=True,
            is_train=True,
        )

        # create dataloader.
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        print(f'Selecting {len(self.data_loader)} images ..')

    def test_maskrcnn_dataloader(self):
        print('\nVisualizing Ground Truth Data for MaskRCNN ..')
        # loop over dataset.
        for i, (image, target) in enumerate(self.data_loader):
            print(f'{i}/{len(self.data_loader)} ..')

            # format data.
            image = np.squeeze(np.array(image)).transpose(1, 2, 0)
            image = np.array(image * (2 ** 8 - 1), dtype=np.uint8).reshape(config.ARL_CROP_SIZE[0], config.ARL_CROP_SIZE[1], -1)
            image, target = arl_affpose_dataset_utils.format_target_data(image, target)

            # Bounding Box.
            bbox_img = arl_affpose_dataset_utils.draw_bbox_on_img(image=image,
                                                                  obj_ids=target['obj_ids'],
                                                                  boxes=target['obj_boxes'],
                                                                  )

            # Original Segmentation Mask.
            color_mask = arl_affpose_dataset_utils.colorize_obj_mask(target['obj_mask'])
            color_mask = cv2.addWeighted(bbox_img, 0.35, color_mask, 0.65, 0)

            # Binary Masks.
            binary_mask = arl_affpose_dataset_utils.get_segmentation_masks(image=image,
                                                                           obj_ids=target['obj_ids'],
                                                                           binary_masks=target['obj_binary_masks'],
                                                                           )
            color_binary_mask = arl_affpose_dataset_utils.colorize_obj_mask(binary_mask)
            color_binary_mask = cv2.addWeighted(bbox_img, 0.35, color_binary_mask, 0.65, 0)

            # print object and affordance class names.
            arl_affpose_dataset_utils.print_class_obj_names(target['obj_ids'])

            # show plots.
            cv2.imshow('rgb', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cv2.imshow('bbox', cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
            cv2.imshow('mask', cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
            cv2.imshow('binary_mask', cv2.cvtColor(color_binary_mask, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)

    def test_affnet_dataloader(self):
        print('\nVisualizing Ground Truth Data for AffNet ..')

        # loop over dataset.
        for i, (image, target) in enumerate(self.data_loader):
            print(f'\n{i}/{len(self.data_loader)} ..')

            # format data.
            image = np.squeeze(np.array(image)).transpose(1, 2, 0)
            image = np.array(image * (2 ** 8 - 1), dtype=np.uint8).reshape(config.ARL_CROP_SIZE[0], config.ARL_CROP_SIZE[1], -1)
            image, target = arl_affpose_dataset_utils.format_target_data(image, target)

            # RGB.
            cv2.imshow('rgb', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Bounding Box.
            bbox_img = arl_affpose_dataset_utils.draw_bbox_on_img(image=image,
                                                                  obj_ids=target['obj_ids'],
                                                                  boxes=target['obj_boxes'],
                                                                  )
            cv2.imshow('bbox', cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))

            # Original Segmentation Mask.
            color_mask = arl_affpose_dataset_utils.colorize_aff_mask(target['aff_mask'])
            color_mask = cv2.addWeighted(bbox_img, 0.35, color_mask, 0.65, 0)
            cv2.imshow('mask', cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))

            # Binary Masks.
            binary_mask = arl_affpose_dataset_utils.get_segmentation_masks(image=image,
                                                                           obj_ids=target['aff_ids'],
                                                                           binary_masks=target['aff_binary_masks'],
                                                                           )
            color_binary_mask = arl_affpose_dataset_utils.colorize_aff_mask(binary_mask)
            color_binary_mask = cv2.addWeighted(bbox_img, 0.35, color_binary_mask, 0.65, 0)
            cv2.imshow('binary_mask', cv2.cvtColor(color_binary_mask, cv2.COLOR_BGR2RGB))

            # show object mask derived from affordance masks.
            obj_part_mask = arl_affpose_dataset_utils.get_obj_part_mask(image=image,
                                                                         obj_ids=target['obj_ids'],
                                                                         aff_ids=target['aff_ids'],
                                                                         bboxs=target['obj_boxes'],
                                                                         binary_masks=target['aff_binary_masks'],
                                                                         )

            obj_mask = arl_affpose_dataset_utils.convert_obj_part_mask_to_obj_mask(obj_part_mask)
            color_obj_mask = arl_affpose_dataset_utils.colorize_obj_mask(obj_mask)
            color_obj_mask = cv2.addWeighted(bbox_img, 0.35, color_obj_mask, 0.65, 0)
            cv2.imshow('obj_mask', cv2.cvtColor(color_obj_mask, cv2.COLOR_BGR2RGB))

            # show plots.
            cv2.waitKey(0)

if __name__ == '__main__':
    # unittest.main()

    # run desired test.
    suite = unittest.TestSuite()
    suite.addTest(ARLAffPoseDatasetTest("test_maskrcnn_dataloader"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

