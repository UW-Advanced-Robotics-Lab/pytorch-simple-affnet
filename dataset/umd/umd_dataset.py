from os import listdir
from os.path import splitext
from glob import glob

import numpy as np

import cv2
from PIL import Image

from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import torch
from torch.utils import data
from torchvision.transforms import functional as F

import config
from dataset.umd import umd_dataset_utils
from dataset import dataset_utils


class UMDDataset(data.Dataset):

    def __init__(self,
                 dataset_dir,
                 # ARLAffPose dataset must be formatted correctly, see ARLAffPoseDatasetUtils.
                 rgb_dir='rgb/',
                 rgb_suffix='',
                 masks_dir='masks/',
                 masks_suffix='_label',
                 depth_folder='depth/',
                 depth_suffix='_depth',
                 # configs for pre-processing our dataset.
                 mean=config.UMD_IMAGE_MEAN,
                 std=config.UMD_IMAGE_STD,
                 resize=config.UMD_RESIZE,
                 crop_size=config.UMD_CROP_SIZE,
                 apply_imgaug=False,
                 # TRAIN OR EVAL
                 is_train=False,
                 is_eval=False,
                 ):

        self.dataset_dir = dataset_dir
        # loading ARL AffPose dataset.
        self.rgb_dir = self.dataset_dir + rgb_dir
        self.rgb_suffix = rgb_suffix
        self.masks_dir = self.dataset_dir + masks_dir
        self.masks_suffix = masks_suffix
        self.depth_dir = self.dataset_dir + depth_folder
        self.depth_suffix = depth_suffix
        # configs for pre-processing our dataset.
        self.mean = mean
        self.std = std
        self.RESIZE = resize
        self.CROP_SIZE = crop_size
        # select fewer images.
        self.is_train = is_train
        self.is_eval = is_eval
        self.transform = dataset_utils.get_transform()

        # Loading images.
        self.rgb_ids = [splitext(file)[0] for file in listdir(self.rgb_dir) if not file.startswith('.')]
        self.masks_ids = [splitext(file)[0] for file in listdir(self.masks_dir) if not file.startswith('.')]
        self.depth_ids = [splitext(file)[0] for file in listdir(self.depth_dir) if not file.startswith('.')]
        assert(len(self.rgb_ids) == len(self.masks_ids))
        print(f'Dataset has {len(self.rgb_ids)} examples .. {dataset_dir}')

        # sorting images.
        self.rgb_ids = np.sort(np.array(self.rgb_ids))
        self.masks_ids = np.sort(np.array(self.masks_ids))
        self.depth_ids = np.sort(np.array(self.depth_ids))

        # Augmenting images.
        self.apply_imgaug = apply_imgaug
        self.affine, self.colour_aug, self.depth_aug = dataset_utils.get_image_augmentations()

    def __len__(self):
        return len(self.rgb_ids)

    def apply_imgaug_to_imgs(self, rgb, depth, mask):
        rgb, depth, mask = np.array(rgb), np.array(depth), np.array(mask)

        H, W, C = rgb.shape[0], rgb.shape[1], rgb.shape[2]

        concat_img = np.zeros(shape=(H, W, C + 1))
        concat_img[:, :, :C] = rgb
        concat_img[:, :, -1] = depth
        concat_img = np.array(concat_img, dtype=np.uint8)

        concat_mask = np.array(mask, dtype=np.uint8)

        segmap = SegmentationMapsOnImage(concat_mask, shape=np.array(rgb).shape)
        aug_concat_img, segmap = self.affine(image=concat_img, segmentation_maps=segmap)
        aug_concat_mask = segmap.get_arr()

        rgb = aug_concat_img[:, :, :C]
        depth = aug_concat_img[:, :, -1]

        mask = aug_concat_mask

        rgb = self.colour_aug(image=rgb)
        depth = self.depth_aug(image=depth)

        rgb = np.array(rgb, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        # Uncomment to check resulting images with augmentation.
        # dataset_utils.print_class_labels(mask)
        # cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        # color_mask = umd_dataset_utils.colorize_aff_mask(mask)
        # cv2.imshow('color_mask', cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)

        return rgb, depth, mask

    def __getitem__(self, index):

        # loading images.
        idx = self.rgb_ids[index]
        img_file = glob(self.rgb_dir + idx + self.rgb_suffix + '.*')
        depth_file = glob(self.depth_dir + idx + self.depth_suffix + '.*')
        mask_file = glob(self.masks_dir + idx + self.masks_suffix + '.*')

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(depth_file) == 1, f'Either no image or multiple images found for the ID {idx}: {depth_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'

        image = Image.open(img_file[0]).convert('RGB')
        depth_16bit = cv2.imread(depth_file[0], -1)
        depth_16bit = np.array(depth_16bit, dtype=np.uint16)
        depth_8bit = dataset_utils.convert_16_bit_depth_to_8_bit(depth_16bit)
        mask = Image.open(mask_file[0])

        image = np.array(image, dtype=np.uint8)
        depth_16bit = np.array(depth_16bit, dtype=np.float16)
        depth_8bit = np.array(depth_8bit, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        # applying image augmentation.
        if self.apply_imgaug:
            image, depth_8bit, mask = self.apply_imgaug_to_imgs(rgb=image, depth=depth_8bit, mask=mask)

        image = np.array(image, dtype=np.uint8)
        H, W = image.shape[0], image.shape[1]

        # Get obj name from bbox.
        obj_name = idx.split("_")[0]
        obj_id = umd_dataset_utils.map_obj_name_to_id(obj_name)

        # Now getting binary masks for MaskRCNN or AffNet Mask Branch.
        aff_ids = np.unique(mask)[1:]

        # Get obj bbox from affordance mask.
        foreground_mask = np.ma.getmaskarray(np.ma.masked_not_equal(mask, 0)).astype(np.uint8)
        obj_boxes = dataset_utils.get_bbox(mask=foreground_mask, obj_ids=np.array([1]), img_width=H, img_height=W)

        # Now getting binary masks for MaskRCNN or AffNet Mask Branch.
        aff_ids = np.unique(mask)[1:]
        aff_binary_masks = []
        for idx, aff_id in enumerate(aff_ids):

            # Object bboxs & masks.
            mask_aff_label = np.ma.getmaskarray(np.ma.masked_equal(mask.copy(), aff_id))
            aff_binary_masks.append(mask_aff_label)

            # # Uncomment to check binary masks.
            # color_label = umd_dataset_utils.colorize_aff_mask(mask_aff_label.copy())
            # color_label = cv2.addWeighted(image, 0.35, color_label, 0.65, 0)
            #
            # x1, y1, x2, y2 = obj_boxes[0]
            # color_label = cv2.rectangle(color_label, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #
            # color_label = cv2.putText(color_label,
            #                           obj_name,
            #                           (x1, y1 - 5),
            #                           cv2.FONT_ITALIC,
            #                           0.4,
            #                           (255, 255, 255))
            #
            # cv2.imshow('color_label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)

        # formatting above.
        obj_ids = np.array([obj_id])
        obj_boxes = np.array(obj_boxes)
        aff_ids = np.array(aff_ids)
        aff_binary_masks = np.array(aff_binary_masks)

        target = {}
        target["image_id"] = torch.tensor([index])
        # original mask and binary masks.
        target["aff_mask"] = torch.as_tensor(np.array(mask, dtype=np.uint8), dtype=torch.uint8)
        target["aff_binary_masks"] = torch.as_tensor(aff_binary_masks, dtype=torch.uint8)
        # ids and bboxs.
        target["obj_ids"] = torch.as_tensor(obj_ids, dtype=torch.int64)
        target["obj_boxes"] = torch.as_tensor(obj_boxes, dtype=torch.float32)
        target["aff_ids"] = torch.as_tensor(aff_ids, dtype=torch.int64)
        # viewing how far objects are from the camera using depth images and object masks.
        # target["depth_8bit"] = torch.as_tensor(depth_8bit, dtype=torch.float32)
        # target["depth_16bit"] = torch.as_tensor(depth_16bit, dtype=torch.float32)
        # target["masked_depth_16bit"] = torch.as_tensor(masked_depth_16bit, dtype=torch.float32)

        # sent to transform.
        if self.is_train or self.is_eval:
            img, target = self.transform(image, target)
        else:
            img = np.array(image, dtype=np.uint8)

        return img, target
