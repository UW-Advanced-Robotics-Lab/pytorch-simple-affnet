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
from dataset.arl_affpose import arl_affpose_dataset_utils
from dataset import dataset_utils


class ARLAffPoseDataset(data.Dataset):

    def __init__(self,
                 dataset_dir,
                 # ARLAffPose dataset must be formatted correctly, see ARLAffPoseDatasetUtils.
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
                 # configs for pre-processing our dataset.
                 mean=config.ARL_IMAGE_MEAN,
                 std=config.ARL_IMAGE_STD,
                 resize=config.ARL_RESIZE,
                 crop_size=config.ARL_CROP_SIZE,
                 apply_imgaug=False,
                 # TRAIN OR EVAL
                 is_train=False,
                 is_eval=False,
                 ):

        self.dataset_dir = dataset_dir
        # loading ARL AffPose dataset.
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
        self.obj_masks_ids = [splitext(file)[0] for file in listdir(self.obj_masks_dir) if not file.startswith('.')]
        self.obj_part_masks_ids = [splitext(file)[0] for file in listdir(self.obj_part_masks_dir) if not file.startswith('.')]
        self.aff_masks_ids = [splitext(file)[0] for file in listdir(self.aff_masks_dir) if not file.startswith('.')]
        self.depth_ids = [splitext(file)[0] for file in listdir(self.depth_dir) if not file.startswith('.')]
        assert(len(self.rgb_ids) == len(self.obj_masks_ids))
        print(f'Dataset has {len(self.rgb_ids)} examples .. {dataset_dir}')

        # sorting images.
        self.rgb_ids = np.sort(np.array(self.rgb_ids))
        self.obj_masks_ids = np.sort(np.array(self.obj_masks_ids))
        self.obj_part_masks_ids = np.sort(np.array(self.obj_part_masks_ids))
        self.aff_masks_ids = np.sort(np.array(self.aff_masks_ids))
        self.depth_ids = np.sort(np.array(self.depth_ids))

        # # TODO: reduce dataset.
        # SELECT_EVERY_ITH_FRAME = 5
        # total_idx = np.arange(0, len(self.rgb_ids), SELECT_EVERY_ITH_FRAME)
        # self.rgb_ids = self.rgb_ids[total_idx]
        # self.obj_masks_ids = self.obj_masks_ids[total_idx]
        # self.obj_part_masks_ids = self.obj_part_masks_ids[total_idx]
        # self.aff_masks_ids = self.aff_masks_ids[total_idx]
        # self.depth_ids = self.depth_ids[total_idx]
        # print(f'Dataset has {len(self.rgb_ids)} examples .. {dataset_dir}')

        # Augmenting images.
        self.apply_imgaug = apply_imgaug
        self.affine, self.colour_aug, self.depth_aug = dataset_utils.get_image_augmentations()

    def __len__(self):
        return len(self.rgb_ids)

    def apply_imgaug_to_imgs(self, rgb, depth, obj_mask, aff_mask, obj_part_mask):
        rgb, depth = np.array(rgb), np.array(depth)
        aff_mask, obj_mask, obj_part_mask = np.array(aff_mask), np.array(obj_mask), np.array(obj_part_mask)

        H, W, C = rgb.shape[0], rgb.shape[1], rgb.shape[2]

        concat_img = np.zeros(shape=(H, W, C + 1))
        concat_img[:, :, :C] = rgb
        concat_img[:, :, -1] = depth
        concat_img = np.array(concat_img, dtype=np.uint8)

        concat_mask = np.zeros(shape=(H, W, 3))
        concat_mask[:, :, 0] = aff_mask
        concat_mask[:, :, 1] = obj_mask
        concat_mask[:, :, 2] = obj_part_mask
        concat_mask = np.array(concat_mask, dtype=np.uint8)

        segmap = SegmentationMapsOnImage(concat_mask, shape=np.array(rgb).shape)
        aug_concat_img, segmap = self.affine(image=concat_img, segmentation_maps=segmap)
        aug_concat_mask = segmap.get_arr()

        rgb = aug_concat_img[:, :, :C]
        depth = aug_concat_img[:, :, -1]

        aff_mask = aug_concat_mask[:, :, 0]
        obj_mask = aug_concat_mask[:, :, 1]
        obj_part_mask = aug_concat_mask[:, :, 2]

        rgb = self.colour_aug(image=rgb)
        depth = self.depth_aug(image=depth)

        rgb = np.array(rgb, dtype=np.uint8)
        aff_mask = np.array(aff_mask, dtype=np.uint8)
        obj_mask = np.array(obj_mask, dtype=np.uint8)
        obj_part_mask = np.array(obj_part_mask, dtype=np.uint8)
        depth = np.array(depth, dtype=np.uint8)

        # Uncomment to check resulting images with augmentation.
        # helper_utils.print_class_labels(aff_mask)
        # helper_utils.print_class_labels(obj_mask)
        # helper_utils.print_class_labels(obj_part_mask)
        # cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        # color_aff_mask = arl_affpose_dataset_utils.colorize_aff_mask(aff_mask)
        # cv2.imshow('color_aff_mask', cv2.cvtColor(color_aff_mask, cv2.COLOR_BGR2RGB))
        # color_obj_mask = arl_affpose_dataset_utils.colorize_obj_mask(obj_mask)
        # cv2.imshow('color_obj_mask', cv2.cvtColor(color_obj_mask, cv2.COLOR_BGR2RGB))
        # color_obj_part_mask = arl_affpose_dataset_utils.convert_obj_part_mask_to_obj_mask(obj_part_mask)
        # color_obj_part_mask = arl_affpose_dataset_utils.colorize_obj_mask(color_obj_part_mask)
        # cv2.imshow('color_obj_part_mask', cv2.cvtColor(color_obj_part_mask, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)

        return rgb, depth, obj_mask, aff_mask, obj_part_mask

    def __getitem__(self, index):

        # loading images.
        idx = self.rgb_ids[index]
        img_file = glob(self.rgb_dir + idx + self.rgb_suffix + '.*')
        depth_file = glob(self.depth_dir + idx + self.depth_suffix + '.*')
        obj_mask_file = glob(self.obj_masks_dir + idx + self.obj_masks_suffix + '.*')
        obj_part_mask_file = glob(self.obj_part_masks_dir + idx + self.obj_part_masks_suffix + '.*')
        aff_mask_file = glob(self.aff_masks_dir + idx + self.aff_masks_suffix + '.*')

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(depth_file) == 1, f'Either no image or multiple images found for the ID {idx}: {depth_file}'
        assert len(obj_mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {obj_mask_file}'
        assert len(obj_part_mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {obj_part_mask_file}'
        assert len(aff_mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {aff_mask_file}'

        image = Image.open(img_file[0]).convert('RGB')
        depth_16bit = cv2.imread(depth_file[0], -1)
        depth_16bit = np.array(depth_16bit, dtype=np.uint16)
        depth_8bit = dataset_utils.convert_16_bit_depth_to_8_bit(depth_16bit)
        obj_mask = Image.open(obj_mask_file[0])
        obj_part_mask = Image.open(obj_part_mask_file[0])
        aff_mask = Image.open(aff_mask_file[0])

        # resize and crop images.
        image = np.array(image, dtype=np.uint8)
        depth_16bit = np.array(depth_16bit, dtype=np.float16)
        depth_8bit = np.array(depth_8bit, dtype=np.uint8)
        obj_mask = np.array(obj_mask, dtype=np.uint8)
        obj_part_mask = np.array(obj_part_mask, dtype=np.uint8)
        aff_mask = np.array(aff_mask, dtype=np.uint8)

        # masked depth for stats.
        # masked_label = np.ma.getmaskarray(np.ma.masked_not_equal(obj_mask, 0)).astype(np.uint8)
        # masked_depth_16bit = masked_label * depth_16bit.copy()

        image = cv2.resize(image, self.RESIZE, interpolation=cv2.INTER_CUBIC)
        depth_8bit = cv2.resize(depth_8bit, self.RESIZE, interpolation=cv2.INTER_CUBIC)
        obj_mask = cv2.resize(obj_mask, self.RESIZE, interpolation=cv2.INTER_NEAREST)
        obj_part_mask = cv2.resize(obj_part_mask, self.RESIZE, interpolation=cv2.INTER_NEAREST)
        aff_mask = cv2.resize(aff_mask, self.RESIZE, interpolation=cv2.INTER_NEAREST)

        image = dataset_utils.crop(image, self.CROP_SIZE, is_img=True)
        depth_8bit = dataset_utils.crop(depth_8bit, self.CROP_SIZE)
        obj_mask = dataset_utils.crop(obj_mask, self.CROP_SIZE)
        obj_part_mask = dataset_utils.crop(obj_part_mask, self.CROP_SIZE)
        aff_mask = dataset_utils.crop(aff_mask, self.CROP_SIZE)

        # applying image augmentation.
        if self.apply_imgaug:
            image, depth_8bit, obj_mask, aff_mask, obj_part_mask = \
                self.apply_imgaug_to_imgs(rgb=image,
                                          depth=depth_8bit,
                                          obj_mask=obj_mask,
                                          aff_mask=aff_mask,
                                          obj_part_mask=obj_part_mask
                                          )

        image = np.array(image, dtype=np.uint8)
        H, W = image.shape[0], image.shape[1]

        # Now getting binary masks for MaskRCNN or AffNet Mask Branch.
        obj_part_ids_list = []
        obj_ids, obj_boxes, aff_ids, aff_boxes = [], [], [], []
        obj_binary_masks, aff_binary_masks = [], []
        _obj_ids = np.unique(obj_mask)[1:]
        _obj_part_ids = np.unique(obj_part_mask)[1:]
        for idx, obj_id in enumerate(_obj_ids):

            # Object bboxs & masks.
            obj_ids.append(obj_id)
            mask_obj_label = np.ma.getmaskarray(np.ma.masked_equal(obj_mask.copy(), obj_id))
            obj_binary_masks.append(mask_obj_label)
            _obj_boxes = dataset_utils.get_bbox(mask=mask_obj_label, obj_ids=np.array([1]), img_width=H, img_height=W)
            obj_boxes.append(_obj_boxes)

            # Object part.
            obj_part_ids = arl_affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
            for obj_part_id in obj_part_ids:
                if obj_part_id in _obj_part_ids:
                    aff_id = arl_affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                    # print(f"Obj Id:{obj_id}, Object: {arl_affpose_dataset_utils.map_obj_id_to_name(obj_id)}, "
                    #       f"Obj_part_id:{obj_part_id}, Aff: {aff_id}")

                    # Affordance.
                    aff_ids.append(aff_id)
                    obj_part_ids_list.append(obj_part_id)
                    mask_obj_part_label = np.ma.getmaskarray(np.ma.masked_equal(obj_part_mask.copy(), obj_part_id))
                    aff_binary_masks.append(mask_obj_part_label)
                    _aff_boxes = dataset_utils.get_bbox(mask=mask_obj_part_label, obj_ids=np.array([1]), img_width=H, img_height=W)
                    aff_boxes.append(_aff_boxes)

            # Uncomment to check binary masks.
            # color_label = arl_affpose_dataset_utils.colorize_obj_mask(mask_obj_label.copy())
            # color_label = cv2.addWeighted(image, 0.35, color_label, 0.65, 0)
            #
            # x1, y1, x2, y2 = _obj_boxes[0]
            # color_label = cv2.rectangle(color_label, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #
            # color_label = cv2.putText(color_label,
            #                           arl_affpose_dataset_utils.map_obj_id_to_name(obj_id),
            #                           (x1, y1 - 5),
            #                           cv2.FONT_ITALIC,
            #                           0.4,
            #                           (255, 255, 255))
            #
            # cv2.imshow('color_label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)

        # formatting above.
        obj_ids = np.array(obj_ids).reshape(-1)
        obj_boxes = np.array(obj_boxes).reshape(-1, 4)
        obj_binary_masks = np.array(obj_binary_masks).reshape(-1, H, W)
        aff_ids = np.array(aff_ids).reshape(-1)
        aff_boxes = np.array(aff_boxes).reshape(-1, 4)
        aff_binary_masks = np.array(aff_binary_masks).reshape(-1, H, W)
        obj_part_ids = np.array(obj_part_ids_list).reshape(-1)

        target = {}
        target["image_id"] = torch.tensor([index])
        # torch maskrcnn
        target["labels"] = torch.as_tensor(obj_ids, dtype=torch.int64)
        target["boxes"] = torch.as_tensor(obj_boxes, dtype=torch.float32)
        target["masks"] = torch.as_tensor(obj_binary_masks, dtype=torch.uint8)
        # original mask and binary masks.
        target['obj_mask'] = torch.as_tensor(np.array(obj_mask, dtype=np.uint8), dtype=torch.uint8)
        target["obj_binary_masks"] = torch.as_tensor(obj_binary_masks, dtype=torch.uint8)
        target["aff_mask"] = torch.as_tensor(np.array(aff_mask, dtype=np.uint8), dtype=torch.uint8)
        target["aff_binary_masks"] = torch.as_tensor(aff_binary_masks, dtype=torch.uint8)
        target["obj_part_mask"] = torch.as_tensor(np.array(obj_part_mask, dtype=np.uint8), dtype=torch.uint8)
        # ids and bboxs.
        target["obj_ids"] = torch.as_tensor(obj_ids, dtype=torch.int64)
        target["obj_boxes"] = torch.as_tensor(obj_boxes, dtype=torch.float32)
        target["aff_ids"] = torch.as_tensor(aff_ids, dtype=torch.int64)
        target["aff_boxes"] = torch.as_tensor(aff_boxes, dtype=torch.float32)
        target["obj_part_ids"] = torch.as_tensor(obj_part_ids, dtype=torch.int64)
        # viewing how far objects are from the camera using depth images and object masks.
        target["depth_8bit"] = torch.as_tensor(depth_8bit, dtype=torch.float32)
        target["depth_16bit"] = torch.as_tensor(depth_16bit, dtype=torch.float32)
        # target["masked_depth_16bit"] = torch.as_tensor(masked_depth_16bit, dtype=torch.float32)

        # print()
        # print(f'{target["obj_binary_masks"].size()}')
        # print(f'{target["obj_ids"].size()}')
        # print(f'{target["obj_boxes"].size()}')

        # sent to transform.
        if self.is_train or self.is_eval:
            img, target = self.transform(image, target)
        else:
            img = np.array(image, dtype=np.uint8)

        return img, target