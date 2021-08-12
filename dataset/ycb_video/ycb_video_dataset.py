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
from dataset.ycb_video import ycb_video_dataset_utils
from dataset import dataset_utils


class YCBVideoPoseDataset(data.Dataset):

    def __init__(self,
                 dataset_dir=config.YCB_DATASET_ROOT_PATH,
                 image_path_txt_file=config.YCB_TRAIN_FILE,
                 image_domain=config.YCB_IMAGE_DOMAIN,
                 # configs for pre-processing our dataset.
                 mean=config.YCB_IMAGE_MEAN,
                 std=config.YCB_IMAGE_STD,
                 resize=config.YCB_RESIZE,
                 crop_size=config.YCB_CROP_SIZE,
                 apply_imgaug=False,
                 # PyTorch Transfroms.
                 is_train=False,
                 is_eval=False,
                 ):

        self.dataset_dir = dataset_dir
        self.image_path_txt_file = image_path_txt_file
        self.image_domain = image_domain
        assert self.image_domain == 'Real' or self.image_domain == 'Syn', 'Images must be Real or Syn'
        # load image paths.
        self.load_images_paths()

        # configs for pre-processing our dataset.
        self.mean = mean
        self.std = std
        self.RESIZE = resize
        self.CROP_SIZE = crop_size

        # PyTorch transforms.
        self.is_train = is_train
        self.is_eval = is_eval
        self.transform = dataset_utils.get_transform()

        # Augmenting images.
        self.apply_imgaug = apply_imgaug
        self.affine, self.colour_aug, self.depth_aug = dataset_utils.get_image_augmentations()

    def __len__(self):
        return len(self.image_paths)

    def load_images_paths(self):

        self.all_image_paths = []
        self.real_image_paths = []
        self.syn_image_paths = []
        input_file = open(self.dataset_dir + '/' + self.image_path_txt_file)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real_image_paths.append(input_line)
            else:
                self.syn_image_paths.append(input_line)
            self.all_image_paths.append(input_line)
        input_file.close()

        if self.image_domain == 'Real':
            self.image_paths = self.real_image_paths
        elif self.image_domain == 'Syn':
            self.image_paths = self.syn_image_paths

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
        # color_aff_mask = ycb_video_dataset_utils.colorize_aff_mask(aff_mask)
        # cv2.imshow('color_aff_mask', cv2.cvtColor(color_aff_mask, cv2.COLOR_BGR2RGB))
        # color_obj_mask = ycb_video_dataset_utils.colorize_obj_mask(obj_mask)
        # cv2.imshow('color_obj_mask', cv2.cvtColor(color_obj_mask, cv2.COLOR_BGR2RGB))
        # color_obj_part_mask = ycb_video_dataset_utils.convert_obj_part_mask_to_obj_mask(obj_part_mask)
        # color_obj_part_mask = ycb_video_dataset_utils.colorize_obj_mask(color_obj_part_mask)
        # cv2.imshow('color_obj_part_mask', cv2.cvtColor(color_obj_part_mask, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)

        return rgb, depth, obj_mask, aff_mask, obj_part_mask

    def __getitem__(self, index):

        # loading images.
        img_file = '{0}/{1}-color.png'.format(self.dataset_dir, self.image_paths[index])
        depth_file = '{0}/{1}-depth.png'.format(self.dataset_dir, self.image_paths[index])
        obj_mask_file = '{0}/{1}-label.png'.format(self.dataset_dir, self.image_paths[index])
        obj_part_mask_file = '{0}/{1}-obj_part_label.png'.format(self.dataset_dir, self.image_paths[index])
        aff_mask_file = '{0}/{1}-aff_label.png'.format(self.dataset_dir, self.image_paths[index])

        image = Image.open(img_file).convert('RGB')
        depth_16bit = cv2.imread(depth_file, -1)
        depth_16bit = np.array(depth_16bit, dtype=np.uint16)
        depth_8bit = dataset_utils.convert_16_bit_depth_to_8_bit(depth_16bit)
        obj_mask = Image.open(obj_mask_file)
        obj_part_mask = Image.open(obj_part_mask_file)
        aff_mask = Image.open(aff_mask_file)

        # format dtypes.
        image = np.array(image, dtype=np.uint8)
        depth_16bit = np.array(depth_16bit, dtype=np.float16)
        depth_8bit = np.array(depth_8bit, dtype=np.uint8)
        obj_mask = np.array(obj_mask, dtype=np.uint8)
        obj_part_mask = np.array(obj_part_mask, dtype=np.uint8)
        aff_mask = np.array(aff_mask, dtype=np.uint8)

        # masked depth for stats.
        # masked_label = np.ma.getmaskarray(np.ma.masked_not_equal(obj_mask, 0)).astype(np.uint8)
        # masked_depth_16bit = masked_label * depth_16bit.copy()

        # resize and crop images.
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
            obj_part_ids = ycb_video_dataset_utils.map_obj_ids_to_obj_part_ids(obj_id)
            for obj_part_id in obj_part_ids:
                if obj_part_id in _obj_part_ids:
                    aff_id = ycb_video_dataset_utils.map_obj_part_ids_to_aff_ids(obj_part_id)
                    # print(f"Obj Id:{obj_id}, Object: {ycb_video_dataset_utils.map_obj_id_to_name(obj_id)}, "
                    #       f"Obj_part_id:{obj_part_id}, Aff: {aff_id}")

                    # Affordance.
                    aff_ids.append(aff_id)
                    obj_part_ids_list.append(obj_part_id)
                    mask_obj_part_label = np.ma.getmaskarray(np.ma.masked_equal(obj_part_mask.copy(), obj_part_id))
                    aff_binary_masks.append(mask_obj_part_label)
                    _aff_boxes = dataset_utils.get_bbox(mask=mask_obj_part_label, obj_ids=np.array([1]), img_width=H, img_height=W)
                    aff_boxes.append(_aff_boxes)

            # Uncomment to check binary masks.
            color_label = ycb_video_dataset_utils.colorize_obj_mask(mask_obj_label.copy())
            color_label = cv2.addWeighted(image, 0.35, color_label, 0.65, 0)

            # x1, y1, x2, y2 = _obj_boxes[0]
            # color_label = cv2.rectangle(color_label, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #
            # color_label = cv2.putText(color_label,
            #                           ycb_video_dataset_utils.map_obj_id_to_name(obj_id),
            #                           (x1, y1 - 5),
            #                           cv2.FONT_ITALIC,
            #                           0.4,
            #                           (255, 255, 255))
            #
            # cv2.imshow('color_label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)

        # formatting above.
        obj_ids = np.squeeze(np.array(obj_ids))
        obj_boxes = np.squeeze(np.array(obj_boxes))
        obj_binary_masks = np.squeeze(np.array(obj_binary_masks))
        aff_ids = np.squeeze(np.array(aff_ids))
        aff_boxes = np.squeeze(np.array(aff_boxes))
        aff_binary_masks = np.squeeze(np.array(aff_binary_masks))
        obj_part_ids = np.squeeze(np.array(obj_part_ids_list))

        target = {}
        target["image_id"] = torch.tensor([index])
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

        # sent to transform.
        if self.is_train or self.is_eval:
            img, target = self.transform(image, target)
        else:
            img = np.array(image, dtype=np.uint8)

        return img, target