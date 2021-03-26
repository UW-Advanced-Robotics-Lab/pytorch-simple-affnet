import bisect
import glob
import os
import re
import time
import copy

import numpy as np
import cv2

import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from utils.vision.engine import train_one_epoch, evaluate
from utils.vision import utils
import utils.vision.transforms as T

from torch.utils import data
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

###########################
###########################

# from pathlib import Path
# ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

######################
######################

from dataset.UMDDataset import BasicDataSet
from dataset.utils.UMD import umd_utils

from model.MaskRCNN import ResNetMaskRCNN

from utils import helper_utils

######################
######################

def get_model_instance_segmentation(pretrained, num_classes):
    print('loading torchvision maskrcnn ..')
    print(f'num classes:{num_classes} ..')
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def main():

    ######################
    # INIT
    ######################
    print('\neval in .. {}'.format(config.EVAL_SAVE_FOLDER))

    if not os.path.exists(config.EVAL_SAVE_FOLDER):
        os.makedirs(config.TEST_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.EVAL_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    #######################
    ### data loader
    #######################
    print()

    test_dataset = BasicDataSet(
        ### REAL
        dataset_dir=config.DATA_DIRECTORY_TEST,
        mean=config.IMG_MEAN,
        std=config.IMG_STD,
        resize=config.RESIZE,
        crop_size=config.INPUT_SIZE,
        ###
        is_train=True,
        ### EXTENDING DATASET
        extend_dataset=False,
        ### IMGAUG
        apply_imgaug=False)

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(test_dataset), 1)
    test_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
    test_dataset = Subset(test_dataset, test_idx)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=config.NUM_WORKERS,
                                              collate_fn=utils.collate_fn)

    print(f"test has {len(test_loader)} images ..")
    assert (len(test_loader) >= int(config.NUM_TEST))

    #######################
    ### model
    #######################
    print()

    # model = get_model_instance_segmentation(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    # model.to(config.DEVICE)

    model = ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    print(f"\nrestoring pre-trained MaskRCNN weights: {config.RESTORE_TRAINED_WEIGHTS} .. ")
    checkpoint = torch.load(config.RESTORE_TRAINED_WEIGHTS, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])

    model.eval() # todo: issues with batch size = 1

    #######################
    ### eval for pred
    #######################
    print()

    for image_idx, (images, targets) in enumerate(test_loader):
        image, target = copy.deepcopy(images), copy.deepcopy(targets)
        images = list(image.to(config.DEVICE) for image in images)

        with torch.no_grad():
            outputs = model(images)
            outputs = [{k: v.to(config.CPU_DEVICE) for k, v in t.items()} for t in outputs]

        #######################
        ### todo: formatting input
        #######################
        image = image[0]
        image = image.to(config.CPU_DEVICE)
        img = np.squeeze(np.array(image)).transpose(1, 2, 0)
        height, width = img.shape[:2]

        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        target = helper_utils.format_target_data(img, target)

        #######################
        ### todo: formatting output
        #######################
        outputs = outputs.pop()

        scores = np.array(outputs['scores'], dtype=np.float32).flatten()

        labels = np.array(outputs['labels'], dtype=np.int32).flatten()
        boxes = np.array(outputs['boxes'], dtype=np.int32).reshape(-1, 4)

        binary_masks = np.squeeze(np.array(outputs['masks'] > config.CONFIDENCE_THRESHOLD, dtype=np.uint8))

        aff_labels = labels.copy()
        if 'aff_labels' in outputs.keys():
            aff_labels = np.array(outputs['aff_labels'], dtype=np.int32)

        #######################
        ### bbox
        #######################
        bbox_img = helper_utils.draw_bbox_on_img(image=img,
                                                 labels=labels,
                                                 boxes=boxes,
                                                 scores=scores)
        cv2.imshow('bbox', cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))

        #######################
        ### masks
        #######################
        mask = helper_utils.get_segmentation_masks(image=img,
                                                   labels=aff_labels,
                                                   binary_masks=binary_masks,
                                                   scores=scores)
        print(f'\nscores:{scores}')
        helper_utils.print_class_labels(target['gt_mask'])
        helper_utils.print_class_labels(mask)

        pred_color_mask = umd_utils.colorize_mask(mask)
        cv2.imshow('pred', pred_color_mask)

        gt_color_mask = umd_utils.colorize_mask(target['gt_mask'])
        cv2.imshow('gt', gt_color_mask)

        gt_name = config.EVAL_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['gt_mask'])

        pred_name = config.EVAL_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, mask)

        #######################
        #######################
        cv2.waitKey(1)

    #######################
    #######################
    os.chdir(config.MATLAB_SCRIPTS_DIR)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    Fwb = eng.evaluate_UMD(config.EVAL_SAVE_FOLDER, nargout=1)
    # print(f'Fwb:{Fwb} ..')
    os.chdir(config.ROOT_DIR_PATH)

if __name__ == "__main__":
    main()