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
import torchvision.transforms as T

from torch.utils import data
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

###########################
###########################

import sys
sys.path.append('../')

import cfg as config

######################
######################

from model.MaskRCNN import ResNetMaskRCNN

from utils import helper_utils

from scripts.utils import dataset_helpers
from scripts.utils import train_helpers

from dataset.utils.UMD import umd_utils
from dataset.utils.Elevator import elevator_utils
from dataset.utils.ARLVicon import arl_vicon_dataset_utils
from dataset.utils.ARLAffPose import affpose_dataset_utils

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

######################
######################

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

    # test_loader = dataset_helpers.load_umd_eval_dataset()
    # test_loader = dataset_helpers.load_arl_vicon_eval_dataset()
    test_loader = dataset_helpers.load_arl_affpose_eval_datasets()

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

    model.eval() # todo: issues with batch size = 1 or frozen_batch_norm()

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
        img = np.array(img*(2**8-1), dtype=np.uint8)
        height, width = img.shape[:2]

        target = target[0]
        ### target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        target = helper_utils.format_target_data(img, target)

        #######################
        ### todo: formatting output
        #######################
        outputs = outputs.pop()

        scores = np.array(outputs['scores'], dtype=np.float32).flatten()
        labels = np.array(outputs['labels'], dtype=np.int32).flatten()
        boxes = np.array(outputs['boxes'], dtype=np.int32).reshape(-1, 4)

        # idx = np.argwhere(scores.copy() > config.CONFIDENCE_THRESHOLD)
        # # print(f'idx:{idx}')
        # scores = scores[idx]
        # labels = labels[idx]
        # # print(f'labels:{labels}')
        # boxes = boxes[idx].reshape(-1, 4)
        # # print(f'boxes:{boxes}')

        binary_masks = np.squeeze(np.array(outputs['masks'] > config.CONFIDENCE_THRESHOLD, dtype=np.uint8))
        aff_labels = labels.copy()
        if 'aff_labels' in outputs.keys():
            aff_labels = np.array(outputs['aff_labels'], dtype=np.int32)

        #######################
        ### bbox
        #######################
        ### gt
        gt_bbox_img = helper_utils.draw_bbox_on_img(image=img,
                                                    labels=target["labels"],
                                                    boxes=target["boxes"],
                                                    is_gt=True)
        # cv2.imshow('gt_bbox', cv2.cvtColor(gt_bbox_img, cv2.COLOR_BGR2RGB))

        ### pred
        pred_bbox_img = helper_utils.draw_bbox_on_img(image=img,
                                                 labels=labels,
                                                 boxes=boxes,
                                                 scores=scores)
        # cv2.imshow('pred_bbox', cv2.cvtColor(pred_bbox_img, cv2.COLOR_BGR2RGB))

        #######################
        ### masks
        #######################
        ### gt
        # TODO
        # gt_color_mask = umd_utils.colorize_aff_mask(target['gt_mask'])
        # gt_color_mask = arl_vicon_dataset_utils.colorize_obj_mask(target['gt_mask'])
        # gt_color_mask = affpose_dataset_utils.colorize_obj_mask(target['gt_mask'])
        gt_color_mask = affpose_dataset_utils.colorize_aff_mask(target['gt_mask'])

        gt_color_mask = cv2.addWeighted(gt_bbox_img, 0.35, gt_color_mask, 0.65, 0)
        cv2.imshow('gt_mask', cv2.cvtColor(gt_color_mask, cv2.COLOR_BGR2RGB))

        ### pred
        mask = helper_utils.get_segmentation_masks(image=img,
                                                   # labels=labels,
                                                   labels=aff_labels,
                                                   binary_masks=binary_masks,
                                                   scores=scores)

        # print(f'\nscores:{scores}')
        # helper_utils.print_class_labels(target['gt_mask'])
        # helper_utils.print_class_labels(mask)

        # TODO
        # pred_color_mask = umd_utils.colorize_aff_mask(mask)
        # pred_color_mask = arl_vicon_dataset_utils.colorize_obj_mask(mask)
        # pred_color_mask = affpose_dataset_utils.colorize_obj_mask(mask)
        pred_color_mask = affpose_dataset_utils.colorize_aff_mask(mask)
        pred_color_mask = cv2.addWeighted(pred_bbox_img, 0.35, pred_color_mask, 0.65, 0)
        cv2.imshow('pred_mask', cv2.cvtColor(pred_color_mask, cv2.COLOR_BGR2RGB))

        #######################
        ### obj_part masks
        #######################
        obj_part_mask = helper_utils.get_obj_part_mask(image=img,
                                                       obj_ids=labels,
                                                       bboxs=boxes,
                                                       aff_ids=aff_labels,
                                                       binary_masks=binary_masks,
                                                       scores=scores)

        # cv2.imshow('obj_part_mask', obj_part_mask*25)
        obj_mask = affpose_dataset_utils.convert_obj_part_mask_to_obj_mask(obj_part_mask)

        color_obj_mask = affpose_dataset_utils.colorize_obj_mask(obj_mask)
        color_obj_mask = cv2.addWeighted(pred_bbox_img, 0.35, color_obj_mask, 0.65, 0)
        cv2.imshow('obj_mask', cv2.cvtColor(color_obj_mask, cv2.COLOR_BGR2RGB))

        #######################
        #######################
        gt_name = config.EVAL_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['gt_mask'])

        pred_name = config.EVAL_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, mask)

        #######################
        #######################
        cv2.waitKey(0)

    #######################
    #######################
    os.chdir(config.MATLAB_SCRIPTS_DIR)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    # TODO
    # Fwb = eng.evaluate_UMD(config.EVAL_SAVE_FOLDER, nargout=1)
    # Fwb = eng.evaluate_ARLVicon(config.EVAL_SAVE_FOLDER, nargout=1)
    Fwb = eng.evaluate_ARLAffPose(config.EVAL_SAVE_FOLDER, nargout=1)
    os.chdir(config.ROOT_DIR_PATH)

if __name__ == "__main__":
    main()
