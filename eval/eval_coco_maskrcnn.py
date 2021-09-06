import bisect
import glob
import os
import re
import time
import copy

import numpy as np
import cv2

import torch
from sklearn.metrics import precision_score

import sys
sys.path.append('../')

import config
from model.maskrcnn import maskrcnn
from model import model_utils

from dataset.coco import coco_dataset_loaders
from dataset.coco import coco_dataset_utils

SHOW_PLOT = True


def main():

    # Init folders
    print('\neval in .. {}'.format(config.COCO_EVAL_SAVE_FOLDER))

    if not os.path.exists(config.COCO_EVAL_SAVE_FOLDER):
        os.makedirs(config.COCO_EVAL_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.COCO_EVAL_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    # Load the Model.
    print()

    # Compare Pytorch-Simple-MaskRCNN. with Torchvision MaskRCNN.
    model = model_utils.get_model_instance_segmentation(pretrained=config.IS_PRETRAINED, num_classes=config.COCO_NUM_CLASSES)
    # model = maskrcnn.ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=config.COCO_NUM_CLASSES)
    model.to(config.DEVICE)

    # Load saved weights.
    print(f"\nrestoring pre-trained MaskRCNN weights: {config.RESTORE_COCO_MASKRCNN_WEIGHTS} .. ")
    checkpoint = torch.load(config.RESTORE_COCO_MASKRCNN_WEIGHTS, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Load the dataset.
    test_loader = coco_dataset_loaders.load_coco_eval_datasets()

    ######################
    ######################
    print('\nstarting eval ..\n')
    SAVED_MODEL_PATH = config.MODEL_SAVE_PATH + 'coco_eval.pth'
    eval_output, iter_eval = coco_dataset_utils.evaluate(model,
                                                      test_loader,
                                                      device=config.DEVICE,
                                                      saved_model_path=SAVED_MODEL_PATH,
                                                      generate=True)
    print(f'\neval_output:{eval_output}')

if __name__ == "__main__":
    main()
