import os
import random

import numpy as np

from PIL import Image

import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

######################
######################

# from pathlib import Path
# ROOT_DIR_PATH = Path(__file__).resolve().parents[1]

import sys
sys.path.append('../../')
# print(sys.path)

import cfg as config

######################
######################

from dataset.COCODataset import COCODataSet

from model.MaskRCNN import ResNetMaskRCNN
from scripts.utils import train_helpers

######################
######################

from scripts.coco_pretrained_weights.pytorch_simple_maskrcnn.engine import train_one_epoch, evaluate
from scripts.tutorial.vision import utils
import torchvision.transforms as T

from utils import helper_utils

######################
######################

def main():

    ######################
    # INIT
    ######################

    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    ######################
    # LOGGING
    ######################
    print('\nsaving run in .. {}'.format(config.SNAPSHOT_DIR))

    if not os.path.exists(config.SNAPSHOT_DIR):
        os.makedirs(config.SNAPSHOT_DIR)

    ######################
    # dataset
    ######################

    ### test
    print("\nloading test ..")
    dataset_test = COCODataSet(dataset_dir=config.COCO_ROOT_DATA_PATH,
                          split=config.COCO_VAL_SPLIT,
                          ###
                          is_eval=True,
                          )

    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(dataset_test), 1)
    test_idx = np.random.choice(total_idx, size=int(100), replace=False)
    dataset_test = torch.utils.data.Subset(dataset_test, test_idx)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    print(f"test has {len(data_loader_test)} images ..")

    #######################
    ### model
    #######################

    model = ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=config.COCO_NUM_CLASSES)
    model.to(config.DEVICE)

    print(f"\nrestoring pre-trained MaskRCNN weights: {config.RESTORE_TRAINED_WEIGHTS} .. ")
    checkpoint = torch.load(config.RESTORE_TRAINED_WEIGHTS, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])

    model.eval()  # todo: issues with batch size = 1

    ######################
    ######################
    print('\nstarting eval ..\n')
    SAVED_MODEL_PATH = config.MODEL_SAVE_PATH + 'coco_eval.pth'
    eval_output, iter_eval = evaluate(model, data_loader_test,
                                         device=config.DEVICE,
                                         saved_model_path=SAVED_MODEL_PATH,
                                         generate=True)
    print(f'\neval_output:{eval_output}')
    
if __name__ == "__main__":
    main()
