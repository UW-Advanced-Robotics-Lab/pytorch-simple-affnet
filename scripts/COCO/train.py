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
    print('\nsaving run in .. {}'.format(config.TRAINED_MODELS_DIR))

    if not os.path.exists(config.TRAINED_MODELS_DIR):
        os.makedirs(config.TRAINED_MODELS_DIR)

    ######################
    # dataset
    ######################

    ### train
    print("\nloading train ..")
    dataset = COCODataSet(dataset_dir=config.COCO_ROOT_DATA_PATH,
                          split=config.COCO_TRAIN_SPLIT,
                          ###
                          is_train=True,
                          )

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=config.NUM_WORKERS,
                                              pin_memory=True,
                                              collate_fn=utils.collate_fn)
    print(f"train has {len(data_loader)} images ..")

    ### test
    print("\nloading test ..")
    dataset_test = COCODataSet(dataset_dir=config.COCO_ROOT_DATA_PATH,
                          split=config.COCO_VAL_SPLIT,
                          ###
                          is_eval=True,
                          )

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=config.NUM_WORKERS,
                                                   pin_memory=True,
                                                   collate_fn=utils.collate_fn)
    print(f"test has {len(data_loader_test)} images ..")

    #######################
    ### model
    #######################

    model = ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=config.COCO_NUM_CLASSES)
    model.to(config.DEVICE)

    ######################
    ######################

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, momentum=0.9)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    ######################
    ######################

    # let's train it for 10 epochs
    num_epochs = 10
    print(f'\nstarting training for {num_epochs} epochs ..')

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, config.DEVICE, epoch, print_freq=int(len(data_loader)/1000))

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        # CHECKPOINT_PATH = config.MODEL_SAVE_PATH + 'coco_epoch_' + np.str(epoch) + '.pth'
        # evaluate(model, data_loader_test, device=config.DEVICE, saved_model_path=CHECKPOINT_PATH)

        # checkpoint_path = config.CHECKPOINT_PATH
        CHECKPOINT_PATH = config.MODEL_SAVE_PATH + 'coco_epoch_' + np.str(epoch) + '.pth'
        train_helpers.save_checkpoint(model, optimizer, epoch, CHECKPOINT_PATH)

    print("That's it!")
    
if __name__ == "__main__":
    main()
