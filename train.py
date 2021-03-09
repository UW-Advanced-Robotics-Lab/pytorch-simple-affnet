import bisect
import glob
import os
import re
import time

import numpy as np
import random

import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from scripts.torchvision_mask_rcnn.vision.engine import train_one_epoch, evaluate
from scripts.torchvision_mask_rcnn.vision import utils
import scripts.torchvision_mask_rcnn.vision.transforms as T

from scripts.torchvision_mask_rcnn.vision import utils
import scripts.torchvision_mask_rcnn.vision.transforms as T

from torch.utils import data
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

###########################
###########################

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# from pathlib import Path
# ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

######################
######################

from dataset.PennFudanDataset import PennFudanDataset
from model.MaskRCNN import ResNetMaskRCNN

from dataset.UMDDataset import BasicDataSet
from dataset.utils import umd_utils

from model.MaskRCNN import ResNetMaskRCNN

from utils import helper_utils

from scripts.utils import train_helpers

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

    ### TENSORBOARD
    writer = SummaryWriter(f'{config.SNAPSHOT_DIR}')

    #######################
    ### data loader
    #######################
    train_loader, val_loader, test_loader = train_helpers.load_umd_real_datasets()

    #######################
    ### model
    #######################
    print()

    # model = get_model_instance_segmentation(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    # model.to(config.DEVICE)

    model = ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    #######################
    #######################
    print()

    # let's train it for 10 epochs
    num_epochs = config.NUM_EPOCHS
    Fwb, best_Fwb = -np.inf, -np.inf

    # print('freezing backbone weights ..\n')
    # for layer in model.backbone.parameters():
    #     layer.requires_grad = False

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, momentum=0.9)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(0 * num_epochs, 1 * num_epochs):
        # train & val for one epoch
        model, optimizer = train_helpers.train_one_epoch(model, optimizer, train_loader, config.DEVICE, epoch, writer)
        model, optimizer = train_helpers.val_one_epoch(model, optimizer, val_loader, config.DEVICE, epoch, writer)
        # eval model
        model = train_helpers.eval_model(model, test_loader)
        Fwb, best_Fwb = train_helpers.eval_Fwb(model=model, optimizer=optimizer,
                                               Fwb=Fwb, best_Fwb=best_Fwb,
                                               epoch=epoch, writer=writer)
        lr_scheduler.step()

        # checkpoint_path = config.CHECKPOINT_PATH
        CHECKPOINT_PATH = config.MODEL_SAVE_PATH + 'affnet_epoch_' + np.str(epoch) + '.pth'
        train_helpers.save_checkpoint(model, optimizer, epoch, CHECKPOINT_PATH)

    # #######################
    # #######################
    # print()
    #
    # print('unfreezing backbone weights ..\n')
    # for layer in model.backbone.parameters():
    #     layer.requires_grad = True
    #
    # # construct an optimizer
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=config.LEARNING_RATE/100, weight_decay=config.WEIGHT_DECAY, momentum=0.9)
    #
    # # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    #
    # for epoch in range(1*num_epochs, 2*num_epochs):
    #     # train & val for one epoch
    #     # train_one_epoch(model, optimizer, train_loader, config.DEVICE, epoch, print_freq=10)
    #     model, optimizer = train_helpers.train_one_epoch(model, optimizer, train_loader, config.DEVICE, epoch, writer)
    #     model, optimizer = train_helpers.val_one_epoch(model, optimizer, val_loader, config.DEVICE, epoch, writer)
    #     # eval model
    #     model = train_helpers.eval_model(model, test_loader)
    #     Fwb, best_Fwb = train_helpers.eval_Fwb(model=model, optimizer=optimizer,
    #                                            Fwb=Fwb, best_Fwb=best_Fwb,
    #                                            epoch=epoch, writer=writer)
    #     lr_scheduler.step()
    #
    #     # checkpoint_path = config.CHECKPOINT_PATH
    #     CHECKPOINT_PATH = config.MODEL_SAVE_PATH + 'affnet_epoch_' + np.str(epoch) + '.pth'
    #     train_helpers.save_checkpoint(model, optimizer, epoch, CHECKPOINT_PATH)

if __name__ == "__main__":
    main()