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
sys.path.append('..')
# print(sys.path)

import cfg as config

######################
######################

from dataset.COCODataset import COCODataSet

from model.MaskRCNN import ResNetMaskRCNN
from scripts.utils import train_helpers

######################
######################

from utils.pytorch_simple_maskrcnn.engine import train_one_epoch, evaluate
from utils.vision import utils
import utils.vision.transforms as T

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

    num_classes = 79 + 1
    dataset_dir = '/data/Akeaveny/Datasets/COCO/'
    train_split = 'train2017'
    val_split = 'val2017'

    ### train
    print("\nloading train ..")
    dataset = COCODataSet(dataset_dir=dataset_dir,
                          split=train_split,
                          ###
                          is_train=True,
                          )

    # np.random.seed(config.RANDOM_SEED)
    # total_idx = np.arange(0, len(dataset), 1)
    # train_idx = np.random.choice(total_idx, size=int(80), replace=False)
    # dataset = torch.utils.data.Subset(dataset, train_idx)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    print(f"train has {len(data_loader)} images ..")

    ### test
    print("\nloading test ..")
    dataset_test = COCODataSet(dataset_dir=dataset_dir,
                          split=val_split,
                          ###
                          is_eval=True,
                          )

    # np.random.seed(config.RANDOM_SEED)
    # total_idx = np.arange(0, len(dataset_test), 1)
    # test_idx = np.random.choice(total_idx, size=int(20), replace=False)
    # dataset_test = torch.utils.data.Subset(dataset_test, test_idx)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    print(f"test has {len(data_loader_test)} images ..")

    #######################
    ### model
    #######################

    # model = get_model_instance_segmentation(pretrained=config.IS_PRETRAINED, num_classes=num_classes)
    # model.to(config.DEVICE)

    model = ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=num_classes)
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
    num_epochs = 5
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