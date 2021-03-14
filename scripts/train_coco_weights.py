# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

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

from model.MaskRCNN import ResNetMaskRCNN

from dataset.COCODataset import COCODataSet

######################
######################

from utils.vision.engine import train_one_epoch, evaluate
from utils.vision import utils
import utils.vision.transforms as T

from utils import helper_utils

######################
######################

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

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
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ######################
    # dataset
    ######################

    num_classes = 91
    dataset_dir = '/data/Akeaveny/Datasets/COCO/'
    split = 'val2017'

    dataset = COCODataSet(dataset_dir=dataset_dir,
                          split=split,
                          ###
                          is_train=True,
                          )

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    dataset_test = COCODataSet(dataset_dir=dataset_dir,
                          split=split,
                          ###
                          is_eval=True,
                          )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

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
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    ######################
    ######################

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    
if __name__ == "__main__":
    main()
