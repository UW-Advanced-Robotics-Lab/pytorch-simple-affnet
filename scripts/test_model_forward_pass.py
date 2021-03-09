from collections import OrderedDict

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

###########################
###########################

# from pathlib import Path
# ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

###########################
###########################

from dataset.UMDDataset import BasicDataSet
from dataset.utils import umd_utils

from dataset.PennFudanDataset import PennFudanDataset
from model.MaskRCNN import ResNetMaskRCNN

######################
######################

def main():

    #######################
    ### model
    #######################
    model = ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    ###########################
    ### freezing resnet layers
    ###########################

    # print(f'\nfreezing backbone')
    # for name, param in model.backbone.named_parameters():
    #     param.requires_grad = False
    #
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print("\tupdate:", name)
    #     else:
    #         print("\tfrozen:", name)
    #
    # print(f'\nunfreezing backbone')
    # for name, param in model.backbone.named_parameters():
    #     param.requires_grad = True
    #
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print("\tupdate:", name)
    #     else:
    #         print("\tfrozen:", name)

    # construct an optimizer
    # params = [p for p in model.parameters() if p.requires_grad]

    #######################
    ### random for debugguing
    #######################
    image = torch.randn(1, 3, 128, 128)

    img_id    = 1
    labels    = torch.tensor([1])
    bbox      = torch.tensor([[99, 17, 113, 114]], dtype=torch.float32)
    mask      = torch.randn(1, 128, 128)

    target = {}
    target["image_id"] = torch.tensor([img_id])
    target["boxes"] = bbox
    target["labels"] = labels
    target["masks"] = mask

    #######################
    ### data loader
    #######################

    # PennFudan
    # root_dataset_path = '/data/Akeaveny/Datasets/PennFudanPed'
    # dataset = PennFudanDataset(root_dataset_path, is_train=True)
    #
    # UMD
    # dataset = BasicDataSet(
    #     ### REAL
    #     dataset_dir=config.DATA_DIRECTORY_TARGET_TEST,
    #     mean=config.IMG_MEAN_TARGET,
    #     std=config.IMG_STD_TARGET,
    #     resize=config.RESIZE_TARGET,
    #     crop_size=config.INPUT_SIZE_TARGET,
    #     ###
    #     is_train=True,
    #     ### EXTENDING DATASET
    #     extend_dataset=False,
    #     ### IMGAUG
    #     apply_imgaug=True)
    #
    # # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    #
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    #
    # _, batch = enumerate(data_loader).__next__()
    # image, target = batch
    #
    # target['boxes'] = target['boxes'].squeeze(0)
    # target['labels'] = target['labels'].squeeze(0)
    # target['masks'] = target['masks'].squeeze(0)

    #######################
    #######################

    image = image.to(config.DEVICE)
    target = {k: v.to(config.DEVICE) for k, v in target.items()}

    with torch.no_grad():
        #######################
        ### eval for pred
        #######################
        model.eval()
        outputs = model(image, target)

    outputs = [{k: v.to(config.CPU_DEVICE) for k, v in t.items()} for t in outputs]
    outputs = outputs.pop()

if __name__ == "__main__":
    main()

