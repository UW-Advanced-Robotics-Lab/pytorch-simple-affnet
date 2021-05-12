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

import sys
sys.path.append('../')

import cfg as config

###########################
###########################

from model.MaskRCNN import ResNetMaskRCNN

######################
######################

def main():

    #######################
    ### model
    #######################
    model = ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

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

