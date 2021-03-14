import os
from os import listdir
from os.path import splitext
from glob import glob

import copy

import logging

import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

######################
######################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

######################
######################

def object_id_to_name(object_id):

    coco_2017_class_names_dict = \
        {0: '__background__',
         1: 'person',
         2: 'bicycle',
         3: 'car',
         4: 'motorcycle',
         5: 'airplane',
         6: 'bus',
         7: 'train',
         8: 'truck',
         9: 'boat',
         10: 'traffic light',
         11: 'fire hydrant',
         12: 'stop sign',
         13: 'parking meter',
         14: 'bench',
         15: 'bird',
         16: 'cat',
         17: 'dog',
         18: 'horse',
         19: 'sheep',
         20: 'cow',
         21: 'elephant',
         22: 'bear',
         23: 'zebra',
         24: 'giraffe',
         25: 'backpack',
         26: 'umbrella',
         27: 'handbag',
         28: 'tie',
         29: 'suitcase',
         30: 'frisbee',
         31: 'skis',
         32: 'snowboard',
         33: 'sports ball',
         34: 'kite',
         35: 'baseball bat',
         36: 'baseball glove',
         37: 'skateboard',
         38: 'surfboard',
         39: 'tennis racket',
         40: 'bottle',
         41: 'wine glass',
         42: 'cup',
         43: 'fork',
         44: 'knife',
         45: 'spoon',
         46: 'bowl',
         47: 'banana',
         48: 'apple',
         49: 'sandwich',
         50: 'orange',
         51: 'broccoli',
         52: 'carrot',
         53: 'hot dog',
         54: 'pizza',
         55: 'donut',
         56: 'cake',
         57: 'chair',
         58: 'couch',
         59: 'potted plant',
         60: 'bed',
         61: 'dining table',
         62: 'toilet',
         63: 'tv',
         64: 'laptop',
         65: 'mouse',
         66: 'remote',
         67: 'keyboard',
         68: 'cell phone',
         69: 'microwave',
         70: 'oven',
         71: 'toaster',
         72: 'sink',
         73: 'refrigerator',
         74: 'book',
         75: 'clock',
         76: 'vase',
         77: 'scissors',
         78: 'teddy bear',
         79: 'hair drier',
         80: 'toothbrush'}

    return coco_2017_class_names_dict[object_id+1]