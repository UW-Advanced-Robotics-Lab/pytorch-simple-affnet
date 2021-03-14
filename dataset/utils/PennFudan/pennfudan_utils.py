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

def object_name_to_id(object_name):
    if object_name == "person":
        return 1
    else:
        assert (" --- Object does not exist in PennFudan dataset --- ")

def object_id_to_name(object_id):
    if object_id == 1:
        return "person"
    else:
        assert (" --- Object does not exist in PennFudan dataset --- ")

######################
######################

def colorize_mask(mask):

    instance_to_color = color_map()
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[mask == key] = instance_to_color[key]

    return np.squeeze(color_mask)

def color_map():
    ''' [red, blue, green]'''

    color_map_dict = {
    0:  [0,   0, 0],          # black
    1:  [255, 0, 0],     # blue
    }
    return color_map_dict