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

import cfg as config

######################
## AFF
######################

def aff_id_to_name(aff_id):

    if aff_id == 1:
        return "grasp"
    elif aff_id == 2:
        return "cut"
    elif aff_id == 3:
        return "scoop"
    elif aff_id == 4:
        return "contain"
    elif aff_id == 5:
        return "pound"
    elif aff_id == 6:
        return "support"
    elif aff_id == 7:
        return "wrap-grasp"
    else:
        assert (" --- Affordance does not exist in UMD dataset --- ")

######################
######################

def colorize_aff_mask(instance_mask):

    instance_to_color = color_map_aff_id()
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[instance_mask == key] = instance_to_color[key]

    return np.squeeze(color_mask)

def color_map_aff_id():
    ''' [red, blue, green]'''

    color_map_dic = {
    0:  [0, 0, 0],
    1:  [235, 17,  17],     # grasp: red
    2:  [235, 209, 17],     # cut: yellow
    3:  [113, 235, 17],     # scoop: green
    4:  [17,  235, 202],    # contain: teal
    5:  [17,   54, 235],    # pound: blue
    6:  [129,  17, 235],    # support: purple
    7:  [235,  17, 179],    # wrap-grasp: pink
    }
    return color_map_dic

######################
######################

def object_name_to_id(object_name):
    if object_name == "bowl":
        return 1
    elif object_name == "cup":
        return 2
    elif object_name == "hammer":
        return 3
    elif object_name == "knife":
        return 4
    elif object_name == "ladle":
        return 5
    elif object_name == "mallet":
        return 6
    elif object_name == "mug":
        return 7
    elif object_name == "pot":
        return 8
    elif object_name == "saw":
        return 9
    elif object_name == "scissors":
        return 10
    elif object_name == "scoop":
        return 11
    elif object_name == "shears":
        return 12
    elif object_name == "shovel":
        return 13
    elif object_name == "spoon":
        return 14
    elif object_name == "tenderizer":
        return 15
    elif object_name == "trowel":
        return 16
    elif object_name == "turner":
        return 17
    else:
        assert (" --- Object does not exist in UMD dataset --- ")

def object_id_to_name(object_id):
    if object_id == 1:
        return "bowl"
    elif object_id == 2:
        return "cup"
    elif object_id == 3:
        return "hammer"
    elif object_id == 4:
        return "knife"
    elif object_id == 5:
        return "ladle"
    elif object_id == 6:
        return "mallet"
    elif object_id == 7:
        return "mug"
    elif object_id == 8:
        return "pot"
    elif object_id == 9:
        return "saw"
    elif object_id == 10:
        return "scissors"
    elif object_id == 11:
        return "scoop"
    elif object_id == 12:
        return "shears"
    elif object_id == 13:
        return "shovel"
    elif object_id == 14:
        return "spoon"
    elif object_id == 15:
        return "tenderizer"
    elif object_id == 16:
        return "trowel"
    elif object_id == 17:
        return "turner"
    else:
        assert (" --- Object does not exist in UMD dataset --- ")

######################
######################

def object_id_to_aff_id(object_ids):
    aff_ids = []
    for i in range(len(object_ids)):
        object_id = object_ids[i]
        if object_id == 0:  # "bowl"
            aff_ids.append([0])
        elif object_id == 1:  # "bowl"
            aff_ids.append([4])
        elif object_id == 2:  # "cup"
            aff_ids.append([4, 7])
        elif object_id == 3:  # "hammer"
            aff_ids.append([1, 5])
        elif object_id == 4:  # "knife"
            aff_ids.append([1, 2])
        elif object_id == 5:  # "ladle"
            aff_ids.append([1, 4])
        elif object_id == 6:  # "mallet"
            aff_ids.append([1, 5])
        elif object_id == 7:  # "mug"
            aff_ids.append([1, 4, 7])
        elif object_id == 8:  # "pot"
            aff_ids.append([4, 7])
        elif object_id == 9:  # "saw"
            aff_ids.append([1, 2])
        elif object_id == 10:  # "scissors"
            aff_ids.append([1, 2])
        elif object_id == 11:  # "scoop"
            aff_ids.append([1, 3])
        elif object_id == 12:  # "shears"
            aff_ids.append([1, 2])
        elif object_id == 13:  # "shovel"
            aff_ids.append([1, 3])
        elif object_id == 14:  # "spoon"
            aff_ids.append([1, 3])
        elif object_id == 15:  # "tenderizer"
            aff_ids.append([1, 5])
        elif object_id == 16:  # "trowel"
            aff_ids.append([1, 3])
        elif object_id == 17:  # "turner"
            aff_ids.append([1, 6])
        else:
            assert (" --- Object does not exist in UMD dataset --- ")
    return aff_ids

def format_obj_ids_to_aff_ids_list(object_ids, aff_ids):
    if len(object_ids) == 0:
        return []
    else:
        _aff_ids_list = []
        for i in range(len(object_ids)):
            _aff_ids_list.append(list(aff_ids))
        return _aff_ids_list

######################
######################

def colorize_bbox(object_id):

    increment = 255 / config.NUM_OBJECT_CLASSES # num objects

    if object_id == "bowl":
        return 1 * increment
    elif object_id == "cup":
        return 2 * increment
    elif object_id == "hammer":
        return 3 * increment
    elif object_id == "knife":
        return 4 * increment
    elif object_id == "ladle":
        return 5 * increment
    elif object_id == "mallet":
        return 6 * increment
    elif object_id == "mug":
        return 7 * increment
    elif object_id == "pot":
        return 8 * increment
    elif object_id == "saw":
        return 9 * increment
    elif object_id == "scissors":
        return 10 * increment
    elif object_id == "scoop":
        return 11 * increment
    elif object_id == "shears":
        return 12 * increment
    elif object_id == "shovel":
        return 13 * increment
    elif object_id == "spoon":
        return 14 * increment
    elif object_id == "tenderizer":
        return 15 * increment
    elif object_id == "trowel":
        return 16 * increment
    elif object_id == "turner":
        return 17 * increment
    else:
        assert (" --- Object does not exist in UMD dataset --- ")