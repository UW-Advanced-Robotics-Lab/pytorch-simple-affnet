import glob
import numpy as np

import cv2
from PIL import Image

import argparse

##################################
##################################

def map_obj_id_to_name(object_id):

    if object_id == 1:
        return 'master_chef_can'
    elif object_id == 2:
        return 'cracker_box'
    elif object_id == 3:
        return 'sugar_box'
    elif object_id == 4:
        return 'tomato_soup_can'
    elif object_id == 5:
        return 'mustard_bottle'
    elif object_id == 6:
        return 'tuna_fish_can'
    elif object_id == 7:
        return 'pudding_box'
    elif object_id == 8:
        return 'gelatin_box'
    elif object_id == 9:
        return 'potted_meat_can'
    elif object_id == 10:
        return 'banana'
    elif object_id == 11:
        return 'pitcher_base'
    elif object_id == 12:
        return 'bleach_cleanser'
    elif object_id == 13:
        return 'bowl'
    elif object_id == 14:
        return 'mug'
    elif object_id == 15:
        return 'power_drill'
    elif object_id == 16:
        return 'wood_block'
    elif object_id == 17:
        return 'scissors'
    elif object_id == 18:
        return 'large_marker'
    elif object_id == 19:
        return 'large_clamp'
    elif object_id == 20:
        return 'extra_large_clamp'
    elif object_id == 21:
        return 'foam_brick'
    else:
        print(" --- Object ID does not exist in UMD --- ")
        exit(1)

##################################
##################################

def colorize_obj_mask(instance_mask):

    instance_to_color = obj_color_map_dict()
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[instance_mask == key] = instance_to_color[key]

    return np.squeeze(color_mask)

def obj_color_map_dict():
    # https://htmlcolorcodes.com/
    # [red, blue, green]

    obj_color_map_dict = {
        0: [0, 0, 0],
        1: [255, 73, 51],    # red 1
        2: [255, 113, 51],   # orange 1
        3: [235, 195, 17],   # gold
        4: [255, 202, 51],   # yellow 1
        5: [255, 240, 51],   # yellow 2
        6: [209, 255, 51],   # light green 1
        7: [125, 255, 51],   # light green 2
        8: [51,  255, 91],   # neon green
        9: [51,  255, 175],  # teal 1
        10: [51, 255, 233],  # teal 2
        11: [51, 199, 233],  # light blue 1
        12: [51, 206, 255],  # light blue 2
        13: [51, 138, 255],  # blue 1
        14: [51, 76,  255],  # blue 2
        15: [113, 51, 255],  # purple 1
        16: [168, 51, 255],  # purple 2
        17: [224, 51, 255],  # pink 1
        18: [255, 51, 252],  # hot pink
        19: [255, 51, 196],  # dark pink 1
        20: [288, 121, 51],  # orange 2
        21: [255, 51, 138],  # dark pink 2
    }

    return obj_color_map_dict

##################################
##################################

def pose_cnn_pred_color():
    return (247, 163, 150) # teal

def densefusion_pred_color():
    return (150, 247, 241) # teal

def obj_color_map(idx):
    ''' [red, blue, green]'''

    if idx == 1:
        return (255, 73, 51)
    elif idx == 2:
        return (255, 113, 51)
    elif idx == 3:
        return (235, 195, 17)
    elif idx == 4:
        return (255, 202, 51)
    elif idx == 5:
        return (255, 240, 51)
    elif idx == 6:
        return (209, 255, 51)
    elif idx == 7:
        return (125, 255, 51)
    elif idx == 8:
        return (51,  255, 91)
    elif idx == 9:
        return (51,  255, 175)
    elif idx == 10:
        return (51, 255, 233)
    elif idx == 11:
        return (51, 199, 233)
    elif idx == 12:
        return (51, 206, 255)
    elif idx == 13:
        return (51, 138, 255)
    elif idx == 14:
        return (51, 76,  255)
    elif idx == 15:
        return (113, 51, 255)
    elif idx == 16:
        return (168, 51, 255)
    elif idx == 17:
        return (224, 51, 255)
    elif idx == 18:
        return (255, 51, 252)
    elif idx == 19:
        return (255, 51, 196)
    elif idx == 20:
        return (288, 121, 51)
    elif idx == 21:
        return (255, 51, 138)
    else:
        print(" --- idx does not map to a colour --- ")
        exit(1)