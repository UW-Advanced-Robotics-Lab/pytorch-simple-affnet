import json
import glob
import cv2
import random

import os

import matplotlib.pyplot as plt

import skimage.draw
from PIL import Image # (pip install Pillow)

import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

###########################################################
###########################################################

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(aff_id, sub_mask, mask, rgb_img):

    #################
    # mask contours
    #################

    h, w = sub_mask.size
    sub_mask = np.array(sub_mask.getdata(), dtype=np.uint8).reshape(w, h)

    sub_mask = cv2.cvtColor(sub_mask, cv2.COLOR_GRAY2BGR)
    sub_mask = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY) * 255

    contours, hierarchy = cv2.findContours(sub_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #################
    # mask contours
    #################

    # # (X coordinate value, Y coordinate value)
    # cv2.rectangle(mask, (bbox[1], bbox[0]), (bbox[3], bbox[2]), 255, 2)
    #
    # # On this output, draw all of the contours that we have detected
    # # in white, and set the thickness to be 3 pixels
    mask = np.array(mask, dtype=np.uint8)
    cv2.drawContours(rgb_img, contours, -1, 255, 3)

    #################
    #################

    region = {}
    region['shape_attributes'] = {}
    region['shape_attributes']["name"] = "polygon"
    region['shape_attributes']["num_contours"] = len(contours)
    # region['shape_attributes']["all_points_x"] = np.array(x_list).tolist()
    # region['shape_attributes']["all_points_y"] = np.array(y_list).tolist()
    region['shape_attributes']["aff_id"] = aff_id

    for contour_idx, k in enumerate(contours):
        x_list = []
        y_list = []
        for i in k:
            for j in i:
                x_list.append(j[0])
                y_list.append(j[1])
        region['shape_attributes']["all_points_x" + str(contour_idx)] = np.array(x_list).tolist()
        region['shape_attributes']["all_points_y" + str(contour_idx)] = np.array(y_list).tolist()

    # cv2.imshow("coco_mask", rgb_img)
    # cv2.waitKey(0)

    return region

###########################################################
###########################################################

def extract_coco_mask_annotations(image_idx, rgb_img, label_img):

    ####################
    ### init
    ####################
    annotations = {}
    annotations[image_idx] = {}

    ###################
    # affmasks
    ###################
    annotations[image_idx]['regions'] = {}
    regions = {}

    label_img = Image.fromarray(label_img)
    sub_masks = create_sub_masks(label_img)

    for aff_id, sub_mask in sub_masks.items():
        if int(aff_id) > 0:
            region = create_sub_mask_annotation(aff_id=int(aff_id), sub_mask=sub_mask, mask=label_img, rgb_img=rgb_img)
            regions[np.str(aff_id)] = region
    annotations[image_idx]['regions'] = regions
    return annotations

###########################################################
###########################################################

def extract_polygon_masks(image_idx, rgb_img, label_img):

    rgb_img = np.array(rgb_img, dtype=np.uint8)
    label_img = np.array(label_img, dtype=np.uint8)
    height, width = rgb_img.shape[:2]

    _aff_labels = np.unique(label_img)[1:]
    # Convert polygons to a bitmap mask of shape
    mask = np.zeros([height, width, len(_aff_labels)], dtype=np.uint8)
    aff_IDs = np.zeros([len(_aff_labels)], dtype=np.int32)

    annotations = extract_coco_mask_annotations(image_idx, rgb_img, label_img)

    annotations = list(annotations.values())
    annotations = [a for a in annotations if a['regions']]

    for a in annotations:
        if type(a['regions']) is dict:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
        else:
            polygons = [r['shape_attributes'] for r in a['regions']]

        for i, p in enumerate(polygons):
            for countour_idx, _ in enumerate(range(p["num_contours"])):
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(p['all_points_y' + str(countour_idx)], p['all_points_x' + str(countour_idx)])
                mask[rr, cc, i] = 1
                aff_IDs[i] = p['aff_id']
    mask = np.array(mask).transpose(2, 0, 1)
    return mask, aff_IDs