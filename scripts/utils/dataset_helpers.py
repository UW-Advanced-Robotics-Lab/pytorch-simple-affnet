import os
import glob
import copy

import math
import sys
import time
import torch

import numpy as np
import cv2

import torchvision.models.detection.mask_rcnn

from scripts.tutorial.vision.coco_utils import get_coco_api_from_dataset
from scripts.tutorial.vision.coco_eval import CocoEvaluator
from scripts.tutorial.vision import utils

from torch.utils import data
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

###############################
###############################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1].absolute().resolve(strict=True)

import cfg as config
from tqdm import tqdm

from utils import helper_utils

from dataset.UMDDataset import UMDDataSet
from dataset.utils.UMD import umd_utils

from dataset.ElevatorDataset import ElevatorDataSet
from dataset.utils.Elevator import elevator_utils

###############################
# TRAIN
###############################

def load_umd_train_datasets():

    ######################
    # train + val
    ######################
    print("\nloading train and val ..")

    dataset = UMDDataSet(
        ### REAL
        dataset_dir=config.DATA_DIRECTORY_TRAIN,
        mean=config.IMAGE_MEAN,
        std=config.IMAGE_STD,
        resize=config.RESIZE,
        crop_size=config.INPUT_SIZE,
        ###
        is_train=True,
        ### EXTENDING DATASET
        extend_dataset=False,
        ### IMGAUG
        apply_imgaug=True)

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(dataset), 1)
    train_idx = np.random.choice(total_idx, size=int(config.NUM_TRAIN+config.NUM_VAL), replace=False)
    dataset = Subset(dataset, train_idx)

    train_dataset, val_dataset = random_split(dataset, [config.NUM_TRAIN, config.NUM_VAL])

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              collate_fn=utils.collate_fn)

    print(f"train has {len(train_loader)} images ..")
    assert (len(train_loader) >= int(config.NUM_TRAIN))

    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=True,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=True,
                            collate_fn=utils.collate_fn)

    print(f"val has {len(val_dataset)} images ..")
    assert (len(val_loader) >= int(config.NUM_VAL))

    ######################
    # test
    ######################
    print("\nloading test ..")
    print('eval in .. {}'.format(config.TEST_SAVE_FOLDER))

    test_dataset = UMDDataSet(
        ### REAL
        dataset_dir=config.DATA_DIRECTORY_VAL,
        mean=config.IMAGE_MEAN,
        std=config.IMAGE_STD,
        resize=config.RESIZE,
        crop_size=config.INPUT_SIZE,
        ###
        is_train=True,
        ### EXTENDING DATASET
        extend_dataset=False,
        ### IMGAUG
        apply_imgaug=False)

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(test_dataset), 1)
    test_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
    test_dataset = Subset(test_dataset, test_idx)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=config.NUM_WORKERS,
                                              collate_fn=utils.collate_fn)

    print(f"test has {len(test_loader)} images ..")
    assert (len(test_loader) >= int(config.NUM_TEST))

    return train_loader, val_loader, test_loader

###############################
# EVAL
###############################

def load_umd_eval_dataset():

    test_dataset = UMDDataSet(
        ### REAL
        dataset_dir=config.DATA_DIRECTORY_TEST,
        mean=config.IMAGE_MEAN,
        std=config.IMAGE_STD,
        resize=config.RESIZE,
        crop_size=config.INPUT_SIZE,
        ###
        is_train=True,
        ### EXTENDING DATASET
        extend_dataset=False,
        ### IMGAUG
        apply_imgaug=False)

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(test_dataset), 1)
    test_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
    test_dataset = Subset(test_dataset, test_idx)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=config.NUM_WORKERS,
                                              collate_fn=utils.collate_fn)

    print(f"test has {len(test_loader)} images ..")
    assert (len(test_loader) >= int(config.NUM_TEST))

    return test_loader

###############################
# TRAIN
###############################

def load_elevator_train_datasets():

    ######################
    # train + val
    ######################
    print("\nloading train and val ..")

    dataset = ElevatorDataSet(
        ### REAL
        dataset_dir=config.DATA_DIRECTORY_TRAIN,
        mean=config.IMAGE_MEAN,
        std=config.IMAGE_STD,
        resize=config.RESIZE,
        crop_size=config.INPUT_SIZE,
        ###
        is_train=True,
        ### EXTENDING DATASET
        extend_dataset=False,
        ### IMGAUG
        apply_imgaug=True)

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(dataset), 1)
    train_idx = np.random.choice(total_idx, size=int(config.NUM_TRAIN+config.NUM_VAL), replace=False)
    dataset = Subset(dataset, train_idx)

    train_dataset, val_dataset = random_split(dataset, [config.NUM_TRAIN, config.NUM_VAL])

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              collate_fn=utils.collate_fn)

    print(f"train has {len(train_loader)} images ..")
    assert (len(train_loader) >= int(config.NUM_TRAIN))

    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=True,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=True,
                            collate_fn=utils.collate_fn)

    print(f"val has {len(val_dataset)} images ..")
    assert (len(val_loader) >= int(config.NUM_VAL))

    ######################
    # test
    ######################
    print("\nloading test ..")
    print('eval in .. {}'.format(config.TEST_SAVE_FOLDER))

    test_dataset = ElevatorDataSet(
        ### REAL
        dataset_dir=config.DATA_DIRECTORY_VAL,
        mean=config.IMAGE_MEAN,
        std=config.IMAGE_STD,
        resize=config.RESIZE,
        crop_size=config.INPUT_SIZE,
        ###
        is_train=True,
        ### EXTENDING DATASET
        extend_dataset=False,
        ### IMGAUG
        apply_imgaug=False)

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(test_dataset), 1)
    test_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
    test_dataset = Subset(test_dataset, test_idx)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=config.NUM_WORKERS,
                                              collate_fn=utils.collate_fn)

    print(f"test has {len(test_loader)} images ..")
    assert (len(test_loader) >= int(config.NUM_TEST))

    return train_loader, val_loader, test_loader

###############################
# EVAL
###############################

def load_elevator_eval_dataset():

    test_dataset = ElevatorDataSet(
        ### REAL
        dataset_dir=config.DATA_DIRECTORY_TEST,
        mean=config.IMAGE_MEAN,
        std=config.IMAGE_STD,
        resize=config.RESIZE,
        crop_size=config.INPUT_SIZE,
        ###
        is_train=True,
        ### EXTENDING DATASET
        extend_dataset=False,
        ### IMGAUG
        apply_imgaug=False)

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(test_dataset), 1)
    test_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
    test_dataset = Subset(test_dataset, test_idx)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=config.NUM_WORKERS,
                                              collate_fn=utils.collate_fn)

    print(f"test has {len(test_loader)} images ..")
    assert (len(test_loader) >= int(config.NUM_TEST))

    return test_loader