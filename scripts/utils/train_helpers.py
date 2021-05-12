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

import cfg as config
from tqdm import tqdm

from utils import helper_utils

###############################
# TRAINING UTILS
###############################

def save_checkpoint(model, optimizer, epochs, checkpoint_path):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["epochs"] = epochs

    torch.save(checkpoint, checkpoint_path)
    print(f'saved model to {checkpoint_path} ..\n')

###############################
###############################

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    assert (len(data_loader) == config.NUM_TRAIN)
    with tqdm(total=config.NUM_TRAIN, desc=f'Epoch:{epoch}', unit='iterations') as pbar:
        for i, batch in enumerate(data_loader):

            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            ## TENSORBOARD
            writer.add_scalar('Learning_rate/train',        optimizer.param_groups[0]['lr'],       int(epoch * config.NUM_TRAIN + i))
            writer.add_scalar('Loss/train',                 loss_value,                            int(epoch * config.NUM_TRAIN + i))
            writer.add_scalar('RPN/train_objectness_loss',  loss_dict_reduced['loss_objectness'],  int(epoch * config.NUM_TRAIN + i))
            writer.add_scalar('RPN/train_box_loss',         loss_dict_reduced['loss_rpn_box_reg'], int(epoch * config.NUM_TRAIN + i))
            writer.add_scalar('RoI/train_classifier_loss',  loss_dict_reduced['loss_classifier'],  int(epoch * config.NUM_TRAIN + i))
            writer.add_scalar('RoI/train_box_loss',         loss_dict_reduced['loss_box_reg'],     int(epoch * config.NUM_TRAIN + i))
            writer.add_scalar('RoI/train_mask_loss',        loss_dict_reduced['loss_mask'],        int(epoch * config.NUM_TRAIN + i))

            pbar.update(config.BATCH_SIZE)
    return model, optimizer

#######################
#######################

def val_one_epoch(model, optimizer, data_loader, device, epoch, writer):
    model.train()

    assert (len(data_loader) == config.NUM_VAL)
    with tqdm(total=config.NUM_VAL, desc=f'Epoch:{epoch}', unit='iterations') as pbar:
        for i, batch in enumerate(data_loader):

            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            # optimizer.zero_grad()
            # losses.backward()
            # optimizer.step()

            ## TENSORBOARD
            writer.add_scalar('Learning_rate/val',       optimizer.param_groups[0]['lr'],           int(epoch * config.NUM_VAL + i))
            writer.add_scalar('Loss/val',                loss_value,                                int(epoch * config.NUM_VAL + i))
            writer.add_scalar('RPN/val_objectness_loss', loss_dict_reduced['loss_objectness'],      int(epoch * config.NUM_VAL + i))
            writer.add_scalar('RPN/val_box_loss',        loss_dict_reduced['loss_rpn_box_reg'],     int(epoch * config.NUM_VAL + i))
            writer.add_scalar('RoI/val_classifier_loss', loss_dict_reduced['loss_classifier'],      int(epoch * config.NUM_VAL + i))
            writer.add_scalar('RoI/val_box_loss',        loss_dict_reduced['loss_box_reg'],         int(epoch * config.NUM_VAL + i))
            writer.add_scalar('RoI/val_mask_loss',       loss_dict_reduced['loss_mask'],            int(epoch * config.NUM_VAL + i))

            pbar.update(config.BATCH_SIZE)
    return model, optimizer

#######################
#######################

def eval_model(model, test_loader):
    print('\nevaluating model ..')
    model.eval()  # todo: issues with batch size = 1

    ######################
    # INIT
    ######################

    if not os.path.exists(config.TEST_SAVE_FOLDER):
        os.makedirs(config.TEST_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.TEST_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    ######################
    ######################

    assert (len(test_loader) == config.NUM_TEST)
    for image_idx, (images, targets) in enumerate(test_loader):
        image, target = copy.deepcopy(images), copy.deepcopy(targets)
        images = list(image.to(config.DEVICE) for image in images)

        with torch.no_grad():
            outputs = model(images)
            outputs = [{k: v.to(config.CPU_DEVICE) for k, v in t.items()} for t in outputs]

        #######################
        ### todo: formatting input
        #######################
        image = image[0]
        image = image.to(config.CPU_DEVICE)
        img = np.squeeze(np.array(image)).transpose(1, 2, 0)
        height, width = img.shape[:2]

        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        target = helper_utils.format_target_data(img, target)

        #######################
        ### todo: formatting output
        #######################
        outputs = outputs.pop()

        scores = np.array(outputs['scores'], dtype=np.float32).flatten()
        labels = np.array(outputs['labels'], dtype=np.int32).flatten()
        binary_masks = np.squeeze(np.array(outputs['masks'] > config.CONFIDENCE_THRESHOLD, dtype=np.uint8))

        aff_labels = labels.copy()
        if 'aff_labels' in outputs.keys():
            aff_labels = np.array(outputs['aff_labels'], dtype=np.int32)

        #######################
        ### masks
        #######################
        mask = helper_utils.get_segmentation_masks(image=img,
                                                   labels=labels,
                                                   # labels=aff_labels,
                                                   binary_masks=binary_masks,
                                                   scores=scores)

        gt_name = config.TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['gt_mask'])

        pred_name = config.TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, mask)
    model.train()
    return model

#######################
#######################

def eval_Fwb(model, optimizer, Fwb, best_Fwb, epoch, writer, matlab_scrips_dir=config.MATLAB_SCRIPTS_DIR):
    print()

    os.chdir(matlab_scrips_dir)
    # print(matlab_scrips_dir)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    # Fwb = eng.evaluate_UMD(config.TEST_SAVE_FOLDER, nargout=1)
    # Fwb = eng.evaluate_ARLVicon(config.TEST_SAVE_FOLDER, nargout=1)
    Fwb = eng.evaluate_ARLAffPose(config.TEST_SAVE_FOLDER, nargout=1)
    writer.add_scalar('eval/Fwb', Fwb, int(epoch * config.NUM_TRAIN))
    os.chdir(config.ROOT_DIR_PATH)

    if Fwb > best_Fwb:
        best_Fwb = Fwb
        writer.add_scalar('eval/Best Fwb', best_Fwb, int(epoch * config.NUM_TRAIN))
        print("Saving best model .. best Fwb={:.5} ..".format(best_Fwb))

        CHECKPOINT_PATH = config.BEST_MODEL_SAVE_PATH
        save_checkpoint(model, optimizer, epoch, CHECKPOINT_PATH)

    return Fwb, best_Fwb