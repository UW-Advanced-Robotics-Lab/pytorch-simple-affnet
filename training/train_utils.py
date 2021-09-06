import os
import sys
import glob
import copy
import math

import cv2
import numpy as np

from tqdm import tqdm

import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

import torch.distributed as dist

import config
from dataset import dataset_utils


def save_checkpoint(model, optimizer, epochs, checkpoint_path):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["epochs"] = epochs

    torch.save(checkpoint, checkpoint_path)

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, is_subsample=True):
    # set the model to train to enable batchnorm.
    model.train()

    train_loader = data_loader
    num_train = len(data_loader.dataset)
    if is_subsample:
        # generate a random subsampler.
        random_idx = np.sort(np.random.choice(len(data_loader.dataset), size=config.NUM_TRAIN, replace=False))
        # Check to see if Subsampler changes every epoch.
        # print(f'\nSubsampler: First 5 Idxs: {random_idx[0:5]}')
        sampler = SubsetRandomSampler(list(random_idx))
        # generate a new dataset with the SubsetRandomSampler..
        train_loader = data.DataLoader(data_loader.dataset,
                                       batch_size=config.BATCH_SIZE,
                                       sampler=sampler,
                                       num_workers=config.NUM_WORKERS,
                                       pin_memory=True,
                                       collate_fn=dataset_utils.collate_fn
                                       )
        num_train = config.NUM_TRAIN

    with tqdm(total=num_train, desc=f'Train Epoch:{epoch}', unit='iterations') as pbar:
        for idx, batch in enumerate(train_loader):

            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward pass.
            loss_dict = model(images, targets)

            # format loss.
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # getting summed loss.
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            # backwards pass.
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # tqdm.
            pbar.update(config.BATCH_SIZE)

            # Tensorboard.
            _global_idx = int(epoch * num_train + idx)
            writer.add_scalar('Learning_rate/train',        optimizer.param_groups[0]['lr'],       _global_idx)
            writer.add_scalar('Loss/train',                 loss_value,                            _global_idx)
            writer.add_scalar('RPN/train_objectness_loss',  loss_dict_reduced['loss_objectness'],  _global_idx)
            writer.add_scalar('RPN/train_box_loss',         loss_dict_reduced['loss_rpn_box_reg'], _global_idx)
            writer.add_scalar('RoI/train_classifier_loss',  loss_dict_reduced['loss_classifier'],  _global_idx)
            writer.add_scalar('RoI/train_box_loss',         loss_dict_reduced['loss_box_reg'],     _global_idx)
            writer.add_scalar('RoI/train_mask_loss',        loss_dict_reduced['loss_mask'],        _global_idx)

    return model, optimizer

def val_one_epoch(model, optimizer, data_loader, device, epoch, writer, is_subsample=True):
    # set the model to train to continue outputting loss.
    model.train()

    val_loader = data_loader
    num_val = len(data_loader.dataset)
    if is_subsample:
        # generate a random subsampler.
        random_idx = np.sort(np.random.choice(len(data_loader.dataset), size=config.NUM_VAL, replace=False))
        # Check to see if Subsampler changes every epoch.
        # print(f'\nSubsampler: First 5 Idxs: {random_idx[0:5]}')
        sampler = SubsetRandomSampler(list(random_idx))
        # generate a new dataset with the SubsetRandomSampler..
        val_loader = data.DataLoader(data_loader.dataset,
                                     batch_size=config.BATCH_SIZE,
                                     sampler=sampler,
                                     num_workers=config.NUM_WORKERS,
                                     pin_memory=True,
                                     collate_fn=dataset_utils.collate_fn
                                     )
        num_val = config.NUM_VAL

    with tqdm(total=num_val, desc=f'Val Epoch:{epoch}', unit='iterations') as pbar:
        for idx, batch in enumerate(val_loader):

            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward pass.
            with torch.no_grad():
                loss_dict = model(images, targets)

            # format loss.
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # getting summed loss.
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            # tqdm.
            pbar.update(config.BATCH_SIZE)

            # Tensorboard.
            _global_idx = int(epoch * num_val + idx)
            writer.add_scalar('Learning_rate/val',       optimizer.param_groups[0]['lr'],       _global_idx)
            writer.add_scalar('Loss/val',                loss_value,                            _global_idx)
            writer.add_scalar('RPN/val_objectness_loss', loss_dict_reduced['loss_objectness'],  _global_idx)
            writer.add_scalar('RPN/val_box_loss',        loss_dict_reduced['loss_rpn_box_reg'], _global_idx)
            writer.add_scalar('RoI/val_classifier_loss', loss_dict_reduced['loss_classifier'],  _global_idx)
            writer.add_scalar('RoI/val_box_loss',        loss_dict_reduced['loss_box_reg'],     _global_idx)
            writer.add_scalar('RoI/val_mask_loss',       loss_dict_reduced['loss_mask'],        _global_idx)

    return model, optimizer

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict