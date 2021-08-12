import os
import sys
import glob
import copy
import math

import cv2
import numpy as np

from tqdm import tqdm

import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torch.distributed as dist

import config
from dataset import dataset_utils
from dataset.arl_affpose import arl_affpose_dataset_utils


def get_model_instance_segmentation(pretrained, num_classes):
    print('loading torchvision maskrcnn ..')
    print(f'num classes:{num_classes} ..')
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def save_checkpoint(model, optimizer, epochs, checkpoint_path):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["epochs"] = epochs

    torch.save(checkpoint, checkpoint_path)
    print(f'saved model to {checkpoint_path} ..')

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer):
    model.train()
    with tqdm(total=len(data_loader.dataset), desc=f'Train Epoch:{epoch}', unit='iterations') as pbar:
        for idx, batch in enumerate(data_loader):

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
            _global_idx = int(epoch * len(data_loader.dataset) + idx)
            writer.add_scalar('Learning_rate/train',        optimizer.param_groups[0]['lr'],       _global_idx)
            writer.add_scalar('Loss/train',                 loss_value,                            _global_idx)
            writer.add_scalar('RPN/train_objectness_loss',  loss_dict_reduced['loss_objectness'],  _global_idx)
            writer.add_scalar('RPN/train_box_loss',         loss_dict_reduced['loss_rpn_box_reg'], _global_idx)
            writer.add_scalar('RoI/train_classifier_loss',  loss_dict_reduced['loss_classifier'],  _global_idx)
            writer.add_scalar('RoI/train_box_loss',         loss_dict_reduced['loss_box_reg'],     _global_idx)
            writer.add_scalar('RoI/train_mask_loss',        loss_dict_reduced['loss_mask'],        _global_idx)

    return model, optimizer

def val_one_epoch(model, optimizer, data_loader, device, epoch, writer):
    model.train()
    with tqdm(total=len(data_loader.dataset), desc=f'Val Epoch:{epoch}', unit='iterations') as pbar:
        for idx, batch in enumerate(data_loader):

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
            _global_idx = int(epoch * len(data_loader.dataset) + idx)
            writer.add_scalar('Learning_rate/val',       optimizer.param_groups[0]['lr'],       _global_idx)
            writer.add_scalar('Loss/val',                loss_value,                            _global_idx)
            writer.add_scalar('RPN/val_objectness_loss', loss_dict_reduced['loss_objectness'],  _global_idx)
            writer.add_scalar('RPN/val_box_loss',        loss_dict_reduced['loss_rpn_box_reg'], _global_idx)
            writer.add_scalar('RoI/val_classifier_loss', loss_dict_reduced['loss_classifier'],  _global_idx)
            writer.add_scalar('RoI/val_box_loss',        loss_dict_reduced['loss_box_reg'],     _global_idx)
            writer.add_scalar('RoI/val_mask_loss',       loss_dict_reduced['loss_mask'],        _global_idx)

    return model, optimizer

def eval_maskrcnn_arl_affpose(model, test_loader):
    print('\nevaluating MaskRCNN ..')
    model.eval()

    # Init folders.
    if not os.path.exists(config.TEST_SAVE_FOLDER):
        os.makedirs(config.TEST_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.TEST_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    for image_idx, (images, targets) in enumerate(test_loader):
        image, target = copy.deepcopy(images), copy.deepcopy(targets)
        images = list(image.to(config.DEVICE) for image in images)

        with torch.no_grad():
            outputs = model(images)
            outputs = [{k: v.to(config.CPU_DEVICE) for k, v in t.items()} for t in outputs]

        # Formatting input.
        image = image[0]
        image = image.to(config.CPU_DEVICE)
        image = np.squeeze(np.array(image)).transpose(1, 2, 0)

        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        image, target = arl_affpose_dataset_utils.format_target_data(image, target)

        # Formatting Output.
        outputs = outputs.pop()
        scores = np.array(outputs['scores'], dtype=np.float32).flatten()
        obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
        obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)

        # Thresholding Binary Masks.
        obj_binary_masks = np.squeeze(np.array(outputs['obj_binary_masks'] > 0.5, dtype=np.uint8))

        # Thresholding predictions based on object confidence score.
        idx = np.argwhere(scores.copy() > config.CONFIDENCE_THRESHOLD)
        scores = scores[idx].reshape(-1)
        obj_ids = obj_ids[idx]
        obj_boxes = obj_boxes[idx].reshape(-1, 4)
        # exception where idx == 1 and binary masks becomes H X W.
        obj_binary_masks = obj_binary_masks[idx, :, :]
        if len(obj_binary_masks.shape) == 2:
            obj_binary_masks = obj_binary_masks[np.newaxis, :]
        obj_binary_masks = obj_binary_masks.reshape(-1, config.CROP_SIZE[0], config.CROP_SIZE[1])

        # getting predicted object mask.
        obj_mask = arl_affpose_dataset_utils.get_segmentation_masks(image=image,
                                                                    obj_ids=obj_ids,
                                                                    binary_masks=obj_binary_masks,
                                                                    )

        gt_name = config.TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['obj_mask'])

        pred_name = config.TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, obj_mask)
    model.train()
    return model

def eval_affnet_arl_affpose(model, test_loader):
    print('\nevaluating MaskRCNN ..')
    model.eval()

    # Init folders.
    if not os.path.exists(config.TEST_SAVE_FOLDER):
        os.makedirs(config.TEST_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.TEST_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    for image_idx, (images, targets) in enumerate(test_loader):
        image, target = copy.deepcopy(images), copy.deepcopy(targets)
        images = list(image.to(config.DEVICE) for image in images)

        with torch.no_grad():
            outputs = model(images)
            outputs = [{k: v.to(config.CPU_DEVICE) for k, v in t.items()} for t in outputs]

        # Formatting input.
        image = image[0]
        image = image.to(config.CPU_DEVICE)
        image = np.squeeze(np.array(image)).transpose(1, 2, 0)

        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        image, target = arl_affpose_dataset_utils.format_target_data(image, target)

        # Formatting Output.
        outputs = outputs.pop()
        # TODO: threshold aff ids using objectiveness.
        aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()

        # Thresholding Binary Masks.
        aff_binary_masks = np.squeeze(np.array(outputs['aff_binary_masks'] > 0.5, dtype=np.uint8))

        # getting predicted object mask.
        aff_mask = arl_affpose_dataset_utils.get_segmentation_masks(image=image,
                                                                    obj_ids=aff_ids,
                                                                    binary_masks=aff_binary_masks,
                                                                    )

        gt_name = config.TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['aff_mask'])

        pred_name = config.TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, aff_mask)
    model.train()
    return model

def eval_Fwb_arl_affpose(model, optimizer, best_Fwb, epoch, writer, matlab_scrips_dir=config.MATLAB_SCRIPTS_DIR):
    print()

    os.chdir(matlab_scrips_dir)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    Fwb = eng.evaluate_ARLAffPose(config.TEST_SAVE_FOLDER, nargout=1)
    writer.add_scalar('eval/Fwb', Fwb, int(epoch))
    os.chdir(config.ROOT_DIR_PATH)

    if Fwb > best_Fwb:
        best_Fwb = Fwb
        writer.add_scalar('eval/Best Fwb', best_Fwb, int(epoch))
        print("Saving best model .. best Fwb={:.5} ..".format(best_Fwb))

        CHECKPOINT_PATH = config.BEST_MODEL_SAVE_PATH
        save_checkpoint(model, optimizer, epoch, CHECKPOINT_PATH)

    return best_Fwb

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

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