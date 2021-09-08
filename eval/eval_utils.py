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

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torch.distributed as dist

import config
from dataset import dataset_utils
from dataset.umd import umd_dataset_utils
from dataset.arl_affpose import arl_affpose_dataset_utils


def eval_affnet_umd(model, test_loader):
    print('\nevaluating AffNet ..')

    # set the model to eval to disable batchnorm.
    model.eval()

    # Init folders.
    if not os.path.exists(config.UMD_TEST_SAVE_FOLDER):
        os.makedirs(config.UMD_TEST_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.UMD_TEST_SAVE_FOLDER + '*')
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
        H, W, C = image.shape

        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        image, target = umd_dataset_utils.format_target_data(image, target)

        # Formatting Output.
        outputs = outputs.pop()
        # TODO: threshold aff ids using objectiveness.
        aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()

        # Thresholding Binary Masks.
        aff_binary_masks = np.squeeze(np.array(outputs['aff_binary_masks'] > 0.5, dtype=np.uint8))

        # getting predicted object mask.
        aff_mask = umd_dataset_utils.get_segmentation_masks(image=image,
                                                            obj_ids=aff_ids,
                                                            binary_masks=aff_binary_masks,
                                                            )

        gt_name = config.UMD_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['aff_mask'])

        pred_name = config.UMD_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, aff_mask)
    model.train()
    return model

def eval_fwb_umd_affnet(matlab_scrips_dir=config.MATLAB_SCRIPTS_DIR):
    print()

    os.chdir(matlab_scrips_dir)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    Fwb = eng.evaluate_umd_affnet(config.UMD_TEST_SAVE_FOLDER, nargout=1)
    os.chdir(config.ROOT_DIR_PATH)

    return Fwb

def eval_maskrcnn_arl_affpose(model, test_loader):
    print('\nevaluating MaskRCNN ..')

    model.eval()

    # Init folders.
    if not os.path.exists(config.ARL_TEST_SAVE_FOLDER):
        os.makedirs(config.ARL_TEST_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.ARL_TEST_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    APs = []
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
        image = np.array(image * (2 ** 8 - 1), dtype=np.uint8)
        H, W, C = image.shape

        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        image, target = arl_affpose_dataset_utils.format_target_data(image, target)

        # format outputs by most confident.
        outputs = outputs.pop()
        image, outputs = arl_affpose_dataset_utils.format_outputs(image, outputs)
        scores = np.array(outputs['scores'], dtype=np.float32).flatten()
        obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
        obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        obj_binary_masks = np.array(outputs['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # match gt to pred.
        target, outputs = get_matched_predictions(image.copy(), target, outputs)
        gt_obj_ids = np.array(target['obj_ids'], dtype=np.int32).flatten()
        gt_obj_boxes = np.array(target['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        gt_obj_binary_masks = np.array(target['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # get average precision.
        AP = compute_ap_range(gt_class_id=gt_obj_ids,
                              gt_box=gt_obj_boxes,
                              gt_mask=gt_obj_binary_masks.reshape(H, W, -1),
                              pred_score=scores,
                              pred_class_id=obj_ids,
                              pred_box=obj_boxes,
                              pred_mask=obj_binary_masks.reshape(H, W, -1),
                              verbose=False,
                              )
        APs.append(AP)

        # threshold outputs for mask.
        image, outputs = arl_affpose_dataset_utils.threshold_outputs(image, outputs)
        obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
        obj_binary_masks = np.array(outputs['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)
        # only visualize "good" masks.
        pred_obj_mask = arl_affpose_dataset_utils.get_segmentation_masks(image=image, obj_ids=obj_ids,binary_masks=obj_binary_masks)

        gt_name = config.ARL_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['obj_mask'])

        pred_name = config.ARL_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, pred_obj_mask)

    model.train()
    return model, np.mean(APs)

def eval_fwb_arl_affpose_maskrcnn(matlab_scrips_dir=config.MATLAB_SCRIPTS_DIR):
    print()

    os.chdir(matlab_scrips_dir)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    Fwb = eng.evaluate_arl_affpose_maskrcnn(config.ARL_TEST_SAVE_FOLDER, nargout=1)
    os.chdir(config.ROOT_DIR_PATH)

    return Fwb

def eval_affnet_arl_affpose(model, test_loader):
    print('\nevaluating AffNet ..')

    # set the model to eval to disable batchnorm.
    model.eval()

    # Init folders.
    if not os.path.exists(config.ARL_TEST_SAVE_FOLDER):
        os.makedirs(config.ARL_TEST_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.ARL_TEST_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    APs = []
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
        image = np.array(image * (2 ** 8 - 1), dtype=np.uint8)
        H, W, C = image.shape

        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        image, target = arl_affpose_dataset_utils.format_target_data(image, target)

        # format outputs by most confident.
        outputs = outputs.pop()
        image, outputs = arl_affpose_dataset_utils.format_affnet_outputs(image, outputs)
        scores = np.array(outputs['scores'], dtype=np.float32).flatten()
        obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
        obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        aff_binary_masks = np.array(outputs['aff_binary_masks'], dtype=np.uint8).reshape(-1, H, W)
        outputs['obj_binary_masks'] = arl_affpose_dataset_utils.get_obj_binary_masks(image, obj_ids, aff_binary_masks)
        obj_binary_masks = np.array(outputs['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # match gt to pred.
        target, outputs = get_affnet_matched_predictions(image.copy(), target, outputs)
        gt_obj_ids = np.array(target['obj_ids'], dtype=np.int32).flatten()
        gt_obj_boxes = np.array(target['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        gt_obj_binary_masks = np.array(target['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # get average precision.
        AP = compute_ap_range(gt_class_id=gt_obj_ids,
                              gt_box=gt_obj_boxes,
                              gt_mask=gt_obj_binary_masks.reshape(H, W, -1),
                              pred_score=scores,
                              pred_class_id=obj_ids,
                              pred_box=obj_boxes,
                              pred_mask=obj_binary_masks.reshape(H, W, -1),
                              verbose=False,
                              )
        APs.append(AP)

        # threshold outputs for mask.
        image, outputs = arl_affpose_dataset_utils.threshold_affnet_outputs(image, outputs)
        aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()
        aff_binary_masks = np.array(outputs['aff_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # only visualize "good" masks.
        aff_mask = arl_affpose_dataset_utils.get_segmentation_masks(image=image,
                                                                    obj_ids=aff_ids,
                                                                    binary_masks=aff_binary_masks,
                                                                    )

        gt_name = config.ARL_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['aff_mask'])

        pred_name = config.ARL_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, aff_mask)

    model.train()
    return model, np.mean(APs)

def eval_fwb_arl_affpose_affnet(matlab_scrips_dir=config.MATLAB_SCRIPTS_DIR):
    print()

    os.chdir(matlab_scrips_dir)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    Fwb = eng.evaluate_arl_affpose_affnet(config.ARL_TEST_SAVE_FOLDER, nargout=1)
    os.chdir(config.ROOT_DIR_PATH)

    return Fwb

def get_matched_predictions(image, target, outputs):
    H, W, C = image.shape

    gt_obj_ids = np.array(target['obj_ids'], dtype=np.int32).flatten()
    gt_obj_boxes = np.array(target['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    gt_obj_binary_masks = np.squeeze(np.array(target['obj_binary_masks'] > config.MASK_THRESHOLD, dtype=np.uint8)).reshape(-1, H, W)

    pred_obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
    pred_obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    pred_obj_binary_masks = np.squeeze(np.array(outputs['obj_binary_masks'] > config.MASK_THRESHOLD, dtype=np.uint8)).reshape(-1, H, W)

    matched_obj_ids = np.zeros_like(pred_obj_ids)
    matched_obj_boxes = np.zeros_like(pred_obj_boxes)
    matched_obj_binary_masks = np.zeros_like(pred_obj_binary_masks)

    # match based on box IoU.
    for pred_idx, pred_obj_id in enumerate(pred_obj_ids):
        # print()
        pred_obj_id = pred_obj_ids[pred_idx]
        pred_obj_box = pred_obj_boxes[pred_idx, :]

        best_iou, best_idx = 0, 0
        for gt_idx, gt_obj_id in enumerate(gt_obj_ids):
            gt_obj_id = gt_obj_ids[gt_idx]
            gt_obj_box = gt_obj_boxes[gt_idx, :]
            pred_iou = get_iou(pred_box=pred_obj_box, gt_box=gt_obj_box)
            if pred_iou > best_iou:
                best_idx = gt_idx
                best_iou = pred_iou
            # print(f'Pred: {pred_obj_id}, GT: {gt_obj_id}, IoU: {pred_iou}')
        # print(f'Best IoU: {best_iou}, Pred Idx: {best_idx}')
        matched_obj_ids[pred_idx] = gt_obj_ids[best_idx]
        matched_obj_boxes[pred_idx, :] = gt_obj_boxes[best_idx, :]
        matched_obj_binary_masks[pred_idx, :, :] = gt_obj_binary_masks[best_idx, :, :]

    target['obj_ids'] = matched_obj_ids
    target['obj_boxes'] = matched_obj_boxes
    target['obj_binary_masks'] = matched_obj_binary_masks

    return target, outputs

def get_affnet_matched_predictions(image, target, outputs):
    H, W, C = image.shape

    gt_obj_ids = np.array(target['obj_ids'], dtype=np.int32).flatten()
    gt_obj_boxes = np.array(target['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    gt_obj_binary_masks = np.squeeze(np.array(target['obj_binary_masks'] > config.MASK_THRESHOLD, dtype=np.uint8)).reshape(-1, H, W)

    pred_obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
    pred_obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    pred_obj_binary_masks = np.squeeze(np.array(outputs['obj_binary_masks'] > config.MASK_THRESHOLD, dtype=np.uint8)).reshape(-1, H, W)

    matched_obj_ids = np.zeros_like(pred_obj_ids)
    matched_obj_boxes = np.zeros_like(pred_obj_boxes)
    matched_obj_binary_masks = np.zeros_like(pred_obj_binary_masks)

    # match based on box IoU.
    for pred_idx, pred_obj_id in enumerate(pred_obj_ids):
        # print()
        pred_obj_id = pred_obj_ids[pred_idx]
        pred_obj_box = pred_obj_boxes[pred_idx, :]

        best_iou, best_idx = 0, 0
        for gt_idx, gt_obj_id in enumerate(gt_obj_ids):
            gt_obj_id = gt_obj_ids[gt_idx]
            gt_obj_box = gt_obj_boxes[gt_idx, :]
            pred_iou = get_iou(pred_box=pred_obj_box, gt_box=gt_obj_box)
            if pred_iou > best_iou:
                best_idx = gt_idx
                best_iou = pred_iou
            # print(f'Pred: {pred_obj_id}, GT: {gt_obj_id}, IoU: {pred_iou}')
        # print(f'Best IoU: {best_iou}, Pred Idx: {best_idx}')
        matched_obj_ids[pred_idx] = gt_obj_ids[best_idx]
        matched_obj_boxes[pred_idx, :] = gt_obj_boxes[best_idx, :]
        matched_obj_binary_masks[pred_idx, :, :] = gt_obj_binary_masks[best_idx, :, :]

    target['obj_ids'] = matched_obj_ids
    target['obj_boxes'] = matched_obj_boxes
    target['obj_binary_masks'] = matched_obj_binary_masks

    return target, outputs

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps

def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps

def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = \
            compute_ap(gt_box, gt_class_id, gt_mask,
                       pred_box, pred_class_id, pred_score, pred_mask,
                       iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(iou_thresholds[0], iou_thresholds[-1], AP))
    return AP