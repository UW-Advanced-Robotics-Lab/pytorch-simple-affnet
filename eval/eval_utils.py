import os
import sys
import glob
import copy
import math

import cv2
import numpy as np

from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import matplotlib.pyplot as plt

import torch

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

import config
from dataset.umd import umd_dataset_utils
from dataset.arl_affpose import arl_affpose_dataset_utils
from dataset.ycb_video import ycb_video_dataset_utils


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def affnet_eval_umd(model, test_loader):
    print('\nevaluating AffNet ..')

    # OBJ_MASK_PROBABILITIES = np.zeros(shape=(len(test_loader), config.UMD_NUM_OBJECT_CLASSES))

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
        image = np.array(image * (2 ** 8 - 1), dtype=np.uint8)
        H, W, C = image.shape

        # Formatting targets.
        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        target = umd_dataset_utils.format_target_data(image.copy(), target.copy())

        # Formatting Output.
        outputs = outputs.pop()
        outputs = affnet_umd_format_outputs(image.copy(), outputs.copy())
        outputs = affnet_umd_threshold_outputs(image.copy(), outputs.copy())
        outputs = affnet_umd_threshold_binary_masks(image.copy(), outputs.copy())
        obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
        aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()
        aff_mask = np.array(outputs['aff_mask'], dtype=np.uint8).reshape(H, W)

        # for idx in range(len(obj_ids)):
        #     OBJ_MASK_PROBABILITIES[image_idx, obj_ids[idx]] = obj_mask_probabilities

        # getting predicted object mask.
        #aff_mask = umd_dataset_utils.get_segmentation_masks(image=image, obj_ids=aff_ids, binary_masks=aff_binary_masks)

        gt_name = config.UMD_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['aff_mask'])

        pred_name = config.UMD_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, aff_mask)

    # print()
    # for obj_id in range(1, config.UMD_NUM_OBJECT_CLASSES):
    #     obj_mask_probabilities = OBJ_MASK_PROBABILITIES[:, obj_id]
    #
    #     mean = obj_mask_probabilities[np.nonzero(obj_mask_probabilities.copy())].mean()
    #     std = obj_mask_probabilities[np.nonzero(obj_mask_probabilities.copy())].std()
    #
    #     obj_name = "{:<13}".format(umd_dataset_utils.map_obj_id_to_name(obj_id))
    #     print(f'Object:{obj_name}'
    #           f'Obj id: {obj_id}, '
    #           f'Mean: {mean: .5f}, '
    #           # f'Std: {std: .5f}'
    #           )

    print()
    # getting Fwb
    os.chdir(config.MATLAB_SCRIPTS_DIR)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    Fwb = eng.evaluate_umd_affnet(config.UMD_TEST_SAVE_FOLDER, nargout=1)
    os.chdir(config.ROOT_DIR_PATH)

    model.train()
    return model, Fwb

def maskrcnn_eval_arl_affpose(model, test_loader):
    print('\nevaluating MaskRCNN ..')

    model.eval()

    # Init folders.
    if not os.path.exists(config.ARL_TEST_SAVE_FOLDER):
        os.makedirs(config.ARL_TEST_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.ARL_TEST_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    APs = []
    gt_obj_ids_list, pred_obj_ids_list = [], []
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

        # Formatting targets.
        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        target = arl_affpose_dataset_utils.format_target_data(image.copy(), target.copy())
        gt_obj_ids = np.array(target['obj_ids'], dtype=np.int32).flatten()
        gt_obj_boxes = np.array(target['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        gt_obj_binary_masks = np.array(target['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # format outputs.
        outputs = outputs.pop()
        outputs = maskrcnn_format_outputs(image.copy(), outputs.copy())
        outputs = maskrcnn_threshold_outputs(image.copy(), outputs.copy())
        matched_outputs = maskrcnn_match_pred_to_gt(image.copy(), target.copy(), outputs.copy())
        scores = np.array(matched_outputs['scores'], dtype=np.float32).flatten()
        obj_ids = np.array(matched_outputs['obj_ids'], dtype=np.int32).flatten()
        obj_boxes = np.array(matched_outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        obj_binary_masks = np.array(matched_outputs['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # confusion matrix.
        gt_obj_ids_list.extend(gt_obj_ids.tolist())
        pred_obj_ids_list.extend(obj_ids.tolist())

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

        # create masks.
        pred_obj_mask = arl_affpose_dataset_utils.get_segmentation_masks(image=image, obj_ids=obj_ids,binary_masks=obj_binary_masks)

        # save masks.
        gt_name = config.ARL_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['obj_mask'])

        pred_name = config.ARL_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, pred_obj_mask)

    # Confusion Matrix.
    cm = sklearn_confusion_matrix(y_true=gt_obj_ids_list, y_pred=pred_obj_ids_list)
    print(f'\n{cm}')

    # mAP
    mAP = np.mean(APs)
    print(f'\nmAP: {mAP:.5f}')

    # getting Fwb
    os.chdir(config.MATLAB_SCRIPTS_DIR)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    Fwb = eng.evaluate_arl_affpose_maskrcnn(config.ARL_TEST_SAVE_FOLDER, nargout=1)
    os.chdir(config.ROOT_DIR_PATH)

    model.train()
    return model, mAP, Fwb

def affnet_eval_arl_affpose(model, test_loader):
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
    gt_obj_ids_list, pred_obj_ids_list = [], []
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

        # Formatting targets.
        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        target = arl_affpose_dataset_utils.format_target_data(image.copy(), target.copy())
        gt_obj_ids = np.array(target['obj_ids'], dtype=np.int32).flatten()
        gt_obj_boxes = np.array(target['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        gt_obj_binary_masks = np.array(target['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # Formatting Output.
        outputs = outputs.pop()
        outputs = affnet_format_outputs(image.copy(), outputs.copy())
        outputs = affnet_threshold_outputs(image.copy(), outputs.copy())
        outputs = maskrcnn_match_pred_to_gt(image.copy(), target.copy(), outputs.copy())
        scores = np.array(outputs['scores'], dtype=np.float32).flatten()
        obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
        obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        obj_binary_masks = np.array(outputs['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)
        aff_scores = np.array(outputs['aff_scores'], dtype=np.float32).flatten()
        obj_part_ids = np.array(outputs['obj_part_ids'], dtype=np.int32).flatten()
        aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()
        aff_binary_masks = np.array(outputs['aff_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # confusion matrix.
        gt_obj_ids_list.extend(gt_obj_ids.tolist())
        pred_obj_ids_list.extend(obj_ids.tolist())

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

        # get aff masks.
        aff_mask = arl_affpose_dataset_utils.get_segmentation_masks(image=image,
                                                                    obj_ids=aff_ids,
                                                                    binary_masks=aff_binary_masks,
                                                                    )

        gt_name = config.ARL_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['aff_mask'])

        pred_name = config.ARL_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, aff_mask)

    # Confusion Matrix.
    cm = sklearn_confusion_matrix(y_true=gt_obj_ids_list, y_pred=pred_obj_ids_list)
    print(f'\n{cm}')

    # mAP
    mAP = np.mean(APs)
    print(f'\nmAP: {mAP:.5f}')

    # getting Fwb
    os.chdir(config.MATLAB_SCRIPTS_DIR)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    Fwb = eng.evaluate_arl_affpose_affnet(config.ARL_TEST_SAVE_FOLDER, nargout=1)
    os.chdir(config.ROOT_DIR_PATH)

    model.train()
    return model, mAP, Fwb

def maskrcnn_eval_ycb_video(model, test_loader):
    print('\nevaluating MaskRCNN ..')

    model.eval()

    # Init folders.
    if not os.path.exists(config.YCB_TEST_SAVE_FOLDER):
        os.makedirs(config.YCB_TEST_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.YCB_TEST_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    APs = []
    gt_obj_ids_list, pred_obj_ids_list = [], []
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

        # Formatting targets.
        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        target = ycb_video_dataset_utils.format_target_data(image.copy(), target.copy())
        gt_obj_ids = np.array(target['obj_ids'], dtype=np.int32).flatten()
        gt_obj_boxes = np.array(target['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        gt_obj_binary_masks = np.array(target['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # format outputs.
        outputs = outputs.pop()
        outputs = maskrcnn_format_outputs(image.copy(), outputs.copy())
        outputs = maskrcnn_threshold_outputs(image.copy(), outputs.copy())
        matched_outputs = maskrcnn_match_pred_to_gt(image.copy(), target.copy(), outputs.copy())
        scores = np.array(matched_outputs['scores'], dtype=np.float32).flatten()
        obj_ids = np.array(matched_outputs['obj_ids'], dtype=np.int32).flatten()
        obj_boxes = np.array(matched_outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        obj_binary_masks = np.array(matched_outputs['obj_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        # confusion matrix.
        gt_obj_ids_list.extend(gt_obj_ids.tolist())
        pred_obj_ids_list.extend(obj_ids.tolist())

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

        # create masks.
        pred_obj_mask = ycb_video_dataset_utils.get_segmentation_masks(image=image, obj_ids=obj_ids,binary_masks=obj_binary_masks)

        # save masks.
        gt_name = config.YCB_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        cv2.imwrite(gt_name, target['obj_mask'])

        pred_name = config.YCB_TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT
        cv2.imwrite(pred_name, pred_obj_mask)

    # Confusion Matrix.
    cm = sklearn_confusion_matrix(y_true=gt_obj_ids_list, y_pred=pred_obj_ids_list)
    print(f'\n{cm}')

    # mAP
    mAP = np.mean(APs)
    print(f'\nmAP: {mAP:.5f}')

    # getting Fwb
    os.chdir(config.MATLAB_SCRIPTS_DIR)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    Fwb = eng.evaluate_arl_affpose_maskrcnn(config.ARL_TEST_SAVE_FOLDER, nargout=1)
    os.chdir(config.ROOT_DIR_PATH)

    model.train()
    return model, mAP, Fwb

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

def maskrcnn_format_outputs(image, outputs):
    height, width = image.shape[0], image.shape[1]

    scores = np.array(outputs['scores'], dtype=np.float32).flatten()
    obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
    obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    obj_binary_masks = np.array(outputs['obj_binary_masks'] > config.MASK_THRESHOLD, dtype=np.uint8).reshape(-1, height, width)

    # sort by class ids (which sorts by score as well).
    idx = np.argsort(obj_ids)
    outputs['scores'] = scores[idx]
    outputs['obj_ids'] = obj_ids[idx]
    outputs['obj_boxes'] = obj_boxes[idx, :]
    outputs['obj_binary_masks'] = obj_binary_masks[idx, :, :]

    # sort by most confident (i.e. -1 to sort in reverse).
    # idx = np.argsort(-1*scores)
    # outputs['scores'] = scores[idx]
    # outputs['obj_ids'] = obj_ids[idx]
    # outputs['obj_boxes'] = obj_boxes[idx, :]
    # outputs['obj_binary_masks'] = obj_binary_masks[idx, :, :]

    return outputs

def maskrcnn_threshold_outputs(image, outputs):
    height, width = image.shape[0], image.shape[1]

    scores = np.array(outputs['scores'], dtype=np.float32).flatten()
    obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
    obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    obj_binary_masks = np.array(outputs['obj_binary_masks'], dtype=np.uint8).reshape(-1, height, width)

    # sort by class ids (which sorts by score as well).
    idx = np.argwhere(scores > config.OBJ_CONFIDENCE_THRESHOLD)
    outputs['scores'] = scores[idx]
    outputs['obj_ids'] = obj_ids[idx]
    outputs['obj_boxes'] = obj_boxes[idx, :]
    outputs['obj_binary_masks'] = obj_binary_masks[idx, :, :]

    return outputs

def maskrcnn_match_pred_to_gt(image, target, outputs):
    H, W, C = image.shape

    gt_obj_ids = np.array(target['obj_ids'], dtype=np.int32).flatten()
    gt_obj_boxes = np.array(target['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    gt_obj_binary_masks = np.squeeze(np.array(target['obj_binary_masks'] > config.MASK_THRESHOLD, dtype=np.uint8)).reshape(-1, H, W)

    pred_scores = np.array(outputs['scores'], dtype=np.float32).flatten()
    pred_obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
    pred_obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    pred_obj_binary_masks = np.squeeze(np.array(outputs['obj_binary_masks'] > config.MASK_THRESHOLD, dtype=np.uint8)).reshape(-1, H, W)

    matched_scores = np.zeros_like(gt_obj_ids, dtype=np.float32)
    matched_obj_ids = np.zeros_like(gt_obj_ids)
    matched_obj_boxes = np.zeros_like(gt_obj_boxes)
    matched_obj_binary_masks = np.zeros_like(gt_obj_binary_masks)

    # match based on box IoU.
    for pred_idx, pred_obj_id in enumerate(pred_obj_ids):
        # print()
        pred_obj_box = pred_obj_boxes[pred_idx, :]

        best_iou, best_idx = 0, 0
        for gt_idx, gt_obj_id in enumerate(gt_obj_ids):
            gt_obj_box = gt_obj_boxes[gt_idx, :]
            pred_iou = get_iou(pred_box=pred_obj_box, gt_box=gt_obj_box)
            if pred_iou > best_iou:
                best_idx = gt_idx
                best_iou = pred_iou
            # print(f'Pred: {pred_obj_id}, GT: {gt_obj_id}, IoU: {pred_iou}')
        # print(f'Best IoU: {best_iou}, Pred Idx: {best_idx}')
        matched_scores[best_idx] = pred_scores[pred_idx]
        matched_obj_ids[best_idx] = pred_obj_ids[pred_idx]
        matched_obj_boxes[best_idx, :] = pred_obj_boxes[pred_idx, :]
        matched_obj_binary_masks[best_idx, :, :] = pred_obj_binary_masks[pred_idx, :, :]

    outputs['scores'] = matched_scores.flatten()
    outputs['obj_ids'] = matched_obj_ids.flatten()
    outputs['obj_boxes'] = matched_obj_boxes
    outputs['obj_binary_masks'] = matched_obj_binary_masks

    return outputs

def maskrcnn_get_best_pred(image, outputs):
    height, width = image.shape[0], image.shape[1]

    scores = outputs['scores'].flatten()
    obj_ids = outputs['obj_ids'].flatten()
    obj_boxes = outputs['obj_boxes'].reshape(-1, 4)
    obj_binary_masks = outputs['obj_binary_masks'].reshape(-1, height, width)

    # Thresholding best confidence score for each object.
    unique_obj_ids = np.unique(obj_ids)
    best_idxs = {obj_id: 0 for obj_id in unique_obj_ids}
    best_scores = {obj_id: 0 for obj_id in unique_obj_ids}
    for idx in range(len(scores)):
        score = scores[idx]
        obj_id = obj_ids[idx]
        best_score = best_scores[obj_id]
        if score > best_score:
            best_scores[obj_id] = score
            best_idxs[obj_id] = idx

    # convert idxs to list.
    idxs = list(best_idxs.values())
    outputs['scores'] = scores[idxs].flatten()
    outputs['obj_ids'] = obj_ids[idxs].flatten()
    outputs['obj_boxes'] = obj_boxes[idxs].reshape(-1, 4)
    outputs['obj_binary_masks'] = obj_binary_masks[idxs, :, :].reshape(-1, height, width)

    return outputs


def affnet_format_outputs(image, outputs):
    height, width = image.shape[0], image.shape[1]

    outputs['scores'] = np.array(outputs['scores'], dtype=np.float32).flatten()
    outputs['obj_ids'] = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
    outputs['obj_boxes'] = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)

    if 'aff_scores' in outputs:
        outputs['aff_scores'] = np.array(outputs['aff_scores'], dtype=np.float32).flatten()
        outputs['obj_part_ids'] = np.array(outputs['obj_part_ids'], dtype=np.int32).flatten()
        outputs['aff_ids'] = np.array(outputs['aff_ids'], dtype=np.int32).flatten()
        outputs['aff_binary_masks'] = np.array(outputs['aff_binary_masks'] > config.MASK_THRESHOLD, dtype=np.uint8).reshape(-1, height, width)
        # get obj mask.
        outputs['obj_binary_masks'] = arl_affpose_dataset_utils.get_obj_binary_masks(image,
                                                                                     outputs['obj_ids'],
                                                                                     outputs['obj_part_ids'],
                                                                                     outputs['aff_binary_masks'])

    else:
        outputs['aff_scores'] = np.zeros_like(outputs['scores'])
        outputs['obj_part_ids'] = np.zeros_like(outputs['obj_ids'])
        outputs['aff_ids'] = np.zeros_like(outputs['obj_ids'])
        outputs['aff_binary_masks'] = np.zeros(shape=(len(outputs['obj_ids']), height, width))
        outputs['obj_binary_masks'] = np.zeros(shape=(len(outputs['obj_ids']), height, width))

    # print(f"obj: {outputs['scores']}")
    # print(f"aff: {outputs['aff_scores']}")

    return outputs


def affnet_threshold_outputs(image, outputs):
    height, width = image.shape[0], image.shape[1]

    # obj
    scores = np.array(outputs['scores'], dtype=np.float32).flatten()
    obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
    obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    # obj_binary_masks = np.array(outputs['obj_binary_masks'], dtype=np.uint8).reshape(-1, height, width)

    idx = np.argwhere(scores > config.OBJ_CONFIDENCE_THRESHOLD)
    outputs['scores'] = scores[idx]
    outputs['obj_ids'] = obj_ids[idx]
    outputs['obj_boxes'] = obj_boxes[idx, :]
    # outputs['obj_binary_masks'] = obj_binary_masks[idx, :, :]

    # aff
    aff_scores = np.array(outputs['aff_scores'], dtype=np.float32).flatten()
    obj_part_ids = np.array(outputs['obj_part_ids'], dtype=np.int32).flatten()
    aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()
    aff_binary_masks = np.array(outputs['aff_binary_masks'], dtype=np.uint8).reshape(-1, height, width)

    idx = np.argwhere(aff_scores > config.OBJ_CONFIDENCE_THRESHOLD)
    outputs['aff_scores'] = aff_scores[idx]
    outputs['obj_part_ids'] = obj_part_ids[idx]
    outputs['aff_ids'] = aff_ids[idx]
    outputs['aff_binary_masks'] = aff_binary_masks[idx, :, :]

    return outputs

def affnet_match_pred_to_gt(image, target, outputs):
    H, W, C = image.shape

    aff_ids = np.array(target['aff_ids'], dtype=np.int32).flatten()
    gt_aff_binary_masks = np.array(target['aff_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

    pred_scores = np.array(outputs['scores'], dtype=np.float32).flatten()
    pred_obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
    pred_obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
    pred_obj_binary_masks = np.squeeze(np.array(outputs['obj_binary_masks'] > config.MASK_THRESHOLD, dtype=np.uint8)).reshape(-1, H, W)

    pred_aff_scores = np.array(outputs['aff_scores'], dtype=np.float32).flatten()
    pred_obj_part_ids = np.array(outputs['obj_part_ids'], dtype=np.int32).flatten()
    pred_aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()
    pred_aff_binary_masks = np.array(outputs['aff_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

    matched_aff_scores = np.zeros_like(aff_ids, dtype=np.float32)
    matched_obj_part_ids = np.zeros_like(aff_ids)
    matched_aff_ids = np.zeros_like(aff_ids)
    matched_aff_binary_masks = np.zeros_like(gt_aff_binary_masks)

    # match based on box IoU.
    for obj_idx, pred_obj_id in enumerate(pred_obj_ids):
        obj_part_ids = arl_affpose_dataset_utils.map_obj_id_to_obj_part_ids(pred_obj_id)
        for obj_part_idx, obj_part_id in enumerate(obj_part_ids):
            pred_idx = np.argwhere(pred_obj_part_ids == obj_part_id)[0]
            matched_aff_scores[obj_part_idx] = pred_aff_scores[pred_idx]
            matched_obj_part_ids[obj_part_idx] = pred_obj_part_ids[pred_idx]
            matched_aff_ids[obj_part_idx] = pred_aff_ids[pred_idx]
            matched_aff_binary_masks[obj_part_idx, :, :] = pred_aff_binary_masks[pred_idx, :, :]

    outputs['aff_scores'] = matched_aff_scores.flatten()
    outputs['obj_part_ids'] = matched_obj_part_ids.flatten()
    outputs['aff_ids'] = matched_aff_ids
    outputs['aff_binary_masks'] = matched_aff_binary_masks

    return outputs

def affnet_umd_format_outputs(image, outputs):
    height, width = image.shape[0], image.shape[1]

    outputs['scores'] = np.array(outputs['scores'], dtype=np.float32).flatten()
    outputs['obj_ids'] = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
    outputs['obj_boxes'] = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)

    if 'aff_scores' in outputs:
        outputs['aff_scores'] = np.array(outputs['aff_scores'], dtype=np.float32).flatten()
        outputs['aff_ids'] = np.array(outputs['aff_ids'], dtype=np.int32).flatten()
        outputs['aff_binary_masks'] = np.array(outputs['aff_binary_masks'], dtype=np.float32).reshape(-1, height, width)

    else:
        outputs['aff_scores'] = np.zeros_like(outputs['scores'])
        outputs['aff_ids'] = np.zeros_like(outputs['obj_ids'])
        outputs['aff_binary_masks'] = np.zeros(shape=(len(outputs['obj_ids']), height, width))

    return outputs

def affnet_umd_threshold_outputs(image, outputs):
    height, width = image.shape[0], image.shape[1]

    # obj
    scores = np.array(outputs['scores'], dtype=np.float32).flatten()
    obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
    obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)

    try:
        idx = np.argmax(scores)
    except:
        idx = np.argwhere(scores > config.OBJ_CONFIDENCE_THRESHOLD)
    outputs['scores'] = scores[idx]
    outputs['obj_ids'] = obj_ids[idx]
    outputs['obj_boxes'] = obj_boxes[idx, :]

    # aff
    aff_scores = np.array(outputs['aff_scores'], dtype=np.float32).flatten()
    aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()
    aff_binary_masks = np.array(outputs['aff_binary_masks'], dtype=np.float32).reshape(-1, height, width)

    idx = np.argwhere(aff_scores > config.OBJ_CONFIDENCE_THRESHOLD)
    outputs['aff_scores'] = aff_scores[idx]
    outputs['aff_ids'] = aff_ids[idx]
    outputs['aff_binary_masks'] = aff_binary_masks[idx, :, :]

    return outputs

def affnet_umd_threshold_binary_masks(image, outputs, SHOW_IMAGES=False):
    height, width = image.shape[0], image.shape[1]
    plt.close('all')

    aff_binary_masks = np.array(outputs['aff_binary_masks'], dtype=np.float32).reshape(-1, height, width)
    aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()
    MEAN = 0

    if len(aff_ids) == 0:
        return outputs, MEAN

    for idx, data in enumerate(zip(aff_ids, aff_binary_masks)):
        aff_id, aff_binary_mask = data

        mean = aff_binary_mask[np.nonzero(aff_binary_mask.copy())].mean()
        std = aff_binary_mask[np.nonzero(aff_binary_mask.copy())].std()
        threshold = mean + std
        MEAN += mean

        threshold = config.MASK_THRESHOLD  # threshold if threshold < config.MASK_THRESHOLD else config.MASK_THRESHOLD
        threshold_mask = np.array(aff_binary_mask > threshold, dtype=np.uint8).reshape(height, width)
        outputs['aff_binary_masks'][idx, :, :] = threshold_mask

        if SHOW_IMAGES:
            print(f'Aff Id: {aff_id}, Mean: {mean:.5f}, Std: {std:.5f}, threshold: {threshold:.5f}')
            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(aff_binary_mask, cmap=plt.cm.plasma)
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(threshold_mask * threshold, cmap=plt.cm.plasma)
            plt.colorbar()
            plt.draw()
            plt.pause(0.0001)

    return outputs, MEAN/len(aff_ids)

# def affnet_umd_threshold_binary_masks(image, outputs, SHOW_IMAGES=False):
#     height, width = image.shape[0], image.shape[1]
#
#     aff_mask = np.array(outputs['aff_mask'] > config.MASK_THRESHOLD, dtype=np.float32).reshape(1, -1, height, width)
#     aff_mask = np.asarray(np.argmax(aff_mask, axis=1), dtype=np.uint8).reshape(height, width)
#     outputs['aff_mask'] = aff_mask
#
#     # print(f'image: {image.shape}')
#     # print(f'aff_mask: {aff_mask.shape}')
#     #
#     # color_aff_mask = umd_dataset_utils.colorize_aff_mask(aff_mask)
#     # color_aff_mask = cv2.addWeighted(image, 0.5, color_aff_mask, 0.5, 0)
#     # cv2.imshow('color_aff_mask', cv2.cvtColor(color_aff_mask, cv2.COLOR_BGR2RGB))
#     # cv2.waitKey(0)
#
#     return outputs