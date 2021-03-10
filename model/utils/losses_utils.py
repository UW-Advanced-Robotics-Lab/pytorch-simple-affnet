import cv2

import torch
import torch.nn.functional as F
from torch import nn

from model.RoIAlign import roi_align
from model.utils.bbox_utils import box_iou


###############################
###############################

def rpn_loss(idx, pos_idx, objectness, label, pred_bbox_delta, regression_target):
    objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])
    box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel()

    return objectness_loss, box_loss


###############################
###############################

def fastrcnn_loss(class_logit, box_regression, label, regression_target):
    classifier_loss = F.cross_entropy(class_logit, label)

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]
    box_idx = torch.arange(num_pos, device=label.device)

    box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N

    return classifier_loss, box_reg_loss

###############################
###############################

def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)

    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)

    # print(f"matched_idx:{matched_idx}")
    # print(f"idx:{idx}, label:{label}")
    # print(f"mask_logit:{mask_logit[idx, label].size()}, mask_target:{mask_target.size()}")

    # for i in range(len(idx)):
    #     _label = label[i]
    #     _gt_mask = mask_target[i, :, :].detach().cpu().numpy()
    #     # print(f"\tlabel:{_label}, gt_mask:{_gt_mask.shape}")
    #     cv2.imshow('gt', _gt_mask)
    #     _mask_logit = mask_logit[idx, label][i, :, :].detach().cpu().numpy()
    #     cv2.imshow('mask_logit', _mask_logit)
    #     cv2.waitKey(0)

    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)
    return mask_loss

