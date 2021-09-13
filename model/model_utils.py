import math

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from model import roi_align
from dataset.arl_affpose import arl_affpose_dataset_utils


def get_model_instance_segmentation(pretrained, num_classes):
    """Function to load torchvision MaskRCNN."""
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

def freeze_backbone(model, verbose=False):
    """Function to Freeze Backbone."""
    # print(f'\nFreezing backbone ..')
    for name, parameter in model.named_parameters():
        if 'backbone' in name:
            parameter.requires_grad_(False)
            if verbose:
                print(f'Frozen: {name}')
        else:
            parameter.requires_grad_(True)
            if verbose:
                print(f'Requires Grad: {name}')
    return model

def freeze_heads(model, verbose=False):
    """Function to Freeze Network Heads."""
    # print(f'\nFreezing backbone ..')
    for name, parameter in model.named_parameters():
        if 'backbone' in name:
            parameter.requires_grad_(True)
            if verbose:
                print(f'Frozen: {name}')
        else:
            parameter.requires_grad_(False)
            if verbose:
                print(f'Requires Grad: {name}')
    return model

def unfreeze_all_layers(model):
    """Function to Un-freeze Backbone."""
    # print(f'\nUn-freezing backbone ..')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(True)
    return model

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def rpn_loss(idx, pos_idx, objectness, label, pred_bbox_delta, regression_target):
    # TODO: try focal loss.
    objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])
    # objectness_loss = sigmoid_focal_loss(objectness[idx], label[idx], reduction='mean')
    box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel()

    return objectness_loss, box_loss

def fastrcnn_loss(class_logit, box_regression, label, regression_target):
    classifier_loss = F.cross_entropy(class_logit, label)

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]
    box_idx = torch.arange(num_pos, device=label.device)

    box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N

    return classifier_loss, box_reg_loss

def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):

    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)

    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align.roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)

    # # TODO: upsample masks for better predictions.
    # size = (640, 640)  # (224, 224)
    # # print(f"\n[before upsample] mask_logit: {mask_logit.size()}")
    # # print(f"[before upsample] mask_target: {mask_target.size()}")
    # mask_logit = F.interpolate(mask_logit, size=size, mode='bilinear', align_corners=False)
    # mask_target = F.interpolate(mask_target[None], size=size, mode='bilinear', align_corners=False)[0]
    # # print(f"[after upsample] mask_logit: {mask_logit.size()}")
    # # print(f""[after upsample] mask_target: {mask_target.size()}")

    # # check size of gt mask.
    # print(f'\ngt mask: {gt_mask.size()}')
    # # check size of mask after ROI pooling.
    # print(f"pred mask: {mask_logit[idx, label].size()}")

    # check gt masks after ROI cropping.
    # for i in range(len(idx)):
    #     current_obj_id = label[i]
    #     current_gt_mask = mask_target[i, :, :].detach().cpu().numpy()
    #     current_pred_mask = mask_logit[idx, label][i, :, :].detach().cpu().numpy()
    #
    #     # print output.
    #     print(f"\nidx: {i}, num idxs {len(idx)}")
    #     print(f"obj id:{current_obj_id}")
    #     print(f"gt mask: {mask_target.shape}")
    #     print(f"gt mask: {mask_logit.shape}")
    #
    #     # visualize.
    #     cv2.imshow('gt', current_gt_mask)
    #     cv2.imshow('pred', current_pred_mask)
    #     cv2.waitKey(0)

    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target,)

    # TODO: try class weights loss.
    # class_weights = arl_affpose_dataset_utils.get_class_weights(aff_ids=label)
    # mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target, reduction='none').sum() / class_weights.sum()

    # TODO: try focal loss.
    # mask_loss = sigmoid_focal_loss(mask_logit[idx, label], mask_target, reduction='mean')

    return mask_loss

class AnchorGenerator:
    def __init__(self, sizes, ratios):
        self.sizes = sizes
        self.ratios = ratios

        self.cell_anchor = None
        self._cache = {}

    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return
        sizes = torch.tensor(self.sizes, dtype=dtype, device=device)
        ratios = torch.tensor(self.ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios

        hs = (sizes[:, None] * h_ratios[None, :]).view(-1)
        ws = (sizes[:, None] * w_ratios[None, :]).view(-1)

        self.cell_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

    def grid_anchor(self, grid_size, stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        shift_x = torch.arange(0, grid_size[1], dtype=dtype, device=device) * stride[1]
        shift_y = torch.arange(0, grid_size[0], dtype=dtype, device=device) * stride[0]

        y, x = torch.meshgrid(shift_y, shift_x)
        x = x.reshape(-1)
        y = y.reshape(-1)
        shift = torch.stack((x, y, x, y), dim=1).reshape(-1, 1, 4)

        anchor = (shift + self.cell_anchor).reshape(-1, 4)
        return anchor

    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride
        if key in self._cache:
            return self._cache[key]
        anchor = self.grid_anchor(grid_size, stride)

        if len(self._cache) >= 3:
            self._cache.clear()
        self._cache[key] = anchor
        return anchor

    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        grid_size = tuple(feature.shape[-2:])
        stride = tuple(int(i / g) for i, g in zip(image_size, grid_size))

        self.set_cell_anchor(dtype, device)

        anchor = self.cached_grid_anchor(grid_size, stride)
        return anchor

class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between
            M ground-truth boxes and N predicted boxes.

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """

        value, matched_idx = iou.max(dim=0)
        label = torch.full((iou.shape[1],), -1, dtype=torch.float, device=iou.device)

        label[value >= self.high_threshold] = 1
        label[value < self.low_threshold] = 0

        if self.allow_low_quality_matches:
            highest_quality = iou.max(dim=1)[0]
            gt_pred_pairs = torch.where(iou == highest_quality[:, None])[1]
            label[gt_pred_pairs] = 1

        return label, matched_idx

class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = torch.where(label == 1)[0]
        negative = torch.where(label == 0)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)

        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx

class BoxCoder:
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_box, proposal):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor[N, 4]): reference boxes
            proposals (Tensor[N, 4]): boxes to be encoded
        """
        
        width = proposal[:, 2] - proposal[:, 0]
        height = proposal[:, 3] - proposal[:, 1]
        ctr_x = proposal[:, 0] + 0.5 * width
        ctr_y = proposal[:, 1] + 0.5 * height

        gt_width = reference_box[:, 2] - reference_box[:, 0]
        gt_height = reference_box[:, 3] - reference_box[:, 1]
        gt_ctr_x = reference_box[:, 0] + 0.5 * gt_width
        gt_ctr_y = reference_box[:, 1] + 0.5 * gt_height

        dx = self.weights[0] * (gt_ctr_x - ctr_x) / width
        dy = self.weights[1] * (gt_ctr_y - ctr_y) / height
        dw = self.weights[2] * torch.log(gt_width / width)
        dh = self.weights[3] * torch.log(gt_height / height)

        delta = torch.stack((dx, dy, dw, dh), dim=1)
        return delta

    def decode(self, delta, box):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            delta (Tensor[N, 4]): encoded boxes.
            boxes (Tensor[N, 4]): reference boxes.
        """
        
        dx = delta[:, 0] / self.weights[0]
        dy = delta[:, 1] / self.weights[1]
        dw = delta[:, 2] / self.weights[2]
        dh = delta[:, 3] / self.weights[3]

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        width = box[:, 2] - box[:, 0]
        height = box[:, 3] - box[:, 1]
        ctr_x = box[:, 0] + 0.5 * width
        ctr_y = box[:, 1] + 0.5 * height

        pred_ctr_x = dx * width + ctr_x
        pred_ctr_y = dy * height + ctr_y
        pred_w = torch.exp(dw) * width
        pred_h = torch.exp(dh) * height

        xmin = pred_ctr_x - 0.5 * pred_w
        ymin = pred_ctr_y - 0.5 * pred_h
        xmax = pred_ctr_x + 0.5 * pred_w
        ymax = pred_ctr_y + 0.5 * pred_h

        target = torch.stack((xmin, ymin, xmax, ymax), dim=1)
        return target
    
def box_iou(box_a, box_b):
    """
    Arguments:
        boxe_a (Tensor[N, 4])
        boxe_b (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in box_a and box_b
    """
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], 1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], 1)
    
    return inter / (area_a[:, None] + area_b - inter)

def process_box(box, score, image_shape, min_size):
    """
    Clip boxes in the image size and remove boxes which are too small.
    """
    
    box[:, [0, 2]] = box[:, [0, 2]].clamp(0, image_shape[1]) 
    box[:, [1, 3]] = box[:, [1, 3]].clamp(0, image_shape[0]) 

    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    keep = torch.where((w >= min_size) & (h >= min_size))[0]
    box, score = box[keep], score[keep]
    return box, score

def nms(box, score, threshold):
    """
    Arguments:
        box (Tensor[N, 4])
        score (Tensor[N]): scores of the boxes.
        threshold (float): iou threshold.

    Returns: 
        keep (Tensor): indices of boxes filtered by NMS.
    """
    
    return torch.ops.torchvision.nms(box, score, threshold)

# just for test. It is too slow. Don't use it during train
def slow_nms(box, nms_thresh):
    idx = torch.arange(box.size(0))
    
    keep = []
    while idx.size(0) > 0:
        keep.append(idx[0].item())
        head_box = box[idx[0], None, :]
        remain = torch.where(box_iou(head_box, box[idx]) <= nms_thresh)[1]
        idx = idx[remain]
    
    return keep