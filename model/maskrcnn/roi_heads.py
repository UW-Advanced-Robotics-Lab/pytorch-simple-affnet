import torch
import torch.nn.functional as F
from torch import nn

import config
from model import model_utils


class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor

        self.mask_roi_pool = None
        self.mask_predictor = None

        self.proposal_matcher = model_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = model_utils.BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = model_utils.BoxCoder(reg_weights)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def select_training_samples(self, proposal, target):
        gt_box = target['obj_boxes']
        gt_label = target['obj_ids']
        proposal = torch.cat((proposal, gt_box))

        iou = model_utils.box_iou(gt_box, proposal)
        pos_neg_label, matched_idx = self.proposal_matcher(iou)
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
        idx = torch.cat((pos_idx, neg_idx))

        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
        proposal = proposal[idx]
        matched_idx = matched_idx[idx]
        label = gt_label[matched_idx]
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0

        return proposal, matched_idx, label, regression_target

    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes = class_logit.shape

        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)

        boxes = []
        labels = []
        scores = []
        for l in range(1, num_classes):
            score, box_delta = pred_score[:, l], box_regression[:, l]

            keep = score >= self.score_thresh
            box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
            box = self.box_coder.decode(box_delta, box)

            box, score = model_utils.process_box(box, score, image_shape, self.min_size)

            keep = model_utils.nms(box, score, self.nms_thresh)[:self.num_detections]
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)

            boxes.append(box)
            labels.append(label)
            scores.append(score)

        results = dict(obj_boxes=torch.cat(boxes), obj_ids=torch.cat(labels), scores=torch.cat(scores))
        return results

    def forward(self, feature, proposal, image_shape, target):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)

        box_feature = self.box_roi_pool(feature, proposal, image_shape)
        class_logit, box_regression = self.box_predictor(box_feature)

        result, losses = {}, {}
        if self.training:
            classifier_loss, box_reg_loss = model_utils.fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(loss_classifier=classifier_loss, loss_box_reg=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)

        if self.has_mask():
            if self.training:
                num_pos = regression_target.shape[0]

                mask_proposal = proposal[:num_pos]
                pos_matched_idx = matched_idx[:num_pos]
                mask_label = label[:num_pos]

                '''
                # -------------- critial ----------------
                box_regression = box_regression[:num_pos].reshape(num_pos, -1, 4)
                idx = torch.arange(num_pos, device=mask_label.device)
                mask_proposal = self.box_coder.decode(box_regression[idx, mask_label], mask_proposal)
                # ---------------------------------------
                '''

                if mask_proposal.shape[0] == 0:
                    losses.update(dict(loss_mask=torch.tensor(0)))
                    return result, losses
            else:
                mask_proposal = result['obj_boxes']

                if mask_proposal.shape[0] == 0:
                    result.update(dict(obj_binary_masks=torch.empty((0, 28, 28))))
                    return result, losses

            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
            mask_logit = self.mask_predictor(mask_feature)

            if self.training:
                gt_mask = target['obj_binary_masks']
                mask_loss = model_utils.maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                losses.update(dict(loss_mask=mask_loss))
            else:
                label = result['obj_ids']
                idx = torch.arange(label.shape[0], device=label.device)
                mask_logit = mask_logit[idx, label]

                mask_prob = mask_logit.sigmoid()
                result.update(dict(obj_binary_masks=mask_prob))

        return result, losses