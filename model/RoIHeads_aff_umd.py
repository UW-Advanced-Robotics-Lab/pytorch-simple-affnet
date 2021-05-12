import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from model.utils.bbox_utils import BoxCoder, box_iou, process_box, nms, Matcher, BalancedPositiveNegativeSampler
from model.utils.losses_utils import fastrcnn_loss, maskrcnn_loss

###############################
###############################

import cfg as config

from dataset.utils.UMD import umd_utils
from dataset.utils.ARLAffPose import affpose_dataset_utils

###############################
###############################

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

        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)

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
        obj_proposal = proposal.clone()

        ### obj
        obj_gt_box = target['boxes']
        obj_gt_label = target['labels']
        obj_proposal = torch.cat((obj_proposal, obj_gt_box))

        obj_iou = box_iou(obj_gt_box, obj_proposal)
        obj_pos_neg_label, obj_matched_idx = self.proposal_matcher(obj_iou)
        obj_pos_idx, obj_neg_idx = self.fg_bg_sampler(obj_pos_neg_label)
        obj_idx = torch.cat((obj_pos_idx, obj_neg_idx))

        obj_regression_target = self.box_coder.encode(obj_gt_box[obj_matched_idx[obj_pos_idx]],
                                                      obj_proposal[obj_pos_idx])
        obj_proposal = obj_proposal[obj_idx]
        obj_matched_idx = obj_matched_idx[obj_idx]
        obj_label = obj_gt_label[obj_matched_idx]
        obj_num_pos = obj_pos_idx.shape[0]
        obj_label[obj_num_pos:] = 0

        return obj_proposal, obj_matched_idx, obj_label, obj_regression_target


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

            box, score = process_box(box, score, image_shape, self.min_size)

            keep = nms(box, score, self.nms_thresh)[:self.num_detections]
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)

            boxes.append(box)
            labels.append(label)
            scores.append(score)

        results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))
        return results

    def forward(self, feature, proposal, image_shape, target):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)

        box_feature = self.box_roi_pool(feature, proposal, image_shape)
        class_logit, box_regression = self.box_predictor(box_feature)

        result, losses = {}, {}
        if self.training:
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(loss_classifier=classifier_loss, loss_box_reg=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)

        if self.has_mask():
            if self.training:
                ####################################################
                ### convert obj labels in a bbox to aff label
                ####################################################
                num_pos = regression_target.shape[0]
                # print(f'\nnum_pos:{num_pos}')

                obj_labels = label[:num_pos].detach().cpu().numpy()
                aff_labels = target['aff_labels'].detach().cpu().numpy()
                # aff_labels = umd_utils.format_object_id_to_aff_id(object_ids=obj_labels, aff_id=aff_labels)
                # print(f'obj_label:{obj_labels}, aff_labels:{aff_labels}')

                _aff_labels = []
                _gt_masks = []
                _mask_proposal = []
                _pos_matched_idx = []
                for i in range(num_pos):

                    aff_label = aff_labels[i]
                    num_aff_label = len(aff_label)
                    # print(f'aff_label: len:{num_aff_label} data:{aff_label}')

                    mask_proposal = proposal[i].detach().cpu().numpy()
                    # print(f'mask_proposal: size:{mask_proposal.shape}, data:{mask_proposal}')
                    mask_proposal = np.tile(mask_proposal, num_aff_label).reshape(-1, 4)
                    # print(f'mask_proposal:{mask_proposal}')

                    # TODO: match pos idx
                    # pos_matched_idx = matched_idx[i].detach().cpu().numpy()
                    # print(f'pos_matched_idx: size:{pos_matched_idx.shape}, data:{pos_matched_idx}')
                    # pos_matched_idx = np.repeat(pos_matched_idx, num_aff_label)
                    pos_matched_idx = np.arange(start=0, stop=num_aff_label)
                    # print(f'pos_matched_idx:{pos_matched_idx}')

                    _aff_labels.extend(aff_label)
                    _mask_proposal.extend(mask_proposal.tolist())
                    _pos_matched_idx.extend(pos_matched_idx.tolist())

                aff_labels = torch.as_tensor(_aff_labels).to(config.DEVICE)
                mask_proposal = torch.as_tensor(_mask_proposal).to(config.DEVICE)
                pos_matched_idx = torch.as_tensor(_pos_matched_idx).to(config.DEVICE)

                # print(f'aff_labels:{aff_labels}')
                # print(f'mask_proposal:{mask_proposal}')
                # print(f'pos_matched_idx:{pos_matched_idx}')

                if mask_proposal.shape[0] == 0:
                    losses.update(dict(loss_mask=torch.tensor(0)))
                    return result, losses
            else:
                ####################################################
                ### convert obj labels in a bbox to aff label
                ####################################################
                mask_proposal = result['boxes']
                num_pos = mask_proposal.shape[0]
                # print(f'\nnum_pos:{num_pos}')
                # print(f'mask_proposal:{mask_proposal}')

                obj_labels = result['labels'].detach().cpu().numpy()
                # aff_labels = umd_utils.object_id_to_aff_id(object_ids=obj_labels)
                aff_labels = affpose_dataset_utils.map_obj_id_to_aff_ids(object_ids=obj_labels)
                # print(f'obj_label:{obj_labels}, aff_labels:{aff_labels}')

                _aff_labels = []
                _mask_proposals = []
                for i in range(num_pos):
                    _aff_label = aff_labels[i]
                    num_aff_label = len(_aff_label)
                    # print(f'\naff_label: len:{num_aff_label} data:{_aff_label}')

                    _mask_proposal = mask_proposal[i].detach().cpu().numpy()
                    # print(f'mask_proposal: size:{mask_proposal.shape}, data:{_mask_proposal}')
                    _mask_proposal = np.tile(_mask_proposal, num_aff_label).reshape(-1, 4)
                    # print(f'mask_proposal:{_mask_proposal}')
                    # # TODO: add random noise to mask_proposal for multi-class seg
                    # for __mask_proposal in _mask_proposal:
                    #     __mask_proposal[0] += np.random.normal(0, 3, 1)
                    #     __mask_proposal[1] += np.random.normal(0, 3, 1)
                    #     __mask_proposal[2] += np.random.normal(0, 3, 1)
                    #     __mask_proposal[3] += np.random.normal(0, 3, 1)
                    # print(f'mask_proposal:{_mask_proposal}')

                    _aff_labels.extend(_aff_label)
                    _mask_proposals.extend(_mask_proposal.tolist())

                aff_labels = torch.as_tensor(_aff_labels).to(config.DEVICE)
                mask_proposal = torch.as_tensor(_mask_proposals).to(config.DEVICE)
                idx = torch.arange(aff_labels.shape[0], device=aff_labels.device)

                # print(f'\naff_labels:{aff_labels}')
                # print(f'mask_proposal:{mask_proposal}')
                # print(f'idx:{idx}')

                if mask_proposal.shape[0] == 0:
                    result.update(dict(masks=torch.empty((0, 28, 28))))
                    return result, losses

            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
            mask_logit = self.mask_predictor(mask_feature)

            # print(f"mask_logit:{mask_logit.size()}")

            if self.training:
                gt_mask = target['masks']
                mask_loss = maskrcnn_loss(mask_logit=mask_logit,
                                          gt_mask=gt_mask,
                                          proposal=mask_proposal,
                                          matched_idx=pos_matched_idx,
                                          label=aff_labels)
                losses.update(dict(loss_mask=mask_loss))
            else:
                mask_logit = mask_logit[idx, aff_labels]

                mask_prob = mask_logit.sigmoid()
                # print(f'mask_prob:{mask_prob.size()}')

                result.update(dict(masks=mask_prob, aff_labels=aff_labels))

                # binary_masks = np.squeeze(np.array(mask_prob.detach().cpu()>0.35, dtype=np.float))
                # print(f'binary_masks:{binary_masks.shape}')
                # for i in range(len(aff_labels)):
                #     aff_label = aff_labels[i].detach().cpu()
                #     binary_mask = binary_masks[i, :, :]
                #     print(f'pre-processing: aff_label:{aff_label}, data:{np.unique(binary_mask, return_counts=True)}')

        return result, losses