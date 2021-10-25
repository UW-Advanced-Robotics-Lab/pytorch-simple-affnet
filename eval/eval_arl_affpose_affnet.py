import bisect
import glob
import os
import re
import time
import copy

import numpy as np
import cv2

from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

import torch

import sys
sys.path.append('../')

import config

from model.affnet import affnet
from dataset.arl_affpose import arl_affpose_dataset_utils
from dataset.arl_affpose import arl_affpose_dataset_loaders
from eval import eval_utils

SHOW_IMAGES = False

NUM_RANDOM = 100
RANDOM_IMAGES = False
SHUFFLE_IMAGES = False

SAVE_AND_EVAL_PRED = True


def main():

    # if SAVE_AND_EVAL_PRED:
    #     # Init folders
    #     print('\neval in .. {}'.format(config.ARL_AFF_EVAL_SAVE_FOLDER))
    #
    #     if not os.path.exists(config.ARL_AFF_EVAL_SAVE_FOLDER):
    #         os.makedirs(config.ARL_AFF_EVAL_SAVE_FOLDER)
    #
    #     gt_pred_images = glob.glob(config.ARL_AFF_EVAL_SAVE_FOLDER + '*')
    #     for images in gt_pred_images:
    #         os.remove(images)

    # Load the Model.
    print()
    model = affnet.ResNetAffNet(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    # Load saved weights.
    print(f"\nrestoring pre-trained MaskRCNN weights: {config.RESTORE_ARL_AFFNET_WEIGHTS} .. ")
    checkpoint = torch.load(config.RESTORE_ARL_AFFNET_WEIGHTS, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Load the dataset.
    test_loader = arl_affpose_dataset_loaders.load_arl_affpose_eval_datasets(random_images=RANDOM_IMAGES,
                                                                             num_random=NUM_RANDOM,
                                                                             shuffle_images=SHUFFLE_IMAGES)

    # run the predictions.
    APs = []
    gt_obj_ids_list, pred_obj_ids_list = [], []
    for image_idx, (images, targets) in enumerate(test_loader):
        print(f'\nImage:{image_idx+1}/{len(test_loader)}')

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
        outputs = eval_utils.affnet_format_outputs(image.copy(), outputs.copy())
        outputs = eval_utils.affnet_threshold_outputs(image.copy(), outputs.copy())
        outputs = eval_utils.maskrcnn_match_pred_to_gt(image.copy(), target.copy(), outputs.copy())
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

        # for obj_binary_mask, gt_obj_binary_mask in zip(obj_binary_masks, gt_obj_binary_masks):
        #     cv2.imshow('obj_binary_mask', obj_binary_mask*20)
        #     cv2.imshow('gt_obj_binary_mask', gt_obj_binary_mask*20)
        #     cv2.waitKey(0)

        # get average precision.
        AP = eval_utils.compute_ap_range(gt_class_id=gt_obj_ids,
                                         gt_box=gt_obj_boxes,
                                         gt_mask=gt_obj_binary_masks.reshape(H, W, -1),
                                         pred_score=scores,
                                         pred_class_id=obj_ids,
                                         pred_box=obj_boxes,
                                         pred_mask=obj_binary_masks.reshape(H, W, -1),
                                         verbose=False,
                                         )
        APs.append(AP)

        # print outputs.
        for gt_idx, pred_idx in zip(range(len(gt_obj_ids)), range(len(obj_ids))):
            gt_obj_name = "{:<15}".format(arl_affpose_dataset_utils.map_obj_id_to_name(gt_obj_ids[gt_idx]))
            pred_obj_name = "{:<15}".format(arl_affpose_dataset_utils.map_obj_id_to_name(obj_ids[pred_idx]))
            score = scores[pred_idx]
            bbox_iou = eval_utils.get_iou(obj_boxes[pred_idx], gt_obj_boxes[gt_idx])

            print(f'GT: {gt_obj_name}',
                  f'Pred: {pred_obj_name}'
                  f'Score: {score:.3f},\t\t',
                  f'IoU: {bbox_iou:.3f},',
                  )
        print("AP @0.5-0.95: {:.5f}".format(AP))

        # visualize bbox.
        pred_bbox_img = arl_affpose_dataset_utils.draw_bbox_on_img(image=image, scores=scores, obj_ids=obj_ids, boxes=obj_boxes)

        # visualize affordance masks.
        pred_aff_mask = arl_affpose_dataset_utils.get_segmentation_masks(image=image,
                                                                         obj_ids=aff_ids,
                                                                         binary_masks=aff_binary_masks,
                                                                         )
        color_aff_mask = arl_affpose_dataset_utils.colorize_aff_mask(pred_aff_mask)
        color_aff_mask = cv2.addWeighted(pred_bbox_img, 0.5, color_aff_mask, 0.5, 0)

        # get obj part mask.
        pred_obj_part_mask = arl_affpose_dataset_utils.get_obj_part_mask(image=image,
                                                                    obj_part_ids=obj_part_ids,
                                                                    aff_binary_masks=aff_binary_masks,
                                                                    )
        # visualize object masks.
        pred_obj_mask = arl_affpose_dataset_utils.convert_obj_part_mask_to_obj_mask(pred_obj_part_mask)
        color_obj_mask = arl_affpose_dataset_utils.colorize_obj_mask(pred_obj_mask)
        color_obj_mask = cv2.addWeighted(pred_bbox_img, 0.5, color_obj_mask, 0.5, 0)

        if SAVE_AND_EVAL_PRED:
            # saving predictions.
            _image_idx = target["image_id"].detach().numpy()[0]
            _image_idx = str(1000000 + _image_idx)[1:]

            gt_name = config.ARL_AFF_EVAL_SAVE_FOLDER + _image_idx + config.TEST_GT_EXT
            pred_name = config.ARL_AFF_EVAL_SAVE_FOLDER + _image_idx + config.TEST_PRED_EXT
            obj_part_name = config.ARL_AFF_EVAL_SAVE_FOLDER + _image_idx + config.TEST_OBJ_PART_EXT

            cv2.imwrite(gt_name, target['aff_mask'])
            cv2.imwrite(pred_name, pred_aff_mask)
            cv2.imwrite(obj_part_name, pred_obj_part_mask)

        # show plot.
        if SHOW_IMAGES:
            cv2.imshow('pred_bbox', cv2.cvtColor(pred_bbox_img, cv2.COLOR_BGR2RGB))
            cv2.imshow('pred_aff_mask', cv2.cvtColor(color_aff_mask, cv2.COLOR_BGR2RGB))
            cv2.imshow('pred_obj_part_mask', cv2.cvtColor(color_obj_mask, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)

    # Confusion Matrix.
    cm = sklearn_confusion_matrix(y_true=gt_obj_ids_list, y_pred=pred_obj_ids_list)
    print(f'\n{cm}')

    # Plot Confusion Matrix.
    # eval_utils.plot_confusion_matrix(cm, arl_affpose_dataset_utils.OBJ_NAMES)

    # mAP
    print("\nmAP @0.5-0.95: over {} test images is {:.3f}".format(len(APs), np.mean(APs)))

    if SAVE_AND_EVAL_PRED:
        print()
        # getting FwB.
        os.chdir(config.MATLAB_SCRIPTS_DIR)
        import matlab.engine
        eng = matlab.engine.start_matlab()
        Fwb = eng.evaluate_arl_affpose_affnet(config.ARL_AFF_EVAL_SAVE_FOLDER, nargout=1)
        os.chdir(config.ROOT_DIR_PATH)

if __name__ == "__main__":
    main()
