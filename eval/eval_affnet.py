import bisect
import glob
import os
import re
import time
import copy

import numpy as np
import cv2

import torch

import sys
sys.path.append('../')

import config

from model.affnet import affnet
from dataset import dataset_loaders
from dataset.arl_affpose import arl_affpose_dataset_utils

def main():

    # Init folders
    print('\neval in .. {}'.format(config.AFF_EVAL_SAVE_FOLDER))

    if not os.path.exists(config.AFF_EVAL_SAVE_FOLDER):
        os.makedirs(config.AFF_EVAL_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.AFF_EVAL_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    # Load the Model.
    print()
    model = affnet.ResNetAffNet(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    # Send model to GPU and load backends (if all inputs are the same size).
    model.to(config.DEVICE)
    torch.backends.cudnn.benchmark = True

    # Load saved weights.
    print(f"\nrestoring pre-trained MaskRCNN weights: {config.RESTORE_AFFNET_WEIGHTS} .. ")
    checkpoint = torch.load(config.RESTORE_AFFNET_WEIGHTS, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Load the dataset.
    test_loader = dataset_loaders.load_arl_affpose_eval_datasets()

    # run the predictions.
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
        image = np.array(image*(2**8-1), dtype=np.uint8).reshape(config.CROP_SIZE[0], config.CROP_SIZE[1], -1)

        target = target[0]
        target = {k: v.to(config.CPU_DEVICE) for k, v in target.items()}
        image, target = arl_affpose_dataset_utils.format_target_data(image, target)

        # Formatting Output.
        outputs = outputs.pop()
        scores = np.array(outputs['scores'], dtype=np.float32).flatten()
        obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
        obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()

        # Thresholding Binary Masks.
        aff_binary_masks = np.squeeze(np.array(outputs['aff_binary_masks'] > 0.5, dtype=np.uint8))

        # Thresholding predictions based on object confidence score.
        idx = np.argwhere(scores.copy() > config.CONFIDENCE_THRESHOLD)
        scores = scores[idx].reshape(-1)
        obj_ids = obj_ids[idx]
        obj_boxes = obj_boxes[idx].reshape(-1, 4)
        # TODO: threshold aff ids using objectiveness.
        aff_ids = aff_ids

        print(f'\nImage:{image_idx}/{len(test_loader)}')
        for score, obj_id in zip(scores, obj_ids):
            print(f'Object:{arl_affpose_dataset_utils.map_obj_id_to_name(obj_id)}, Score:{score:.2f}')

        # bbox
        pred_bbox_img = arl_affpose_dataset_utils.draw_bbox_on_img(image=image,
                                                                   obj_ids=obj_ids,
                                                                   boxes=obj_boxes,
                                                                   )
        cv2.imshow('pred_bbox', cv2.cvtColor(pred_bbox_img, cv2.COLOR_BGR2RGB))

        # getting predicted affordnace mask.
        pred_aff_mask = arl_affpose_dataset_utils.get_segmentation_masks(image=image,
                                                                         obj_ids=aff_ids,
                                                                         binary_masks=aff_binary_masks,
                                                                         )
        # Original Segmentation Mask.
        color_mask = arl_affpose_dataset_utils.colorize_aff_mask(pred_aff_mask)
        color_mask = cv2.addWeighted(pred_bbox_img, 0.35, color_mask, 0.65, 0)
        cv2.imshow('pred_mask', cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))

        pred_obj_part_mask = arl_affpose_dataset_utils.get_obj_part_mask(image=image,
                                                                    obj_ids=target['obj_ids'],
                                                                    aff_ids=target['aff_ids'],
                                                                    bboxs=target['obj_boxes'],
                                                                    binary_masks=target['aff_binary_masks'],
                                                                    )

        # saving predictions.
        _image_idx = target["image_id"].detach().numpy()[0]
        _image_idx = str(1000000 + _image_idx)[1:]

        gt_name = config.AFF_EVAL_SAVE_FOLDER + _image_idx + config.TEST_GT_EXT
        pred_name = config.AFF_EVAL_SAVE_FOLDER + _image_idx + config.TEST_PRED_EXT
        obj_part_name = config.AFF_EVAL_SAVE_FOLDER + _image_idx + config.TEST_OBJ_PART_EXT

        cv2.imwrite(gt_name, target['aff_mask'])
        cv2.imwrite(pred_name, pred_aff_mask)
        cv2.imwrite(obj_part_name, pred_obj_part_mask)

        # show plot.
        cv2.waitKey(1)

    print()
    # getting FwB.
    os.chdir(config.MATLAB_SCRIPTS_DIR)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    Fwb = eng.evaluate_ARLAffPose(config.AFF_EVAL_SAVE_FOLDER, nargout=1)
    os.chdir(config.ROOT_DIR_PATH)

if __name__ == "__main__":
    main()