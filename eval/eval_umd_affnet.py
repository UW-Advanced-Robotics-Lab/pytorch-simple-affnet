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
from dataset.umd import umd_dataset_utils
from dataset.umd import umd_dataset_loaders
from eval import eval_utils

SHOW_IMAGES = True

NUM_RANDOM = 250
RANDOM_IMAGES = True
SHUFFLE_IMAGES = False

SAVE_AND_EVAL_PRED = True

NUM_UMD_TEST = 1298
OBJ_MASK_PROBABILITIES = np.zeros(shape=(NUM_UMD_TEST*5, config.UMD_NUM_OBJECT_CLASSES))

def main():

    if SAVE_AND_EVAL_PRED:
        # Init folders
        print('\neval in .. {}'.format(config.UMD_AFF_EVAL_SAVE_FOLDER))

        if not os.path.exists(config.UMD_AFF_EVAL_SAVE_FOLDER):
            os.makedirs(config.UMD_AFF_EVAL_SAVE_FOLDER)

        gt_pred_images = glob.glob(config.UMD_AFF_EVAL_SAVE_FOLDER + '*')
        for images in gt_pred_images:
            os.remove(images)

    # Load the Model.
    print()
    model = affnet.ResNetAffNet(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    # Load saved weights.
    print(f"\nrestoring pre-trained AffNet weights for UMD: {config.RESTORE_UMD_AFFNET_WEIGHTS} .. ")
    checkpoint = torch.load(config.RESTORE_UMD_AFFNET_WEIGHTS, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Load the dataset.
    test_loader = umd_dataset_loaders.load_umd_eval_datasets(random_images=RANDOM_IMAGES,
                                                             num_random=NUM_RANDOM,
                                                             shuffle_images=SHUFFLE_IMAGES)

    # run the predictions.
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
        target = umd_dataset_utils.format_target_data(image.copy(), target.copy())

        # Formatting Output.
        outputs = outputs.pop()
        outputs = eval_utils.affnet_umd_format_outputs(image.copy(), outputs.copy())
        outputs = eval_utils.affnet_umd_threshold_outputs(image.copy(), outputs.copy())
        outputs, obj_mask_probabilities = eval_utils.affnet_umd_threshold_binary_masks(image.copy(), outputs.copy(), False)
        scores = np.array(outputs['scores'], dtype=np.float32).flatten()
        obj_ids = np.array(outputs['obj_ids'], dtype=np.int32).flatten()
        obj_boxes = np.array(outputs['obj_boxes'], dtype=np.int32).reshape(-1, 4)
        aff_ids = np.array(outputs['aff_ids'], dtype=np.int32).flatten()
        aff_binary_masks = np.array(outputs['aff_binary_masks'], dtype=np.uint8).reshape(-1, H, W)

        for idx in range(len(obj_ids)):
            OBJ_MASK_PROBABILITIES[image_idx, obj_ids[idx]] = obj_mask_probabilities
            obj_name = "{:<13}".format(umd_dataset_utils.map_obj_id_to_name(obj_ids[idx]))
            print(f'Object:{obj_name}'
                  f'Obj id: {obj_ids[idx]}, '
                  f'Score:{scores[idx]:.3f}, ',
                  )

        # Pred bbox.
        print(f'1')
        pred_bbox_img = umd_dataset_utils.draw_bbox_on_img(image=image, scores=scores, obj_ids=obj_ids, boxes=obj_boxes)

        # Pred affordance mask.
        print(f'2')
        pred_aff_mask = umd_dataset_utils.get_segmentation_masks(image=image,
                                                                 obj_ids=aff_ids,
                                                                 binary_masks=aff_binary_masks,
                                                                 )
        color_aff_mask = umd_dataset_utils.colorize_aff_mask(pred_aff_mask)
        color_aff_mask = cv2.addWeighted(pred_bbox_img, 0.5, color_aff_mask, 0.5, 0)

        # gt affordance masks.
        print(f'3')
        binary_mask = umd_dataset_utils.get_segmentation_masks(image=image,
                                                               obj_ids=target['aff_ids'],
                                                               binary_masks=target['aff_binary_masks'],
                                                               )
        color_binary_mask = umd_dataset_utils.colorize_aff_mask(binary_mask)
        color_binary_mask = cv2.addWeighted(image, 0.5, color_binary_mask, 0.5, 0)

        if SAVE_AND_EVAL_PRED:
            # saving predictions.
            _image_idx = target["image_id"].detach().numpy()[0]
            _image_idx = str(1000000 + _image_idx)[1:]

            gt_name = config.UMD_AFF_EVAL_SAVE_FOLDER + _image_idx + config.TEST_GT_EXT
            pred_name = config.UMD_AFF_EVAL_SAVE_FOLDER + _image_idx + config.TEST_PRED_EXT

            cv2.imwrite(gt_name, target['aff_mask'])
            cv2.imwrite(pred_name, pred_aff_mask)

        # show plot.
        if SHOW_IMAGES:
            cv2.imshow('pred_bbox', cv2.cvtColor(pred_bbox_img, cv2.COLOR_BGR2RGB))
            cv2.imshow('pred_aff_mask', cv2.cvtColor(color_aff_mask, cv2.COLOR_BGR2RGB))
            cv2.imshow('gt_aff_mask', cv2.cvtColor(color_binary_mask, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)

    print()
    for obj_id in range(1, config.UMD_NUM_OBJECT_CLASSES):
        obj_mask_probabilities = OBJ_MASK_PROBABILITIES[:, obj_id]

        mean = obj_mask_probabilities[np.nonzero(obj_mask_probabilities.copy())].mean()
        std = obj_mask_probabilities[np.nonzero(obj_mask_probabilities.copy())].std()

        obj_name = "{:<13}".format(umd_dataset_utils.map_obj_id_to_name(obj_id))
        print(f'Object:{obj_name}'
              f'Obj id: {obj_id}, '
              f'Mean: {mean: .5f}, '
              # f'Std: {std: .5f}'
              )

    if SAVE_AND_EVAL_PRED:
        print()
        # getting FwB.
        os.chdir(config.MATLAB_SCRIPTS_DIR)
        import matlab.engine
        eng = matlab.engine.start_matlab()
        Fwb = eng.evaluate_umd_affnet(config.UMD_AFF_EVAL_SAVE_FOLDER, nargout=1)
        os.chdir(config.ROOT_DIR_PATH)

if __name__ == "__main__":
    main()