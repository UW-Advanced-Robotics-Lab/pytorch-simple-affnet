import cv2
import numpy as np

import config
from dataset import dataset_utils


def colorize_obj_mask(instance_mask):

    np.random.seed(0)
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)

    obj_ids = np.unique(instance_mask)[1:]
    for obj_id in obj_ids:
        rgb = np.random.randint(low=0, high=255, size=3,)
        color_mask[instance_mask == obj_id] = rgb

    return np.squeeze(color_mask)

def format_target_data(image, target):
    height, width = image.shape[0], image.shape[1]

    # original mask and binary masks.
    target['obj_binary_masks'] = np.array(target['obj_binary_masks'], dtype=np.uint8).reshape(-1, height, width)

    # ids and bboxs.
    target['obj_ids'] = np.array(target['obj_ids'], dtype=np.int32).flatten()
    target['obj_boxes'] = np.array(target['obj_boxes'], dtype=np.int32).reshape(-1, 4)

    return image, target

def draw_bbox_on_img(image, obj_ids, boxes, color=(255, 255, 255)):
    bbox_img = image.copy()

    for obj_id, bbox in zip(obj_ids, boxes):
        bbox = dataset_utils.format_bbox(bbox)
        # see dataset_utils.get_bbox for output of bbox.
        # x1,y1 ------
        # |          |
        # |          |
        # |          |
        # --------x2,y2
        bbox_img = cv2.rectangle(bbox_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, 1)

        cv2.putText(bbox_img,
                    f'{obj_id}',
                    (bbox[0], bbox[1] - 5),
                    cv2.FONT_ITALIC,
                    0.4,
                    color)

    return bbox_img

def get_segmentation_masks(image, obj_ids, binary_masks):

    height, width = image.shape[0], image.shape[1]
    instance_masks = np.zeros((height, width), dtype=np.uint8)
    instance_mask_one = np.ones((height, width), dtype=np.uint8)

    if len(binary_masks.shape) == 2:
        binary_masks = binary_masks[np.newaxis, :, :]

    for idx, obj_id in enumerate(obj_ids):
        binary_mask = binary_masks[idx, :, :]

        instance_mask = instance_mask_one * obj_id
        instance_masks = np.where(binary_mask, instance_mask, instance_masks).astype(np.uint8)

    return instance_masks