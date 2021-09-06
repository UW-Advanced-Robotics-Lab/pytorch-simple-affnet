import sys
import re
import time
import copy

import cv2
import numpy as np

import torch

import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

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

class TextArea:
    def __init__(self):
        self.buffer = []

    def write(self, s):
        self.buffer.append(s)

    def __str__(self):
        return "".join(self.buffer)

    def get_AP(self):
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        values = [int(v) / 10 for v in values]
        result = {"bbox AP": values[0], "mask AP": values[12]}
        return result


class Meter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:sum={sum:.2f}, avg={avg:.4f}, count={count}"
        return fmtstr.format(**self.__dict__)


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types="bbox"):
        if isinstance(iou_types, str):
            iou_types = [iou_types]

        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        # self.ann_labels = ann_labels
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type)
                          for iou_type in iou_types}

    def accumulate(self, coco_results):  # input all predictions
        image_ids = list(set([res["image_id"] for res in coco_results]))
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_dt = self.coco_gt.loadRes(coco_results) if coco_results else COCO()  # use the method loadRes

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = image_ids  # ids of images to be evaluated
            coco_eval.evaluate()  # 15.4s
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            coco_eval.accumulate()  # 3s

    def summarize(self):
        for iou_type in self.iou_types:
            print("IoU metric: {}".format(iou_type))
            self.coco_eval[iou_type].summarize()

def prepare_for_coco(predictions, ann_labels):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]
        masks = prediction["masks"]

        x1, y1, x2, y2 = boxes.unbind(1)
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)
        boxes = boxes.tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        labels = [ann_labels[l] for l in labels]

        masks = masks > 0.5
        rles = [
            mask_util.encode(np.array(mask[:, :, :], dtype=np.uint8, order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[i],
                    "bbox": boxes[i],
                    "segmentation": rle,
                    "score": scores[i],
                }
                for i, rle in enumerate(rles)
            ]
        )
    return coco_results


def evaluate(model, data_loader, device, saved_model_path, generate=True):
    if generate:
        iter_eval = generate_results(model, data_loader, device, saved_model_path)

    dataset = data_loader.dataset.dataset
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(saved_model_path, map_location="cpu")

    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp

    return output, iter_eval


# generate results file
@torch.no_grad()
def generate_results(model, data_loader, device, saved_model_path):
    iters = len(data_loader)
    ann_labels = data_loader.dataset.dataset.ann_labels

    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (images, targets) in enumerate(data_loader):
        T = time.time()

        # image = image.to(device)
        # target = {k: v.to(device) for k, v in target.items()}

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        S = time.time()
        torch.cuda.synchronize()
        outputs = model(images)
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        output = outputs.pop()
        m_m.update(time.time() - S)

        targets = targets[0]
        prediction = {targets["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction, ann_labels))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break

    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000 * A / iters, 1000 * t_m.avg, 1000 * m_m.avg))

    S = time.time()
    print("all gather: {:.1f}s".format(time.time() - S))
    torch.save(coco_results, saved_model_path)

    return A / iters