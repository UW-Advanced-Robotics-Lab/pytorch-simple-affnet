import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from PIL import Image

from pycocotools.coco import COCO

import torch
from torch.utils import data

import config
from dataset import dataset_utils


class COCODataSet(data.Dataset):

    def __init__(self,
                 dataset_dir,
                 split,
                 is_train=False,
                 is_eval=False,
                 ):

        # init dataset.
        self.dataset_dir = dataset_dir
        self.split = split
        self.is_train = is_train
        self.is_eval = is_eval
        self.transform = dataset_utils.get_transform()

        # init annotations.
        ann_file = os.path.join(self.dataset_dir, "annotations/instances_{}.json".format(split))
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]

        self._classes = {k: v["name"] for k, v in self.coco.cats.items()}
        self.classes = tuple(self.coco.cats[k]["name"] for k in sorted(self.coco.cats))
        # resutls' labels convert to annotation labels
        self.ann_labels = {self.classes.index(v): k for k, v in self._classes.items()}

        # remove files w/o annotations.
        checked_id_file = os.path.join(self.dataset_dir, "checked_{}.txt".format(split))
        if is_train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            self.check_dataset(checked_id_file)

    def __len__(self):
        return len(self.ids)

    def check_dataset(self, checked_id_file):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems listed in method `_check`.
        """

        if os.path.exists(checked_id_file):
            info = [line.strip().split(", ") for line in open(checked_id_file)]
            self.ids, self.aspect_ratios = zip(*info)
            return

        since = time.time()
        print("Checking the dataset...")

        executor = ThreadPoolExecutor(max_workers=config.NUM_WORKERS)
        seqs = torch.arange(len(self)).chunk(config.NUM_WORKERS)
        tasks = [executor.submit(self._check, seq.tolist()) for seq in seqs]

        outs = []
        for future in as_completed(tasks):
            outs.extend(future.result())
        if not hasattr(self, "id_compare_fn"):
            self.id_compare_fn = lambda x: int(x)
        outs.sort(key=lambda x: self.id_compare_fn(x[0]))

        with open(checked_id_file, "w") as f:
            for img_id, aspect_ratio in outs:
                f.write("{}, {:.4f}\n".format(img_id, aspect_ratio))

        info = [line.strip().split(", ") for line in open(checked_id_file)]
        self.ids, self.aspect_ratios = zip(*info)
        print("checked id file: {}".format(checked_id_file))
        print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - since))

    def _check(self, seq):
        out = []
        for i in seq:
            img_id = self.ids[i]
            target = self.get_target(img_id)
            boxes = target["boxes"]
            labels = target["labels"]
            masks = target["masks"]

            try:
                assert len(boxes) > 0, "{}: len(boxes) = 0".format(i)
                assert len(boxes) == len(labels), "{}: len(boxes) != len(labels)".format(i)
                assert len(boxes) == len(masks), "{}: len(boxes) != len(masks)".format(i)

                out.append((img_id, self._aspect_ratios[i]))
            except AssertionError as e:
                # print(img_id, e)
                pass
        return out

    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.dataset_dir, "{}".format(self.split), img_info["file_name"]))
        return image.convert("RGB")

    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []
        area = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                name = self._classes[ann["category_id"]]
                labels.append(self.classes.index(name))
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)

                area.append((ann['bbox'][3] - ann['bbox'][1]) * (ann['bbox'][2] - ann['bbox'][0]))
            # suppose all instances are not crowd
            # iscrowd.append(torch.zeros((len(anns),), dtype=torch.int64))

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
            area = torch.tensor(area)
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        # format target.
        # target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks,
        #              area=area, iscrowd=iscrowd)

        target = {}
        target["image_id"] = torch.tensor([img_id])
        # torch maskrcnn
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        # ids and bboxs and binary masks.
        target["obj_ids"] = torch.as_tensor(labels, dtype=torch.int64)
        target["obj_boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["obj_binary_masks"] = torch.as_tensor(masks, dtype=torch.uint8)

        return target

    @staticmethod
    def convert_to_xyxy(box):  # box format: (xmin, ymin, w, h)
        new_box = torch.zeros_like(box)
        new_box[:, 0] = box[:, 0]
        new_box[:, 1] = box[:, 1]
        new_box[:, 2] = box[:, 0] + box[:, 2]
        new_box[:, 3] = box[:, 1] + box[:, 3]
        return new_box  # new_box format: (xmin, ymin, xmax, ymax)

    def __getitem__(self, i):

        img_id = self.ids[i]
        image = self.get_image(img_id)
        target = self.get_target(img_id)

        # sent to transform.
        if self.is_train or self.is_eval:
            img, target = self.transform(image, target)
        else:
            img = np.array(image, dtype=np.uint8)

        return img, target