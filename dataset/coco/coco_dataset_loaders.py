import cv2

import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader, random_split, Subset

import config
from dataset.coco import coco_dataset
from dataset import dataset_utils


def load_coco_train_datasets():

    # Train dataset.
    print("\nloading train ..")
    # Load COCO dataset.
    train_dataset = coco_dataset.COCODataSet(
        dataset_dir=config.COCO_ROOT_DATA_PATH,
        split=config.COCO_TRAIN_SPLIT,
        is_train=True,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              collate_fn=dataset_utils.collate_fn
                              )

    print(f"train has {len(train_loader)} images ..")

    # Val dataset.
    print("\nloading val ..")
    val_dataset = coco_dataset.COCODataSet(
        dataset_dir=config.COCO_ROOT_DATA_PATH,
        split=config.COCO_VAL_SPLIT,
        is_train=True,
    )

    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=True,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=True,
                            collate_fn=dataset_utils.collate_fn
                            )

    print(f"val has {len(val_dataset)} images ..")

    return train_loader, val_loader

def load_coco_eval_datasets():
    # Test dataset.
    print("\nloading test ..")
    test_dataset = coco_dataset.COCODataSet(
        dataset_dir=config.COCO_ROOT_DATA_PATH,
        split=config.COCO_VAL_SPLIT,
        is_train=True,
    )

    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(test_dataset), 1)
    test_idx = np.random.choice(total_idx, size=int(100), replace=False)
    test_dataset = torch.utils.data.Subset(test_dataset, test_idx)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=config.NUM_WORKERS,
                                              pin_memory=True,
                                              collate_fn=dataset_utils.collate_fn
                                              )

    print(f"test has {len(test_loader)} images ..")

    return test_loader