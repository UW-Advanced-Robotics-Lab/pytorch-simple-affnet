import cv2

import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader, random_split, Subset

import config
from dataset.umd import umd_dataset
from dataset import dataset_utils


def load_umd_train_datasets():
    # Train dataset.
    print("\nloading train ..")
    train_dataset = umd_dataset.UMDDataset(
        dataset_dir=config.UMD_DATA_DIRECTORY_TRAIN,
        mean=config.UMD_IMAGE_MEAN,
        std=config.UMD_IMAGE_STD,
        resize=config.UMD_RESIZE,
        crop_size=config.UMD_CROP_SIZE,
        apply_imgaug=True,
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
    val_dataset = umd_dataset.UMDDataset(
        dataset_dir=config.UMD_DATA_DIRECTORY_VAL,
        mean=config.UMD_IMAGE_MEAN,
        std=config.UMD_IMAGE_STD,
        resize=config.UMD_RESIZE,
        crop_size=config.UMD_CROP_SIZE,
        apply_imgaug=False,
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

    # Test dataset.
    print("\nloading test ..")

    test_dataset = umd_dataset.UMDDataset(
        dataset_dir=config.UMD_DATA_DIRECTORY_TEST,
        mean=config.UMD_IMAGE_MEAN,
        std=config.UMD_IMAGE_STD,
        resize=config.UMD_RESIZE,
        crop_size=config.UMD_CROP_SIZE,
        apply_imgaug=False,
        is_train=True,
    )

    # Selecting a subset of test images.
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(test_dataset), 1)
    test_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
    test_dataset = Subset(test_dataset, test_idx)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=config.NUM_WORKERS,
                                              pin_memory=True,
                                              collate_fn=dataset_utils.collate_fn
                                              )

    print(f"Selecting {len(test_loader)} test images and evaluating in {config.UMD_TEST_SAVE_FOLDER} ..")

    return train_loader, val_loader, test_loader

def load_umd_eval_datasets(random_images=True):
    # Test dataset.
    print("\nloading test ..")
    test_dataset = umd_dataset.UMDDataset(
        dataset_dir=config.UMD_DATA_DIRECTORY_TEST,
        mean=config.UMD_IMAGE_MEAN,
        std=config.UMD_IMAGE_STD,
        resize=config.UMD_RESIZE,
        crop_size=config.UMD_CROP_SIZE,
        apply_imgaug=False,
        is_eval=True,
    )

    if random_images:
        # Selecting a subset of test images.
        np.random.seed(config.RANDOM_SEED)
        total_idx = np.arange(0, len(test_dataset), 1)
        test_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
        test_dataset = Subset(test_dataset, test_idx)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=config.NUM_WORKERS,
                                              pin_memory=True,
                                              collate_fn=dataset_utils.collate_fn
                                              )

    print(f"test has {len(test_loader)} images ..")

    return test_loader