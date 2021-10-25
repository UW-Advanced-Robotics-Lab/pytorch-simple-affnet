import cv2

import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader, random_split, Subset

import config
from dataset.ycb_video import ycb_video_dataset
from dataset import dataset_utils


def load_ycb_video_train_datasets():

    # Train dataset.
    print("\nloading train ..")
    dataset = ycb_video_dataset.YCBVideoPoseDataset(
        image_path_txt_file=config.YCB_TRAIN_FILE,
        mean=config.YCB_IMAGE_MEAN,
        std=config.YCB_IMAGE_STD,
        resize=config.YCB_RESIZE,
        crop_size=config.YCB_CROP_SIZE,
        apply_imgaug=True,
        is_train=True,
        )

    # Selecting a subset of test images.
    train_split = 0.8
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(dataset), 1)
    train_idx = np.sort(np.random.choice(total_idx, size=int(len(dataset)*train_split), replace=False))
    val_idx = np.sort(np.delete(total_idx, train_idx))
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              collate_fn=dataset_utils.collate_fn
                              )
    print(f"train has {len(train_loader)} images ..")

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

    test_dataset = ycb_video_dataset.YCBVideoPoseDataset(
        image_path_txt_file=config.YCB_TEST_FILE,
        mean=config.YCB_IMAGE_MEAN,
        std=config.YCB_IMAGE_STD,
        resize=config.YCB_RESIZE,
        crop_size=config.YCB_CROP_SIZE,
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

    print(f"Selecting {len(test_loader)} test images and evaluating in {config.ARL_TEST_SAVE_FOLDER} ..")

    return train_loader, val_loader, test_loader

def load_ycb_video_eval_datasets(random_images=False, num_random=config.NUM_TEST, shuffle_images=False):

    # Test dataset.
    print("\nloading test ..")
    test_dataset = ycb_video_dataset.YCBVideoPoseDataset(
        image_path_txt_file=config.YCB_TEST_FILE,
        mean=config.YCB_IMAGE_MEAN,
        std=config.YCB_IMAGE_STD,
        resize=config.YCB_RESIZE,
        crop_size=config.YCB_CROP_SIZE,
        apply_imgaug=False,
        is_eval=True,
    )

    if random_images:
        # Selecting a subset of test images.
        np.random.seed(config.RANDOM_SEED)
        total_idx = np.arange(0, len(test_dataset), 1)
        test_idx = np.random.choice(total_idx, size=int(num_random), replace=False)
        test_dataset = Subset(test_dataset, test_idx)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=shuffle_images,
                                              num_workers=config.NUM_WORKERS,
                                              pin_memory=True,
                                              collate_fn=dataset_utils.collate_fn
                                              )

    print(f"test has {len(test_loader)} images ..")

    return test_loader