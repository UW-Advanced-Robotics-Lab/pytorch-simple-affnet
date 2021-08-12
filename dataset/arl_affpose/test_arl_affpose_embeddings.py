import os

import unittest

import numpy as np
import scipy.io as scio

import cv2
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import torch
from torch.utils import data
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')

import config

from training import train_utils
from dataset.arl_affpose import arl_affpose_dataset
from dataset.arl_affpose import arl_affpose_dataset_utils
from dataset.arl_affpose import load_arl_affpose_obj_ply_files

_NUM_IMAGES = 10

class DatasetStatisticsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(DatasetStatisticsTest, self).__init__(*args, **kwargs)

        # Load ARL AffPose dataset.
        dataset = arl_affpose_dataset.ARLAffPoseDataset(
            dataset_dir=config.DATA_DIRECTORY_TEST,
            mean=config.IMAGE_MEAN,
            std=config.IMAGE_STD,
            resize=config.RESIZE,
            crop_size=config.CROP_SIZE,
            apply_imgaug=False,
        )

        # creating subset.
        np.random.seed(0)
        total_idx = np.arange(0, len(dataset), 1)
        test_idx = np.random.choice(total_idx, size=int(_NUM_IMAGES), replace=True)
        dataset = torch.utils.data.Subset(dataset, test_idx)

        # create dataloader.
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        print(f'Selecting {len(self.data_loader)} images ..')

    def test_embeddings(self):

        # setup model.
        model = models.resnet50(pretrained=True)
        model.eval()

        # loop over dataset.
        for i, (images, target) in enumerate(self.data_loader):
            print(f'\n{i}/{len(self.data_loader)} ..')

            # format rgb.
            image = np.squeeze(np.array(images))
            image, target = arl_affpose_dataset_utils.format_target_data(image, target)

            # get class label.
            obj_mask = target['obj_mask']
            labels = np.unique(obj_mask)[1:]

            with torch.no_grad():
                images = images.view(config.BATCH_SIZE, image.shape[2], image.shape[0], image.shape[1]).type(torch.float32)
                outputs = model(images)

            current_outputs = outputs.cpu().numpy()
            features = np.concatenate((outputs, current_outputs))

            tsne = TSNE(n_components=2).fit_transform(features)

            # TODO: plot embeddings.

if __name__ == '__main__':
    # unittest.main()

    # run desired test.
    suite = unittest.TestSuite()
    suite.addTest(DatasetStatisticsTest("test_occlusion"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

