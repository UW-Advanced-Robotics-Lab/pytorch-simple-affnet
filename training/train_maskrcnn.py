import os

import numpy as np
import random

import torch
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')

import config

from model.maskrcnn import maskrcnn
from dataset import dataset_loaders
from training import train_utils


def main():

    # Init random seeds.
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed(config.RANDOM_SEED)

    # Setup Tensorboard.
    print('\nsaving run in .. {}'.format(config.TRAINED_MODELS_DIR))
    if not os.path.exists(config.TRAINED_MODELS_DIR):
        os.makedirs(config.TRAINED_MODELS_DIR)
    writer = SummaryWriter(f'{config.TRAINED_MODELS_DIR}')

    # Load the Model.
    print()
    # Compare Pytorch-Simple-MaskRCNN. with Torchvision MaskRCNN.
    # model = train_utils.get_model_instance_segmentation(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    model = maskrcnn.ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    # send model to GPU and load backends (if all inputs are the same size)
    model.to(config.DEVICE)
    torch.backends.cudnn.benchmark = True

    # Load the dataset.
    train_loader, val_loader, test_loader = dataset_loaders.load_arl_affpose_train_datasets()

    # Construct an optimizer.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
                                momentum=config.MOMENTUM)

    # Init a learning rate scheduler.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Main training loop.
    num_epochs = config.NUM_EPOCHS
    best_Fwb = -np.inf

    for epoch in range(0 * num_epochs, 1 * num_epochs):
        print()
        # train & val for one epoch
        model, optimizer = train_utils.train_one_epoch(model, optimizer, train_loader, config.DEVICE, epoch, writer)
        model, optimizer = train_utils.val_one_epoch(model, optimizer, val_loader, config.DEVICE, epoch, writer)
        lr_scheduler.step()
        # eval model
        model = train_utils.eval_maskrcnn_arl_affpose(model, test_loader)
        best_Fwb = train_utils.eval_Fwb_arl_affpose(model=model, optimizer=optimizer,
                                                    best_Fwb=best_Fwb, epoch=epoch, writer=writer)
        # checkpoint_path
        CHECKPOINT_PATH = config.MODEL_SAVE_PATH + 'maskrcnn_epoch_' + np.str(epoch) + '.pth'
        train_utils.save_checkpoint(model, optimizer, epoch, CHECKPOINT_PATH)

if __name__ == "__main__":
    main()