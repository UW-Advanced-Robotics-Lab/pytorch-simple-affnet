import os

import numpy as np
import random

import torch
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')

import config

from model.maskrcnn import maskrcnn
from model import model_utils
from training import train_utils
from eval import eval_utils

from dataset.arl_affpose import arl_affpose_dataset_loaders


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
    model = maskrcnn.ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    # Load the dataset.
    train_loader, val_loader, test_loader = arl_affpose_dataset_loaders.load_arl_affpose_train_datasets()

    # Main training loop.
    num_epochs = config.NUM_EPOCHS
    best_Fwb, best_mAP = -np.inf, -np.inf

    for epoch in range(0, num_epochs):
        print()

        if epoch < config.NUM_EPOCHS_HEADS:
            print(f'Epoch {epoch+1}: Freezing Network Heads ..')
            model = model_utils.freeze_heads(model)
            # Construct an optimizer.
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, momentum=config.MOMENTUM)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.MILESTONES, gamma=config.GAMMA)
        else:
            print(f'Epoch {epoch+1}: Training all layers ..')
            model = model_utils.unfreeze_all_layers(model)
            # Construct an optimizer.
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, momentum=config.MOMENTUM)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.MILESTONES, gamma=config.GAMMA)

        if epoch < config.EPOCH_TO_TRAIN_FULL_DATASET:
            is_subsample = True
        else:
            is_subsample = False

        # train & val for one epoch
        model, optimizer = train_utils.train_one_epoch(model, optimizer, train_loader, config.DEVICE, epoch, writer, is_subsample=is_subsample)
        model, optimizer = train_utils.val_one_epoch(model, optimizer, val_loader, config.DEVICE, epoch, writer, is_subsample=True)
        # update learning rate.
        lr_scheduler.step()

        if epoch >= config.EPOCH_TO_TRAIN_FULL_DATASET:
            # eval mAP
            model, mAP = eval_utils.eval_maskrcnn_arl_affpose(model, test_loader)
            writer.add_scalar('eval/mAP', mAP, int(epoch))
            print(f'mAP: {mAP:.5f}')
            # eval FwB
            # Fwb = eval_utils.eval_fwb_arl_affpose_maskrcnn()
            # writer.add_scalar('eval/Fwb', Fwb, int(epoch))
            # save best model.
            # if Fwb > best_Fwb:
            #     best_Fwb = Fwb
            #     writer.add_scalar('eval/Best_Fwb', best_Fwb, int(epoch))
            if mAP > best_mAP:
                best_mAP = mAP
                writer.add_scalar('eval/Best_mAP', best_mAP, int(epoch))
                checkpoint_path = config.BEST_MODEL_SAVE_PATH
                train_utils.save_checkpoint(model, optimizer, epoch, checkpoint_path)
                print("Saving best model .. best mAP={:.5f} ..".format(best_mAP))

        # checkpoint_path
        checkpoint_path = config.MODEL_SAVE_PATH + 'maskrcnn_epoch_' + np.str(epoch) + '.pth'
        train_utils.save_checkpoint(model, optimizer, epoch, checkpoint_path)

if __name__ == "__main__":
    main()