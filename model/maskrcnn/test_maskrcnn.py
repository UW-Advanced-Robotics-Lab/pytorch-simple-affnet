import unittest

import torch

import sys
sys.path.append('../../')

import config
from model.maskrcnn import maskrcnn
from training import train_utils

from dataset.arl_affpose import arl_affpose_dataset_loaders
from dataset.ycb_video import ycb_video_dataset_loaders


class MaskRCNNTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load model.
        self.model = maskrcnn.ResNetMaskRCNN(pretrained=config.IS_PRETRAINED, num_classes=config.NUM_CLASSES)
        self.model.to(config.DEVICE)

        # Load the dataset.
        train_loader, val_loader, test_loader = ycb_video_dataset_loaders.load_ycb_video_train_datasets()
        # create dataloader.
        self.data_loader = train_loader

    def test_freeze_backbone(self):
        # freeze backbone layers
        for name, parameter in self.model.named_parameters():
            if 'backbone' in name:
                print(f'Frozen: {name}')
                parameter.requires_grad_(False)
            else:
                print(f'Requires Grad: {name}')
                parameter.requires_grad_(True)

    def test_random_input(self):

        # we can load values more easily for PyCharm debugging with random inputs.
        image = torch.randn(1, 3, 128, 128)

        img_id = 1
        labels = torch.tensor([1])
        bbox = torch.tensor([[99, 17, 113, 114]], dtype=torch.float32)
        mask = torch.randn(1, 128, 128)

        target = {}
        target["image_id"] = torch.tensor([img_id])
        target["obj_boxes"] = bbox
        target["obj_ids"] = labels
        target["obj_binary_masks"] = mask

        image = image.to(config.DEVICE)
        target = {k: v.to(config.DEVICE) for k, v in target.items()}

        with torch.no_grad():
            self.model.eval()
            outputs = self.model(image, target)

        outputs = [{k: v.to(config.CPU_DEVICE) for k, v in t.items()} for t in outputs]
        outputs = outputs.pop()

    def test_maskrcnn_train(self):
        self.model.train()
        # get one item from dataloader.
        # batch = iter(self.data_loader).__next__()

        for idx, batch in enumerate(self.data_loader):
            print(f'{idx}/{len(self.data_loader)} ..')

            images, targets = batch
            images = list(image.to(config.DEVICE) for image in images)
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

        # losses = sum(loss for loss in loss_dict.values())
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = train_utils.reduce_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # # getting summed loss.
        # loss_value = losses_reduced.item()
        #
        # print(f'\nloss_value: {loss_value}')
        # print(f'\nloss_objectness: {loss_dict_reduced["loss_objectness"]}')
        # print(f'loss_rpn_box_reg: {loss_dict_reduced["loss_rpn_box_reg"]}')
        # print(f'loss_classifier: {loss_dict_reduced["loss_classifier"]}')
        # print(f'loss_box_reg: {loss_dict_reduced["loss_box_reg"]}')
        # print(f'loss_mask: {loss_dict_reduced["loss_mask"]}')

    def test_maskrcnn_eval(self):
        # get one item from dataloader.
        batch = iter(self.data_loader).__next__()
        images, targets = batch
        images = list(image.to(config.DEVICE) for image in images)

        with torch.no_grad():
            self.model.eval()
            outputs = self.model(images)

        outputs = [{k: v.to(config.CPU_DEVICE) for k, v in t.items()} for t in outputs]
        outputs = outputs.pop()
        print(f'\noutputs:{outputs.keys()}')

if __name__ == '__main__':
    # unittest.main()

    # run desired test.
    suite = unittest.TestSuite()
    suite.addTest(MaskRCNNTest("test_maskrcnn_train"))
    runner = unittest.TextTestRunner()
    runner.run(suite)


