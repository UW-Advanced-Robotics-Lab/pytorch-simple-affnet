from torch import nn
from torchvision import models
from torchvision.ops import misc
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models._utils import IntermediateLayerGetter

import config

def resnet_backbone(backbone_name,pretrained):
    return ResNetBackbone(backbone_name, pretrained)

class ResNetBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()

        # # TODO: load VGG16 backbone.
        # # backbone = models.vgg16(pretrained=True)
        # print(backbone)
        # backbone = list(backbone.children())[:-2]
        # backbone = nn.Sequential(*backbone)
        # print(backbone)

        if pretrained:
            print(f'\n{config.BACKBONE_FEAT_EXTRACTOR} backbone using pretrained weights ..')
        # TODO: FrozenBatchNorm2d to help with Batch Size = 1.
        body = models.resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048  # Resnet50:2048 vs Resnet18:512
        self.out_channels = 256

        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x


def resnet_fpn_backbone(
        backbone_name,
        pretrained,
        trainable_layers=5,
        returned_layers=None,
        extra_blocks=None
):

    backbone = models.resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


class BackboneWithFPN(nn.Module):

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
