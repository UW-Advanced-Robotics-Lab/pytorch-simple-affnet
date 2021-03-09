import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torchvision.ops import misc

###########################
###########################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

#########################
#########################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = misc.FrozenBatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,dilation=rate, padding=rate, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = misc.FrozenBatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        self.bn3 = misc.FrozenBatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#########################
#########################

class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #########################
        # RGB
        #########################

        self.inplanes = 64

        # ResNet Block 0
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = misc.FrozenBatchNorm2d(64)

        # ResNet Block 1
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        # ResNet Block 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        # ResNet Block 3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        # ResNet Block 4
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        #########################
        #########################

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()
        else:
            print("training from scratch .. ")

    #########################
    #########################

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
                misc.FrozenBatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
                misc.FrozenBatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    #########################
    #########################

    def forward(self, rgb):
        # ResNet Block 0
        x_resnet0 = self.conv1(rgb)
        x_resnet0 = self.bn1(x_resnet0)
        x_resnet0 = self.relu(x_resnet0)
        x_resnet0 = self.maxpool(x_resnet0)

        # ResNet Block 1
        x_resnet1 = self.layer1(x_resnet0)
        low_level_feat = x_resnet1
        # ResNet Block 2
        x_resnet2 = self.layer2(x_resnet1)
        # ResNet Block 3
        x_resnet3 = self.layer3(x_resnet2)
        # ResNet Block 4
        x_resnet4 = self.layer4(x_resnet3)

        return x_resnet4

    #########################
    #########################

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, nn.BatchNorm2d):
            elif isinstance(m, misc.FrozenBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        print("loading pre-trained ResNet101 weights .. {}".format(config.RESNET_PRETRAINED_WEIGHTS))
        pretrain_dict = model_zoo.load_url(config.RESNET_PRETRAINED_WEIGHTS)
        model_dict = {}
        state_dict = self.state_dict()
        #######################
        # D OR RBG+D INPUT
        #######################
        if config.NUM_CHANNELS != 3:
            print("not rgb input, pruning saved weights ..")
            pruned_pretrain_dict = {}
            for i in pretrain_dict:
                i_parts = i.split('.')
                if i_parts[0] != 'conv1' and i_parts[0] != 'bn1':
                    pruned_pretrain_dict[i] = pretrain_dict[i]
            pretrain_dict = pruned_pretrain_dict
        #######################
        #######################
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

#########################
#########################

def ResNet101(nInputChannels=config.NUM_CHANNELS, os=config.OUTPUT_STRIDE, pretrained=config.IS_PRETRAINED):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model

#########################
#########################

if __name__ == "__main__":
    model = ResNet101(nInputChannels=config.NUM_CHANNELS, os=config.OUTPUT_STRIDE, pretrained=config.IS_PRETRAINED)
    model.to(device=config.DEVICE)
    model.eval()

    ### print(model.optim_parameters(config.LEARNING_RATE))

    # from torchsummary import summary
    # TORCH_SUMMARY = (config.NUM_CHANNELS, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    # summary(model, TORCH_SUMMARY)

    image = torch.randn(config.BATCH_SIZE, config.NUM_CHANNELS, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    print("\nImage: ", image.size())
    with torch.no_grad():
        pred_target_main = model.forward(image.to(device=config.DEVICE))
    print("pred_target_main: {}".format(pred_target_main.size()))