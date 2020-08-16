"""
INFO8010: Mushroom classification
Authors: Folon Nora, Horbach Amadis, Tahiraj Arian

Parts of the code are inspired from:
    - Title: PyTorch Image Classification
      Authors: Ben Trevett
      Availability: https://github.com/bentrevett/pytorch-image-classification

    - Title: Pytorch-cifar100
      Authors: weiaicunzai
      Availability: https://github.com/weiaicunzai/pytorch-cifar100

    - Title: Bag of Tricks for Image Classification with Convolutional Neural Networks
      Authors: weiaicunzai
      Availability: https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks
"""

import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple
import torch.optim as optim

ResNetConfig = namedtuple("ResNetConfig", ["block", "n_blocks", "channels", ],)


class ResNet(nn.Module):
    def __init__(
        self, config, output_dim,
    ):
        super().__init__()

        (block, n_blocks, channels,) = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(
            3,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0],)
        self.layer2 = self.get_resnet_layer(
            block, n_blocks[1], channels[1], stride=2,
        )
        self.layer3 = self.get_resnet_layer(
            block, n_blocks[2], channels[2], stride=2,
        )
        self.layer4 = self.get_resnet_layer(
            block, n_blocks[3], channels[3], stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1,))
        self.fc = nn.Linear(self.in_channels, output_dim,)

    def get_resnet_layer(
        self, block, n_blocks, channels, stride=1,
    ):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample,))

        for i in range(1, n_blocks,):
            layers.append(block(block.expansion * channels, channels,))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(
        self, x,
    ):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = torch.flatten(x, 1,)
        x = self.fc(h)

        return (
            x,
            h,
        )


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=False,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            self.expansion * out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(
                in_channels,
                self.expansion * out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn,)
        else:
            downsample = None

        self.downsample = downsample

    def forward(
        self, x,
    ):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=False,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn,)
        else:
            downsample = None

        self.downsample = downsample

    def forward(
        self, x,
    ):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


def get_resnet_model(
    model_name, output_dim
):
    """ Helper function to get a model

    :param model_name: Name of the model
    :type model_name: str
    :param output_dim: Output dimension of the model
    :type output_dim: int

    :return model: The model
    :rtype: ResNet
    """
    # Getting the model
    if model_name == "resnet18":
        resnet_config = ResNetConfig(
            block=BasicBlock,
            n_blocks=[2, 2, 2, 2, ],
            channels=[64, 128, 256, 512, ],
        )
        pretrained_model = models.resnet18(pretrained=True)
    elif model_name == "resnet34":
        resnet_config = ResNetConfig(
            block=BasicBlock,
            n_blocks=[3, 4, 6, 3, ],
            channels=[64, 128, 256, 512, ],
        )
        pretrained_model = models.resnet34(pretrained=True)
    elif model_name == "resnet50":
        resnet_config = ResNetConfig(
            block=Bottleneck,
            n_blocks=[3, 4, 6, 3, ],
            channels=[64, 128, 256, 512, ],
        )
        pretrained_model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        resnet_config = ResNetConfig(
            block=Bottleneck,
            n_blocks=[3, 4, 23, 3, ],
            channels=[64, 128, 256, 512, ],
        )
        pretrained_model = models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        resnet_config = ResNetConfig(
            block=Bottleneck,
            n_blocks=[3, 8, 36, 3, ],
            channels=[64, 128, 256, 512, ],
        )
        pretrained_model = models.resnet152(pretrained=True)
    else:
        resnet_config = ResNetConfig(
            block=Bottleneck,
            n_blocks=[3, 8, 36, 3, ],
            channels=[64, 128, 256, 512, ],
        )
        pretrained_model = models.resnet152(pretrained=True)

    IN_FEATURES = pretrained_model.fc.in_features

    fc = nn.Linear(IN_FEATURES, output_dim,)

    pretrained_model.fc = fc

    model = ResNet(resnet_config, output_dim,)
    model.load_state_dict(pretrained_model.state_dict())

    return model
