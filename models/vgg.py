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
cfg = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M", ],
    "vgg13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(
        self, features, output_dim,
    ):
        super(VGG, self,).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7,))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096,),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096,),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_dim,),
        )

    def forward(
        self, x,
    ):
        x = self.features(x)
        x = self.avgpool(x)
        h = torch.flatten(x, 1,)
        x = self.classifier(h)
        return (
            x,
            h,
        )


def make_vgg_layers(
    cfg, batch_norm=False,
):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2,)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1,)
            if batch_norm:
                layers += [
                    conv2d,
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
            else:
                layers += [
                    conv2d,
                    nn.ReLU(inplace=True),
                ]
            in_channels = v
    return nn.Sequential(*layers)


def get_vgg_model(
    model_name, output_dim,
):
    """ Helper function to get a model

    :param model_name: Name of the model
    :type model_name: str
    :param output_dim: Output dimension of the model
    :type output_dim: int

    :return model: The model
    :rtype: VGG
    """
    # Getting the model
    if "vgg11" in model_name:
        model = VGG(
            make_vgg_layers(cfg["vgg11"], batch_norm=True,),
            output_dim=output_dim,
        )
        pretrained_model = models.vgg11_bn(pretrained=True)
    if "vgg13" in model_name:
        model = VGG(
            make_vgg_layers(cfg["vgg13"], batch_norm=True,),
            output_dim=output_dim,
        )
        pretrained_model = models.vgg13_bn(pretrained=True)
    if "vgg16" in model_name:
        model = VGG(
            make_vgg_layers(cfg["vgg16"], batch_norm=True,),
            output_dim=output_dim,
        )
        pretrained_model = models.vgg16_bn(pretrained=True)
    if "vgg19" in model_name:
        model = VGG(
            make_vgg_layers(cfg["vgg19"], batch_norm=True,),
            output_dim=output_dim,
        )
        pretrained_model = models.vgg19_bn(pretrained=True)
    # model_pretrained = VGG(vgg11_layers, OUTPUT_DIM)
    IN_FEATURES = pretrained_model.classifier[-1].in_features

    final_fc = nn.Linear(IN_FEATURES, output_dim,)

    pretrained_model.classifier[-1] = final_fc

    model.load_state_dict(pretrained_model.state_dict())

    return model
