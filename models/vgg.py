"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
"""VGG11/13/16/19 in Pytorch."""


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
