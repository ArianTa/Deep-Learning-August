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

"""
    Since we removed all the models but MobileNet, we should rework this function
"""
def get_all_model(
    model_name, output_dim,
):
    """ Helper function to get a model

    :param model_name: Name of the model
    :type model_name: str
    :param output_dim: Output dimension of the model
    :type output_dim: int

    :return model: The model
    :rtype: nn.Module
    """

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc

    else:
        model = models.mobilenet_v2(pretrained=True)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc

    return model
