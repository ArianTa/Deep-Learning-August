import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple
import torch.optim as optim


def get_all_model(
    model_name, output_dim,
):
    """ Helper function
    """
    # Getting the model
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc

    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=True)
        IN_FEATURES = model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.fc = fc

    elif model_name == "resnext101_32x8d":
        model = models.resnext101_32x8d(pretrained=True)
        IN_FEATURES = model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.fc = fc

    elif model_name == "mnasnet0_5":
        model = models.mnasnet0_5(pretrained=True)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc

    elif model_name == "mnasnet1_0":
        model = models.mnasnet1_0(pretrained=True)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc

    else:
        model = models.mobilenet_v2(pretrained=True)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc

    return model
