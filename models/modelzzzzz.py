import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple
import torch.optim as optim


def get_all_model(
    model_name, learning_rate, output_dim,
):
    """ Helper function
    """
    # Getting the model
    if model_name == "inception_v3":
        model = models.inception_v3(pretrained=False)
        IN_FEATURES = model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.fc = fc
        
        pretrained_model = models.inception_v3(pretrained=True)
        IN_FEATURES = pretrained_model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        pretrained_model.fc = fc
        
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc
        
        pretrained_model = models.mobilenet_v2(pretrained=True)
        IN_FEATURES = pretrained_model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        pretrained_model.classifier[-1] = fc
        
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=False)
        IN_FEATURES = model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.fc = fc
        
        pretrained_model = models.resnext50_32x4d(pretrained=True)
        IN_FEATURES = pretrained_model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        pretrained_model.fc = fc
        
    elif model_name == "resnext101_32x8d":
        model = models.resnext101_32x8d(pretrained=False)
        IN_FEATURES = model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.fc = fc
        
        pretrained_model = models.resnext101_32x8d(pretrained=True)
        IN_FEATURES = pretrained_model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        pretrained_model.fc = fc
        
        
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=False)
        IN_FEATURES = model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.fc = fc
        
        pretrained_model = models.wide_resnet50_2(pretrained=True)
        IN_FEATURES = pretrained_model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        pretrained_model.fc = fc
        
    elif model_name == "wide_resnet101_2":
        model = models.wide_resnet101_2(pretrained=False)
        IN_FEATURES = model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.fc = fc
        
        pretrained_model = models.wide_resnet101_2(pretrained=True)
        IN_FEATURES = pretrained_model.fc.in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        pretrained_model.fc = fc
        
    elif model_name == "mnasnet0_5":
        model = models.mnasnet0_5(pretrained=False)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc
        
        pretrained_model = models.mnasnet0_5(pretrained=True)
        IN_FEATURES = pretrained_model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        pretrained_model.classifier[-1] = fc
        
    elif model_name == "mnasnet0_75":
        model = models.mnasnet0_75(pretrained=False)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc
        
        pretrained_model = models.mnasnet0_75(pretrained=True)
        IN_FEATURES = pretrained_model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        pretrained_model.classifier[-1] = fc
        
    elif model_name == "mnasnet1_0":
        model = models.mnasnet1_0(pretrained=False)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc
        
        pretrained_model = models.mnasnet1_0(pretrained=True)
        IN_FEATURES = pretrained_model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        pretrained_model.classifier[-1] = fc
    
    elif model_name == "mnasnet1_3":
        model = models.mnasnet1_3(pretrained=False)
        IN_FEATURES = model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        model.classifier[-1] = fc
        
        pretrained_model = models.mnasnet1_3(pretrained=True)
        IN_FEATURES = pretrained_model.classifier[-1].in_features
        fc = nn.Linear(IN_FEATURES, output_dim,)
        pretrained_model.classifier[-1] = fc
    
    else:
        resnet_config = ResNetConfig(
            block=Bottleneck,
            n_blocks=[3, 8, 36, 3, ],
            channels=[64, 128, 256, 512, ],
        )
        pretrained_model = models.resnet152(pretrained=True)


    model.load_state_dict(pretrained_model.state_dict())

    return (
        model,
        model.parameters(),
    )
