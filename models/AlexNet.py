import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple
import torch.optim as optim

class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), #in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2), #kernel_size
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 192, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 384, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h



def get_alexnet_model(model_name, learning_rate, output_dim):
    """ Helper function
    """
    # Getting the model
    model = AlexNet(OUTPUT_DIM)
    pretrained_model = models.alexnet(pretrained=True)

    IN_FEATURES = pretrained_model.classifier[-1].in_features

    fc = nn.Linear(IN_FEATURES, output_dim)

    pretrained_model.classifier[-1] = fc

    model.load_state_dict(pretrained_model.state_dict())


    return model, model.parameters()
