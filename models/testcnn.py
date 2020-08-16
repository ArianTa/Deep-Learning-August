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

import torch.nn as nn
import torch.nn.functional as F


class TestCNN(nn.Module):
    def __init__(
        self, outputdim,
    ):
        super(TestCNN, self,).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                3, 1, kernel_size=(3, 3,), stride=1, padding=0, dilation=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                1, 3, kernel_size=(16, 16,), stride=1, padding=0, dilation=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 8, kernel_size=3,),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(
                8, 1, kernel_size=(3, 3,), stride=1, padding=0, dilation=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                1, 3, kernel_size=(16, 16,), stride=1, padding=0, dilation=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 8, kernel_size=3,),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(
                8, 1, kernel_size=(3, 3,), stride=1, padding=0, dilation=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                1, 3, kernel_size=(16, 16,), stride=1, padding=0, dilation=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 8, kernel_size=3,),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1800, 4096,),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096,),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2,),
        )

    def forward(
        self, x,
    ):
        x = self.features(x)
        x = x.view(x.size()[0], -1,)
        x = self.classifier(x)
        return x
