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
"""
This file contains the imagenet tansforms
"""

import torchvision.transforms as transforms

size = 224
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose(
    [
        transforms.Resize(size),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomCrop(size, padding=20,),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds,),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds,),
    ]
)
