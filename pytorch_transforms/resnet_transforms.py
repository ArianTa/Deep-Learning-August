import torchvision.transforms as transforms


"""
means = [
    0.4459,
    0.4182,
    0.3441,
]
stds = [
    0.2210,
    0.2137,
    0.2109,
]
"""

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
