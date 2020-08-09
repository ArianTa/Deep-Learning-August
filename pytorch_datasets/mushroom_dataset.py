import os
import pandas as pd
from skimage import (
    io,
    transform,
)
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from torchvision import (
    transforms,
    utils,
)
import glob
from PIL import Image
import json


class MushroomDataset(Dataset):
    def __init__(
        self, root, annotation, transform=None,
    ):
        """
        Args:
            root (string): Path to the root dir of the dataset.
            annotation (string): Path to json annotation file
            transform (callable, optional): Optional transform to appy.
        """
        self.transform = transform
        self.root = root
        with open(annotation) as json_file:
            self.annotation = json.load(json_file)

    def __len__(self,):
        """
        Returns:
            then number of images in the dataset
        """
        return len(self.annotation["annotations"])

    def get_category(idx):
        """
        """
        return self.annotation['categories'][idx]['name']

    def __getitem__(
        self, idx,
    ):
        """
        Args:
            idx (integer): Index of the queried image
        Returns:
            dictionary containing 3 entries:
                -"image": the image
                -"superclass": its superclass
                -"subclass": its subclass
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        category = self.annotation['annotations'][idx]['category_id']

        path = self.annotation['images'][idx]['file_name'].replace(
            " ", "_").replace(
            ".", "_").replace(
            ":", "_")
        path = list(path)
        path[-4] = "."
        path = "".join(path)

        image = os.path.join(self.root, path)  # KEK
        image = Image.open(image)
        image = self.transform(image)

        return image, category
