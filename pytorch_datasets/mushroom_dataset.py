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
        """ Contrstructor for the MushroomDataset class

        :param root: Path to the root dir of the dataset.
        :type root: str
        :param annotation: Path to json annotation file
        :type annotation: str
        :param transform: Transform to apply
        :type transform: Callable

        :return model: The mushroom dataset
        :rtype: MushroomDataset
        """

        self.transform = transform
        self.root = root
        with open(annotation) as json_file:
            self.annotation = json.load(json_file)

        categories = self.annotation["categories"]
        self.classes = [None] * len(categories)
        self.superclasses = [None] * len(categories)
        self.subclasses = [None] * len(categories)

        for i in range(len(categories)):
            klass_id = categories[i]['id']
            klass = categories[i]['name']
            superclass = categories[i]['supercategory']
            subclass = klass.split()[1]

            self.classes[klass_id] = klass
            self.superclasses[klass_id] = superclass
            self.subclasses[klass_id] = subclass


    def __len__(self,):
        """ Returns the length of the dataset

        :rtype: int
        """
        return len(self.annotation["annotations"])

    def get_category(idx):
        """ Returns the category of a object given given its index
        
        :param idx: Index of the object in the dataset
        :type idx: int

        :rtype: str
        """
        return self.annotation['categories'][idx]['name']

    def __getitem__(
        self, idx,
    ):
        """ Contrstructor for the MushroomDataset class

        :param idx: Index of the queried image
        :type idx: 

        :return image: The transformed image
        :rtype image: torch.Tensor
        :return category: The category index of the image
        :rtype category: int
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
