import os
import pandas as pd
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
from PIL import Image

class MushroomDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Path to the root dir of the dataset.
            transform (callable, optional): Optional transform to appy.
        """
        self.root = root
        self.transform = transform
        self.superclasses = set()
        self.subclasses = set()
        self.classes = set()

        # Create the dictionary
        self.dico = dict()

        idx = 0
        dir_list = os.listdir(root)
        for direct in dir_list:
            class_path = os.path.join(root, direct)
            self.superclasses.add(direct.split("_")[0])
            self.subclasses.add(direct.split("_")[1])
            self.classes.add(direct)
            img_path_list = glob.glob(class_path + "/*")
            for img_path in img_path_list:
                self.dico[idx] = img_path
                idx += 1

    def __len__(self):
        """
        Returns:
            then number of images in the dataset
        """
        return len(self.dico)

    def __getitem__(self, idx):
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

        image = io.imread(self.dico[idx])
        image = Image.fromarray(image) 

        direct = self.dico[idx].split("/")[-2]

        sample = {"image": image,
                "superclass": direct.split("_")[0],
                "subclass": direct.split("_")[1],
                "class": direct}

        if(self.transform):
            sample["image"] = self.transform(sample["image"])

        return sample