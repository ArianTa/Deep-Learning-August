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

from collections import Counter
from numpy import log
import argparse
import os


def shannon_entropy(path):
    """ Computes the shannon entropy of a given dataset

    :param path: Path to the dataset
    :type path: str

    :rtype: None
    """

    classes = os.listdir(path)
    # Counting the number of classes
    n = 0
    for kls in classes:
        kls_path = os.path.join(path, kls)
        n += len(os.listdir(kls_path))

    k = len(os.listdir(path))
    H = 0
    for kls in classes:
        kls_path = os.path.join(path, kls)
        c = len(os.listdir(kls_path))
        p = c / n
        H += - p * log(p)

    return H / log(k)


if __name__ == "__main__":
    # Parsing arguments and setting up metadata
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--path",
        type=str,
        default="data/images",
        help="Path to root directory of the dataset",
    )
    # Add more stuff here maybe ?
    args = parser.parse_args()

    # Measure the balance of the dataset:
    print(f'shannon_entropy = {shannon_entropy(args.path)}')
