
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models

import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
import time

import skimage

from pydoc import locate
from torch.utils.data import (
    Dataset,
    DataLoader,
)

import utils


def get_predictions(model, iterator):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument(
        "--source_path", default="data/images", help="Path to the dataset",
    )
    parser.add_argument(
        "--train_path",
        default="data/train",
        help="Path to the train set to be created",
    )
    parser.add_argument(
        "--test_path",
        default="data/test",
        help="Path to the test set to be created",
    )
    parser.add_argument(
        "--train_ratio",
        default=0.8,
        help="Train ratio - (dataset size / train set size)",
    )

    args = parser.parse_args()

	images, labels, probs = get_predictions(model, test_iterator)