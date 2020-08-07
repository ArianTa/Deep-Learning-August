import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.datasets as datasets
import torchvision.models as models

import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import argparse
import os
import shutil
import time
import random


import copy

from collections import namedtuple

import skimage

from pydoc import locate
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from utils import measure_time

from find_lr import *

from models import *

import utils

def train(
    model, iterator, optimizer, criterion, scheduler, device
):

    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.train()

    if DEBUG:
        i = 0
        total = len(iterator)

    for (x, y,) in iterator:
        if DEBUG:
            i += 1
            print(f"{i}th batch: total is {total}")

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        y_pred = model(x)

        if isinstance(y_pred, tuple):
            y_pred, _ = y_pred

        loss = criterion(y_pred, y,)

        if DEBUG:
            print(f"{i}th batch loss: {loss.item()}")

        (acc_1, acc_5,) = utils.calculate_topk_accuracy(y_pred, y,)

        if DEBUG:
            print(f"{i}th batch train accuracies: top1: {acc_1.item()*100:6.2f} | top5: {acc_5.item()*100:6.2f}")

        loss.backward()

        optimizer.step()

        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    return (
        epoch_loss,
        epoch_acc_1,
        epoch_acc_5,
    )


def evaluate(
    model, iterator, criterion, device,
):

    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.eval()

    with torch.no_grad():

        for (x, y,) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            if isinstance(y_pred, tuple):
                y_pred, _ = y_pred

            loss = criterion(y_pred, y,)

            (acc_1, acc_5,) = utils.calculate_topk_accuracy(y_pred, y,)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    return (
        epoch_loss,
        epoch_acc_1,
        epoch_acc_5,
    )


def epoch_time(
    start_time, end_time,
):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return (
        elapsed_mins,
        elapsed_secs,
    )


def find_learning_rate(
    star_lr, end_lr, iterator, optimizer, nb_iter = 100
):
    lr_finder = LRFinder(model, optimizer, criterion, device,)
    (lrs, losses,) = lr_finder.range_test(iterator, end_lr,)
    plot_lr_finder(
        lrs, losses, skip_start=30, skip_end=30,
    )


def main(**kwargs,):
    # can't modify locals()
    globals().update(kwargs)

    model.to(device)
    criterion.to(device)
    if DEBUG:
        print(
            f"The model has {count_parameters(model):,} trainable parameters"
        )

    # Creating dataset, splitting train/validation set and creating dataloaders
    train_data = dataset(root=data_dir, transform=train_transforms,)
    classes = [utils.format_label(c) for c in train_data.classes]

    """
    #TO GET THE NUMBER OF CLASSES ETC
    unique_classes = set(classes)
    number_of_unique_values = len(unique_classes)
    max_element = max(classes,key=classes.count)
    nb_max = classes.count(max_element)
    print('number of superclasses: ', number_of_unique_values)
    print('max occurence of subclasses: ', nb_max)
    exit(0)
    """


    n_train_examples = int(len(train_data) * valid_ratio)
    n_valid_examples = len(train_data) - n_train_examples

    (train_data, valid_data,) = data.random_split(
        train_data, [n_train_examples, n_valid_examples, ],
    )

    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms

    train_iterator = dataloader(
        train_data, shuffle=True, batch_size=batch_size,
    )

    valid_iterator = dataloader(valid_data, batch_size=batch_size,)

    # Find learning rate instead of training
    if find_lr:
        find_learning_rate(
            star_lr=1e-7, end_lr=end_lr, iterator=train_iterator, optimizer = optimizer
        )
        exit(0)

    # We can start the training of the model
    TOTAL_STEPS = epochs * len(train_iterator)

    MAX_LRS = [p["lr"] for p in optimizer.param_groups]

    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LRS, total_steps=TOTAL_STEPS,
    )

    best_valid_loss = float("inf")
    for epoch in range(epochs):

        start_time = time.time()

        (train_loss, train_acc_1, train_acc_5,) = train(
            model, train_iterator, optimizer, criterion, scheduler, device,
        )
        (valid_loss, valid_acc_1, valid_acc_5,) = evaluate(
            model, valid_iterator, criterion, device,
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                model.state_dict(), file_name,
            )

        end_time = time.time()

        (epoch_mins, epoch_secs,) = epoch_time(start_time, end_time,)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | "
            f"Train Acc @5: {train_acc_5*100:6.2f}%")
        print(
            f"\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | "
            f"Valid Acc @5: {valid_acc_5*100:6.2f}%")


if __name__ == "__main__":
    # Parsing arguments and setting up metadata
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--path",
        type=str,
        default="data/train",
        help="Path to root directory of the dataset",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use gpu",
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="Batch size",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs",
    )
    parser.add_argument(
        "--model", type=str, default="resnet152", help="CNN model to be used",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="model_weights.pt",
        help="Name of the saved weights",
    )
    parser.add_argument(
        "--lr", type=float, default=10e-3, help="Sarting learning rate",
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Which optimizer to use",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="CrossEntropyLoss",
        help="Which criterion to use",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.9,
        help="Ratio between the train set and validation set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Specify a seed, for reproducability",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        default="resnet_transforms",
        help="The pytorch transforms to be used",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print debug info",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Specify a path for existing weights",
    )
    parser.add_argument(
        "--find_lr",
        action="store_true",
        help="Find the starting learning rate, if set, --lr becomes the lowest learning rate considered",
    )
    parser.add_argument(
        "--end_lr",
        type=float,
        default=10,
        help="The highest learning rate considered if --flind_lr is set"
    )

    # Add more stuff here maybe ?
    args = parser.parse_args()

    globals()["DEBUG"] = True if args.debug else False

    # For reproducability
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # criterion
    if args.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    # else blabla

    output_dim = len(os.listdir(args.path))
    # Get the model and is parameters that are to be optimized
    if "resnet" in args.model:
        (model, params,) = get_resnet_model(args.model, args.lr, output_dim,)
    elif "vgg" in args.model:
        (model, params,) = get_vgg_model(args.model, args.lr, output_dim,)
    elif "googlenet" in args.model:
        (model, params,) = get_googlenet_model(args.lr, output_dim,)
    elif "densenet" in args.model:
        (model, params,) = get_densenet_model(args.model, args.lr, output_dim,)
    elif "shufflenet" in args.model:
        (model, params,) = get_shufflenet_model(
            args.model, args.lr, output_dim,
        )
    elif "alexnet" in args.model:
        (model, params) = get_alexnet_model(args.model, args.lr, output_dim,)
    else:
        (model, params) = get_all_model(args.model, args.lr, output_dim)
    # else blabla

    if args.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=args.lr,)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(
            params, lr=args.lr, momentum=0.9, weight_decay=5e-4,
        )

    # Load the weight into the model
    if args.weights:
        with utils.measure_time(
            "Loading weights"
        ) if DEBUG else utils.dummy_context_mgr():
            model.load_state_dict(torch.load(args.weights))

    main(
        data_dir=args.path,
        device=torch.device(
            "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"
        ),
        batch_size=args.batch,
        epochs=args.epochs,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_transforms=locate(
            "pytorch_transforms." + args.transforms + ".train_transforms"
        ),
        test_transforms=locate(
            "pytorch_transforms." + args.transforms + ".test_transforms"
        ),
        dataset=datasets.ImageFolder,
        dataloader=data.DataLoader,
        valid_ratio=args.valid_ratio,
        file_name=args.save,
        find_lr=True if args.find_lr else False,
        end_lr = args.end_lr
    )
