import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import (
    Dataset,
    DataLoader,
)

import numpy as np
import argparse
import os
import random
import copy
from pydoc import locate

from tensorboardX import SummaryWriter


from train import train
from test import test
from find_lr import find_learning_rate
from pytorch_datasets import MushroomDataset

from utils import *

if __name__ == "__main__":
    # Parsing arguments and setting up metadata
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to root directory of the dataset",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="data/train.json",
        help="Path to JSON annotation file",
    )
    parser.add_argument(
        "--mode",
        default="train",
        help="Mode, train or test model."
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
        "--valid_ratio",
        type=float,
        default=0.90,
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
        default="imagenet_transforms",
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
    parser.add_argument(
        "--log",
        type=str,
        default="results",
        help="Log file path for tensorboard"
    )

    # Add more stuff here maybe ?
    args = parser.parse_args()

    # For reproducability
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    # criterion
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # Get the model and is parameters that are to be optimized
    model = get_model(args)
    model.to(device)

    if args.model == "resnet152":
        params = get_params(model, args.lr)

    else:
        params = model.parameters()

    # Optimizer
    optimizer = optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=5e-4,
    )

    # Load the weight into the model
    if args.weights:
        with utils.measure_time(
            "Loading weights"
        ) if args.debug else utils.dummy_context_mgr():
            model.load_state_dict(torch.load(args.weights))

    meta_data = {
        "DEBUG": True if args.debug else False,
        "device": device,
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion
    }

    train_transforms = locate(
        "pytorch_transforms." + args.transforms + ".train_transforms"
    )

    test_transforms = locate(
        "pytorch_transforms." + args.transforms + ".test_transforms"
    )

    dataset = MushroomDataset(
        root=args.data_path,
        annotation=args.json_path,
        transform=None)

    if args.mode == "train":
        train_data = dataset
        train_data.transform = train_transforms

        n_train_examples = int(len(train_data) * args.valid_ratio)
        n_valid_examples = len(train_data) - n_train_examples

        (train_data, valid_data,) = data.random_split(
            train_data, [n_train_examples, n_valid_examples, ],
        )

        valid_data = copy.deepcopy(valid_data)
        valid_data.dataset.transform = test_transforms

        train_iterator = data.DataLoader(
            train_data, shuffle=True, batch_size=args.batch,
        )

        valid_iterator = data.DataLoader(valid_data, batch_size=args.batch,)

        meta_data.update({
            "epochs": args.epochs,
            "train_iterator": train_iterator,
            "valid_iterator": valid_iterator,
            "file_name": args.save
        })

        train(**meta_data)

    elif args.mode == "test":
        test_data = dataset
        test_data.transform = test_transforms

        meta_data.update({
            "classes": [utils.format_label(c) for c in test_data.classes],
            "test_iterator": data.DataLoader(
                test_data, shuffle=True, batch_size=args.batch,
            )
        })
        test(**meta_data)

    elif args.mode == "find_lr":
        test_data = dataset
        test_data.transform = test_transforms

        meta_data.update({
            "end_lr": args.end_lr,
            "classes": [utils.format_label(c) for c in train_data.classes],
            "iterator": data.DataLoader(
                train_data, shuffle=True, batch_size=args.batch,
            )
        })

        find_learning_rate(**meta_data)
