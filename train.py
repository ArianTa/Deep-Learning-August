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
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler

from find_lr import count_parameters

from utils import *


def train_epoch(
    model, iterator, optimizer, criterion, scheduler, device
):
    """ Trains the model on a single epoch

    :param model: A NN model
    :type model: torch.nn.module
    :param iterator: A dataloader
    :type iterator: torch.utils.data.Datalodaer
    :param optimizer: A torch optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: A loss criterion
    :type criterion: torch.nn.module
    :param scheduler: A learning rate scheduler
    :type scheduler: _LRScheduler
    :param device: The device on which the operations will be done
    :type device: torch.device

    :return epoch_loss: The average training loss of the epoch
    :rtype epoch_loss: float
    :return epoch_acc_1: The average top1 training accuracy of the epoch
    :rtype epoch_acc_1: float
    :return epoch_acc_5: The average top5 training accuracy of the epoch
    :rtype epoch_acc_5: float
    """

    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.train()

    if DEBUG:
        i = 0
        total = len(iterator)

    for n_batch, (x, y) in enumerate(iterator):
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

        (acc_1, acc_5,) = calculate_topk_accuracy(y_pred, y,)

        if DEBUG:
            print(
                f"{i}th batch train accuracies: top1: {acc_1.item()*100:6.2f} | top5: {acc_5.item()*100:6.2f}")

        loss.backward()

        optimizer.step()

        scheduler.step()

        n_iter = n_epoch * len(iterator) + n_batch + 1

        if hasattr(model, "fc"):
            last_layer = model.fc
        else:
            last_layer = model.classifier[-1]

        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()



    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram(f"{layer}/{attr}", param, n_epoch)

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
    """ Evaluates a model on a dataset for a single epoch

    :param model: A NN model
    :type model: torch.nn.module
    :param iterator: A dataloader
    :type iterator: torch.utils.data.Datalodaer
    :param criterion: A loss criterion
    :type criterion: torch.nn.module
    :param device: The device on which the operations will be done
    :type device: torch.device

    :return epoch_loss: The average training loss of the epoch
    :rtype epoch_loss: float
    :return epoch_acc_1: The average top1 training accuracy of the epoch
    :rtype epoch_acc_1: float
    :return epoch_acc_5: The average top5 training accuracy of the epoch
    :rtype epoch_acc_5: float
    """
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

            (acc_1, acc_5,) = calculate_topk_accuracy(y_pred, y,)

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
    """ Measures the time taken for a single epoch

    :param start_time: The time at which the epoch started
    :type start_time: int
    :param end_time: The time at which the epoch ended
    :type end_time: int

    :return elapsed_mins: Elapsed minutes
    :rtype elapsed_mins: int
    :return elapsed_secs: Elapsed seconds
    :rtype elapsed_secs: int
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return (
        elapsed_mins,
        elapsed_secs,
    )


def train(**kwargs):
    """ Trains the model

    :param model: A NN model
    :type model: torch.nn.module
    :param start_epoch: The starting epoch
    :type start_epoch: int
    :param epochs: The total number of epochs to be done. (epochs - start_epoch) gives the actual number of epochs.
    :type epochs: int
    :param train_iterator: A dataloader for the training set
    :type train_iterator: torch.utils.data.Datalodaer
    :param valid_iterator: A dataloader for the validation set
    :type valid_iterator: torch.utils.data.Datalodaer
    :param optimizer: A torch optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: A loss criterion
    :type criterion: torch.nn.module
    :param scheduler: A learning rate scheduler
    :type scheduler: _LRScheduler
    :param device: The device on which the operations will be done
    :type device: torch.device
    :param file_name: Name of the file to save the weights in
    :type file_name: str
    :param writer: Tensorboard writer
    :type writer: torch.utils.tensorboard.SummaryWriter
    :param DEBUG: Print debug information or not
    :type DEBUG: bool


    :rtype: None
    """

    globals().update(kwargs)
    if DEBUG:
        print(
            f"The model has {count_parameters(model):,} trainable parameters"
        )

    best_valid_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        globals()['n_epoch'] = epoch
        start_time = time.time()

        (train_loss, train_acc_1, train_acc_5,) = train_epoch(
            model, train_iterator, optimizer, criterion, scheduler, device,
        )
        (valid_loss, valid_acc_1, valid_acc_5,) = evaluate(
            model, valid_iterator, criterion, device,
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1
            }, file_name)

        end_time = time.time()

        (epoch_mins, epoch_secs,) = epoch_time(start_time, end_time,)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | "
            f"Train Acc @5: {train_acc_5*100:6.2f}%")
        print(
            f"\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | "
            f"Valid Acc @5: {valid_acc_5*100:6.2f}%")

        writer.add_scalar(f'epoch/train_average_loss', train_loss, epoch + 1)
        writer.add_scalar(f'epoch/train_Top1_acc', train_acc_5, epoch + 1)
        writer.add_scalar(f'epoch/train_Top5_acc', train_acc_1, epoch + 1)
        writer.add_scalar(f'epoch/valid_Average_loss', valid_loss, epoch + 1)
        writer.add_scalar(f'epoch/valid_Top5_acc', valid_acc_5, epoch + 1)
        writer.add_scalar(f'epoch/valid_Top1_acc', valid_acc_1, epoch + 1)
