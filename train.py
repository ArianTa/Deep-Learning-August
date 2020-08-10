import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from find_lr import count_parameters

from utils import *


def train_epoch(
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

        (acc_1, acc_5,) = calculate_topk_accuracy(y_pred, y,)

        if DEBUG:
            print(
                f"{i}th batch train accuracies: top1: {acc_1.item()*100:6.2f} | top5: {acc_5.item()*100:6.2f}")

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
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return (
        elapsed_mins,
        elapsed_secs,
    )


def train(**kwargs):
    globals().update(kwargs)

    if DEBUG:
        print(
            f"The model has {count_parameters(model):,} trainable parameters"
        )

    # We can start the training of the model
    TOTAL_STEPS = epochs * len(train_iterator)

    MAX_LRS = [p["lr"] for p in optimizer.param_groups]

    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LRS, total_steps=TOTAL_STEPS,
    )

    best_valid_loss = float("inf")
    for epoch in range(epochs):

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
