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
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler


class LRFinder:
    def __init__(
        self, model, optimizer, criterion, device,
    ):
        """ Constructor for the LRFinder class, saves the initial parameters in 'init_params.pt'

        :param model: A NN model
        :type model: torch.nn.module
        :param optimizer: A torch optimizer
        :type optimizer: torch.optim.Optimizer
        :param criterion: A loss criterion
        :type criterion: torch.nn.module
        :param device: The device on which the operations will be done
        :type device: torch.device

        :rtype: LRFinder
        """


        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device

        torch.save(
            model.state_dict(), "init_params.pt",
        )

    def range_test(
        self, iterator, end_lr=10, num_iter=100, smooth_f=0.05, diverge_th=5,
    ):
        """ Tries to train the model in a range of learning rates

        :param iterator: A dataloader
        :type iterator: torch.utils.data.Datalodaer
        :param end_lr: A maximum lr
        :type end_lr: float
        :param num_iter: The number of iteration
        :type num_iter: int
        :param smooth_f: A smoothing to apply to the loss
        :type num_iter: float
        :param diverge_th: The factor after which the losses are assumed to have diverged
        :type diverge_th: float

        :return lrs: A list of the learning rates
        :rtype lrs: List
        :return lrs: A list of the losses
        :rtype lrs: List
        """


        lrs = []
        losses = []
        best_loss = float("inf")

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter,)

        iterator = IteratorWrapper(iterator)

        for iteration in range(num_iter):
            loss = self._train_batch(iterator)

            # update lr
            lr_scheduler.step()

            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss)

            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break

        # reset model to initial parameters
        self.model.load_state_dict(torch.load("init_params.pt"))

        return (
            lrs,
            losses,
        )

    def _train_batch(
        self, iterator,
    ):
        """ Trains the model on a single batch, returns the loss on that batch

        :param iterator: A dataloader for the dataset
        :type iterator: torch.utils.data.DataLoader

        :rtype: float
        """

        self.model.train()

        self.optimizer.zero_grad()

        (x, y,) = iterator.get_batch()

        x = x.to(self.device)
        y = y.to(self.device)

        (y_pred, _,) = self.model(x)

        loss = self.criterion(y_pred, y,)

        loss.backward()

        self.optimizer.step()

        return loss.item()


class ExponentialLR(_LRScheduler):
    """ This class implements a exponential learning rate scheduler
    """
    def __init__(
        self, optimizer, end_lr, num_iter, last_epoch=-1,
    ):
        """ Constructor for the ExponentialLR class

        :param optimizer: A torch optimizer
        :type optimizer: torch.optim.Optimizer
        :param end_lr: A maximum lr
        :type end_lr: float
        :param num_iter: The number of iteration
        :type num_iter: int
        :param last_epoch: The last epoch for which the learning rates were computed
        :type last_epoch: int
        :rtype: ExponentialLR
        """
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self,).__init__(
            optimizer, last_epoch,
        )

    def get_lr(self,):
        """ Gets the next learning rates

        :return: A list containing the next learing rates
        :rtype: List
        """
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [
            base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs
        ]


class IteratorWrapper:
    """ This class is a wrapper around an iterator that provides safer
    iterations.
    """
    def __init__(
        self, iterator,
    ):
        """ Constructor for the IteratorWrapper class
        
        :param iterator: A dataloader for which to build a wrapper around
        :type iterator: torch.utils.data.Datalodaer
        
        :rtype: IteratorWrapper
        """
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self,):
        """ Gets the next item of the iterator.

        :return inputs: The transformed image
        :rtype inputs: torch.Tensor
        :return labels: The category index of the image
        :rtype labels: int
        """
        try:
            (inputs, labels,) = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            (inputs, labels, *_,) = next(self._iterator)

        return (
            inputs,
            labels,
        )

    def get_batch(self,):
        """ Wrapper around __next__

        :return inputs: The transformed image
        :rtype inputs: torch.Tensor
        :return labels: The category index of the image
        :rtype labels: int
        """
        return next(self)


def plot_lr_finder(
    lrs, losses, skip_start=5, skip_end=5,
):
    """ Plots the losses a s a function of the learning rates

    :param lrs: A list of the learning rates
    :type lrs: List
    :param lrs: A list of the losses
    :type lrs: List
    :param skip_start: The number of datapoints to skip at the start of the graph
    :type skip_start: int 
    :param skip_end: The number of datapoints to skip at the end of the graph
    :type skip_end: int 
    """
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]

    fig = plt.figure(figsize=(16, 8,))
    ax = fig.add_subplot(1, 1, 1,)
    ax.plot(
        lrs, losses,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    ax.grid(
        True, "both", "x",
    )
    plt.show()


def find_learning_rate(nb_iter=100, **kwargs):
    """ Wrapper around the LRFinder class, plots the losses.
    All parameters must be named to pass them with **kwargs for simplicity
    
    :param model: The model to find a lr for
    :type model: torch.nn.Module
    :param optimizer: A torch optimizer
    :type optimizer: torch.optim.Optimizer
    :param criterion: A loss criterion
    :type criterion: torch.nn.module
    :param device: The device on which the operations will be done
    :type device: torch.device

    :rtype: None
    """

    globals().update(kwargs)
    model.to(device)
    criterion.to(device)
    lr_finder = LRFinder(model, optimizer, criterion, device,)
    (lrs, losses,) = lr_finder.range_test(iterator, end_lr,)
    plot_lr_finder(
        lrs, losses, skip_start=30, skip_end=30,
    )

def count_parameters(model,):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)