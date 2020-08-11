import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch_optimizer as optimizer
import time
import datetime
from contextlib import contextmanager
from models import *


@contextmanager
def measure_time(label,):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'
    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    print("{}...".format(label))
    start = time.time()
    yield
    end = time.time()
    print(
        "Duration of [{}]: {}".format(
            label, datetime.timedelta(seconds=end - start),
        )
    )


class dummy_context_mgr:
    def __enter__(self,):
        return None

    def __exit__(
        self, exc_type, exc_value, traceback,
    ):
        return False


def format_label(label,):
    # label = label.split("_")[1]  # takes only the superclass
    return label


def calculate_topk_accuracy(
    y_pred, y, k=5,
):
    with torch.no_grad():
        batch_size = y.shape[0]
        (_, top_pred,) = y_pred.topk(k, 1,)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1,).expand_as(top_pred))
        correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True,)
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True,)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return (
        acc_1,
        acc_k,
    )


def show_img(
    dataset, n_images=25,
):
    (images, labels,) = zip(
        *[
            (image, label,)
            for image, label in [dataset[i] for i in range(n_images)]
        ]
    )

    dataset.classes = [format_label(c) for c in dataset.classes]
    classes = dataset.classes
    plot_images(
        images, labels, classes,
    )


def normalize_image(image,):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(
        min=image_min, max=image_max,
    )
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def plot_images(
    images, labels, classes, normalize=True,
):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(15, 15,))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i + 1,)

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0,).cpu().numpy())
        label = classes[labels[i]]
        ax.set_title(label)
        ax.axis("off")
    plt.show()


def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


def get_model(args):

    output_dim = len(os.listdir(os.path.join(args.data_path, "images")))
    if "resnet" in args.model:
        model = get_resnet_model(args.model, output_dim)
    elif "vgg" in args.model:
        model = get_vgg_model(args.model, output_dim)
    elif "googlenet" in args.model:
        model = get_googlenet_model(output_dim)
    elif "densenet" in args.model:
        model = get_densenet_model(args.model, output_dim)
    elif "shufflenet" in args.model:
        model = get_shufflenet_model(
            args.model, output_dim,
        )
    elif "alexnet" in args.model:
        model = get_alexnet_model(args.model, output_dim)
    else:
        model = get_all_model(args.model, output_dim)

    return model


def get_params(resnet, args):
    """
    """
    assert isinstance(resnet, ResNet)
    base_lr = args.lr

    if args.lr_decay:
        params = [
            {"params": resnet.conv1.parameters(), "lr": base_lr / 10},
            {"params": resnet.bn1.parameters(), "lr": base_lr / 10},
            {"params": resnet.layer1.parameters(), "lr": base_lr / 8},
            {"params": resnet.layer2.parameters(), "lr": base_lr / 6},
            {"params": resnet.layer3.parameters(), "lr": base_lr / 4},
            {"params": resnet.layer4.parameters(), "lr": base_lr / 2},
            {"params": resnet.fc.parameters()},
        ]
    elif args.no_bias:
        def no_bias_decay(net, lr):
            decay = []
            no_decay = []
            for m in net.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    decay.append(m.weight)

                    if m.bias is not None:
                        no_decay.append(m.bias)

                else:
                    if hasattr(m, 'weight'):
                        no_decay.append(m.weight)
                    if hasattr(m, 'bias'):
                        no_decay.append(m.bias)

            return [
                dict(
                    params=decay, lr=lr), dict(
                    params=no_decay, weight_decay=0, lr=lr)]

        length = 0
        params = []
        increasing_lr = [(resnet.conv1, base_lr /
                          10), (resnet.bn1, base_lr /
                                10), (resnet.layer1, base_lr /
                                      8), (resnet.layer2, base_lr /
                                           6), (resnet.layer3, base_lr /
                                                4), (resnet.layer4, base_lr /
                                                     2), (resnet.fc, base_lr)]

        for layer, lr in increasing_lr:
            layer_params = no_bias_decay(layer, lr)
            length += len(layer_params[0]['params']) + \
                len(layer_params[1]['params'])
            params += layer_params

        assert len(list(resnet.parameters())) == length

    else:
        params = resnet.parameters()

    return params

def get_criterion(args):
    return nn.CrossEntropyLoss()

def get_optimizer(params, args):
    if args.optimizer == "SGD":
        return optim.SGD(
                params, lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay, nesterov=args.nesterov
            )
    elif args.optimizer == "DiffGrad":
        return optimizer.DiffGrad(
                params,
                lr= 1e-3,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
            )
    elif args.optimizer == "Adam":
        return optim.Adam(params, lr=args.lr)
    elif args.optimizer == "Adagrad":
        return optim.Adagrad(params, lr=args.lr)
    elif args.optimizer == "RMSprop":
        return optim.RMSprop(params, lr=args.lr)
    elif args.optimizer == "Adadelta"
        return optim.Adadelta(params, lr=args.lr)
