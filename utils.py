import os
import torch
import matplotlib.pyplot as plt
import numpy as np

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



def get_params(net, lr):
    """
    """
    decay = []
    no_decay = []

    for m in net.modules():
        print("aa")