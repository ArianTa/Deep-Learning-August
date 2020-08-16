
import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np

from utils import *

import json


def get_predictions(model, iterator):
    """ Gets the predictions of a model

    :param model: A NN model
    :type model: torch.nn.module
    :param iterator: A dataloader
    :type iterator: torch.utils.data.Datalodaer
-   
    :rtype: None
    """
    model.eval()

    test_acc_1 = 0
    test_acc_5 = 0

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

            acc_1, acc_5 = calculate_topk_accuracy(y_pred.cpu(), y.cpu())
            test_acc_1 += acc_1.item()
            test_acc_5 += acc_5.item()

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)
    
    test_acc_1 /= len(iterator)
    test_acc_5 /= len(iterator)


    return images, labels, probs, test_acc_1, test_acc_5




def save_confusion_matrix(labels, pred_labels, classes):
    """ Saves the confusion matrix in the 'results/confuision_matrix.png' file

    :param labels: A list of the actual labels of the examples
    :type labels: List
    :param pred_labels: A list of the predicted labels of the examples
    :type pred_labels: List
    :param classes: A list of the classes of the dataset
    :type classes: List

    :rtype: None
    """


    fig = plt.figure(figsize=(250, 250))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels, normalize='true')
    with open('confusion_matrix.json', 'w') as outfile:
        json.dump(cm.tolist(), outfile)
    cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    cm.plot(include_values=False, cmap='Blues', ax=ax)
    fig.delaxes(fig.axes[1])  # delete colorbar
    plt.xticks(rotation=90)
    plt.xlabel('Predicted Label', fontsize=50)
    plt.ylabel('True Label', fontsize=50)
    plt.savefig('results/confusion_matrix.png', dpi=100)


def save_most_incorrect(incorrect, classes, n_images, normalize=True):
    """ Saves the most incorrect predictions in the 'results/most_incorrect.png' file
    
    :param incorrect: A list of tuples (images, actual labels of the images, predicted probabilities)
    :type incorrect: List
    :param classes: A list of the classes of the dataset
    :type classes: List
    :param n_images: Number of images to plot
    :type n_images: int
    :param normalize: Whether to normalize the images or not
    :type normalize: bool

    :rtype: None
    """

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(25, 20))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i + 1)

        image, true_label, probs = incorrect[i]
        image = image.permute(1, 2, 0)
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        true_class = classes[true_label]
        incorrect_class = classes[incorrect_label]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f'T: {true_class} ({true_prob:.3f})\n'
                     f'P: {incorrect_class} ({incorrect_prob:.3f})')
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)
    plt.savefig('results/most_incorrect.png')


def save_filtered_images(images, filters, n_filters=None, normalize=True):
    """ Saves filtered images in the 'results/filtered_image.png' file
    
    :param images: A tensor consisting in the images to plot
    :type images: torch.Tensor
    :param filters: A tensor consisting in the filters to apply to the images
    :type filters: torch.Tensor
    :param n_filters: The number of filters to apply to the images
    :type n_filters: int
    :param normalize: Whether to normalize the images or not
    :type normalize: bool

    :rtype: None
    """
    images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
    filters = filters.cpu()

    if n_filters is not None:
        filters = filters[:n_filters]

    n_images = images.shape[0]
    n_filters = filters.shape[0]

    filtered_images = F.conv2d(images, filters)

    fig = plt.figure(figsize=(30, 30))

    for i in range(n_images):

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax = fig.add_subplot(n_images, n_filters + 1, i + 1 + (i * n_filters))
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title('Original')
        ax.axis('off')

        for j in range(n_filters):
            image = filtered_images[i][j]

            if normalize:
                image = normalize_image(image)

            ax = fig.add_subplot(n_images, n_filters + 1,
                                 i + 1 + (i * n_filters) + j + 1)
            ax.imshow(image.numpy(), cmap='bone')
            ax.set_title(f'Filter {j+1}')
            ax.axis('off')

    fig.subplots_adjust(hspace=-0.7)
    plt.savefig('results/filtered_image.png')

def save_filters(filters, normalize=True):
    """ Saves filters in the 'results/filters.png' file
    
    :param filters: A tensor consisting in the filters to apply to the images
    :type filters: torch.Tensor
    :param normalize: Whether to normalize the images or not
    :type normalize: bool

    :rtype: None
    """
    filters = filters.cpu()

    n_filters = filters.shape[0]

    rows = int(np.sqrt(n_filters))
    cols = int(np.sqrt(n_filters))

    fig = plt.figure(figsize=(30, 15))

    for i in range(rows * cols):

        image = filters[i]

        if normalize:
            image = normalize_image(image)

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image.permute(1, 2, 0))
        ax.axis('off')

    fig.subplots_adjust(wspace=-0.9)
    plt.savefig('results/filters.png')

def test(**kwargs):
    """ Tests a ResNet152 the model on a dataset.
    Saves multiple images:
        - 'results/filters.png': The filters of the first layer
        - 'results/filtered_image.png': The first 7 layers applid on randomly drawn images
        - 'results/confusion_matrix.png': The confusion matrix
        - 'results/most_incorrect.png': The most incorrect precisions
    
    :param model: A ResNet model
    :type model: models.ResNet
    :param test_iterator: A dataloader for the testing set
    :type test_iterator: torch.utils.data.Datalodaer
    :param device: The device on which the operations will be done
    :type device: torch.device
    :param classes: A list of all the classes
    :type classes: List
    :param writer: Tensorboard writer (not used as of now)
    :type writer: torch.utils.tensorboard.SummaryWriter
    :param DEBUG: Print debug information or not
    :type DEBUG: bool

    :rtype: None
    """

    globals().update(kwargs)

    with measure_time("Getting predictions"):
        images, labels, probs, acc_1, acc_5 = get_predictions(model, test_iterator)

    pred_labels = torch.argmax(probs, 1)

    with measure_time("Big confusion matrix"):
        save_confusion_matrix(labels, pred_labels, classes)

    print(f"Test Acc @1: {acc_1*100:6.2f}% | Test Acc @5: {acc_5*100:6.2f}%")

    corrects = torch.eq(labels, pred_labels)

    incorrect_examples = []
    with measure_time("Getting most incorrect predictions"):
        for image, label, prob, correct in zip(images, labels, probs, corrects):
            if not correct:
                incorrect_examples.append((image, label, prob))

    incorrect_examples.sort(reverse = True, key = lambda x: torch.max(x[2], dim = 0).values)

    N_IMAGES = 36

    save_most_incorrect(incorrect_examples, classes, N_IMAGES)

    outputs, labels = get_representations(model, test_iterator)

    output_pca_data = get_pca(outputs)
    save_representations(output_pca_data, labels, classes)

    N_IMAGES = 5
    N_FILTERS = 7

    images = [image for image, label in [test_data[i] for i in range(N_IMAGES)]]
    filters = model.conv1.weight.data

    save_filtered_images(images, filters, N_FILTERS)

    save_filters(filters)

