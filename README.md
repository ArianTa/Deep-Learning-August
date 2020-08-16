# Mushroom Classification

> Project for the INFO8010-1 course.

This project consists in classifying a huge dataset of mushrooms. 
The dataset comes from the [FGCVx Fungi Classification Challenge dataset](https://www.kaggle.com/c/fungi-challenge-fgvc-2018/overview) 
and the links to download it may be found in [this repo](https://github.com/visipedia/fgvcx_fungi_comp#data).


## Set up

The different packages needed are listed in the file `environment.yml` and can be installed with the following instructions: 

```sh
conda env create -f environment.yml
```

NOTE: this will create and environment called 'mushroom_classification'.

## Usage 

The entrypoint for training and testing is `main.py`.

### Train with the default parameters

The dataset with the different classes of mushrooms has to be in the folder `data/images/`.
The JSON file containing the annotation information should be in `data/`

Please run
```sh

```

### Train with other parameters

This is a list that contains all the different parameters that can be used.
- `--debug`: print debug information
- `--data_path`: path to root directory of the dataset
- `--json_path`: path to JSON annotation file
- `--mode`: specify whether we want to train or test the model
- `--gpu`: whether or not we want to use the gpu
- `--workers`: set the number of workers for dataloaders
- `--batch`: specify the batch size
- `--epochs`: specify the number of epochs
- `--model`: specify the CNN model to be used
- `--optimizer`: specify the optimizer to be used. default: SGD
- `--criterion`: specify the criterion to be used. default: CrossEntropyLoss
- `-- scheduler`: specify the scheduler to be used. default: OneCycleLR
- `--save`: specify under which name the weights should be saved. default:model_weights.pt
- `--lr`: specify the value of the starting learning rate. default 10e-3
- `--valid_ratio`: specify the ratio between the train set and validation set. default: 0.90
- `--seed`: specify a seed, for reproducability. default 1234
- `--transforms`: the pytorch transforms to be used. default: imagenet_transforms
- `--load`: specify  a path for a checkpoint
- `--find_lr`: find the starting learning rate, if set, --lr becomes the lowest learning rate considered
- `--no_bias`: the highest learning rate considered if --flind_lr is set. default 10
- `--lr_decay`: use decaying learning rate
- `--momentum`: specify the value of the momentum to be used. default 0.9
- `--wdecay`: specify the value of the weight decay. default 5e-4
- `--nesterov`: specify whether or not to use the nesterov momentum. 
- `--save_log`: specify where to save  the results. default `results/`




Please add the parameter with the value picked 

The code can be run with the different parameters tested.

Please choose your optimizer from the 

## Test

Our final model may be trained with the following command
```sh

```

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)


<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki