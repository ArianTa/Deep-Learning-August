# Mushroom Classification

> Project for the INFO8010-1 course.

This project consists in classifying a huge dataset of mushrooms. 
The dataset comes from the [FGCVx Fungi Classification Challenge dataset](https://www.kaggle.com/c/fungi-challenge-fgvc-2018/overview) 
and links to download it may be found in [this repository](https://github.com/visipedia/fgvcx_fungi_comp#data).


## Set up

The different packages needed are listed in the file `environment.yml` and can be installed with the following instructions: 

```sh
conda env create -f environment.yml
```

NOTE: this will create and environment called 'mushroom_classification'.

## Usage 

The entrypoint for training and testing is `main.py`.

### Train with the default parameters

By default, the script expects to have a `data` composed of the JSON files `train.json` and `test.json` as well as a `images` directory containing the dataset.

To train the default model with the default parameters, run:
```sh
python main.py --gpu
```
See the next section for details about the script's paramerts.

### Train with other parameters

This is a list that contains all the different parameters that can be used.
- `--debug`: print debug information
- `--data_path`: path to root directory of the dataset. The default is set to `data/`
- `--json_path`: path to JSON annotation file. The default is set to `data/`
- `--mode`: specify whether we want to train or test the model
	- The command `train` and `test` are accepted
- `--gpu`: whether or not we want to use the gpu
- `--workers`: set the number of workers for dataloaders. The default is set to 2
- `--batch`: specify the batch size. The default is set to 32
- `--epochs`: specify the number of epochs. The default value is set to 10
- `--model`: specify the CNN model to be used. The default is set to `resnet152`. The different possible models are listed in the table below.
- `--optimizer`: specify the optimizer to be used. The default is set to `SGD`. The different possible models are listed in the table below.
- `--criterion`: specify the criterion to be used. The default is set to `CrossEntropyLoss`
- `-- scheduler`: specify the scheduler to be used. The default is set to `OneCycleLR`. The different possible scheduler are listed in the table below.
- `--save`: specify under which name the weights should be saved. The default is set to `model_weights.pt`
- `--lr`: specify the value of the starting learning rate. The default is set to 10e-3
- `--valid_ratio`: specify the ratio between the train set and validation set. The default value is set to 0.90
- `--seed`: specify a seed, for reproducability. The default is set to 1234
- `--transforms`: the pytorch transforms to be used. The default is set to `imagenet_transforms`
- `--load`: specify  a path for a checkpoint
- `--find_lr`: find the starting learning rate, if set, --lr becomes the lowest learning rate considered
- `--no_bias`: the highest learning rate considered if --flind_lr is set. The default is set to 10
- `--lr_decay`: use decaying learning rate
- `--momentum`: specify the value of the momentum to be used. The default is set to 0.9
- `--wdecay`: specify the value of the weight decay. The default is set to 5e-4
- `--nesterov`: specify whether or not to use the nesterov momentum. 
- `--save_log`: specify where to save  the results. The default is set to `results/`

The different models that are supported are listed in the table below.

| Models | Command | ||
| --- | --- | --- | --- | 
|**AlexNet**| `alexnet` | | 
|**DenseNet**| `densenet` ?????| |
|**GoogLeNet**| `googlenet`| | 
|**MobileNet** | `mobilenet_v2`|  | 
|**ResNet**|`resnet18`|`resnet34`|`resnet50`|
| | `resnet101`|`resnet152` |  
|**ShuffleNet**| `shufflenet_v2_x0_5`| `shufflenet_v2_x1_0`|  
| | `shufflenet_v2_x1_5`|`shufflenet_v2_x2_0`|
|**VGG**| `vgg11`| `vgg13`|
||`vgg16`|`vgg19`|

|Optimizer|Command|
| --- | --- |
|**AdaDelta**|`Adadelta`|
|**AdaGrad**| `Adagrad`| 
|**Adam**| `Adam`|
|**AdaBound**| `Adabound`|
|**DiffGrad**|`DiffGrad`|
|**RMSprop**|`RMSprop`|
|**SGD**| `SGD`|


Please add the parameter with the value picked 

The code can be run with the different parameters tested.

Please choose your optimizer from the 

## Test

Our final model may be trained with the following command:
```sh
python main.py --gpu \
	--workers 4 \
	--batch 32 \
	--epoch 100 \
	--model resnet152 \
	--optimizer SGD \
	--scheduler OneCycleLR \
	--wdecay 0.0005 \
	--no_bias \
	--nesterov
```
The tensorboard logs and the weights of the model will be saved in the `results` directory.
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