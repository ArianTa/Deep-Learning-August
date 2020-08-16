# Mushroom Classification

> Project for the INFO8010-1 course of the ULiege University under the supervision of Professor Louppe.

This project consists in classifying a huge dataset of mushrooms. 
The dataset comes from the [FGCVx Fungi Classification Challenge dataset](https://www.kaggle.com/c/fungi-challenge-fgvc-2018/overview) 
and links to download it may be found in [this repository](https://github.com/visipedia/fgvcx_fungi_comp#data).


## Setup

Make sure you have an anaconda installation working on your machine.

The different packages needed are listed in the file `environment.yml` which can be used to create a new environment with the following instructions: 

```sh
conda env create -f environment.yml
```

NOTE: this will create and environment called 'mushroom_classification'.

To activate the environment, run
```sh
conda activate mushroom_classification
```

## Usage 

The entrypoint for training and testing is `main.py`.

### Train with the default parameters

By default, the script expects to have a `data` directory composed of the JSON files `train.json` and `test.json` as well as a `images` directory containing the dataset.

To train the default model with the default parameters, run
```sh
python main.py --gpu
```
See the next section for details about the script's parameters.

### Train with other parameters

This is a list that contains all the different parameters that can be used.
- `--debug`: print debug information when this parameter is specified
- `--data_path`: specify the path to root directory of the dataset. The default is set to `data/`
- `--json_path`: specify the path to JSON annotation file. The default is set to `data/train.json`
- `--mode`: specify in which mode the script is to be run: `train` or `test`
- `--gpu`: set this to use the gpu
- `--workers`: set the number of workers for dataloaders. The default value is `2`
- `--batch`: specify the batch size. The default value is `32`
- `--epochs`: specify the number of epochs. The default value is `10`
- `--model`: specify the CNN model to be used. The default is set to `resnet152`. The different possible models are listed in the table below.
- `--optimizer`: specify the optimizer to be used. The default is set to `SGD`. The different possible optimizers are listed in the table below.
- `--criterion`: specify the criterion to be used. The default is set to `CrossEntropyLoss`. As of now, only this criterion is supported
- `--scheduler`: specify the scheduler to be used. The default is set to `OneCycleLR`. The different possible schedulers are listed in the table below.
- `--save`: specify under which name the weights should be saved. The default is set to `results/model_weights.pt`
- `--lr`: specify the value of the starting learning rate. The default value is `10e-3`
- `--valid_ratio`: specify the ratio between the train set and validation set. The default value is `0.90`
- `--seed`: specify a seed, for reproducability. The default value is `1234`
- `--transforms`: specify the pytorch transforms to be used. The default is set to `imagenet_transforms`. The different possible pytorch transforms are listed in the table below.
- `--load`: specify  a path for a checkpoint to continue training or to test
- `--find_lr`: find the starting learning rate, if set, `--lr` becomes the lowest learning rate considered
- `--end_lr`: the highest learning rate considered if `--flind_lr` is set. The default value is `10`
- `--no_bias`: Specify if the bias weight decay is set to `0`.
- `--lr_decay`: use different learning rate per layer, works only for the model `resnet152`
- `--momentum`: specify the value of the momentum to be used. This parameter only works with the `SGD`optimizer and its default value is `0.9`
- `--wdecay`: specify the value of the weight decay. This parameter only works with the `SGD`optimizer and its default value is `5e-4`
- `--nesterov`: specify whether or not to use the nesterov momentum. This parameter only works with the `SGD`optimizer
- `--save_log`: specify the path where the tensorboard logs should be saved. The default is set to `results/`

The different models that are available are listed in the table below.

| Model | Commands | |||
| --- | --- | --- | --- | --- | 
|**AlexNet**| `alexnet` | | 
|**DenseNet**| `densenet` | |
|**GoogLeNet**| `googlenet`| | 
|**MobileNet** | `mobilenet_v2`|  | 
|**ResNet**|`resnet18`|`resnet34`|`resnet50`|`resnet101`|
| |`resnet152` |  
|**ShuffleNet**| `shufflenet_v2_x0_5`| `shufflenet_v2_x1_0`| `shufflenet_v2_x1_5`|`shufflenet_v2_x2_0`|
|**VGG**| `vgg11`| `vgg13`|`vgg16`|`vgg19`|

NOTE: the densenet model is not supported anymore.

The different optimizers that are supported are listed in the table below.

|Optimizer|Commands|
| --- | --- |
|**AdaDelta**|`Adadelta`|
|**AdaGrad**| `Adagrad`| 
|**Adam**| `Adam`|
|**AdaBound**| `Adabound`|
|**DiffGrad**|`DiffGrad`|
|**RMSprop**|`RMSprop`|
|**SGD**| `SGD`|

The different schedulers that are supported are listed in the table below.
|Scheduler |Commands|
| --- | --- |
|**Cosine Annealing**| `CosineAnnealingLR`|
|**Cyclic LR** |`CyclicLR`|
|**One Cycle** | `OneCycleLR`|
| **Step decay**|`StepLR`|

The different pytorch transforms that are supported are listed in the table below.

|Transforms |Commands| Description|
| --- | --- |--- |
|**ImageNet**|`imagenet_transforms`|Standard ImageNet transforms|
|**Erasure**| `erasure_transforms`| ImageNet with added random erasure |
|**Custom**|`mushroom_transform`| Our own transforms|

The pytorch transforms are modules of the `pytorch_transforms` package that define the train transforms and test transforms.


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

## Links
The state dicts of the model, the optimizer and the scheduler as well as the final number of epochs after training (i.e after running the previous command) can be downloaded [here](https://we.tl/t-LkrbHnah5V).

## Meta


Authors : 
- Nora Folon - nfolon@student.ulg.ac.be
- Amadis Horbach - a.horbach@student.uliege.be
- Arian Tahiraj - atahiraj@student.ulg.ac.be
