#!/bin/bash

python3 train.py --gpu --model resnet151 --epochs 10 --batch 32 --optimizer Adam --save resnet151_e10_b32_adam.pt > resnet151_e10_b32_adam_log.txt
python3 train.py --gpu --model alexnet --epoch 10 --batch 128 --optimizer SGD --save alexnet_e10_b128_sgd.pt > alexnet_e10_b128_sgd_log.txt
python3 train.py --gpu --model googlenet --epoch 10 --batch 128 --optimizer SGD --save googlenet_e10_b128_sgd.pt > googlenet_e10_b128_sgd.txt
