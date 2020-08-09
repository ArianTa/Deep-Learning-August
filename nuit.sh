#!/bin/bash

#python3 -u main.py --mode train --gpu --model resnet151 --epochs 10 --batch 32 --optimizer SGD --save ./results/resnet151_e10_b32_sgd.pt | tee ./results/resnet151_e10_b32_sgd.log
#python3 -u main.py --mode train --gpu --model alexnet --epoch 10 --batch 128 --optimizer SGD --save ./results/alexnet_e10_b128_sgd.pt | tee ./results/alexnet_e10_b128_sgd.log
#python3 -u main.py --mode train --gpu --model vgg16 --epoch 10 --batch 32 --optimizer SGD --save ./results/vgg16_e10_b32_sgd.pt | tee ./results/vgg16_e10_b32_sgd.log
#python3 -u main.py --mode train --gpu --model googlenet --epoch 10 --batch 128 --optimizer SGD --save ./results/googlenet_e10_b128_sgd.pt | tee ./results/googlenet_e10_b128_sgd.log
#python3 -u main.py --mode train --gpu --model shufflenet_v2_x1_0 --epoch 10 --batch 128 --optimizer SGD --save ./results/shufflenet_v2_x1_0_e10_b128_sgd.pt | tee ./results/shufflenet_v2_x1_0_e10_b128_sgd.log

#python3 -u main.py --mode train --gpu --model mobilenet_v2 --epochs 10 --batch 64 --optimizer SGD --save ./results/mobilenet_v2_e10_b128_sgd.pt | tee ./results/mobilenet_v2_e10_b128_sgd.log


#python3 -u main.py --mode train --gpu --model resnext50_32x4d --epoch 10 --batch 32 --optimizer SGD --save ./results/resnext50_32x4d_e10_b32_sgd.pt | tee ./results/resnext50_32x4d_e10_b32_sgd.log
#python3 -u main.py --mode train --gpu --model resnet151 --epochs 10 --batch 32 --optimizer SGD --save ./results/resnet151_decay_e10_b32_sgd.pt | tee ./results/resnet151_decay_e10_b32_sgd.log
#python3 -u main.py --mode train --gpu --model resnext101_32x8d --epoch 10 --batch 16 --optimizer SGD --save ./results/resnext101_32x8d_e10_b16_sgd.pt | tee ./results/resnext101_32x8d_e10_b16_sgd.log
#python3 -u main.py --mode train --gpu --model vgg19 --epoch 10 --batch 32 --optimizer SGD --save ./results/vgg19_e10_b32_sgd.pt | tee ./results/vgg19_e10_b32_sgd.log
python3 -u main.py --mode train --gpu --model resnet151 --epochs 20 --batch 32 --save ./results/resnet151_decay_e20_b32_sgd.pt | tee ./results/resnet151_decay_e20_b32_sgd.log
