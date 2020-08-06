#!/bin/bash

python3 -u train.py --gpu --model resnet151 --epochs 10 --batch 32 --optimizer SGD --save ./results/resnet151_e10_b32_sgd.pt --debug | tee ./results/resnet151_e10_b32_sgd.log
python3 -u train.py --gpu --model vgg16 --epoch 10 --batch 32 --optimizer SGD --save ./results/vgg16_e10_b32_sgd.pt --debug | tee ./results/vgg16_e10_b32_sgd.log
python3 -u train.py --gpu --model googlenet --epoch 10 --batch 128 --optimizer SGD --save ./results/googlenet_e10_b128_sgd.pt --debug | tee ./results/googlenet_e10_b128_sgd.log
python3 -u train.py --gpu --model alexnet --epoch 10 --batch 128 --optimizer SGD --save ./results/alexnet_e10_b128_sgd.pt --debug | tee ./results/alexnet_e10_b128_sgd.log
python3 -u train.py --gpu --model shufflenet_v2_x1_0 --epoch 10 --batch 128 --optimizer SGD --save ./results/shufflenet_v2_x1_0_e10_b128_sgd.pt --debug | tee ./results/shufflenet_v2_x1_0_e10_b128_sgd.log

python3 -u train.py --gpu --model mobilenet_v2 --epochs 10 --batch 128 --optimizer SGD --save ./results/mobilenet_v2_e10_b128_sgd.pt --debug | tee ./results/mobilenet_v2_e10_b128_sgd.log
python3 -u train.py --gpu --model resnext50_32x4d --epoch 10 --batch 32 --optimizer SGD --save ./results/resnext50_32x4d_e10_b32_sgd.pt --debug | tee ./results/resnext50_32x4d_e10_b32_sgd.log
python3 -u train.py --gpu --model resnext101_32x8d --epoch 10 --batch 16 --optimizer SGD --save ./results/resnext101_32x8d_e10_b16_sgd.pt --debug | tee ./results/resnext101_32x8d_e10_b16_sgd.log
python3 -u train.py --gpu --model mnasnet0_5 --epoch 10 --batch 128 --optimizer SGD --save ./results/mnasnet0_5_e10_b128_sgd.pt --debug | tee ./results/mnasnet0_5_e10_b128_sgd.log
python3 -u train.py --gpu --model mnasnet1_0 --epoch 10 --batch 128 --optimizer SGD --save ./results/mnasnet1_0_e10_b128_sgd.pt --debug | tee ./results/mnasnet1_0_e10_b128_sgd.log

