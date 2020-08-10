#!/usr/bin/env bash
#
# Slurm arguments
#SBATCH --job-name=arian_test
#SBATCH --output=/home/arian/results/resnet152_e20_batch_32_cyclic.log
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1 
#SBATCH --partition=debug
#SBATCH --time=15:00:00              # Max execution time
#

# Activate your Anaconda environment
conda activate deep

# Run your Python script
cd /home/arian/Deep-Learning-August
python main.py --gpu --workers 2 --scheduler StepLR --batch 32 --epoch 20 --model resnet152 --optimizer SGD --save /home/arian/results/resnet152_e20_batch_32_cyclic.pt
