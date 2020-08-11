#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=ADM            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=/home/arian/Deep-Learning-August/results/resnet152_e20_batch_32_adam.log
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=00:15:00              # Max execution time
#SBATCH --partition=debug


# Activate your Anaconda environment
conda activate deep

LOG_DIR=/home/arian/Deep-Learning-August/results/adam

mkdir $LOG_DIR


# Run your Python script
cd /home/arian/Deep-Learning-August
python main.py --gpu --workers 4 --scheduler CosineAnnealingLR --batch 32 --epoch 20 --model resnet152 \
	--optimizer Adam --save $LOG_DIR/resnet152_e20_batch_32_adam.pt \
	--data_path /scratch/users/arian/data --json_path /scratch/users/arian/data/test.json \
	--save_log $LOG_DIR
