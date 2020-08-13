#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=MOMNU            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=/home/arian/Deep-Learning-August/results/resnet152_e20_batch_32_wdecay45.log
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=15:00:00              # Max execution time
#

# Activate your Anaconda environment
conda activate deep

LOG_DIR=/home/arian/Deep-Learning-August/results/wdecay45

mkdir $LOG_DIR


# Run your Python script
cd /home/arian/Deep-Learning-August
python main.py --gpu --workers 4 --batch 32 --epoch 20 --model resnet152 \
	--save $LOG_DIR/resnet152_e20_batch_32_wdecay45.pt \
	--data_path /scratch/users/arian/data --json_path /scratch/users/arian/data/train.json --wdecay 0.00005 \
	--save_log $LOG_DIR