#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=ADD            # Name of the job 
#SBATCH --export=WADD                # Export all environment variables
#SBATCH --output=/home/arian/Deep-Learning-August/results/resnet152_e20_batch_32_OneCycle_worst.log
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=2-15:00:00              # Max execution time
#

# Activate your Anaconda environment
conda activate deep

LOG_DIR=/home/arian/Deep-Learning-August/results/OneCycle_worst

mkdir $LOG_DIR


# Run your Python script
cd /home/arian/Deep-Learning-August
python main.py --gpu --workers 4 --batch 32 --epoch 100 --model resnet152 \
	--save $LOG_DIR/resnet152_e20_batch_32_OneCycle_worst.pt \
	--data_path /scratch/users/arian/data --json_path /scratch/users/arian/data/train.json \
	--save_log $LOG_DIR
