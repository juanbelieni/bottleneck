#!/bin/bash
#SBATCH --job-name=bottleneck
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=2-8

source /home/parentd1/.bashrc
conda activate cooperative_sheaves

python run-drgnn-2-8.py --depth=$SLURM_ARRAY_TASK_ID
