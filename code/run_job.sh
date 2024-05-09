#!/bin/bash

#SBATCH --job-name="train_model"
#SBATCH --partition=gpus
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=4-10:00:00
#SBATCH -e job_%x_%j.e
#SBATCH -o job_%x_%j.o
#SBATCH --nodes=1
#SBATCH --exclude=gpu[1601-1605, 1701-1708, 1801-1802, 1901-1904, 1905-1506, 2002-2003, 2201]
#SBATCH --gpus-per-node=1

source activate ~/miniconda3/envs/CS1430-CV-Project
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
cd ~/CS1430-CV-Project/cs1430-final-project/code
nvidia-smi
python train_model.py