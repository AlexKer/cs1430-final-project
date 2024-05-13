#!/bin/bash

#SBATCH --job-name="vgg_model"
#SBATCH --partition=gpus
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=4-10:00:00
#SBATCH -e job_%x_%j.e
#SBATCH -o job_%x_%j.o
#SBATCH --nodes=1
#SBATCH --exclude=gpu[1601-1605,1701-1708,1801-1802,1905-1906]
#SBATCH --gpus-per-node=1

source activate ~/miniconda3/envs/CVfinal_env
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
cd ~/newCS1430-CV-Project/cs1430-final-project/code
nvidia-smi
python train.py