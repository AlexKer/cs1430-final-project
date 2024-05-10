#!/bin/bash

#SBATCH --job-name="VisionTransformer"
#SBATCH --partition=gpus
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=4-10:00:00
#SBATCH -e job_%x_%j.e
#SBATCH -o job_%x_%j.o
#SBATCH --nodes=1
#SBATCH --exclude=gpu[1601-1605,1701-08,1801-02,1901-04,1905-06,2002-03]
#SBATCH --gpus-per-node=1

source activate ~/miniconda3/envs/DLproject
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
cd /home/rdemello/CSCI1430/cs1430-final-project/code
nvidia-smi
python train.py