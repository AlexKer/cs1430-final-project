#!/bin/bash

#SBATCH --job-name="vgg_cpu2_batchsize218"
#SBATCH --partition=compute
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
#SBATCH --time=4-10:00:00
#SBATCH -e job_%x_%j.e
#SBATCH -o job_%x_%j.o
#SBATCH --nodes=1


source activate ~/miniconda3/envs/CVfinal_env
cd ~/CS1430-CV-Project/cs1430-final-project/code
nvidia-smi
python main.py --task 3