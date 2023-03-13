#!/bin/bash
#SBATCH --job-name=inpainting
#SBATCH --output=inpainting.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu

module load anaconda

source ~/.bashrc

conda init bash

source ~/.bashrc

conda activate Paint-by-Example

python3 -u main.py \
--logdir models/Paint-by-Example \
--pretrained_model pretrained_models/model.ckpt \
--base configs/v1.yaml \
--scale_lr False

