#!/bin/bash
#SBATCH --job-name=inpainting
#SBATCH --output=inpainting.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:2

module load anaconda

source ~/.bashrc

conda init bash

source ~/.bashrc

conda activate project-diffuser

export OMP_PROC_BIND=true

python3 multi_inpaint_by_examples.py
