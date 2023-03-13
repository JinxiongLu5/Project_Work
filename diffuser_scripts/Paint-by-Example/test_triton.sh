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

conda activate Paint-by-Example

export OMP_PROC_BIND=true

python3 scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_1.png \
--mask_path examples/mask/example_1.png \
--reference_path examples/reference/example_1.jpg \
--seed 321 \
--scale 5

