# Installation

## diffusion 

```bash
conda create --name img-generation

conda install -c conda-forge diffusers

python3 -m pip install opencv-python

python3 -m pip install --upgrade diffusers accelerate transformers

```

## detector

`pip3 install torch torchvision`

`pip3 install opencv-python`

# Usage

Put example image data into corresponding folder

```bash
source ~/.bashrc

conda env list

conda activate img-generation
```

## Diffusion

`resize.py` (resize background images)

`maskImg_generating.py`

`multi_inpaint_by_examples.py`

## Detector

`python3 train_ssd.py --dataset-type=voc --data=data/voc --td=data/voc_test --model-dir=models/voc_with_test --batch-size=64 --epochs=50000`

`python3 run_ssd_example.py `

`/m/home/home3/30/luj4/unix/.local/bin/tensorboard --logdir sign_light_ped_detection_ssd/src/models/voc/tensorboard/xx/`

# Triton 

```bash
sbatch run_ppmc.sh

sacct

scancel

slurm q

sbatch

srun --jobid=x nvidia-smi

slurm prio
```

# Reference

https://huggingface.co/docs/diffusers/api/pipelines/paint_by_example 