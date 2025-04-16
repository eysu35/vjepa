#!/bin/bash
#SBATCH --job-name=vjepaTraining
#SBATCH --account=csci_ga_3033-2025sp
#SBATCH --output=/scratch/eys8549/vjepa_training/train.out
#SBATCH --error=/scratch/eys8549/vjepa_training/train.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=g2-standard-48

singularity exec --bind /scratch/eys8549 \
  --overlay /scratch/eys8549/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
  /bin/bash -c "source /home/eys8549/miniconda3/etc/profile.d/conda.sh; conda activate vjepa; python -u -m app.main --fname configs/pretrain/vitl16.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3"
