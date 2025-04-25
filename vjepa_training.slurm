#!/bin/bash
#
#SBATCH --job-name=vjepaTraining
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=eys8549@nyu.edu
#SBATCH --mail-type=END,FAIL

singularity exec --overlay /scratch/eys8549/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
  /bin/bash -c "\
    source /home/eys8549/miniconda3/etc/profile.d/conda.sh && \
    conda activate vjepa && \
    python -m app.main --fname configs/pretrain/vitl16.yaml --devices cuda:0 cuda:1 cuda:2"
