#!/bin/bash
#SBATCH --job-name=vjepaTraining_a100
#SBATCH --account=csci_ga_3033-2025sp
#SBATCH --output=/scratch/eys8549/vjepa_training/train_a100_%j.out
#SBATCH --error=/scratch/eys8549/vjepa_training/train_a100_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --partition=c24m170-a100-2

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CODE_DIR=/home/eys8549/vjepa

srun --mpi=none singularity exec --nv \
      --bind /scratch/eys8549 \
      --bind $CODE_DIR:$CODE_DIR \
      --overlay /scratch/eys8549/overlay-50G-10M.ext3:ro \
      /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
    bash -lc "
      source /home/eys8549/miniconda3/etc/profile.d/conda.sh
      conda activate vjepa
      cd $CODE_DIR
      export PYTHONPATH=$CODE_DIR
      export MASTER_ADDR=127.0.0.1
      export MASTER_PORT=29500
      torchrun \
        --standalone \
        --nproc_per_node=1 \
        app/main_distributed.py \
        --fname configs/pretrain/vitl16.yaml "
