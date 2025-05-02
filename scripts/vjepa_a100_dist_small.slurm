#!/bin/bash
#SBATCH --job-name=SM_vjepaTraining_a100
#SBATCH --account=csci_ga_3033-2025sp
#SBATCH --output=/scratch/eys8549/vjepa_training/small/rain_a100_%j.out
#SBATCH --error=/scratch/eys8549/vjepa_training/small/train_a100_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # Change: Request only 1 task for the whole node
#SBATCH --gres=gpu:2          # Keep: Request 2 GPUs for this single task
#SBATCH --cpus-per-task=8     # Change: Request enough CPUs for all GPU processes (2 GPUs * 4 CPUs/GPU = 8)
#SBATCH --partition=c24m170-a100-2

# Optional: Keep these for debugging if necessary
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

CODE_DIR=/home/eys8549/vjepa

# srun will launch the command within the single allocated task
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
     export WANDB_API_KEY=c7cffe3d10cad3b4cadee1fc83a0b3eb91476295
     export WANDB_ENTITY=ellensu-new-york-university
     export WANDB_PROJECT=vjepa-pretraining
     
     torchrun \
       --nproc_per_node=2 \
       app/main_distributed.py \
       --fname configs/pretrain/vits16.yaml "
