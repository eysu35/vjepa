#!/bin/bash
#SBATCH --job-name=vjepaTraining
#SBATCH --account=csci_ga_3033-2025sp
#SBATCH --output=/scratch/eys8549/vjepa_training/train_%j.out
#SBATCH --error=/scratch/eys8549/vjepa_training/train_%j.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --ntasks=4                      # 4 tasks for 4 GPUs              # optional, good for data loading
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4              # optional, good for data loading
#SBATCH --partition=g2-standard-48

export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CODE_DIR=/home/eys8549/vjepa
singularity exec --bind /scratch/eys8549 \
  --bind $CODE_DIR:$CODE_DIR \
  --overlay /scratch/eys8549/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif \
  /bin/bash -c "
    source /home/eys8549/miniconda3/etc/profile.d/conda.sh && \
    conda activate vjepa && \
    cd /home/eys8549/vjepa && \
    export PYTHONPATH=$CODE_DIR && \
    echo 'Running on: ' \$(hostname) && \
    echo 'Using script at: ' \$(realpath app/main_distributed.py) && \
    torchrun \
      --nproc_per_node=4 \
      --nnodes=1 \
      --node_rank=0 \
      --master_port=29500 \
      app/main_distributed.py \
      --fname configs/pretrain/vitl16.yaml"

