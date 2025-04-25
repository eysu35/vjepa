# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Simplified version: torchrun + Slurm only
#

import argparse
import os
import pprint
import yaml
import torch
import torch.distributed as dist
import wandb
from pathlib import Path
from app.scaffold import main as app_main
from src.utils.logging import get_logger

logger = get_logger(force=True)


# --------------------------
# Distributed setup
# --------------------------
def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    logger.info(f"[RANK {rank}] LOCAL_RANK={local_rank}, "
                f"CUDA Device={torch.cuda.current_device()} "
                f"Name={torch.cuda.get_device_name(local_rank)}")

    return rank, local_rank, world_size


# --------------------------
# Training wrapper
# --------------------------
class Trainer:
    def __init__(self, args_pretrain):
        self.app = args_pretrain['app']
        self.args_pretrain = args_pretrain

    def __call__(self):
        logger.info('Loaded training params:')
        pprint.pprint(self.args_pretrain)
        app_main(self.app, args=self.args_pretrain, resume_preempt=False)


# --------------------------
# Main launch function
# --------------------------
def launch():
    with open(args.fname, 'r') as y_file:
        config = yaml.load(y_file, Loader=yaml.FullLoader)

    logger.info(f"Loaded config from: {args.fname}")
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        logger.info(f"Launching training on {world_size} GPUs...")
    
    wandb.init(project="vjepa-pretraining",
               entity="ellensu-new-york-university",
               config=config,
               group="vjepa_distributed_run")

    trainer = Trainer(config)
    trainer()


# --------------------------
# Argument parsing
# --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fname', type=str,
        help='Path to YAML config file',
        default='configs/pretrain/vitl16.yaml')
    args = parser.parse_args()

    launch()

