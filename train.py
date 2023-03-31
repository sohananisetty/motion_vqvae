
import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch
from glob import glob
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
import json

from core.models.vqvae import VQMotionModel
from ctl.trainer import VQVAEMotionTrainer

def main(args):


    model = VQMotionModel()
    

    trainer = VQVAEMotionTrainer(
        vqvae_model = model,
        args = args,
        training_args = args,
    ).cuda()


    trainer.train(args.resume)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='/srv/scratch/sanisetty3/music_motion/AIST', help="folder with train and test data")
    parser.add_argument('--pretrained', default='')
    parser.add_argument('--resume', default=True, type = bool)
    parser.add_argument('--output_dir', default="./checkpoints/vanilla_aist")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fp16', default=True, type=bool)
    parser.add_argument("--dataset_name", type=str, default='aist', help="t2m or kit or aist")


    parser.add_argument('--train_bs', default=96, type=int,)
    parser.add_argument('--eval_bs', default=1, type=int,)
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int,)
    
    parser.add_argument("--commit", type=float, default=1, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss_vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l2', help='reconstruction loss')
    parser.add_argument('--max_seq_length', type=int, default=200, help='max sequence length')

    parser.add_argument("--num_train_iters",  default=200000,type=int)
    parser.add_argument("--save_steps",  default=2000,type=int)
    parser.add_argument("--logging_steps",  default=10,type=int)
    parser.add_argument("--wandb_every",  default=100,type=int)
    parser.add_argument("--evaluate_every",  default=4000,type=int)

    ## optimization
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--warmup_steps', default=4000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--lr_scheduler_type', default="cosine", help="learning rate schedule type")


    args = parser.parse_args()



    main(args)




#accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py




# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml  