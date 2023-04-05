
import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path

from core.models.vqvae import VQMotionModel
from ctl.trainer import VQVAEMotionTrainer
from configs.config import cfg

def main():


    model = VQMotionModel(cfg.vqvae)
    

    trainer = VQVAEMotionTrainer(
        vqvae_model = model,
        args = cfg.vqvae,
        training_args = cfg.train,
        dataset_args = cfg.dataset,
        model_name = cfg.vqvae_model_name,
    ).cuda()


    trainer.train(cfg.train.resume)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_model_name", type=str, default='vqvae_motion_vl')

    parser.add_argument('--data_folder', default="/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D", help="folder with train and test data")
    parser.add_argument('--pretrained', default='')
    parser.add_argument('--resume', default=False, type = bool)
    parser.add_argument('--output_dir', default="/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/var_len/vq_768_768")
    # parser.add_argument('--evaluate', action='store_true')
    # parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--fp16', default=True, type=bool)
    parser.add_argument("--dataset_name", type=str, default='t2m', help="t2m or kit or aist")
    parser.add_argument('--var_len', default=True, type=bool)


    parser.add_argument('--train_bs', default=48, type=int,)
    # parser.add_argument('--eval_bs', default=64, type=int,)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int,)

    # parser.add_argument('--motion_dim', type=int, default=263, help='Input motion dimension dimension')
    # parser.add_argument('--enc_dec_dim', type=int, default=768, help='Encoder and Decoder dimension')
    # parser.add_argument('--depth', type=int, default=12, help='Encoder Decoder depth')
    # parser.add_argument('--heads', type=int, default=10, help='Encoder Decoder number of heads')
    # parser.add_argument('--codebook_dim', type=int, default=128, help='codeboook dimension')
    # parser.add_argument('--codebook_size', type=int, default=1024, help='number of codebook embeddings')



    
    # parser.add_argument("--commit", type=float, default=1, help="hyper-parameter for the commitment loss")
    # parser.add_argument('--loss_vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    # parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    # parser.add_argument('--max_seq_length', type=int, default=200, help='max sequence length')

    parser.add_argument("--num_train_iters",  default=500000,type=int)
    # parser.add_argument("--save_steps",  default=5000,type=int)
    # parser.add_argument("--logging_steps",  default=10,type=int)
    # parser.add_argument("--wandb_every",  default=100,type=int)
    # parser.add_argument("--evaluate_every",  default=10000,type=int)

    # ## optimization
    # parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--warmup_steps', default=4000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='max learning rate')
    # parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    # parser.add_argument('--lr_scheduler_type', default="cosine", help="learning rate schedule type")


    args = parser.parse_args()

    cfg.merge_from_list([
        "train.train_bs" , args.train_bs,
        "dataset.var_len" , args.var_len,
        "train.gradient_accumulation_steps",args.gradient_accumulation_steps,
        "train.num_train_iters", args.num_train_iters,
        "vqvae_model_name" , args.vqvae_model_name,
        "dataset.dataset_name" , args.dataset_name,
        "dataset.data_folder" ,args.data_folder,
        "output_dir" , args.output_dir,
        "train.learning_rate" , args.learning_rate,
        "train.warmup_steps" , args.warmup_steps,
        "train.resume" , args.resume
    ])



    main()




#accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py




# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml  