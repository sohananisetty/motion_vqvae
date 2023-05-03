
import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch
from core.models.vqvae import VQMotionModel
from core.models.motion_regressor import MotionRegressorModel
from ctl.trainer_regressor_style import RegressorMotionTrainerStyle
from configs.config import cfg, get_cfg_defaults
import clip

def main():


    trans_model = MotionRegressorModel(args = cfg.motion_trans ,pad_value=cfg.train.pad_index )
    vqvae_model = VQMotionModel(cfg.vqvae).eval()

    pkg = torch.load(f"/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/var_len/vq_768_768_mix/vqvae_motion_best_fid.pt", map_location = 'cpu')
    vqvae_model.load_state_dict(pkg["model"])
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    trainer = RegressorMotionTrainerStyle(
        trans_model = trans_model,
        vqvae_model = vqvae_model,
        clip_model = clip_model,
        args = cfg.motion_trans,
        training_args = cfg.train,
        dataset_args = cfg.dataset,
        eval_args = cfg.eval_model,
        model_name = cfg.motion_trans_model_name,
        
    ).cuda()


    trainer.train(cfg.train.resume)
    


if __name__ == '__main__':

   
    cfg = get_cfg_defaults()
    print("loading config from:" , "/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_albi_aist_styl/var_len_768_768_aist_style.yaml")
    cfg.merge_from_file("/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_albi_aist_style/var_len_768_768_aist_style.yaml")
    cfg.freeze()
    print("\n cfg: \n", cfg)

    # cfg_vq = get_cfg_defaults()
    # cfg_vq.merge_from_file("/srv/scratch/sanisetty3/music_motion/motion_vqvae/configs/var_len_768_768_aist.yaml")
    ##


    
    main()




#accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py




# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml  