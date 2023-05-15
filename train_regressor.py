
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
from ctl.trainer_regressor import RegressorMotionTrainer
from configs.config import cfg, get_cfg_defaults

def main():


    trans_model = MotionRegressorModel(args = cfg.motion_trans ,pad_value=cfg.train.pad_index )
    vqvae_model = VQMotionModel(cfg.vqvae).eval()

    pkg = torch.load(f"/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/var_len/vq_768_768_mix/vqvae_motion_best_fid.pt", map_location = 'cpu')
    vqvae_model.load_state_dict(pkg["model"])

    trainer = RegressorMotionTrainer(
        trans_model = trans_model,
        vqvae_model = vqvae_model,
        args = cfg.motion_trans,
        training_args = cfg.train,
        dataset_args = cfg.dataset,
        eval_args = cfg.eval_model,
        model_name = cfg.motion_trans_model_name,
        
    ).cuda()


    trainer.train(cfg.train.resume)
    


if __name__ == '__main__':
    
    encodec_cfg_path = "/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_albi_aist/var_len_768_768_aist.yaml"
    librosa_cfg_path = "/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_albi_aist_35/var_len_768_768_aist_35.yaml"
    encodec_sine_cfg_path = "/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_sine_aist/var_len_768_768_sine_aist.yaml"
    encodec_mask_cfg_path = "/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_albi_aist_mask_prob50/trans_768_768_albi_aist_mask_prob50.yaml"


    cfg = get_cfg_defaults()
    print("loading config from:" , encodec_cfg_path)
    cfg.merge_from_file(encodec_cfg_path)
    cfg.freeze()
    print("\n cfg: \n", cfg)
    
   ### ALBI 35 no style
    # cfg = get_cfg_defaults()
    # print("loading config from:" , "/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_albi_aist_35/var_len_768_768_aist_35.yaml")
    # cfg.merge_from_file("/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_albi_aist_35/var_len_768_768_aist_35.yaml")
    # cfg.freeze()
    # print("\n cfg: \n", cfg)
    
    
    
    
    ### Sinusoidal 128 no style
    
    # cfg = get_cfg_defaults()
    # print("loading config from:" , "/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_sine_aist/var_len_768_768_sine_aist.yaml")
    # cfg.merge_from_file("/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_sine_aist/var_len_768_768_sine_aist.yaml")
    # cfg.freeze()
    # print("\n cfg: \n", cfg)

    ### Mask prob 50
    
    # cfg = get_cfg_defaults()
    # print("loading config from:" , "/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_albi_aist_mask_prob50/trans_768_768_albi_aist_mask_prob50.yaml")
    # cfg.merge_from_file("/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/generator/var_len/trans_768_768_albi_aist_mask_prob50/trans_768_768_albi_aist_mask_prob50.yaml")
    # cfg.freeze()
    # print("\n cfg: \n", cfg)

    ##Default
   

    
    main()




#accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py




# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml  