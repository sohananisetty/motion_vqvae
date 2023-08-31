
import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path

from core.models.vqvae import VQMotionModel, VQMotionModelPositional
from ctl.trainer import VQVAEMotionTrainer
from configs.config import cfg, get_cfg_defaults

def main():


    model = VQMotionModelPositional(cfg.vqvae)
    

    trainer = VQVAEMotionTrainer(
        vqvae_model = model,
        args = cfg.vqvae,
        training_args = cfg.train,
        dataset_args = cfg.dataset,
        eval_args = cfg.eval_model,
        model_name = cfg.vqvae_model_name,
        
    ).cuda()


    trainer.train(cfg.train.resume)
    


if __name__ == '__main__':

   

    cfg = get_cfg_defaults()
    print("loading config from:" , "/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/var_len/vq_768_768_mix_albi/var_len_768_768_mix_albi.yaml")
    cfg.merge_from_file("/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/var_len/vq_768_768_mix_albi/var_len_768_768_mix_albi.yaml")
    cfg.freeze()
    print("output_dir: ", cfg.output_dir , cfg.train.output_dir)
    
    
    main()




#accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py




# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml  