
import argparse
import os
import numpy as np

from pathlib import Path

from ctl.trainer_extractors import AISTExtractorMotionTrainer
from configs.config import cfg, get_cfg_defaults
from core.models.eval_modules import AISTEncoderBiGRUCo

def main():
    
    motion_extractor = AISTEncoderBiGRUCo(cfg.extractor.motion_input_size,cfg.extractor.hidden_size,cfg.extractor.output_size)
    music_extractor =  AISTEncoderBiGRUCo(cfg.extractor.music_input_size,cfg.extractor.hidden_size,cfg.extractor.output_size)
    

    trainer = AISTExtractorMotionTrainer(
        motion_extractor = motion_extractor,
        music_extractor = music_extractor,
        args = cfg.extractor,
        training_args = cfg.train,
        dataset_args = cfg.dataset,
        eval_args = cfg.eval_model,
        model_name = cfg.extractors_model_name,
        
    ).cuda()


    trainer.train(cfg.train.resume)
    


if __name__ == '__main__':

   

    # cfg = get_cfg_defaults()
    # print("loading config from:" , "/srv/scratch/sanisetty3/music_motion/motion_vqvae/configs/var_len_768_768_aist_vq.yaml")
    # cfg.merge_from_file("/srv/scratch/sanisetty3/music_motion/motion_vqvae/configs/var_len_768_768_aist_vq.yaml")
    # cfg.freeze()
    # print("output_dir: ", cfg.output_dir , cfg.train.output_dir)

    cfg = get_cfg_defaults()
    print("loading config from:" , "/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/extractors/small/aist_extractor.yaml")
    cfg.merge_from_file("/srv/scratch/sanisetty3/music_motion/motion_vqvae/checkpoints/extractors/small/aist_extractor.yaml")
    cfg.freeze()
    print("output_dir: ", cfg.output_dir , cfg.train.output_dir)
    
    
    main()




#accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py




# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml  