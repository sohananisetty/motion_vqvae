'''
Default config
'''
from yacs.config import CfgNode as CN
# import argparse
# import yaml
import os
from glob import glob
from utils.word_vectorizer import POS_enumerator


cfg = CN()


cfg.abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg.device = 'cuda'

cfg.vqvae_model_name = "vqvae"

cfg.pretrained_modelpath = os.path.join(cfg.abs_dir, f"checkpoints/{cfg.vqvae_model_name}/vqvae_motion.pt")
cfg.output_dir = os.path.join(cfg.abs_dir , "checkpoints/")

cfg.eval_output_dir = os.path.join(cfg.abs_dir , "eval/")

cfg.eval_model_path = os.path.join(cfg.abs_dir , f"checkpoints/{cfg.vqvae_model_name}/vqvae_motion.pt")


cfg.dataset = CN()
cfg.dataset.dataset_name = "t2m" #"t2m or kit or aist"
cfg.dataset.var_len = False
cfg.dataset.data_folder =  "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D"




cfg.train = CN()
cfg.train.resume = True
cfg.train.seed = 42
cfg.train.fp16 = True
cfg.train.output_dir = cfg.output_dir

cfg.train.num_train_iters = 500000 #'Number of training steps
cfg.train.save_steps = 5000
cfg.train.logging_steps = 10
cfg.train.wandb_every = 100
cfg.train.evaluate_every = 5000
cfg.train.eval_bs = 24
cfg.train.train_bs = 24
cfg.train.gradient_accumulation_steps = 4

cfg.train.write_summary = True
cfg.train.log_dir = os.path.join(cfg.abs_dir , f"logs/{cfg.vqvae_model_name}")

## optimization

cfg.train.learning_rate = 2e-4
cfg.train.weight_decay = 0.0
cfg.train.warmup_steps = 4000
cfg.train.gamma = 0.05
cfg.train.lr_scheduler_type = "cosine"

cfg.vqvae = CN()

cfg.vqvae.nb_joints = 22 if cfg.dataset.dataset_name == "t2m" else 21
cfg.vqvae.motion_dim = 263 #'Input motion dimension dimension'
cfg.vqvae.enc_dec_dim = 768 #'Encoder and Decoder dimension'
cfg.vqvae.depth = 12
cfg.vqvae.heads=8
cfg.vqvae.codebook_dim = 768
cfg.vqvae.codebook_size = 1024

## Loss
cfg.vqvae.commit = 1  #"hyper-parameter for the commitment loss"
cfg.vqvae.loss_vel = 0.1
cfg.vqvae.recons_loss = "l1_smooth" #l1_smooth , l1 , l2
cfg.vqvae.max_seq_length = 200
cfg.vqvae.min_seq_length = 20


cfg.eval_model = CN()

cfg.eval_model.device = cfg.device
cfg.eval_model.dataset_name = 't2m'
cfg.eval_model.checkpoints_dir = "/srv/scratch/sanisetty3/music_motion/T2M-GPT/checkpoints"

cfg.eval_model.max_text_len = 20
cfg.eval_model.max_motion_length = 196
cfg.eval_model.unit_length = 4

cfg.eval_model.dim_word = 300
cfg.eval_model.dim_text_hidden= 512
cfg.eval_model.dim_z= 128
cfg.eval_model.dim_pose = 263
cfg.eval_model.dim_pos_ohot = len(POS_enumerator)
cfg.eval_model.dim_motion_hidden = 1024
cfg.eval_model.dim_coemb_hidden = 512
cfg.eval_model.dim_msd_hidden = 512
cfg.eval_model.dim_pos_hidden = 1024
cfg.eval_model.dim_pri_hidden = 1024
cfg.eval_model.dim_seq_de_hidden = 512
cfg.eval_model.dim_seq_en_hidden = 512

cfg.eval_model.dim_movement_enc_hidden = 512
cfg.eval_model.dim_movement2_dec_hidden = 512
cfg.eval_model.dim_movement_dec_hidden= 512
cfg.eval_model.dim_movement_latent = 512


