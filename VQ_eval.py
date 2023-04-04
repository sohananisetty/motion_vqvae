import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from configs.config import cfg


import core.models.vqvae as vqvae
import utils.utils_model as utils_model
from core.datasets import dataset_TM_eval
import utils.eval_trans as eval_trans
from core.models.evaluator_wrapper import EvaluatorModelWrapper
from core.models.vqvae import VQMotionModel


##### ---- Exp dirs ---- #####
torch.manual_seed(42)
dataname = "t2m"
exp_name = "vq_768_128"
out_dir = "/srv/scratch/sanisetty3/music_motion/motion_vqvae/evals"

out_dir = os.path.join(out_dir, f'{exp_name}')
os.makedirs(out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(out_dir)
writer = SummaryWriter(out_dir)
# logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('/srv/scratch/sanisetty3/music_motion/T2M-GPT/glove', 'our_vab')

checkpoint_dir = "/srv/scratch/sanisetty3/music_motion/T2M-GPT/checkpoints"
# dataset_opt_path = f'{checkpoint_dir}/kit/Comp_v6_KLD005/opt.txt' if dataname == 'kit' else f'{checkpoint_dir}/t2m/Comp_v6_KLD005/opt.txt'


eval_wrapper = EvaluatorModelWrapper(cfg.eval_model)


##### ---- Dataloader ---- #####
nb_joints = 21 if dataname == 'kit' else 22

val_loader = dataset_TM_eval.DATALoader(dataname, True, 4, w_vectorizer, unit_length=2**2)

##### ---- Network ---- #####
net = VQMotionModel(cfg.vqvae)

if cfg.train.resume: 
    logger.info('loading checkpoint from {}'.format(cfg.pretrained_modelpath))
    ckpt = torch.load(cfg.pretrained_modelpath, map_location='cpu')
    net.load_state_dict(ckpt['model'], strict=True)
net.train()
net.cuda()

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
repeat_time = 20
for i in range(repeat_time):
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, draw=False, save=False, savenpy=(i==0))
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)