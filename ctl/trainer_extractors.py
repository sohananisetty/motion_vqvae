from math import sqrt
from pathlib import Path

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from torchvision import transforms
import itertools
from transformers import AdamW, get_scheduler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs,InitProcessGroupKwargs
from accelerate import DistributedType
import wandb
import transformers
from datetime import timedelta
from core.optimizer import get_optimizer
from render_final import render
from core.datasets.vqa_motion_dataset import VQMotionDataset,DATALoader,VQVarLenMotionDataset,MotionCollator,VQVarLenMotionDatasetConditional, TransMotionDatasetConditional, MotionCollatorConditional
from tqdm import tqdm
from collections import Counter

from core.models.eval_modules import AISTEncoderBiGRUCo
from core.models.loss import InfoNceLoss, CLIPLoss
from core.datasets.evaluator_dataset import EvaluatorMotionDataset, EvaluatorVarLenMotionDataset,EvaluatorMotionCollator

def exists(val):
	return val is not None

def noop(*args, **kwargs):
	pass

def cycle(dl):
	while True:
		for data in dl:
			yield data

def cast_tuple(t):
	return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
	answer = input(f'{question} (y/n) ')
	return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
	for key, new_value in new_logs.items():
		old_value = log.get(key, 0.)
		log[key] = old_value + new_value
	return log

# auto data to module keyword argument routing functions

def has_duplicates(tup):
	counts = dict()
	for el in tup:
		if el not in counts:
			counts[el] = 0
		counts[el] += 1
	return any(filter(lambda count: count > 1, counts.values()))


# main trainer class

class AISTExtractorMotionTrainer(nn.Module):
	def __init__(
		self,
		motion_extractor: AISTEncoderBiGRUCo,
		music_extractor : AISTEncoderBiGRUCo,
		args,
		training_args,
		dataset_args,
		eval_args,
		model_name = "",
		apply_grad_penalty_every = 4,
		valid_frac = 0.01,
		max_grad_norm = 0.5,
		accelerate_kwargs: dict = dict(),
	):
		super().__init__()

		kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
		initkwargs = InitProcessGroupKwargs(timeout = timedelta(seconds=6000))
		self.accelerator = Accelerator(kwargs_handlers = [kwargs,initkwargs], **accelerate_kwargs)

		transformers.set_seed(42)

		self.args = args
		self.training_args = training_args
		self.dataset_args = dataset_args
		self.dataset_name = dataset_args.dataset_name
		self.model_name = model_name
		self.enable_var_len = dataset_args.var_len
		self.num_train_steps = self.training_args.num_train_iters
		self.num_stages = self.training_args.num_stages
		self.output_dir = Path(self.training_args.output_dir)
		self.output_dir.mkdir(parents = True, exist_ok = True)


		print("self.enable_var_len: ", self.enable_var_len)

		self.stage_steps = list(np.linspace(0,200000, self.num_stages , dtype = np.uint))
		print("stage_steps: " , self.stage_steps )
		self.stage = 0


		

		self.register_buffer('steps', torch.Tensor([0]))
		self.motion_extractor = motion_extractor
		self.music_extractor = music_extractor
		total = sum(p.numel() for p in self.motion_extractor.parameters() if p.requires_grad) + sum(p.numel() for p in self.music_extractor.parameters() if p.requires_grad)
		print("Total training params: %.2fM" % (total / 1e6))

		
	
		
		self.grad_accum_every = self.training_args.gradient_accumulation_steps

		self.loss_fnc = InfoNceLoss(temperature=args.temparature)

		self.optim_motion = get_optimizer(self.motion_extractor.parameters(), lr = self.training_args.learning_rate, wd = self.training_args.weight_decay)
		self.optim_music = get_optimizer(self.music_extractor.parameters(), lr = self.training_args.learning_rate, wd = self.training_args.weight_decay)

		self.lr_scheduler_motion = get_scheduler(
			name = self.training_args.lr_scheduler_type,
			optimizer=self.optim_motion,
			num_warmup_steps=self.training_args.warmup_steps,
			num_training_steps=self.num_train_steps,
		)
  
		self.lr_scheduler_music = get_scheduler(
			name = self.training_args.lr_scheduler_type,
			optimizer=self.optim_music,
			num_warmup_steps=self.training_args.warmup_steps,
			num_training_steps=self.num_train_steps,
		)


		self.max_grad_norm = max_grad_norm


		if self.enable_var_len:
			train_ds = EvaluatorVarLenMotionDataset(split = "train", data_root = self.dataset_args.data_folder , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
			valid_ds = EvaluatorMotionDataset(data_root = self.dataset_args.data_folder , split = "val",window_size=self.args.window_size, init_0 = True)
			self.render_ds = EvaluatorMotionDataset(data_root = self.dataset_args.data_folder , split = "render" ,window_size=self.args.window_size)

		else:

			train_ds = EvaluatorMotionDataset(data_root = self.dataset_args.data_folder,split = "train", window_size=self.args.window_size)
			valid_ds = EvaluatorMotionDataset(data_root = self.dataset_args.data_folder , split = "val",window_size=self.args.window_size)
			self.render_ds = EvaluatorMotionDataset(data_root = self.dataset_args.data_folder , split = "render" ,window_size=self.args.window_size)

		self.print(f'training with training and valid dataset of {len(train_ds)} and  {len(valid_ds)} samples and test of  {len(self.render_ds)}')

		# dataloader
		collate_fn = EvaluatorMotionCollator()

		

		self.dl = DATALoader(train_ds , batch_size = self.training_args.train_bs,collate_fn=collate_fn)
		self.valid_dl = DATALoader(valid_ds , batch_size = self.training_args.eval_bs, shuffle = False,collate_fn=collate_fn)
		self.render_dl = DATALoader(self.render_ds , batch_size = 1,shuffle = False,collate_fn=collate_fn)
		# self.valid_dl = dataset_TM_eval.DATALoader(self.dataset_name, True, self.training_args.eval_bs, self.w_vectorizer, unit_length=4)
		
		# prepare with accelerator

		(
			self.motion_extractor,
			self.music_extractor,
			self.optim_motion,
			self.optim_music,
			self.dl,
			self.valid_dl,
			self.render_dl

		) = self.accelerator.prepare(
			self.motion_extractor,
			self.music_extractor,
			self.optim_motion,
			self.optim_music,
			self.dl,
			self.valid_dl,
			self.render_dl
		)

		self.accelerator.register_for_checkpointing(self.lr_scheduler_motion)
		self.accelerator.register_for_checkpointing(self.lr_scheduler_music)

		self.dl_iter = cycle(self.dl)
		self.valid_dl_iter = cycle(self.valid_dl)

		self.save_model_every = self.training_args.save_steps
		self.log_losses_every = self.training_args.logging_steps
		self.evaluate_every = self.training_args.evaluate_every
		self.wandb_every = self.training_args.wandb_every

		self.apply_grad_penalty_every = apply_grad_penalty_every



		self.best_fid_k = float("inf")
		self.best_fid_g = float("inf")
		self.best_div_k = float("-inf")
		self.best_div_g= float("-inf")
		self.best_beat_align= float("-inf")




		hps = {"num_train_steps": self.num_train_steps,"window size":self.args.window_size,"max_seq_length": self.args.max_seq_length, "learning_rate": self.training_args.learning_rate}
		self.accelerator.init_trackers(f"{self.model_name}", config=hps)    

		if self.is_main:
			wandb.login()
			wandb.init(project=self.model_name)

			


	def print(self, msg):
		self.accelerator.print(msg)
			
	@property
	def device(self):
		return self.accelerator.device

	@property
	def is_distributed(self):
		return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

	@property
	def is_main(self):
		return self.accelerator.is_main_process

	@property
	def is_local_main(self):
		return self.accelerator.is_local_main_process

	def save(self, path ,loss = None ):
		pkg = dict(
			motion_extractor = self.accelerator.get_state_dict(self.motion_extractor),
			music_extractor = self.accelerator.get_state_dict(self.music_extractor),
			optim_music = self.optim_music.state_dict(),
			optim_motion = self.optim_motion.state_dict(),
			steps = self.steps,
			total_loss = self.best_loss if loss is None else loss,
		)
		torch.save(pkg, path)

	@property
	def unwrapped_motion_extractor(self):
		return self.accelerator.unwrap_model(self.motion_extractor)

	@property
	def unwrapped_music_extractor(self):
		return self.accelerator.unwrap_model(self.music_extractor)

	def load(self, path):
		path = Path(path)
		assert path.exists()
		pkg = torch.load(str(path), map_location = 'cpu')

		self.unwrapped_motion_extractor.load_state_dict(pkg['motion_extractor'])
		self.unwrapped_music_extractor.load_state_dict(pkg['music_extractor'])

		self.optim_music.load_state_dict(pkg['optim_music'])
		self.optim_motion.load_state_dict(pkg['optim_motion'])
		
		self.steps = pkg["steps"]
		self.best_loss = pkg["total_loss"]
		
		if self.enable_var_len:
			self.stage = max(np.searchsorted(self.stage_steps , int(self.steps.item())) - 1 , 0)
			print("starting at step: ", self.steps ,"and stage", self.stage)
			self.dl.dataset.set_stage(self.stage)
		else:
			print("starting at step: ", self.steps)





	def train_step(self):

		steps = int(self.steps.item())

		if self.enable_var_len:
	  
	
			if steps in self.stage_steps:
				# self.stage += 1 
				self.stage = self.stage_steps.index(steps)
				# self.stage  = min(self.stage , len(self.stage_steps))
				print("changing to stage" , self.stage)
				self.dl.dataset.set_stage(self.stage)


		apply_grad_penalty = self.apply_grad_penalty_every > 0 and not (steps % self.apply_grad_penalty_every)
		log_losses = self.log_losses_every > 0 and not (steps % self.log_losses_every)

		self.motion_extractor.train()
		self.music_extractor.train()

		# logs

		logs = {}


		for _ in range(self.grad_accum_every):
			batch = next(self.dl_iter)
   
			motion = batch["motion"]
			motion_mask = batch["motion_mask"][:,:,None]
			motion_lengths =  batch["motion_lengths"]

			condition = batch["condition"]
			condition_mask = batch["condition_mask"][:,:,None]
   
			motion_encodings = self.motion_extractor(motion*motion_mask, motion_lengths)
			music_encodings = self.music_extractor(condition *condition_mask ,motion_lengths)
   
			loss = self.loss_fnc(motion_encodings , music_encodings)
   
		
   

			self.accelerator.backward(loss / self.grad_accum_every)

			accum_log(logs, dict(
				loss = loss/self.grad_accum_every,
				avg_max_length = int(max(motion_lengths)) / self.grad_accum_every,
				))

	
		if exists(self.max_grad_norm):
			self.accelerator.clip_grad_norm_(self.motion_extractor.parameters(), self.max_grad_norm)

		self.optim_motion.step()
		self.optim_music.step()
  
		self.lr_scheduler_motion.step()
		self.lr_scheduler_music.step()
		self.optim_motion.zero_grad()
		self.optim_music.zero_grad()



		losses_str = f"{steps}: regressor model total loss: {logs['loss'].float():.3}, average max length: {logs['avg_max_length']}"

		if log_losses:
			self.accelerator.log({
				"total_loss": logs["loss"],
				"average_max_length":logs["avg_max_length"],
			}, step=steps)

		if self.is_main and (steps%self.wandb_every == 0):
			for key , value in logs.items():
				wandb.log({f'train_loss/{key}': value})           

		self.print(losses_str)

		if self.is_main and not (steps % self.save_model_every) and steps>0:
			os.makedirs(os.path.join(self.output_dir , "checkpoints" ) , exist_ok=True)
			model_path = os.path.join(self.output_dir , "checkpoints", f'extractors.{steps}.pt')
			self.save(model_path , logs["loss"])

			if float(loss) < self.best_loss :

				model_path = os.path.join(self.output_dir, f'extractors.pt')
				self.save(model_path)
				self.best_loss = loss

			self.print(f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}')



		if self.is_main and (steps % self.evaluate_every == 0):
			print(f"validation start")
			self.validation_step()
			# print("calculating metrics")
			# self.calculate_metrics(logs['loss'])
			# print("rendering pred outputs")
			# self.sample_render(os.path.join(self.output_dir , "samples"))
			# print("test generating from <bos>")
			# self.sample_render_generative(os.path.join(self.output_dir , "generative") , seq_len=400, num_start_indices  =1)

		# self.accelerator.wait_for_everyone()
				
		# save model
		
		
		self.steps += 1
		return logs

	def validation_step(self):
		self.motion_extractor.eval()
		self.music_extractor.eval()
		val_loss_ae = {}

		with torch.no_grad():

			for batch in tqdm((self.valid_dl), position=0, leave=True):


				motion = batch["motion"]
				motion_mask = batch["motion_mask"][:,:,None]
				motion_lengths =  batch["motion_lengths"]

				condition = batch["condition"]
				condition_mask = batch["condition_mask"][:,:,None]
	
				motion_encodings = self.motion_extractor(motion*motion_mask, motion_lengths)
				music_encodings = self.music_extractor(condition *condition_mask ,motion_lengths)
	
				loss = self.loss_fnc(motion_encodings , music_encodings)

				loss_dict = {
				"total_loss": loss,
				}	
			   
				val_loss_ae.update(loss_dict)

				sums_ae = dict(Counter(val_loss_ae) + Counter(loss_dict))
				means_ae = {k: sums_ae[k] / float((k in val_loss_ae) + (k in loss_dict)) for k in sums_ae}
				val_loss_ae.update(means_ae)



		for key , value in val_loss_ae.items():
			wandb.log({f'val_loss_vqgan/{key}': value})                
		
		print(f"val/total_loss " ,val_loss_ae["total_loss"], )

		self.motion_extractor.train()
		self.music_extractor.train()



	def train(self, resume = False, log_fn = noop):

		self.best_loss = float("inf")
		print(self.output_dir)



		if resume:
			save_path = os.path.join(self.output_dir , "checkpoints")
			chk = sorted(os.listdir(save_path) , key = lambda x: int(x.split('.')[1]))[-1]
			print("resuming from ", os.path.join(save_path, f'{chk}'))
			self.load(os.path.join(save_path , f"{chk}"))

		while self.steps <= self.num_train_steps:
			logs = self.train_step()
			log_fn(logs)

		self.print('training complete')
