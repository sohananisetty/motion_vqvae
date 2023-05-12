from math import sqrt
from pathlib import Path

import os
import numpy as np
import torch
from torch import nn
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
from core.datasets.vqa_motion_dataset import VQMotionDataset,DATALoader,VQVarLenMotionDataset,MotionCollator,VQVarLenMotionDatasetConditional, TransMotionDatasetConditional, MotionCollatorConditional,MotionCollatorConditionalStyle
from tqdm import tqdm
from collections import Counter
import visualize.plot_3d_global as plot_3d
from core.datasets import dataset_TM_eval
from core.models.vqvae import VQMotionModel
from core.models.loss import ReConsLoss
from utils.motion_process import recover_from_ric
from utils.eval_trans import evaluation_vqvae_loss
from core.models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from core.models.motion_regressor import MotionRegressorModel,top_k
from utils.eval_music import evaluate_music_motion_vqvae, evaluate_music_motion_trans,get_target_indices, evaluate_music_motion_generative_style


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

class RegressorMotionTrainerStyle(nn.Module):
	def __init__(
		self,
		trans_model: MotionRegressorModel,
		vqvae_model : VQMotionModel,
		clip_model,
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
		self.trans_model = trans_model
		self.vqvae_model = vqvae_model
		self.clip_model = clip_model
		total = sum(p.numel() for p in self.trans_model.parameters() if p.requires_grad)
		print("Total training params: %.2fM" % (total / 1e6))
  
  

		
	
		
		self.grad_accum_every = self.training_args.gradient_accumulation_steps

		self.loss_fnc = nn.CrossEntropyLoss(ignore_index=self.training_args.pad_index )

		self.optim = get_optimizer(self.trans_model.parameters(), lr = self.training_args.learning_rate, wd = self.training_args.weight_decay)
		
		self.lr_scheduler = get_scheduler(
			name = self.training_args.lr_scheduler_type,
			optimizer=self.optim,
			num_warmup_steps=self.training_args.warmup_steps,
			num_training_steps=self.num_train_steps,
		)

		self.max_grad_norm = max_grad_norm
		# self.w_vectorizer = W ordVectorizer('/srv/scratch/sanisetty3/music_motion/T2M-GPT/glove', 'our_vab')
		# self.eval_wrapper = EvaluatorModelWrapper(eval_args)
		self.best_fid = float("-inf")
  
		if self.training_args.use_mixture:

			if self.enable_var_len:
				hml_train_ds = VQVarLenMotionDatasetConditional("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D", datafolder="joint_indices_max_400", num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
				aist_train_ds = VQVarLenMotionDatasetConditional("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST" , datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
				
				train_ds = torch.utils.data.ConcatDataset([hml_train_ds, aist_train_ds])
				weights_train = [
				[train_ds.__len__() / (hml_train_ds.__len__())] * hml_train_ds.__len__(),
				[train_ds.__len__() / (aist_train_ds.__len__())] * aist_train_ds.__len__(),
				]
	
				weights_train = list(itertools.chain.from_iterable(weights_train))
				sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights_train, num_samples=len(weights_train))
				print("train weights: ", train_ds.__len__() / (hml_train_ds.__len__()), train_ds.__len__() / (aist_train_ds.__len__()))

					
				# hml_valid_ds = TransMotionDatasetConditional("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D" , split = "val",datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder, window_size = 400,force_len=True)
				aist_valid_ds = TransMotionDatasetConditional("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST" , split = "val",datafolder="joint_indices_max_400", window_size = 400,force_len=True)
				# valid_ds = torch.utils.data.ConcatDataset([hml_valid_ds, aist_valid_ds])
				
				# weights_valid = [
				# [valid_ds.__len__() / (hml_valid_ds.__len__())] * hml_valid_ds.__len__(),
				# [valid_ds.__len__() / (aist_valid_ds.__len__())] * aist_valid_ds.__len__(),
				# ]
				# weights_valid = list(itertools.chain.from_iterable(weights_valid))
				# sampler_valid = torch.utils.data.WeightedRandomSampler(weights=weights_valid, num_samples=len(weights_valid))
				# print("valid weights: ", valid_ds.__len__() / (hml_valid_ds.__len__()), valid_ds.__len__() / (aist_valid_ds.__len__()))



				aist_render_ds = TransMotionDatasetConditional("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST", split = "render" , datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder,window_size=400)
				hml_render_ds = TransMotionDatasetConditional("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D" , split = "render" , datafolder="joint_indices_max_400",window_size=400)
				self.render_ds = torch.utils.data.ConcatDataset([hml_render_ds, aist_render_ds])


   
			else:
       
				hml_train_ds = TransMotionDatasetConditional("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D", datafolder="joint_indices_max_400", window_size = 400)
				aist_train_ds = TransMotionDatasetConditional("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST" , datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder , window_size = 400)
				
				train_ds = torch.utils.data.ConcatDataset([hml_train_ds, aist_train_ds])
				weights_train = [
				[train_ds.__len__() / (hml_train_ds.__len__())] * hml_train_ds.__len__(),
				[train_ds.__len__() / (aist_train_ds.__len__())] * aist_train_ds.__len__(),
				]
	
				weights_train = list(itertools.chain.from_iterable(weights_train))
				sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights_train, num_samples=len(weights_train))
				print("train weights: ", train_ds.__len__() / (hml_train_ds.__len__()), train_ds.__len__() / (aist_train_ds.__len__()))

					
				# hml_valid_ds = TransMotionDatasetConditional("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D" , split = "val",datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder, window_size = 400,force_len=True)
				aist_valid_ds = TransMotionDatasetConditional("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST" , split = "val",datafolder="joint_indices_max_400", window_size = 400,force_len=True)
				# valid_ds = torch.utils.data.ConcatDataset([hml_valid_ds, aist_valid_ds])
				
				# weights_valid = [
				# [valid_ds.__len__() / (hml_valid_ds.__len__())] * hml_valid_ds.__len__(),
				# [valid_ds.__len__() / (aist_valid_ds.__len__())] * aist_valid_ds.__len__(),
				# ]
				# weights_valid = list(itertools.chain.from_iterable(weights_valid))
				# sampler_valid = torch.utils.data.WeightedRandomSampler(weights=weights_valid, num_samples=len(weights_valid))
				# print("valid weights: ", valid_ds.__len__() / (hml_valid_ds.__len__()), valid_ds.__len__() / (aist_valid_ds.__len__()))



				aist_render_ds = TransMotionDatasetConditional("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST", split = "render" , datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder,window_size=400)
				hml_render_ds = TransMotionDatasetConditional("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D" , split = "render" , datafolder="joint_indices_max_400",window_size=400)
				self.render_ds = torch.utils.data.ConcatDataset([hml_render_ds, aist_render_ds])
    
    
			collate_fn = MotionCollatorConditionalStyle( clip_model = clip_model, bos = self.training_args.bos_index, pad = self.training_args.pad_index, eos = self.training_args.eos_index)

			self.dl = DATALoader(train_ds , batch_size = self.training_args.train_bs, sampler = sampler_train, shuffle = False, collate_fn=collate_fn)
			self.valid_dl = DATALoader(aist_valid_ds , batch_size = self.training_args.eval_bs, shuffle = False,collate_fn=collate_fn)
			self.render_dl = DATALoader(self.render_ds , batch_size = 1, shuffle = False,collate_fn=collate_fn)
			


				
		else:
			print("no mixture, enable_var_len ", self.enable_var_len)
			if self.enable_var_len:
				train_ds = VQVarLenMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "train",datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
				valid_ds = TransMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "val",datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder , window_size = 400)
				self.render_ds = TransMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "render" , datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder,window_size=400,force_len=True)

				# valid_ds = VQVarLenMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "val" ,datafolder="joint_indices_max_400", num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
				# self.render_ds = VQVarLenMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "render" ,datafolder="joint_indices_max_400", num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
			else:

				train_ds = TransMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder,split = "train", datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder , window_size=self.args.window_size)
				valid_ds = TransMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "val", datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder,window_size=self.args.window_size)
				self.render_ds = TransMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "render" , datafolder="joint_indices_max_400", musicfolder = self.dataset_args.music_folder,window_size=self.args.window_size,force_len=True)

			self.print(f'training with training and valid dataset of {len(train_ds)} and  {len(valid_ds)} samples and test of  {len(self.render_ds)}')


			# dataloader
			collate_fn = MotionCollatorConditionalStyle(clip_model = self.clip_model , bos = self.training_args.bos_index, pad = self.training_args.pad_index, eos = self.training_args.eos_index)

			

			self.dl = DATALoader(train_ds , batch_size = self.training_args.train_bs,collate_fn=collate_fn)
			self.valid_dl = DATALoader(valid_ds , batch_size = self.training_args.eval_bs, shuffle = False,collate_fn=collate_fn)
			self.render_dl = DATALoader(self.render_ds , batch_size = 1,shuffle = False,collate_fn=collate_fn)
		
		# prepare with accelerator

		(
			self.trans_model,
			self.vqvae_model,
   			self.clip_model,
			self.optim,
			self.dl,
			self.valid_dl,
			self.render_dl

		) = self.accelerator.prepare(
			self.trans_model,
			self.vqvae_model,
			self.clip_model,
			self.optim,
			self.dl,
			self.valid_dl,
			self.render_dl
		)

		self.accelerator.register_for_checkpointing(self.lr_scheduler)

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
			model = self.accelerator.get_state_dict(self.trans_model),
			optim = self.optim.state_dict(),
			steps = self.steps,
			total_loss = self.best_loss if loss is None else loss,
		)
		torch.save(pkg, path)

	@property
	def unwrapped_trans_model(self):
		return self.accelerator.unwrap_model(self.trans_model)

	def load(self, path):
		path = Path(path)
		assert path.exists()
		pkg = torch.load(str(path), map_location = 'cpu')

		self.unwrapped_trans_model.load_state_dict(pkg['model'])

		self.optim.load_state_dict(pkg['optim'])
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

		self.trans_model.train()

		# logs

		logs = {}


		for _ in range(self.grad_accum_every):
			batch = next(self.dl_iter)
			right_num = 0


			inp, target = batch["motion"][:, :-1], batch["motion"][:, 1:]
			lengths = batch["motion_lengths"]
			total_tokens = int(sum(lengths))
   
			# print(batch.keys() , batch["style"].shape)

			logits = self.trans_model(motion = inp , mask = batch["motion_mask"][:,:-1]  , \
				context = batch["condition"], context_mask = batch["condition_mask"], style_context = batch["style"])	
					
			loss = self.loss_fnc(logits.contiguous().view(-1, logits.shape[-1]), target.contiguous().view(-1))

			for i in range(self.training_args.train_bs):
			# if not self.enable_var_len:

				probs = torch.softmax(logits[i][:int(lengths[i])], dim=-1)

				if self.args.sample_max:
					_, cls_pred_index = torch.max(probs, dim=-1)

				else:
					dist = torch.distributions.Categorical(probs)
					cls_pred_index = dist.sample()

				right_num += (cls_pred_index.flatten(0) == target[i][:int(lengths[i])].flatten(0)).sum().item()

				# right_num += (cls_pred_index.flatten(0) == target.flatten(0)).sum().item()

			
			self.accelerator.backward(loss / self.grad_accum_every)

			accum_log(logs, dict(
				loss = loss/self.grad_accum_every,
				avg_max_length = int(max(lengths)) / self.grad_accum_every,
				accuracy = right_num/((total_tokens) * self.grad_accum_every),
				))

	
		if exists(self.max_grad_norm):
			self.accelerator.clip_grad_norm_(self.trans_model.parameters(), self.max_grad_norm)

		self.optim.step()
		self.lr_scheduler.step()
		self.optim.zero_grad()



		losses_str = f"{steps}: regressor model total loss: {logs['loss'].float():.3} , Accuracy: {logs['accuracy']:.3}, average max length: {logs['avg_max_length']}"

		if log_losses:
			self.accelerator.log({
				"total_loss": logs["loss"],
				"average_max_length":logs["avg_max_length"],
				"accuracy" : logs["accuracy"]
			}, step=steps)

		if self.is_main and (steps%self.wandb_every == 0):
			for key , value in logs.items():
				wandb.log({f'train_loss/{key}': value})           

		self.print(losses_str)

		if self.is_main and not (steps % self.save_model_every) and steps>0:
			os.makedirs(os.path.join(self.output_dir , "checkpoints" ) , exist_ok=True)
			model_path = os.path.join(self.output_dir , "checkpoints", f'trans_motion.{steps}.pt')
			self.save(model_path , logs["loss"])

			if float(loss) < self.best_loss :

				model_path = os.path.join(self.output_dir, f'trans_motion.pt')
				self.save(model_path)
				self.best_loss = loss

			self.print(f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}')



		if self.is_main and (steps % self.evaluate_every == 0):
			print(f"validation start")
			self.validation_step()
			print("calculating metrics")
			self.calculate_metrics(logs['loss'])
			print("rendering pred outputs")
			self.sample_render(os.path.join(self.output_dir , "samples"))
			print("test generating from <bos>")
			self.sample_render_generative(os.path.join(self.output_dir , "generative") , seq_len=100, num_start_indices  =1)

		# self.accelerator.wait_for_everyone()
				
		# save model
		
		
		self.steps += 1
		return logs
	
	def calculate_metrics(self ,loss):
		best_fid_k, best_fid_g,best_div_k,best_div_g,best_beat_align = evaluate_music_motion_trans(self.valid_dl,self.vqvae_model , self.trans_model)

		if (best_fid_k+best_fid_g)/2 < (self.best_fid_k + self.best_fid_g)/2:
			model_path = os.path.join(self.output_dir, f'trans_motion_best_fid.pt')
			self.save(model_path , loss=loss)

		wandb.log({f'best_fid_k': best_fid_k})  
		wandb.log({f'best_fid_g': best_fid_g})  
		wandb.log({f'best_div_k': best_div_k})  
		wandb.log({f'best_div_g': best_div_g})  
		wandb.log({f'best_beat_align': best_beat_align})  


		self.best_fid_k, self.best_fid_g, self.best_div_k, self.best_div_g, self.best_beat_align = \
			best_fid_k, best_fid_g, best_div_k, best_div_g, best_beat_align

		

	def validation_step(self):
		self.trans_model.eval()
		val_loss_ae = {}

		


		with torch.no_grad():

			for batch in tqdm((self.valid_dl), position=0, leave=True):


				right_num = 0
				inp, target = batch["motion"][:, :-1], batch["motion"][:, 1:]
				lengths = batch["motion_lengths"]
				total_tokens = int(sum(lengths))

				logits = self.trans_model(motion = inp , mask = batch["motion_mask"][:,:-1]  , \
					context = batch["condition"], context_mask = batch["condition_mask"], style_context = batch["style"])	
						
				loss = self.loss_fnc(logits.contiguous().view(-1, logits.shape[-1]), target.contiguous().view(-1))
				
				for i in range(inp.shape[0]):
					# probs = torch.softmax(logits, dim=-1)

					# if self.args.sample_max:
					# 	_, cls_pred_index = torch.max(probs, dim=-1)

					# else:
					# 	dist = torch.distributions.Categorical(probs)
					# 	cls_pred_index = dist.sample()
					# right_num += (cls_pred_index.flatten(0) == target.flatten(0)).sum().item()


					probs = torch.softmax(logits[i][:int(lengths[i])], dim=-1)

					if self.args.sample_max:
						_, cls_pred_index = torch.max(probs, dim=-1)

					else:
						dist = torch.distributions.Categorical(probs)
						cls_pred_index = dist.sample()

					right_num += (cls_pred_index.flatten(0) == target[i][:int(lengths[i])].flatten(0)).sum().item()


				loss_dict = {
				"total_loss": loss,
				"accuracy":right_num/(total_tokens)
				}	
			   
				val_loss_ae.update(loss_dict)

				sums_ae = dict(Counter(val_loss_ae) + Counter(loss_dict))
				means_ae = {k: sums_ae[k] / float((k in val_loss_ae) + (k in loss_dict)) for k in sums_ae}
				val_loss_ae.update(means_ae)



		for key , value in val_loss_ae.items():
			wandb.log({f'val_loss_vqgan/{key}': value})                
		
		print(f"val/total_loss " ,val_loss_ae["total_loss"], )

		self.trans_model.train()


	 
	def sample_render(self , save_path):

		save_file = os.path.join(save_path , f"{int(self.steps.item())}")
		os.makedirs(save_file , exist_ok=True)

		render_mean = self.render_ds.mean 
		render_std = self.render_ds.std 

		
		# assert self.render_dl.batch_size == 1 , "Batch size for rendering should be 1!"

		self.vqvae_model.eval()
		self.trans_model.eval()
		print(f"render start")
		with torch.no_grad():
			for batch in tqdm(self.render_dl):


				gt_motion = batch["motion"]
				name = str(batch["names"][0])


				

				gen_motion_indices , gt_motion_indices = get_target_indices(batch , self.trans_model)

				_ , pred_motion = self.vqvae_model.module.decode(gen_motion_indices.cuda())
				_ , gt_motion = self.vqvae_model.module.decode(gt_motion_indices.cuda())


				gt_motion_xyz = recover_from_ric(gt_motion.cpu().float()*render_std+render_mean, 22)
				gt_motion_xyz = gt_motion_xyz.reshape(gt_motion.shape[0],-1, 22, 3)

				pred_motion_xyz = recover_from_ric(pred_motion.cpu().float()*render_std+render_mean, 22)
				pred_motion_xyz = pred_motion_xyz.reshape(pred_motion.shape[0],-1, 22, 3)

				

				gt_pose_vis = plot_3d.draw_to_batch(gt_motion_xyz.numpy(),None, [os.path.join(save_file,name + "_gt.gif")])
				pred_pose_vis = plot_3d.draw_to_batch(pred_motion_xyz.numpy(),None, [os.path.join(save_file,name + "_pred.gif")])

				# render(pred_motion_xyz, outdir=save_path, step=self.steps, name=f"{name}", pred=True)
				# render(gt_motion_xyz, outdir=save_path, step=self.steps, name=f"{name}", pred=False)

			

		self.trans_model.train()

	
	def process_gen_output(self , gen_motion_indices , seq_len):
		eos_index = (gen_motion_indices == self.training_args.eos_index).nonzero().flatten().tolist()
		# print(eos_index)
		pad_index = (gen_motion_indices == self.training_args.pad_index).nonzero().flatten().tolist()
		# print(pad_index)
		bos_index = (gen_motion_indices == self.training_args.bos_index).nonzero().flatten().tolist()
		# print(bos_index)
		stop_index = min([*eos_index , *pad_index , *bos_index, seq_len])

		gen_motion_indices_ = gen_motion_indices[:int(stop_index)]

		return gen_motion_indices_




	def sample_render_generative(self , save_path , seq_len = 400 , num_start_indices = 1):

		save_file = os.path.join(save_path , f"{int(self.steps.item())}")
		os.makedirs(save_file , exist_ok=True)

		self.trans_model.eval()
		print(f"render start")
		with torch.no_grad():
			for batch in tqdm(self.render_dl):

				gt_motion_indices = batch["motion"][:,:seq_len]
				if seq_len == -1:
					seq_len = gt_motion_indices.shape[1]


				name = batch["names"]
				start_tokens = gt_motion_indices[:,:num_start_indices]

				gen_motion_indices = self.trans_model.module.generate(start_tokens = start_tokens, seq_len=seq_len , context = batch["condition"], context_mask = batch["condition_mask"], style_context = batch["style"])
				
				gen_motion_indices_ = self.process_gen_output(gen_motion_indices , seq_len)

				gt_motion_indices_ = gt_motion_indices[gt_motion_indices<self.training_args.bos_index]

				_ , pred_motion = self.vqvae_model.module.decode(gen_motion_indices_.reshape(gt_motion_indices.shape[0],-1))
				_ , gt_motion = self.vqvae_model.module.decode(gt_motion_indices_.reshape(gt_motion_indices.shape[0],-1))


				try:

					gt_motion_xyz = recover_from_ric(gt_motion.cpu().float()*self.render_ds.std+self.render_ds.mean, 22)
					gt_motion_xyz = gt_motion_xyz.reshape(gt_motion.shape[0],-1, 22, 3)

					pred_motion_xyz = recover_from_ric(pred_motion.cpu().float()*self.render_ds.std+self.render_ds.mean, 22)
					pred_motion_xyz = pred_motion_xyz.reshape(pred_motion.shape[0],-1, 22, 3)

					

					gt_pose_vis = plot_3d.draw_to_batch(gt_motion_xyz.numpy(),None, [os.path.join(save_file,name[0] + "_gt.gif")])
					pred_pose_vis = plot_3d.draw_to_batch(pred_motion_xyz.numpy(),None, [os.path.join(save_file,name[0] + "_pred.gif")])
				except:
					print("ERROR cant render motion gt_motion shape", gt_motion.shape , "pred_motion shape: ", pred_motion.shape)
					continue

		self.trans_model.train()

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
