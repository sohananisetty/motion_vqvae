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
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import DistributedType
import wandb
import transformers

from core.optimizer import get_optimizer
from render_final import render
from core.datasets.vqa_motion_dataset import VQMotionDataset,DATALoader,VQVarLenMotionDataset,MotionCollator
from tqdm import tqdm
from collections import Counter
import visualize.plot_3d_global as plot_3d
from core.datasets import dataset_TM_eval
from core.models.vqvae import VQMotionModel
from core.models.loss import ReConsLoss
from utils.motion_process import recover_from_ric
from utils.eval_trans import evaluation_vqvae_loss,evaluation_vqvae
from core.models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer



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

class VQVAEMotionTrainer(nn.Module):
	def __init__(
		self,
		vqvae_model: VQMotionModel,
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
		self.accelerator = Accelerator(kwargs_handlers = [kwargs], **accelerate_kwargs)

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

		self.stage_steps = list(np.linspace(200000,self.num_train_steps, self.num_stages , dtype = np.uint))
		print("stage_steps: " , self.stage_steps )
		self.stage = 0


		

		self.register_buffer('steps', torch.Tensor([0]))
		self.vqvae_model = vqvae_model
		total = sum(p.numel() for p in self.vqvae_model.parameters() if p.requires_grad)
		print("Total training params: %.2fM" % (total / 1e6))

		if args.freeze_model:
			print("freezing encoder and decoder")
			for name, param in self.vqvae_model.motionEncoder.named_parameters():
				param.requires_grad = False
			for name, param in self.vqvae_model.motionDecoder.named_parameters():
				param.requires_grad = False

			total = sum(p.numel() for p in self.vqvae_model.parameters() if p.requires_grad)
			print("Total training params: %.2fM" % (total / 1e6))
			
		self.grad_accum_every = self.training_args.gradient_accumulation_steps

		self.loss_fnc = ReConsLoss(self.args.recons_loss, self.args.nb_joints)

		self.optim = get_optimizer(self.vqvae_model.parameters(), lr = self.training_args.learning_rate, wd = self.training_args.weight_decay)
		
		self.lr_scheduler = get_scheduler(
			name = self.training_args.lr_scheduler_type,
			optimizer=self.optim,
			num_warmup_steps=self.training_args.warmup_steps,
			num_training_steps=self.num_train_steps,
		)

		self.max_grad_norm = max_grad_norm
		

		if self.training_args.use_mixture:

			if self.enable_var_len:
				
				hml_train_ds = VQVarLenMotionDataset("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D" , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
				aist_train_ds = VQVarLenMotionDataset("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST" , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)

				train_ds = torch.utils.data.ConcatDataset([hml_train_ds, aist_train_ds])
				weights_train = [
				[train_ds.__len__() / (hml_train_ds.__len__())] * hml_train_ds.__len__(),
				[train_ds.__len__() / (aist_train_ds.__len__())] * aist_train_ds.__len__(),
				]

				weights_train = list(itertools.chain.from_iterable(weights_train))
				sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights_train, num_samples=len(weights_train))
				print("train weights: ", train_ds.__len__() / (hml_train_ds.__len__()), train_ds.__len__() / (aist_train_ds.__len__()))

				# hml_valid_ds = VQVarLenMotionDataset("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D" , split = "val",min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
				# aist_valid_ds = VQVarLenMotionDataset("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST" , split = "val",min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
				# valid_ds = torch.utils.data.ConcatDataset([hml_valid_ds, aist_valid_ds])
				
				# weights_valid = [
				# [valid_ds.__len__() / (hml_valid_ds.__len__())] * hml_valid_ds.__len__(),
				# [valid_ds.__len__() / (aist_valid_ds.__len__())] * aist_valid_ds.__len__(),
				# ]

				# weights_valid = list(itertools.chain.from_iterable(weights_valid))
				# sampler_valid = torch.utils.data.WeightedRandomSampler(weights=weights_valid, num_samples=len(weights_valid))
				# print("valid weights: ", valid_ds.__len__() / (hml_valid_ds.__len__()), valid_ds.__len__() / (aist_valid_ds.__len__()))

				hml_render_ds = VQVarLenMotionDataset("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D", split = "render" , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
				aist_render_ds = VQVarLenMotionDataset("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST" , split = "render" , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
				self.render_ds = torch.utils.data.ConcatDataset([hml_render_ds, aist_render_ds])

			else:

				hml_train_ds = VQMotionDataset("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D" , window_size = self.args.window_size)
				# hml_valid_ds = VQMotionDataset("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D" , split = "val", window_size = self.args.window_size)
				hml_render_ds = VQMotionDataset("t2m", data_root = "/srv/scratch/sanisetty3/music_motion/HumanML3D/HumanML3D", split = "render" , window_size = self.args.window_size)
			
				aist_train_ds = VQMotionDataset("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST" , window_size = self.args.window_size)
				valid_ds = VQMotionDataset("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST" , split = "val", window_size = self.args.window_size)
				aist_render_ds = VQMotionDataset("aist", data_root = "/srv/scratch/sanisetty3/music_motion/AIST" , split = "render" , window_size = self.args.window_size)
			

				train_ds = torch.utils.data.ConcatDataset([hml_train_ds, aist_train_ds])
				weights_train = [
				[train_ds.__len__() / (hml_train_ds.__len__())] * hml_train_ds.__len__(),
				[train_ds.__len__() / (aist_train_ds.__len__())] * aist_train_ds.__len__(),
				]

				weights_train = list(itertools.chain.from_iterable(weights_train))
				sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights_train, num_samples=len(weights_train))
				print("train weights: ", train_ds.__len__() / (hml_train_ds.__len__()), train_ds.__len__() / (aist_train_ds.__len__()))


				# valid_ds = torch.utils.data.ConcatDataset([hml_valid_ds, aist_valid_ds])
				# weights_valid = [
				# [valid_ds.__len__() / (hml_valid_ds.__len__())] * hml_valid_ds.__len__(),
				# [valid_ds.__len__() / (aist_valid_ds.__len__())] * aist_valid_ds.__len__(),
				# ]

				# weights_valid = list(itertools.chain.from_iterable(weights_valid))
				# sampler_valid = torch.utils.data.WeightedRandomSampler(weights=weights_valid, num_samples=len(weights_valid))
				# print("valid weights: ", valid_ds.__len__() / (hml_valid_ds.__len__()), valid_ds.__len__() / (aist_valid_ds.__len__()))

				self.render_ds = torch.utils.data.ConcatDataset([hml_render_ds, aist_render_ds])
				


			self.print(f'training with training and valid dataset of {len(train_ds)} and  {0} samples and test of  {len(self.render_ds)}')

			# dataloader
			collate_fn = MotionCollator() if self.enable_var_len else None

			

			self.dl = DATALoader(train_ds , batch_size = self.training_args.train_bs, sampler = sampler_train, shuffle = False, collate_fn=collate_fn)
			# self.valid_dl = DATALoader(valid_ds , batch_size = self.training_args.eval_bs, shuffle = False,collate_fn=collate_fn)
			self.render_dl = DATALoader(self.render_ds , batch_size = 1, shuffle = False,collate_fn=collate_fn)
			
			

		else:
			if self.enable_var_len:
				train_ds = VQVarLenMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
				valid_ds = VQMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "val", max_length_seconds = self.args.max_length_seconds)
				self.render_ds = VQVarLenMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "render" , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
			
			
			else:

				train_ds = VQMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder,max_length_seconds = self.args.max_length_seconds)
				valid_ds = VQMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "val", max_length_seconds = self.args.max_length_seconds)
				self.render_ds = VQMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "render" , max_length_seconds = self.args.max_length_seconds)



			self.print(f'training with training and valid dataset of {len(train_ds)} and  {len(valid_ds)} samples and test of  {len(self.render_ds)}')

			# dataloader
			collate_fn = MotionCollator() if self.enable_var_len else None

			

			self.dl = DATALoader(train_ds , batch_size = self.training_args.train_bs,collate_fn=collate_fn)
			self.valid_dl = DATALoader(valid_ds , batch_size = self.training_args.eval_bs, shuffle = False,collate_fn=None)
			self.render_dl = DATALoader(self.render_ds , batch_size = 1,shuffle = False,collate_fn=collate_fn)
		
		if self.is_main:
			self.w_vectorizer = WordVectorizer('/srv/scratch/sanisetty3/music_motion/T2M-GPT/glove', 'our_vab')
			self.eval_wrapper = EvaluatorModelWrapper(eval_args)
			self.tm_eval = dataset_TM_eval.DATALoader(self.dataset_args.dataset_name, True, self.training_args.eval_bs, self.w_vectorizer, unit_length=4)
			
		# prepare with accelerator

		(
			self.vqvae_model,
			self.optim,
			self.dl,
			# self.valid_dl,
			self.render_dl,

		) = self.accelerator.prepare(
			self.vqvae_model,
			self.optim,
			self.dl,
			# self.valid_dl,
			self.render_dl,
			
		)

		self.accelerator.register_for_checkpointing(self.lr_scheduler)

		self.dl_iter = cycle(self.dl)
		# self.valid_dl_iter = cycle(self.valid_dl)

		self.save_model_every = self.training_args.save_steps
		self.log_losses_every = self.training_args.logging_steps
		self.evaluate_every = self.training_args.evaluate_every
		self.calc_metrics_every = self.training_args.evaluate_every
		self.wandb_every = self.training_args.wandb_every

		self.apply_grad_penalty_every = apply_grad_penalty_every




		hps = {"num_train_steps": self.num_train_steps, "max_seq_length": self.args.max_seq_length, "learning_rate": self.training_args.learning_rate}
		self.accelerator.init_trackers(f"{self.model_name}", config=hps)  


		self.best_fid = float("inf")
		self.best_div = float("-inf")
		self.best_top1 = float("-inf")
		self.best_top2= float("-inf")
		self.best_top3= float("-inf")
		self.best_matching  = float("inf")

		if self.is_main:
			wandb.login()
			wandb.init(project="vqvae_768_768_vl_mix")

		    


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

	def save(self, path, loss = None):
		pkg = dict(
			model = self.accelerator.get_state_dict(self.vqvae_model),
			optim = self.optim.state_dict(),
			steps = self.steps,
			total_loss = self.best_loss if loss is None else loss,
		)
		torch.save(pkg, path)

	@property
	def unwrapped_vqvae_model(self):
		return self.accelerator.unwrap_model(self.vqvae_model)

	def load(self, path):
		path = Path(path)
		assert path.exists()
		pkg = torch.load(str(path), map_location = 'cpu')

		self.unwrapped_vqvae_model.load_state_dict(pkg['model'])

		self.optim.load_state_dict(pkg['optim'])
		self.steps = pkg["steps"]
		self.best_loss = pkg["total_loss"]
		# print("Loading at stage" , np.searchsorted(self.stage_steps , int(self.steps.item())) - 1)
		self.stage = max(np.searchsorted(self.stage_steps , int(self.steps.item())) - 1 , 0)
		print("starting at step: ", self.steps ,"and stage", self.stage)

		if not self.training_args.use_mixture:
			self.dl.dataset.set_stage(self.stage)
		else:
			self.dl.dataset.datasets[0].set_stage(self.stage)
			self.dl.dataset.datasets[1].set_stage(self.stage)




	def train_step(self):

		steps = int(self.steps.item())

		if steps in self.stage_steps:

			if not self.training_args.use_mixture:
				self.stage = self.stage_steps.index(steps)
				print("changing to stage" , self.stage)
				self.dl.dataset.set_stage(self.stage)
			else:
				self.stage = self.stage_steps.index(steps)
				print("changing to stage" , self.stage)
				self.dl.dataset.datasets[0].set_stage(self.stage)
				self.dl.dataset.datasets[1].set_stage(self.stage)



		apply_grad_penalty = self.apply_grad_penalty_every > 0 and not (steps % self.apply_grad_penalty_every)
		log_losses = self.log_losses_every > 0 and not (steps % self.log_losses_every)

		self.vqvae_model.train()

		# logs

		logs = {}


		for _ in range(self.grad_accum_every):
			batch = next(self.dl_iter)

			gt_motion = batch["motion"]

			if self.enable_var_len is False:

				pred_motion , indices, commit_loss = self.vqvae_model(gt_motion)				
				loss_motion = self.loss_fnc(pred_motion, gt_motion)
				loss_vel = self.loss_fnc.forward_vel(pred_motion, gt_motion)
				loss = loss_motion + self.args.commit * commit_loss + self.args.loss_vel * loss_vel

			else:
				mask = batch["motion_mask"]
				lengths = batch["motion_lengths"]

				pred_motion , indices, commit_loss = self.vqvae_model(gt_motion , mask)				
				loss_motion = self.loss_fnc(pred_motion, gt_motion , mask)
				loss_vel = self.loss_fnc.forward_vel(pred_motion, gt_motion , mask)
				loss = loss_motion + self.args.commit * commit_loss + self.args.loss_vel * loss_vel



			# print(loss,loss.shape)
			
			self.accelerator.backward(loss / self.grad_accum_every)

			accum_log(logs, dict(
				loss = loss.detach().cpu()/self.grad_accum_every,
				loss_motion = loss_motion.detach().cpu()/ self.grad_accum_every,
				loss_vel = loss_vel.detach().cpu() / self.grad_accum_every,
				commit_loss = commit_loss.detach().cpu() / self.grad_accum_every,
				avg_max_length = int(max(lengths)) / self.grad_accum_every
				))

	
		if exists(self.max_grad_norm):
			self.accelerator.clip_grad_norm_(self.vqvae_model.parameters(), self.max_grad_norm)

		self.optim.step()
		self.lr_scheduler.step()
		self.optim.zero_grad()


		# build pretty printed losses

		losses_str = f"{steps}: vqvae model total loss: {logs['loss'].float():.3} reconstruction loss: {logs['loss_motion'].float():.3} loss_vel: {logs['loss_vel'].float():.3} commitment loss: {logs['commit_loss'].float():.3} average max length: {logs['avg_max_length']}"

		if log_losses:
			self.accelerator.log({
				"total_loss": logs["loss"],
				"loss_motion":logs["loss_motion"],
				"loss_vel": logs["loss_vel"],
				"commit_loss" :logs["commit_loss"],
				"average_max_length":logs["avg_max_length"],

			}, step=steps)

		# log
		if self.is_main and (steps%self.wandb_every == 0):
			for key , value in logs.items():
				wandb.log({f'train_loss/{key}': value})           

		self.print(losses_str)

		

		if self.is_main and (steps % self.evaluate_every == 0):
			# self.validation_step()
			self.sample_render(os.path.join(self.output_dir , "samples"))

		if self.is_main and (steps % self.calc_metrics_every == 0):
			self.calculate_metrics(steps, logs["loss"])

			
			
				
		# save model
		
		if self.is_main and not (steps % self.save_model_every) and steps>0:
			os.makedirs(os.path.join(self.output_dir , "checkpoints" ) , exist_ok=True)
			model_path = os.path.join(self.output_dir , "checkpoints", f'vqvae_motion.{steps}.pt')
			self.save(model_path)

			if float(logs["loss"]) <= self.best_loss:

				model_path = os.path.join(self.output_dir, f'vqvae_motion.pt')
				self.save(model_path)
				self.best_loss = logs["loss"]

			self.print(f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}')


		self.steps += 1
		return logs
	

	def calculate_metrics(self , steps , loss):
		best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_vqvae_loss(
				best_fid = self.best_fid, 
				best_div= self.best_div, 
				best_top1= self.best_top1, 
				best_top2= self.best_top2, 
				best_top3= self.best_top3, 
				best_matching= self.best_matching,
				val_loader = self.tm_eval, 
				net= self.vqvae_model,
				nb_iter= steps, 
				eval_wrapper = self.eval_wrapper,
				save = False,

				)
		if best_fid < self.best_fid:
			model_path = os.path.join(self.output_dir, f'vqvae_motion_best_fid.pt')
			self.save(model_path , loss=loss)

		wandb.log({f'best_fid': best_fid})  
		wandb.log({f'best_div': best_div})  
		wandb.log({f'best_top1': best_top1})  
		wandb.log({f'best_top2': best_top2})  
		wandb.log({f'best_top3': best_top3})  
		wandb.log({f'best_matching': best_matching})  


		self.best_fid, self.best_iter, self.best_div, self.best_top1, self.best_top2, self.best_top3, self.best_matching = best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching

		
		

	def validation_step(self):
		self.vqvae_model.eval()
		val_loss_ae = {}
		all_loss = 0.

		print(f"validation start")

		with torch.no_grad():

			for batch in tqdm((self.valid_dl), position=0, leave=True):


				gt_motion = batch["motion"]

				pred_motion , indices, commit_loss = self.vqvae_model(gt_motion)
				loss_motion = self.loss_fnc(pred_motion, gt_motion)
				loss_vel = self.loss_fnc.forward_vel(pred_motion, gt_motion)

				loss = loss_motion + self.args.commit * commit_loss + self.args.loss_vel * loss_vel

				loss_dict = {
				"total_loss": loss.detach().cpu(),
				"loss_motion":loss_motion.detach().cpu(),
				"loss_vel": loss_vel.detach().cpu(),
				"commit_loss" :commit_loss.detach().cpu()
				}	
			   
				val_loss_ae.update(loss_dict)

				sums_ae = dict(Counter(val_loss_ae) + Counter(loss_dict))
				means_ae = {k: sums_ae[k] / float((k in val_loss_ae) + (k in loss_dict)) for k in sums_ae}
				val_loss_ae.update(means_ae)



		for key , value in val_loss_ae.items():
			wandb.log({f'val_loss_vqgan/{key}': value})                
		
		print("val/rec_loss" ,val_loss_ae["loss_motion"], )
		print(f"val/total_loss " ,val_loss_ae["total_loss"], )

		self.vqvae_model.train()
	
	def sample_render(self , save_path):

		save_file = os.path.join(save_path , f"{int(self.steps.item())}")
		os.makedirs(save_file , exist_ok=True)

		
		# assert self.render_dl.batch_size == 1 , "Batch size for rendering should be 1!"

		self.vqvae_model.eval()
		print(f"render start")
		with torch.no_grad():
			for batch in tqdm(self.render_dl):


				gt_motion = batch["motion"]
				name = str(batch["names"][0])

				if self.training_args.use_mixture:

					render_mean = self.render_ds.datasets[0].mean if len(name.split("_")) <2 else self.render_ds.datasets[1].mean
					render_std = self.render_ds.datasets[0].std if len(name.split("_")) <2 else self.render_ds.datasets[1].std
				else:
					render_mean = self.render_ds.mean 
					render_std = self.render_ds.std 



				motion_len = int(batch.get("motion_lengths" , [gt_motion.shape[1]])[0])

				gt_motion = gt_motion[:,:motion_len,:]

				pred_motion , _, _ = self.vqvae_model(gt_motion)

				gt_motion_xyz = recover_from_ric(gt_motion.cpu().float()*render_std+render_mean, 22)
				gt_motion_xyz = gt_motion_xyz.reshape(gt_motion.shape[0],-1, 22, 3)

				pred_motion_xyz = recover_from_ric(pred_motion.cpu().float()*render_std+render_mean, 22)
				pred_motion_xyz = pred_motion_xyz.reshape(pred_motion.shape[0],-1, 22, 3)

				

				gt_pose_vis = plot_3d.draw_to_batch(gt_motion_xyz.numpy(),None, [os.path.join(save_file,name + "_gt.gif")])
				pred_pose_vis = plot_3d.draw_to_batch(pred_motion_xyz.numpy(),None, [os.path.join(save_file,name + "_pred.gif")])

				# render(pred_motion_xyz, outdir=save_path, step=self.steps, name=f"{name}", pred=True)
				# render(gt_motion_xyz, outdir=save_path, step=self.steps, name=f"{name}", pred=False)

			

		self.vqvae_model.train()


	def train(self, resume = False, log_fn = noop):

		self.best_loss = float("inf")
		print(self.output_dir)



		if resume:
			save_path = os.path.join(self.output_dir , "checkpoints")
			chk = sorted(os.listdir(save_path) , key = lambda x: int(x.split('.')[1]))[-1]
			print("resuming from ", os.path.join(save_path, f'{chk}'))
			self.load(os.path.join(save_path ,  f'{chk}'))

		while self.steps < self.num_train_steps:
			logs = self.train_step()
			log_fn(logs)

		self.print('training complete')
