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
from core.datasets.vqa_motion_dataset import VQMotionDataset,DATALoader,VQVarLenMotionDataset,MotionCollator,VQVarLenMotionDatasetConditional, TransMotionDatasetConditional, MotionCollatorConditional
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

class RegressorMotionTrainer(nn.Module):
	def __init__(
		self,
		trans_model: MotionRegressorModel,
		vqvae_model : VQMotionModel,
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

		self.stage_steps = list(np.linspace(0,self.num_train_steps, self.num_stages , dtype = np.uint))
		print("stage_steps: " , self.stage_steps )
		self.stage = 0


		

		self.register_buffer('steps', torch.Tensor([0]))
		self.trans_model = trans_model
		self.vqvae_model = vqvae_model
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

		if self.enable_var_len:
			train_ds = VQVarLenMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
			valid_ds = TransMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "val", max_length_seconds = self.args.max_length_seconds)
			self.render_ds = VQVarLenMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "render" , num_stages=self.num_stages ,min_length_seconds=self.args.min_length_seconds, max_length_seconds=self.args.max_length_seconds)
		else:

			train_ds = TransMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder,split = "train", max_length_seconds = self.args.max_length_seconds , window_size=self.args.window_size)
			valid_ds = TransMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "val", max_length_seconds = self.args.max_length_seconds,window_size=self.args.window_size)
			self.render_ds = TransMotionDatasetConditional(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "render" , max_length_seconds = self.args.max_length_seconds,window_size=self.args.window_size)

		self.print(f'training with training and valid dataset of {len(train_ds)} and  {len(valid_ds)} samples and test of  {len(self.render_ds)}')

		# dataloader
		collate_fn = MotionCollatorConditional(dataset_name = self.dataset_args.dataset_name , bos = self.training_args.bos_index, pad = self.training_args.pad_index, eos = self.training_args.eos_index)

		

		self.dl = DATALoader(train_ds , batch_size = self.training_args.train_bs,collate_fn=collate_fn)
		self.valid_dl = DATALoader(valid_ds , batch_size = self.training_args.eval_bs, shuffle = False,collate_fn=collate_fn)
		self.render_dl = DATALoader(self.render_ds , batch_size = 1,shuffle = False,collate_fn=collate_fn)
		# self.valid_dl = dataset_TM_eval.DATALoader(self.dataset_name, True, self.training_args.eval_bs, self.w_vectorizer, unit_length=4)
		
		# prepare with accelerator

		(
			self.trans_model,
			self.vqvae_model,
			self.optim,
			self.dl,
			self.valid_dl,
			self.render_dl

		) = self.accelerator.prepare(
			self.trans_model,
			self.vqvae_model,
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

	def save(self, path):
		pkg = dict(
			model = self.accelerator.get_state_dict(self.trans_model),
			optim = self.optim.state_dict(),
			steps = self.steps,
			total_loss = self.best_loss,
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
			logits = self.trans_model(motion = inp , mask = batch["motion_mask"][:,:-1]  , \
				context = batch["condition"], context_mask = batch["condition_mask"])	
					
			loss = self.loss_fnc(logits.contiguous().view(-1, logits.shape[-1]), target.contiguous().view(-1))

			if not self.enable_var_len:
				probs = torch.softmax(logits, dim=-1)

				if self.args.sample_max:
					_, cls_pred_index = torch.max(probs, dim=-1)

				else:
					dist = torch.distributions.Categorical(probs)
					cls_pred_index = dist.sample()
				right_num += (cls_pred_index.flatten(0) == target.flatten(0)).sum().item()

			
			self.accelerator.backward(loss / self.grad_accum_every)

			accum_log(logs, dict(
				loss = loss/self.grad_accum_every,
				avg_max_length = int(max(lengths)) / self.grad_accum_every,
				accuracy = right_num/((logits.shape[0]*logits.shape[1]) * self.grad_accum_every),
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

		

		if self.is_main and (steps % self.evaluate_every == 0):
			self.validation_step()
			# print("test generating samples from gt motion")
			# self.sample_render_generative(os.path.join(self.output_dir , "samples") , seq_len=self.args.window_size, num_start_indices  =-1)
			print("test generating from <bos>")
			self.sample_render_generative(os.path.join(self.output_dir , "generative") , seq_len=-1, num_start_indices  =1)
			
				
		# save model
		
		if self.is_main and not (steps % self.save_model_every) and steps>0:
			os.makedirs(os.path.join(self.output_dir , "checkpoints" ) , exist_ok=True)
			model_path = os.path.join(self.output_dir , "checkpoints", f'vqvae_motion.{steps}.pt')
			self.save(model_path)

			if float(loss) < self.best_loss :

				model_path = os.path.join(self.output_dir, f'vqvae_motion.pt')
				self.save(model_path)
				self.best_loss = loss

			self.print(f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}')


		self.steps += 1
		return logs
	
	def validation_step(self):
		self.trans_model.eval()
		val_loss_ae = {}

		print(f"validation start")

		with torch.no_grad():

			for batch in tqdm((self.valid_dl), position=0, leave=True):


				right_num = 0
				inp, target = batch["motion"][:, :-1], batch["motion"][:, 1:]
				logits = self.trans_model(motion = inp , mask = batch["motion_mask"][:,:-1]  , \
					context = batch["condition"], context_mask = batch["condition_mask"])	
						
				loss = self.loss_fnc(logits.contiguous().view(-1, logits.shape[-1]), target.contiguous().view(-1))
				if not self.enable_var_len:
					probs = torch.softmax(logits, dim=-1)

					if self.args.sample_max:
						_, cls_pred_index = torch.max(probs, dim=-1)

					else:
						dist = torch.distributions.Categorical(probs)
						cls_pred_index = dist.sample()
					right_num += (cls_pred_index.flatten(0) == target.flatten(0)).sum().item()

				loss_dict = {
				"total_loss": loss,
				"accuracy":right_num/(cls_pred_index.shape[0]*cls_pred_index.shape[1])
				}	
			   
				val_loss_ae.update(loss_dict)

				sums_ae = dict(Counter(val_loss_ae) + Counter(loss_dict))
				means_ae = {k: sums_ae[k] / float((k in val_loss_ae) + (k in loss_dict)) for k in sums_ae}
				val_loss_ae.update(means_ae)



		for key , value in val_loss_ae.items():
			wandb.log({f'val_loss_vqgan/{key}': value})                
		
		print(f"val/total_loss " ,val_loss_ae["total_loss"], )

		self.trans_model.train()

	def sample_render_generative(self , save_path , seq_len = 100 , num_start_indices = 1):

		save_file = os.path.join(save_path , f"{int(self.steps.item())}")
		os.makedirs(save_file , exist_ok=True)

		self.trans_model.eval()
		print(f"render start")
		with torch.no_grad():
			for batch in tqdm(self.render_dl):

				gt_motion_indices = batch["motion"][:,:-1]
				if seq_len == -1:
					seq_len = gt_motion_indices.shape[1]


				name = batch["names"]
				start_tokens = gt_motion_indices[:,:num_start_indices]

				gen_motion_indices = self.trans_model.module.generate(start_tokens = start_tokens, seq_len=seq_len , context = batch["condition"], context_mask = batch["condition_mask"])
				
				gt_motion_indices_ = gt_motion_indices[gt_motion_indices<self.training_args.bos_index]
				gen_motion_indices_ = gen_motion_indices[gen_motion_indices<self.training_args.bos_index]

				_ , pred_motion = self.vqvae_model.module.decode(gen_motion_indices_.reshape(gt_motion_indices.shape[0],-1))
				_ , gt_motion = self.vqvae_model.module.decode(gt_motion_indices_.reshape(gt_motion_indices.shape[0],-1))



				gt_motion_xyz = recover_from_ric(gt_motion.cpu().float()*self.render_ds.std+self.render_ds.mean, 22)
				gt_motion_xyz = gt_motion_xyz.reshape(gt_motion.shape[0],-1, 22, 3)

				pred_motion_xyz = recover_from_ric(pred_motion.cpu().float()*self.render_ds.std+self.render_ds.mean, 22)
				pred_motion_xyz = pred_motion_xyz.reshape(pred_motion.shape[0],-1, 22, 3)

				

				gt_pose_vis = plot_3d.draw_to_batch(gt_motion_xyz.numpy(),None, [os.path.join(save_file,name[0] + "_gt.gif")])
				pred_pose_vis = plot_3d.draw_to_batch(pred_motion_xyz.numpy(),None, [os.path.join(save_file,name[0] + "_pred.gif")])


		self.trans_model.train()

	def train(self, resume = False, log_fn = noop):

		self.best_loss = float("inf")
		print(self.output_dir)



		if resume:
			save_path = os.path.join(self.output_dir , "checkpoints")
			chk = sorted(os.listdir(save_path) , key = lambda x: int(x.split('.')[1]))[-1]
			print("resuming from ", os.path.join(save_path, f'{chk}'))
			self.load(os.path.join(save_path , f"{chk}"))

		while self.steps < self.num_train_steps:
			logs = self.train_step()
			log_fn(logs)

		self.print('training complete')
