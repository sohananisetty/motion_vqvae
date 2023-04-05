from math import sqrt
from pathlib import Path

import os

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

from core.models.vqvae import VQMotionModel
from core.models.loss import ReConsLoss
from utils.motion_process import recover_from_ric

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

		print("self.enable_var_len: ", self.enable_var_len)

		self.stage_steps = [0 , 100000 , 140000,180000,220000 , 300000 ]
		self.stage = -1


		if self.is_main:
			wandb.login()
			wandb.init(project="vqvae_768_768_vl")

		self.output_dir = Path(self.training_args.output_dir)
		self.output_dir.mkdir(parents = True, exist_ok = True)

		self.register_buffer('steps', torch.Tensor([0]))
		self.vqvae_model = vqvae_model

		total = sum(p.numel() for p in self.vqvae_model.parameters() if p.requires_grad)
		print("Total training params: %.2fM" % (total / 1e6))



		self.args.nb_joints = 22 if self.dataset_name == "t2m" else 21

		
		self.num_train_steps = self.training_args.num_train_iters
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

		if self.enable_var_len:
			train_ds = VQVarLenMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder)
			valid_ds = VQMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "val", max_motion_length = self.args.max_seq_length)
			self.render_ds = VQVarLenMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "render" )
		else:

			train_ds = VQMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder,max_motion_length = self.args.max_seq_length)
			valid_ds = VQMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "val", max_motion_length = self.args.max_seq_length)
			self.render_ds = VQMotionDataset(self.dataset_args.dataset_name, data_root = self.dataset_args.data_folder , split = "render" , max_motion_length = self.args.max_seq_length)

		self.print(f'training with training and valid dataset of {len(train_ds)} and {len(valid_ds)} samples and test of  {len(self.render_ds)}')

		# dataloader
		collate_fn = MotionCollator(self.args.max_seq_length) if self.enable_var_len else None

		

		self.dl = DATALoader(train_ds , batch_size = self.training_args.train_bs,collate_fn=collate_fn)
		self.valid_dl = DATALoader(valid_ds , batch_size = self.training_args.eval_bs, shuffle = False,collate_fn=collate_fn)
		self.render_dl = DATALoader(self.render_ds , batch_size = 1,shuffle = False,collate_fn=collate_fn)

		# prepare with accelerator

		(
			self.vqvae_model,
			self.optim,
			self.dl,
			self.valid_dl,
			self.render_dl

		) = self.accelerator.prepare(
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

		hps = {"num_train_steps": self.num_train_steps, "max_seq_length": self.args.max_seq_length, "learning_rate": self.training_args.learning_rate}
		self.accelerator.init_trackers(f"{self.model_name}", config=hps)        


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
			model = self.accelerator.get_state_dict(self.vqvae_model),
			optim = self.optim.state_dict(),
			steps = self.steps,
			total_loss = self.best_loss,
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
		print("starting at step: ", self.steps)



	def train_step(self):

		


		steps = int(self.steps.item())

		if steps in self.stage_steps:
			self.stage = min(self.stage+1 , len(self.stage_steps))
			print("stage" , self.stage)
			self.dl.dataset.set_stage(self.stage)


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
				loss = loss/self.grad_accum_every,
				loss_motion = loss_motion/ self.grad_accum_every,
				loss_vel = loss_vel / self.grad_accum_every,
				commit_loss = commit_loss / self.grad_accum_every,
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
			self.validation_step()
			self.sample_render(os.path.join(self.output_dir , "samples"))
			
				
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
				"total_loss": loss,
				"loss_motion":loss_motion,
				"loss_vel": loss_vel,
				"commit_loss" :commit_loss
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

		print(self.render_dl.batch_size)
		
		# assert self.render_dl.batch_size == 1 , "Batch size for rendering should be 1!"

		self.vqvae_model.eval()
		print(f"render start")
		with torch.no_grad():
			for batch in tqdm(self.render_dl):


				gt_motion = batch["motion"]
				name = batch["names"]

				motion_len = int(batch.get("motion_lengths" , [gt_motion.shape[1]])[0])

				gt_motion = gt_motion[:,:motion_len,:]

				pred_motion , _, _ = self.vqvae_model(gt_motion)

				gt_motion_xyz = recover_from_ric(gt_motion.cpu().float()*self.render_ds.std+self.render_ds.mean, 22)
				gt_motion_xyz = gt_motion_xyz.reshape(gt_motion.shape[0],-1, 22, 3)

				pred_motion_xyz = recover_from_ric(pred_motion.cpu().float()*self.render_ds.std+self.render_ds.mean, 22)
				pred_motion_xyz = pred_motion_xyz.reshape(pred_motion.shape[0],-1, 22, 3)

				

				gt_pose_vis = plot_3d.draw_to_batch(gt_motion_xyz.numpy(),None, [os.path.join(save_file,name[0] + "_gt.gif")])
				pred_pose_vis = plot_3d.draw_to_batch(pred_motion_xyz.numpy(),None, [os.path.join(save_file,name[0] + "_pred.gif")])

				# render(pred_motion_xyz, outdir=save_path, step=self.steps, name=f"{name}", pred=True)
				# render(gt_motion_xyz, outdir=save_path, step=self.steps, name=f"{name}", pred=False)

		self.vqvae_model.train()


	def train(self, resume = False, log_fn = noop):

		self.best_loss = float("inf")


		if resume:
			save_path = os.path.join(self.output_dir)
			#chk = sorted(os.listdir(save_path) , key = lambda x: int(x.split('.')[1]))[-1]
			print("resuming from ", os.path.join(self.output_dir, f'vqvae_motion.pt'))
			self.load(os.path.join(save_path , "vqvae_motion.pt"))

		while self.steps < self.num_train_steps:
			logs = self.train_step()
			log_fn(logs)

		self.print('training complete')
