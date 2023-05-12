import numpy as np
import torch
from tqdm import tqdm
import librosa
import os

from utils.motion_process import recover_from_ric
from utils.aist_metrics.calculate_fid_scores import calculate_avg_distance, extract_feature,calculate_frechet_feature_distance,calculate_frechet_distance
from utils.aist_metrics.features import kinetic,manual
from utils.aist_metrics.calculate_beat_scores import motion_peak_onehot,alignment_score

import clip
import random

genre_dict = {
	"mBR" : "Break",
	"mPO" : "Pop",
	"mLO" : "Lock",
	"mMH" : "Middle Hip-hop",
	"mLH" : "LA style Hip-hop",
	"mHO" : "House",    
	"mWA" : "Waack",
	"mKR" : "Krump",
	"mJS" : "Street Jazz",
	"mJB" : "Ballet Jazz",
}




@torch.no_grad()        
def evaluate_music_motion_vqvae(
	val_loader, net,audio_feature_dir = "/srv/scratch/sanisetty3/music_motion/AIST/audio_features",
	best_fid_k=1000,best_fid_g=1000,
	best_div_k=-100, best_div_g=-100,
	best_beat_align=-100):



  
	result_features = {"kinetic": [], "manual": []}
	real_features = {"kinetic": [], "manual": []}

	mean = val_loader.dataset.mean
	std = val_loader.dataset.std

	beat_scores_real = []
	beat_scores_pred = []

	for i,aist_batch in enumerate(tqdm(val_loader)):
		

		mot_len = aist_batch["motion_lengths"][0]
		motion_name = aist_batch["names"][0]
		
		try:
			ind = net.encode(aist_batch["motion"].cuda())
		except:
			ind = net.module.encode(aist_batch["motion"].cuda())

		quant , out_motion = net.decode(ind)

		keypoints3d_gt = recover_from_ric(aist_batch["motion"][0,:mot_len] , 22).detach().cpu().numpy()
		keypoints3d_pred = recover_from_ric(out_motion[0,:mot_len] , 22).detach().cpu().numpy()
		
		real_features["kinetic"].append(extract_feature(keypoints3d_gt, "kinetic"))
		real_features["manual"].append(extract_feature(keypoints3d_gt, "manual"))
		
		result_features["kinetic"].append(extract_feature(keypoints3d_pred, "kinetic"))
		result_features["manual"].append(extract_feature(keypoints3d_pred, "manual"))




		motion_beats = motion_peak_onehot(keypoints3d_gt)
		# get real data music beats
		audio_name = motion_name.split("_")[-2]
		
		audio_feature = np.load(os.path.join(audio_feature_dir, f"{audio_name}.npy"))
		audio_beats = audio_feature[:mot_len, -1] # last dim is the music beats
		# get beat alignment scores
		beat_score = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_real.append(beat_score)
		
		
		motion_beats = motion_peak_onehot(keypoints3d_pred)
		beat_score_pred = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_pred.append(beat_score_pred)


	FID_k, Dist_k = calculate_frechet_feature_distance(real_features["kinetic"], result_features["kinetic"])
	FID_g, Dist_g = calculate_frechet_feature_distance(real_features["manual"], result_features["manual"])
 
	FID_k2, Dist_k2 = calculate_frechet_feature_distance(real_features["kinetic"], real_features["kinetic"])
	FID_g2, Dist_g2 = calculate_frechet_feature_distance(real_features["manual"], real_features["manual"])
	
	
	print("FID_k: ",FID_k,"Diversity_k:",Dist_k)
	print("FID_g: ",FID_g,"Diversity_g:",Dist_g)
 
	print("FID_k_real: ",FID_k2,"Diversity_k_real:",Dist_k2)
	print("FID_g_real: ",FID_g2,"Diversity_g_real:",Dist_g2)


	print ("\nBeat score on real data: %.3f\n" % (np.mean(beat_scores_real)))
	print ("\nBeat score on generated data: %.3f\n" % (np.mean(beat_scores_pred)))

	best_fid_k = FID_k if FID_k < best_fid_k else best_fid_k
	best_fid_g = FID_g if FID_g < best_fid_g else best_fid_g
	best_div_k = Dist_k if Dist_k > best_div_k else best_div_k
	best_div_g = Dist_g if Dist_g > best_div_g else best_div_g

	best_beat_align = np.mean(beat_scores_real) if np.mean(beat_scores_real) > best_beat_align else best_beat_align 



	return best_fid_k, best_fid_g,best_div_k,best_div_g,best_beat_align
		
		
	
	
def get_target_indices(batch , trans_model , sample_max = False , bos = 1024,pad = 1025,eos = 1026):
##batch size = 1

	inp, target = batch["motion"][:, :-1], batch["motion"][:, 1:]
	# print(target.shape)

	##inp: b seqlen-1 target: b seqlen-1
	length = int(batch["motion_lengths"][0])
	# print(length)
 
	try:

		logits = trans_model(motion = inp , mask = batch["motion_mask"][:,:-1]  , \
			context = batch["condition"], context_mask = batch["condition_mask"], style_context = batch["style"])
	except:
		logits = trans_model(motion = inp , mask = batch["motion_mask"][:,:-1]  , \
			context = batch["condition"], context_mask = batch["condition_mask"])
	
		
	probs = torch.softmax(logits[0][:length], dim=-1)
	if sample_max:
		_, cls_pred_index = torch.max(probs, dim=-1)

	else:
		dist = torch.distributions.Categorical(probs)
		cls_pred_index = dist.sample()
	# print(cls_pred_index.shape)


	## cls_pred_index - list

	eos_index = (cls_pred_index == eos).nonzero().flatten().tolist()
	# print(eos_index)
	pad_index = (cls_pred_index == pad).nonzero().flatten().tolist()
	# print(pad_index)
	bos_index = (cls_pred_index == bos).nonzero().flatten().tolist()
	# print(bos_index)
	stop_index = min([*eos_index , *pad_index , *bos_index, length-1])

	gen_motion_indices_ = cls_pred_index[:int(stop_index)]
	gt_motion_indices_ = target[target<bos]

	# print(gen_motion_indices_.dtype , gt_motion_indices_.dtype)

	
	gen_motion_indices_ = (gen_motion_indices_).contiguous().view(1,-1)
	gt_motion_indices_ = gt_motion_indices_.contiguous().view(1,-1)
	# print(gen_motion_indices_.shape,gt_motion_indices_.shape)


	return gen_motion_indices_ , gt_motion_indices_




@torch.no_grad()        
def evaluate_music_motion_trans(
	val_loader, net,trans,audio_feature_dir = "/srv/scratch/sanisetty3/music_motion/AIST/audio_features",
	best_fid_k=1000,best_fid_g=1000,
	best_div_k=-100, best_div_g=-100,
	best_beat_align=-100):



  
	result_features = {"kinetic": [], "manual": []}
	real_features = {"kinetic": [], "manual": []}

	mean = val_loader.dataset.mean
	std = val_loader.dataset.std

	beat_scores_real = []
	beat_scores_pred = []

	for i,aist_batch in enumerate(tqdm(val_loader)):



		mot_len = int(aist_batch["motion_lengths"][0])
		motion_name = aist_batch["names"][0]

		

		gen_motion_indices , gt_motion_indices = get_target_indices(aist_batch , trans) ## 1 seq_len, 1 seq_len
		
		try:
			_ , pred_motion = net.module.decode(gen_motion_indices.cuda())
			_ , gt_motion = net.module.decode(gt_motion_indices.cuda())
		except:
			_ , gt_motion = net.decode(gt_motion_indices.cuda())
			_ , pred_motion = net.decode(gen_motion_indices.cuda())

		# print(gt_motion.shape)
			

		keypoints3d_gt = recover_from_ric(gt_motion.detach().cpu()*std+mean , 22)[0].numpy()
		keypoints3d_pred = recover_from_ric(pred_motion.detach().cpu()*std+mean , 22)[0].numpy()
		# print(keypoints3d_gt.shape,keypoints3d_pred.shape)

		try:


			real_features["kinetic"].append(extract_feature(keypoints3d_gt, "kinetic"))
			real_features["manual"].append(extract_feature(keypoints3d_gt, "manual"))
			
			result_features["kinetic"].append(extract_feature(keypoints3d_pred, "kinetic"))
			result_features["manual"].append(extract_feature(keypoints3d_pred, "manual"))
		except:
			continue




		motion_beats = motion_peak_onehot(keypoints3d_gt)
		# get real data music beats
		audio_name = motion_name.split("_")[-2]
		
		audio_feature = np.load(os.path.join(audio_feature_dir, f"{audio_name}.npy"))
		audio_beats = audio_feature[:mot_len, -1] # last dim is the music beats
		# get beat alignment scores
		beat_score = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_real.append(beat_score)
		
		
		motion_beats = motion_peak_onehot(keypoints3d_pred)
		beat_score_pred = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_pred.append(beat_score_pred)


	FID_k, Dist_k = calculate_frechet_feature_distance(real_features["kinetic"], result_features["kinetic"])
	FID_g, Dist_g = calculate_frechet_feature_distance(real_features["manual"], result_features["manual"])
	
	
	print("FID_k: ",FID_k,"Diversity_k:",Dist_k)
	print("FID_g: ",FID_g,"Diversity_g:",Dist_g)
	print ("Beat score on real data: %.3f\n" % (np.mean(beat_scores_real)))
	print ("Beat score on generated data: %.3f\n" % (np.mean(beat_scores_pred)))

	best_fid_k = FID_k if FID_k < best_fid_k else best_fid_k
	best_fid_g = FID_g if FID_g < best_fid_g else best_fid_g
	best_div_k = Dist_k if Dist_k > best_div_k else best_div_k
	best_div_g = Dist_g if Dist_g > best_div_g else best_div_g

	best_beat_align = np.mean(beat_scores_real) if np.mean(beat_scores_real) > best_beat_align else best_beat_align 



	return best_fid_k, best_fid_g,best_div_k,best_div_g,best_beat_align
		





@torch.no_grad()        
def evaluate_music_motion_generative(
	val_loader, vqvae_model,net,use35 = False,audio_feature_dir = "/srv/scratch/sanisetty3/music_motion/AIST/audio_features", audio_encoding_dir = "/srv/scratch/sanisetty3/music_motion/AIST/music",
	best_fid_k=1000,best_fid_g=1000,
	best_div_k=-100, best_div_g=-100,
	best_beat_align=-100, seq_len=400 ):



  
	result_features = {"kinetic": [], "manual": []}
	real_features = {"kinetic": [], "manual": []}

	mean = val_loader.dataset.mean
	std = val_loader.dataset.std

	beat_scores_real = []
	beat_scores_pred = []
	real_pfc = []
	pred_pfc = []
 
	audio_dir = audio_feature_dir if use35 else audio_encoding_dir

	for i,aist_batch in enumerate(tqdm(val_loader)):
		

		mot_len = aist_batch["motion_lengths"][0]
		motion_name = aist_batch["names"][0]
  
		music_name = motion_name.split('_')[-2]
		music_encoding=  np.load(os.path.join(audio_dir , music_name + ".npy"))
  
		gen_motion_indices = torch.randint(0 , 1024 , (1,1))
		# print(gen_motion_indices.shape)
		while gen_motion_indices.shape[1]<=seq_len:
			
			try:
				gen_motion_indices = net.module.generate(start_tokens =gen_motion_indices.cuda(),\
												seq_len=seq_len , \
												context = torch.Tensor(music_encoding)[None,...].cuda(), \
												context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda()
												)
			except:
				gen_motion_indices = net.generate(start_tokens =gen_motion_indices.cuda(),\
														seq_len=seq_len , \
														context = torch.Tensor(music_encoding)[None,...].cuda(), \
														context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda()
														)
	

			
			gen_motion_indices = gen_motion_indices[gen_motion_indices<1024][None,...]
   

   
		try:
			out_motion = torch.zeros((aist_batch["motion"].shape[0] ,gen_motion_indices.shape[-1] , aist_batch["motion"].shape[-1]))
			for i in range(0 , seq_len, 200):
				quant , out_motion_= vqvae_model.decode(gen_motion_indices[:,i:i+200])
				out_motion[:,i:i+200] = out_motion_

			# quant , out_motion = vqvae_model.module.decode(gen_motion_indices[:,:mot_len])
		except:
			# quant , out_motion = vqvae_model.decode(gen_motion_indices[:,:mot_len])
			out_motion = torch.zeros((aist_batch["motion"].shape[0] ,gen_motion_indices.shape[-1] , aist_batch["motion"].shape[-1]))
			for i in range(0 , seq_len, 200):
				quant , out_motion_= vqvae_model.decode(gen_motion_indices[:,i:i+200])
				out_motion[:,i:i+200] = out_motion_


		keypoints3d_gt = recover_from_ric(aist_batch["motion"][0,:mot_len] , 22).detach().cpu().numpy()
		keypoints3d_pred = recover_from_ric(out_motion[0,:mot_len] , 22).detach().cpu().numpy()
		
		real_features["kinetic"].append(extract_feature(keypoints3d_gt, "kinetic"))
		real_features["manual"].append(extract_feature(keypoints3d_gt, "manual"))
		
		result_features["kinetic"].append(extract_feature(keypoints3d_pred, "kinetic"))
		result_features["manual"].append(extract_feature(keypoints3d_pred, "manual"))
  
		real_pfc.append(calc_physical_score(keypoints3d_gt))
		pred_pfc.append(calc_physical_score(keypoints3d_pred))




		motion_beats = motion_peak_onehot(keypoints3d_gt[:mot_len])
		# get real data music beats
		audio_name = motion_name.split("_")[-2]
		
		audio_feature = np.load(os.path.join(audio_feature_dir, f"{audio_name}.npy"))
		audio_beats = audio_feature[:mot_len, -1] # last dim is the music beats
		# get beat alignment scores
		beat_score = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_real.append(beat_score)
		
		
		motion_beats = motion_peak_onehot(keypoints3d_pred[:mot_len])
		beat_score_pred = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_pred.append(beat_score_pred)


	FID_k, Dist_k = calculate_frechet_feature_distance(real_features["kinetic"], result_features["kinetic"])
	FID_g, Dist_g = calculate_frechet_feature_distance(real_features["manual"], result_features["manual"])
	
	
	print("FID_k: ",FID_k,"Diversity_k:",Dist_k)
	print("FID_g: ",FID_g,"Diversity_g:",Dist_g)


	print ("\nBeat score on real data: %.3f\n" % (np.mean(beat_scores_real)))
	print ("\nBeat score on generated data: %.3f\n" % (np.mean(beat_scores_pred)))
 
	print ("\PFC score on real data: %.3f\n" % (np.mean(real_pfc)))
	print ("\PFC score on generated data: %.3f\n" % (np.mean(pred_pfc)))

	best_fid_k = FID_k if FID_k < best_fid_k else best_fid_k
	best_fid_g = FID_g if FID_g < best_fid_g else best_fid_g
	best_div_k = Dist_k if Dist_k > best_div_k else best_div_k
	best_div_g = Dist_g if Dist_g > best_div_g else best_div_g

	best_beat_align = np.mean(beat_scores_real) if np.mean(beat_scores_real) > best_beat_align else best_beat_align 



	return best_fid_k, best_fid_g,best_div_k,best_div_g,best_beat_align
		
		
  

@torch.no_grad()        
def evaluate_music_motion_generative_style(
	val_loader, vqvae_model,net,clip_model, style = None, audio_feature_dir = "/srv/scratch/sanisetty3/music_motion/AIST/audio_features", audio_encoding_dir = "/srv/scratch/sanisetty3/music_motion/AIST/music",
	best_fid_k=1000,best_fid_g=1000,
	best_div_k=-100, best_div_g=-100,
	best_beat_align=-100, seq_len = 400):



  
	result_features = {"kinetic": [], "manual": []}
	real_features = {"kinetic": [], "manual": []}

	mean = val_loader.dataset.mean
	std = val_loader.dataset.std

	beat_scores_real = []
	beat_scores_pred = []

	for i,aist_batch in enumerate(tqdm(val_loader)):
		

		mot_len = aist_batch["motion_lengths"][0]
		motion_name = aist_batch["names"][0]
  
		music_name = motion_name.split('_')[-2]
		music_encoding=  np.load(os.path.join(audio_encoding_dir , music_name + ".npy"))
  
		genre = (genre_dict.get(music_name[:3])) if style is None else style
		
		text = clip.tokenize([genre], truncate=True).cuda()
		style_embeddings = clip_model.encode_text(text).cpu().float().reshape(-1) if clip_model is not None else None
		
  
		gen_motion_indices = torch.randint(0 , 1024 , (1,1))
		# print(gen_motion_indices.shape)
		while gen_motion_indices.shape[1]<=seq_len:
			
			try:
				gen_motion_indices = net.module.generate(start_tokens =gen_motion_indices.cuda(),\
												seq_len=seq_len , \
												context = torch.Tensor(music_encoding)[None,...].cuda(), \
												context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda(), \
												style_context = torch.Tensor(style_embeddings.reshape(-1))[None,...].cuda(),
			
												)
			except:
				gen_motion_indices = net.generate(start_tokens =gen_motion_indices.cuda(),\
														seq_len=seq_len , \
														context = torch.Tensor(music_encoding)[None,...].cuda(), \
														context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda(),\
														style_context = torch.Tensor(style_embeddings.reshape(-1))[None,...].cuda(),
														)
	

			
			gen_motion_indices = gen_motion_indices[gen_motion_indices<1024][None,...]
   
		try:
			out_motion = torch.zeros((aist_batch["motion"].shape[0] ,gen_motion_indices.shape[-1] , aist_batch["motion"].shape[-1]))
			for i in range(0 , seq_len, 200):
				quant , out_motion_= vqvae_model.decode(gen_motion_indices[:,i:i+200])
				out_motion[:,i:i+200] = out_motion_

			# quant , out_motion = vqvae_model.module.decode(gen_motion_indices[:,:mot_len])
		except:
			# quant , out_motion = vqvae_model.decode(gen_motion_indices[:,:mot_len])
			out_motion = torch.zeros((aist_batch["motion"].shape[0] ,gen_motion_indices.shape[-1] , aist_batch["motion"].shape[-1]))
			for i in range(0 , seq_len, 200):
				quant , out_motion_= vqvae_model.decode(gen_motion_indices[:,i:i+200])
				out_motion[:,i:i+200] = out_motion_


		keypoints3d_gt = recover_from_ric(aist_batch["motion"][0,:mot_len] , 22).detach().cpu().numpy()
		keypoints3d_pred = recover_from_ric(out_motion[0,:mot_len] , 22).detach().cpu().numpy()
		
		real_features["kinetic"].append(extract_feature(keypoints3d_gt, "kinetic"))
		real_features["manual"].append(extract_feature(keypoints3d_gt, "manual"))
		
		result_features["kinetic"].append(extract_feature(keypoints3d_pred, "kinetic"))
		result_features["manual"].append(extract_feature(keypoints3d_pred, "manual"))




		motion_beats = motion_peak_onehot(keypoints3d_gt[:mot_len])
		# get real data music beats
		audio_name = motion_name.split("_")[-2]
		
		audio_feature = np.load(os.path.join(audio_feature_dir, f"{audio_name}.npy"))
		audio_beats = audio_feature[:mot_len, -1] # last dim is the music beats
		# get beat alignment scores
		beat_score = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_real.append(beat_score)
		
		
		motion_beats = motion_peak_onehot(keypoints3d_pred[:mot_len])
		beat_score_pred = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_pred.append(beat_score_pred)


	FID_k, Dist_k = calculate_frechet_feature_distance(real_features["kinetic"], result_features["kinetic"])
	FID_g, Dist_g = calculate_frechet_feature_distance(real_features["manual"], result_features["manual"])
	
	
	print("FID_k: ",FID_k,"Diversity_k:",Dist_k)
	print("FID_g: ",FID_g,"Diversity_g:",Dist_g)


	print ("\nBeat score on real data: %.3f\n" % (np.mean(beat_scores_real)))
	print ("\nBeat score on generated data: %.3f\n" % (np.mean(beat_scores_pred)))

	best_fid_k = FID_k if FID_k < best_fid_k else best_fid_k
	best_fid_g = FID_g if FID_g < best_fid_g else best_fid_g
	best_div_k = Dist_k if Dist_k > best_div_k else best_div_k
	best_div_g = Dist_g if Dist_g > best_div_g else best_div_g

	best_beat_align = np.mean(beat_scores_real) if np.mean(beat_scores_real) > best_beat_align else best_beat_align 



	return best_fid_k, best_fid_g,best_div_k,best_div_g,best_beat_align
		
  

@torch.no_grad()        
def evaluate_music_motion_generative_style2(
	val_loader, vqvae_model,net,clip_model, style = None, audio_feature_dir = "/srv/scratch/sanisetty3/music_motion/AIST/audio_features", audio_encoding_dir = "/srv/scratch/sanisetty3/music_motion/AIST/music",
	best_fid_k=1000,best_fid_g=1000,
	best_div_k=-100, best_div_g=-100,
	best_beat_align=-100, seq_len = 400):



  
	result_features = {"kinetic": [], "manual": []}
	real_features = {"kinetic": [], "manual": []}

	mean = val_loader.dataset.mean
	std = val_loader.dataset.std

	beat_scores_real = []
	beat_scores_pred = []

	for i,aist_batch in enumerate(tqdm(val_loader)):
     
		if len(real_features["kinetic"]) > 40:
			
			break
		

		mot_len = aist_batch["motion_lengths"][0]
		motion_name = aist_batch["names"][0]
  

		

		mot_len = aist_batch["motion_lengths"][0]
  
		if mot_len < seq_len:
			continue
  
		music_name = motion_name.split('_')[-2]
		music_encoding=  np.load(os.path.join(audio_encoding_dir , music_name + ".npy"))
  
		genre = (genre_dict.get(music_name[:3])) if style is None else style
		
		text = clip.tokenize([genre], truncate=True).cuda()
		style_embeddings = clip_model.encode_text(text).cpu().float().reshape(-1) if clip_model is not None else None
		
  
		gen_motion_indices = torch.randint(0 , 1024 , (1,1))
		# print(gen_motion_indices.shape)
		while gen_motion_indices.shape[1]<=seq_len:
			
			try:
				gen_motion_indices = net.module.generate(start_tokens =gen_motion_indices.cuda(),\
												seq_len=seq_len , \
												context = torch.Tensor(music_encoding)[None,...].cuda(), \
												context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda(), \
												style_context = torch.Tensor(style_embeddings.reshape(-1))[None,...].cuda(),
			
												)
			except:
				gen_motion_indices = net.generate(start_tokens =gen_motion_indices.cuda(),\
														seq_len=seq_len , \
														context = torch.Tensor(music_encoding)[None,...].cuda(), \
														context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda(),\
														style_context = torch.Tensor(style_embeddings.reshape(-1))[None,...].cuda(),
														)
	

			
			gen_motion_indices = gen_motion_indices[gen_motion_indices<1024][None,...]
   
		try:
			out_motion = torch.zeros((aist_batch["motion"].shape[0] ,gen_motion_indices.shape[-1] , aist_batch["motion"].shape[-1]))
			for i in range(0 , seq_len, 200):
				quant , out_motion_= vqvae_model.decode(gen_motion_indices[:,i:i+200])
				out_motion[:,i:i+200] = out_motion_

			# quant , out_motion = vqvae_model.module.decode(gen_motion_indices[:,:mot_len])
		except:
			# quant , out_motion = vqvae_model.decode(gen_motion_indices[:,:mot_len])
			out_motion = torch.zeros((aist_batch["motion"].shape[0] ,gen_motion_indices.shape[-1] , aist_batch["motion"].shape[-1]))
			for i in range(0 , seq_len, 200):
				quant , out_motion_= vqvae_model.decode(gen_motion_indices[:,i:i+200])
				out_motion[:,i:i+200] = out_motion_


		keypoints3d_gt = recover_from_ric(aist_batch["motion"][0,:seq_len] , 22).detach().cpu().numpy()
		keypoints3d_pred = recover_from_ric(out_motion[0,:seq_len] , 22).detach().cpu().numpy()
		
		real_features["kinetic"].append(extract_feature(keypoints3d_gt, "kinetic"))
		real_features["manual"].append(extract_feature(keypoints3d_gt, "manual"))
		
		result_features["kinetic"].append(extract_feature(keypoints3d_pred, "kinetic"))
		result_features["manual"].append(extract_feature(keypoints3d_pred, "manual"))




		motion_beats = motion_peak_onehot(keypoints3d_gt[:seq_len])
		# get real data music beats
		audio_name = motion_name.split("_")[-2]
		
		audio_feature = np.load(os.path.join(audio_feature_dir, f"{audio_name}.npy"))
		audio_beats = audio_feature[:mot_len, -1] # last dim is the music beats
		# get beat alignment scores
		beat_score = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_real.append(beat_score)
		
		
		motion_beats = motion_peak_onehot(keypoints3d_pred[:seq_len])
		beat_score_pred = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_pred.append(beat_score_pred)


	FID_k, Dist_k = calculate_frechet_feature_distance(real_features["kinetic"], result_features["kinetic"])
	FID_g, Dist_g = calculate_frechet_feature_distance(real_features["manual"], result_features["manual"])
	
	
	print("FID_k: ",FID_k,"Diversity_k:",Dist_k)
	print("FID_g: ",FID_g,"Diversity_g:",Dist_g)


	print ("\nBeat score on real data: %.3f\n" % (np.mean(beat_scores_real)))
	print ("\nBeat score on generated data: %.3f\n" % (np.mean(beat_scores_pred)))

	best_fid_k = FID_k if FID_k < best_fid_k else best_fid_k
	best_fid_g = FID_g if FID_g < best_fid_g else best_fid_g
	best_div_k = Dist_k if Dist_k > best_div_k else best_div_k
	best_div_g = Dist_g if Dist_g > best_div_g else best_div_g

	best_beat_align = np.mean(beat_scores_real) if np.mean(beat_scores_real) > best_beat_align else best_beat_align 



	return best_fid_k, best_fid_g,best_div_k,best_div_g,best_beat_align
		
			
  
  



@torch.no_grad()        
def evaluate_music_motion_generative2(
	val_loader, vqvae_model,net,use35 = False, audio_feature_dir = "/srv/scratch/sanisetty3/music_motion/AIST/audio_features", audio_encoding_dir = "/srv/scratch/sanisetty3/music_motion/AIST/music",
	best_fid_k=1000,best_fid_g=1000,
	best_div_k=-100, best_div_g=-100,
	best_beat_align=-100, seq_len=400 ):



  
	result_features = {"kinetic": [], "manual": []}
	real_features = {"kinetic": [], "manual": []}

	mean = val_loader.dataset.mean
	std = val_loader.dataset.std

	beat_scores_real = []
	beat_scores_pred = []
 
	real_pfc = []
	pred_pfc = []
 
	audio_dir = audio_feature_dir if use35 else audio_encoding_dir

	for i,aist_batch in enumerate(tqdm(val_loader)):
		if len(real_features["kinetic"]) > 40:
			
			break
		

		mot_len = aist_batch["motion_lengths"][0]
  
		if mot_len < seq_len:
			continue
	  
		motion_name = aist_batch["names"][0]
  
		music_name = motion_name.split('_')[-2]
		music_encoding=  np.load(os.path.join(audio_dir , music_name + ".npy"))
  
		gen_motion_indices = torch.randint(0 , 1024 , (1,1))
		# print(gen_motion_indices.shape)
		while gen_motion_indices.shape[1]<=seq_len:
			
			try:
				gen_motion_indices = net.module.generate(start_tokens =gen_motion_indices.cuda(),\
												seq_len=seq_len , \
												context = torch.Tensor(music_encoding)[None,...].cuda(), \
												context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda()
												)
			except:
				gen_motion_indices = net.generate(start_tokens =gen_motion_indices.cuda(),\
														seq_len=seq_len , \
														context = torch.Tensor(music_encoding)[None,...].cuda(), \
														context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda()
														)
	

			
			gen_motion_indices = gen_motion_indices[gen_motion_indices<1024][None,...]
   
	
   
		try:
			out_motion = torch.zeros((aist_batch["motion"].shape[0] ,gen_motion_indices.shape[-1] , aist_batch["motion"].shape[-1]))
			for i in range(0 , seq_len, 200):
				quant , out_motion_= vqvae_model.decode(gen_motion_indices[:,i:i+200])
				out_motion[:,i:i+200] = out_motion_

			# quant , out_motion = vqvae_model.module.decode(gen_motion_indices[:,:mot_len])
		except:
			# quant , out_motion = vqvae_model.decode(gen_motion_indices[:,:mot_len])
			out_motion = torch.zeros((aist_batch["motion"].shape[0] ,gen_motion_indices.shape[-1] , aist_batch["motion"].shape[-1]))
			for i in range(0 , seq_len, 200):
				quant , out_motion_= vqvae_model.decode(gen_motion_indices[:,i:i+200])
				out_motion[:,i:i+200] = out_motion_
   
   


		keypoints3d_gt = recover_from_ric(aist_batch["motion"][0,:seq_len] , 22).detach().cpu().numpy()
		keypoints3d_pred = recover_from_ric(out_motion[0,:seq_len] , 22).detach().cpu().numpy()
		
		real_features["kinetic"].append(extract_feature(keypoints3d_gt, "kinetic"))
		real_features["manual"].append(extract_feature(keypoints3d_gt, "manual"))
		
		result_features["kinetic"].append(extract_feature(keypoints3d_pred, "kinetic"))
		result_features["manual"].append(extract_feature(keypoints3d_pred, "manual"))
  
		real_pfc.append(calc_physical_score(keypoints3d_gt))
		pred_pfc.append(calc_physical_score(keypoints3d_pred))
  
  




		motion_beats = motion_peak_onehot(keypoints3d_gt[:seq_len])
		# get real data music beats
		audio_name = motion_name.split("_")[-2]
		
		audio_feature = np.load(os.path.join(audio_feature_dir, f"{audio_name}.npy"))
		audio_beats = audio_feature[:seq_len, -1] # last dim is the music beats
		# get beat alignment scores
		beat_score = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_real.append(beat_score)
		
		
		motion_beats = motion_peak_onehot(keypoints3d_pred[:seq_len])
		beat_score_pred = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_pred.append(beat_score_pred)
  
	
	FID_k, Dist_k = calculate_frechet_feature_distance(real_features["kinetic"], result_features["kinetic"])
	FID_g, Dist_g = calculate_frechet_feature_distance(real_features["manual"], result_features["manual"])
	
	
	print("FID_k: ",FID_k,"Diversity_k:",Dist_k)
	print("FID_g: ",FID_g,"Diversity_g:",Dist_g)
 
	print ("\PFC score on real data: %.3f\n" % (np.mean(real_pfc)))
	print ("\PFC score on generated data: %.3f\n" % (np.mean(pred_pfc)))


	print ("\nBeat score on real data: %.3f\n" % (np.mean(beat_scores_real)))
	print ("\nBeat score on generated data: %.3f\n" % (np.mean(beat_scores_pred)))
 
 

	best_fid_k = FID_k if FID_k < best_fid_k else best_fid_k
	best_fid_g = FID_g if FID_g < best_fid_g else best_fid_g
	best_div_k = Dist_k if Dist_k > best_div_k else best_div_k
	best_div_g = Dist_g if Dist_g > best_div_g else best_div_g

	best_beat_align = np.mean(beat_scores_real) if np.mean(beat_scores_real) > best_beat_align else best_beat_align 



	return best_fid_k, best_fid_g,best_div_k,best_div_g,best_beat_align , np.mean(real_pfc), np.mean(pred_pfc)
		
		


@torch.no_grad()        
def evaluate_music_motion_generative_parts(
	val_loader, vqvae_model,net,use35 = False, audio_feature_dir = "/srv/scratch/sanisetty3/music_motion/AIST/audio_features", audio_encoding_dir = "/srv/scratch/sanisetty3/music_motion/AIST/music",
	best_fid_k=1000,best_fid_g=1000,
	best_div_k=-100, best_div_g=-100,
	best_beat_align=-100, seq_len=400 , part_len = 200,):



  
	result_features = {"kinetic": [], "manual": []}
	real_features = {"kinetic": [], "manual": []}

	mean = val_loader.dataset.mean
	std = val_loader.dataset.std

	beat_scores_real = []
	beat_scores_pred = []
 
	real_pfc = []
	pred_pfc = []
 
	audio_dir = audio_feature_dir if use35 else audio_encoding_dir

	for i,aist_batch in enumerate(tqdm(val_loader)):
		if len(real_features["kinetic"]) > 40:
			
			break
		

		mot_len = aist_batch["motion_lengths"][0]
  
		if mot_len < seq_len:
			continue
	  
		motion_name = aist_batch["names"][0]
  
		music_name = motion_name.split('_')[-2]
		music_encoding=  np.load(os.path.join(audio_dir , music_name + ".npy"))
  
		gen_motion_indices = torch.randint(0 , 1024 , (1,1))
		context = torch.Tensor(music_encoding)[None,...].cuda()
		context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda()
		# print(gen_motion_indices.shape)
		while gen_motion_indices.shape[1]<=seq_len:
      
			# start_tokens = gen_motion_indices[:,-1]

      
			# context_part = context[]
      
			
      
			
			
			try:
				gen_motion_indices = net.module.generate(start_tokens =gen_motion_indices.cuda(),\
												seq_len=part_len , \
												context = torch.Tensor(music_encoding)[None,...].cuda(), \
												context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda()
												)
			except:
				gen_motion_indices = net.generate(start_tokens =gen_motion_indices.cuda(),\
														seq_len=part_len , \
														context = torch.Tensor(music_encoding)[None,...].cuda(), \
														context_mask=torch.ones((1 ,music_encoding.shape[0]) , dtype = torch.bool).cuda()
														)
	

			
			gen_motion_indices = gen_motion_indices[gen_motion_indices<1024][None,...]
   
	
   
		try:
			quant , out_motion = vqvae_model.module.decode(gen_motion_indices)
		except:
			quant , out_motion = vqvae_model.decode(gen_motion_indices)
   
   


		keypoints3d_gt = recover_from_ric(aist_batch["motion"][0,:seq_len] , 22).detach().cpu().numpy()
		keypoints3d_pred = recover_from_ric(out_motion[0,:seq_len] , 22).detach().cpu().numpy()
		
		real_features["kinetic"].append(extract_feature(keypoints3d_gt, "kinetic"))
		real_features["manual"].append(extract_feature(keypoints3d_gt, "manual"))
		
		result_features["kinetic"].append(extract_feature(keypoints3d_pred, "kinetic"))
		result_features["manual"].append(extract_feature(keypoints3d_pred, "manual"))
  
		real_pfc.append(calc_physical_score(keypoints3d_gt))
		pred_pfc.append(calc_physical_score(keypoints3d_pred))
  
  




		motion_beats = motion_peak_onehot(keypoints3d_gt[:seq_len])
		# get real data music beats
		audio_name = motion_name.split("_")[-2]
		
		audio_feature = np.load(os.path.join(audio_feature_dir, f"{audio_name}.npy"))
		audio_beats = audio_feature[:seq_len, -1] # last dim is the music beats
		# get beat alignment scores
		beat_score = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_real.append(beat_score)
		
		
		motion_beats = motion_peak_onehot(keypoints3d_pred[:seq_len])
		beat_score_pred = alignment_score(audio_beats, motion_beats, sigma=1)
		beat_scores_pred.append(beat_score_pred)
  
	
	FID_k, Dist_k = calculate_frechet_feature_distance(real_features["kinetic"], result_features["kinetic"])
	FID_g, Dist_g = calculate_frechet_feature_distance(real_features["manual"], result_features["manual"])
	
	
	print("FID_k: ",FID_k,"Diversity_k:",Dist_k)
	print("FID_g: ",FID_g,"Diversity_g:",Dist_g)
 
	print ("\PFC score on real data: %.3f\n" % (np.mean(real_pfc)))
	print ("\PFC score on generated data: %.3f\n" % (np.mean(pred_pfc)))


	print ("\nBeat score on real data: %.3f\n" % (np.mean(beat_scores_real)))
	print ("\nBeat score on generated data: %.3f\n" % (np.mean(beat_scores_pred)))
 
 

	best_fid_k = FID_k if FID_k < best_fid_k else best_fid_k
	best_fid_g = FID_g if FID_g < best_fid_g else best_fid_g
	best_div_k = Dist_k if Dist_k > best_div_k else best_div_k
	best_div_g = Dist_g if Dist_g > best_div_g else best_div_g

	best_beat_align = np.mean(beat_scores_real) if np.mean(beat_scores_real) > best_beat_align else best_beat_align 



	return best_fid_k, best_fid_g,best_div_k,best_div_g,best_beat_align , np.mean(real_pfc), np.mean(pred_pfc)
		
	 
  
  
  
  
def calc_physical_score(joint3d):
	
	# print(joint3d.shape)
	
	##joints3D seqxjointx3
	# scores = []
	# names = []
	# accelerations = []
	up_dir = 2  # y is up
	flat_dirs = [i for i in range(3) if i != up_dir]
	DT = 1 / 30

	root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT  # root velocity (S-1, 3)
	root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
	# clamp the up-direction of root acceleration
	root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
	# l2 norm
	root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
	scaling = root_a.max()
	root_a /= scaling

	foot_idx = [7, 10, 8, 11]
	feet = joint3d[:, foot_idx]  # foot positions (S, 4, 3)
	foot_v = np.linalg.norm(
		feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
	)  # (S-2, 4) horizontal velocity
	foot_mins = np.zeros((len(foot_v), 2))
	foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
	foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

	foot_loss = (
		foot_mins[:, 0] * foot_mins[:, 1] * root_a
	)  # min leftv * min rightv * root_a (S-2,)
	foot_loss = foot_loss.mean()
	
	score = foot_loss	
	accelerations = foot_mins[:, 0].mean()
 
	return score

	
 
 
 
#  out = np.mean(scores) * 10000
# 	print(f"{dir} has a mean PFC of {out}")
