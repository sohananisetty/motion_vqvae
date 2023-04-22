import numpy as np
import torch
from tqdm import tqdm
import librosa
import os

from utils.motion_process import recover_from_ric
from utils.aist_metrics.calculate_fid_scores import calculate_avg_distance, extract_feature,calculate_frechet_feature_distance,calculate_frechet_distance
from utils.aist_metrics.features import kinetic,manual
from utils.aist_metrics.calculate_beat_scores import motion_peak_onehot,alignment_score





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
		
		
	
	
def get_target_indices(batch , trans_model , sample_max = False , bos = 1024,pad = 1025,eos = 1026):
##batch size = 1

	inp, target = batch["motion"][:, :-1], batch["motion"][:, 1:]
	# print(target.shape)

	##inp: b seqlen-1 target: b seqlen-1
	length = int(batch["motion_lengths"][0])
	# print(length)

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

		

		gen_motion_indices , gt_motion_indices = get_target_indices(aist_batch , trans)
		
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
		
	


