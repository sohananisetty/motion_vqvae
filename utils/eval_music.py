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
    best_div_k=100, best_div_g=100,
    best_beat_align=100):



  
    result_features = {"kinetic": [], "manual": []}
    real_features = {"kinetic": [], "manual": []}

    beat_scores_real = []
    beat_scores_pred = []

    for i,aist_batch in enumerate(tqdm(val_loader)):
        

        mot_len = aist_batch["motion_lengths"][0]
        motion_name = aist_batch["names"][0]
        
        ind = net.encode(aist_batch["motion"].cuda())
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
        
        
    
    



