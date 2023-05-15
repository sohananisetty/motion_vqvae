import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
from torch.utils.data.dataloader import default_collate 
from glob import glob
import clip

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


class EvaluatorMotionCollator():
    def __init__(self):
        self.bos = torch.LongTensor(([0]))
        self.eos = torch.LongTensor(([2]))
        self.pad = torch.LongTensor(([1]))

    def __call__(self, samples):

        
        pad_batch_inputs = []
        pad_batch_mask = []
        condition_batch_masks = []
        motion_lengths = []
        condition_list = []
        names = []
        
        max_len = max([sample.shape[0] for sample, _,_ in samples])


        for inp,name,condition in samples:
            n,d = inp.shape
            diff = max_len - n
            mask = torch.BoolTensor([1]*n + [0]*diff)
            padded = torch.concatenate((torch.tensor(inp) , torch.ones((diff,d))*self.pad))
            pad_batch_inputs.append(padded)
            pad_batch_mask.append(mask)
            motion_lengths.append(n)
            names.append(name)
            
            ## Conditioning 
            music_encoding = condition
               
            condition_padded = torch.concatenate(
                (
                    torch.tensor(music_encoding) ,
                    torch.ones((diff,condition.shape[-1]))*self.pad,
                    
                    ))
            c_mask = torch.BoolTensor([1]*(n) + [0]*diff)
            condition_list.append(condition_padded)
            condition_batch_masks.append(c_mask)

        condition_embeddings = torch.stack(condition_list , 0)

        batch = {
            "motion": torch.stack(pad_batch_inputs , 0),
            "motion_lengths": torch.Tensor(motion_lengths),
            "motion_mask" : torch.stack(pad_batch_mask , 0),
            "names" : np.array(names),
            "condition" : condition_embeddings.float(),## b seq_len
            "condition_mask" : torch.stack(condition_batch_masks , 0),## b seq_len

        }

   
        return batch    

class EvaluatorVarLenMotionDataset(data.Dataset):
    def __init__(self, data_root, max_length_seconds = 10, min_length_seconds = 3, fps = 20, split = "train" , num_stages = 6):
        self.fps = fps
        self.min_length_seconds = min_length_seconds
        self.max_length_seconds = max_length_seconds

        self.min_motion_length = self.fps*min_length_seconds
        self.max_motion_length = self.fps*max_length_seconds
        self.split = split
        self.num_stages = num_stages
        self.set_stage(0)

        self.data_root = data_root
        self.motion_dir = os.path.join(self.data_root, 'new_joint_vecs')
        self.music_dir =  os.path.join(self.data_root, "music")
        self.joints_num = 22
        self.meta_dir = ''
        self.condition = "music"        

        mean = np.load(os.path.join(self.data_root, 'Mean.npy'))
        std = np.load(os.path.join(self.data_root, 'Std.npy'))

        split_file = os.path.join(self.data_root, f'{split}.txt')


        # self.data = []
        lengths = []
        self.id_list = []
        data_dict = {}
        new_name_list = []
        
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        for name in tqdm(self.id_list):
            try:
                motion = np.load(os.path.join(self.motion_dir, name + '.npy'))

                
         
                
                music_name = name.split("_")[-2]

                music_encoding=  np.load(os.path.join(self.music_dir , music_name + ".npy"))
                music_len = len(music_encoding)
                motion_len = len(motion)

                min_l = min(music_len , motion_len)
                data_dict[name] = {'motion': motion[:min_l],
                                    'length': min_l,
                                    'music':music_encoding[:min_l]}

               
                lengths.append(min_l)
                new_name_list.append(name)
                
                
            except:
                pass

            
            
        self.mean = mean
        self.std = std
        self.length_arr = np.array(lengths)
        self.data_dict = data_dict
        self.name_list = new_name_list
        
        print("Total number of motions {}".format(len(self.data_dict)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def set_stage(self, stage):

        lengths = list(np.array(np.logspace(np.log(self.min_motion_length), np.log(self.fps*self.max_length_seconds), self.num_stages, base=np.exp(1)) + 1 , dtype = np.uint))

        self.max_motion_length = lengths[stage]
        print(f'changing range to: {self.min_motion_length} - {self.max_motion_length}')
        
    
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        
        data = self.data_dict[self.name_list[item]]
        motion, motion_len, music = data['motion'], data['length'], data['music']
        condition = music
        
        try:
            self.window_size = np.random.randint(self.min_motion_length , min(motion_len , self.max_motion_length))
        except:
            self.window_size = min(motion_len , self.min_motion_length)

        
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx:idx+self.window_size]
        
        condition = (condition[idx:idx+self.window_size])

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion , self.id_list[item], condition



class EvaluatorMotionDataset(data.Dataset):
    def __init__(self, data_root, window_size = 60 , fps = 20, split = "train", init_0 = False):
        self.fps = fps
        self.window_size = window_size

        self.split = split
       

        self.data_root = data_root
        self.motion_dir = os.path.join(self.data_root, 'new_joint_vecs')
        self.music_dir =  os.path.join(self.data_root, "music")
        self.joints_num = 22
        self.meta_dir = ''
        self.condition = "music"        

        mean = np.load(os.path.join(self.data_root, 'Mean.npy'))
        std = np.load(os.path.join(self.data_root, 'Std.npy'))

        split_file = os.path.join(self.data_root, f'{split}.txt')


        # self.data = []
        lengths = []
        self.id_list = []
        data_dict = {}
        new_name_list = []
        
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        for name in tqdm(self.id_list):
            try:
                motion = np.load(os.path.join(self.motion_dir, name + '.npy'))

                
         
                
                music_name = name.split("_")[-2]

                music_encoding=  np.load(os.path.join(self.music_dir , music_name + ".npy"))
                music_len = len(music_encoding)
                motion_len = len(motion)

                min_l = min(music_len , motion_len)
                data_dict[name] = {'motion': motion[:min_l],
                                    'length': min_l,
                                    'music':music_encoding[:min_l]}

               
                lengths.append(min_l)
                new_name_list.append(name)
                
                
            except:
                pass

            
            
        self.mean = mean
        self.std = std
        self.length_arr = np.array(lengths)
        self.data_dict = data_dict
        self.name_list = new_name_list
        
        print("Total number of motions {}".format(len(self.data_dict)))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        
        data = self.data_dict[self.name_list[item]]
        motion, motion_len, music = data['motion'], data['length'], data['music']
        condition = music
        
        window_size = min(self.window_size , motion_len)
        

        if self.init_0:
            idx = 0
        else:
            idx = random.randint(0, motion_len - window_size)

        
        motion = motion[idx:idx+self.window_size]
        
        condition = (condition[idx:idx+self.window_size])

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion , self.id_list[item], condition

