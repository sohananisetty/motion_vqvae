import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
from torch.utils.data.dataloader import default_collate 

class MotionCollator():
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.bos = torch.LongTensor(([0]))
        self.eos = torch.LongTensor(([2]))
        self.pad = torch.LongTensor(([1]))

    def __call__(self, samples):
        

        pad_batch_inputs = []
        pad_batch_mask = []
        motion_lengths = []
        names = []
        max_len = max([sample.shape[0] for sample, name in samples])


        for inp,name in samples:
            n,d = inp.shape
            diff = max_len - n
            mask = torch.BoolTensor([1]*n + [0]*diff)
            padded = torch.concatenate((torch.tensor(inp) , torch.ones((diff,d))*self.pad))
            pad_batch_inputs.append(padded)
            pad_batch_mask.append(mask)
            motion_lengths.append(n)
            names.append(name)

    
        batch = {
            "motion": torch.stack(pad_batch_inputs , 0),
            "motion_lengths": torch.Tensor(motion_lengths),
            "motion_mask" : torch.stack(pad_batch_mask , 0),
            "names" : np.array(names)

        }

   
        return batch    
    
def id_collate(samples):
    new_batch = []
    ids = []
    for inp,name in samples:
        new_batch.append(torch.Tensor(inp))
        ids.append(name)

    batch = {
            "motion": torch.stack(new_batch , 0),
            "names" : np.array(ids)

        }
    
    return batch

class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, data_root, max_motion_length = 196, window_size = 64, fps = 20, split = "train"):
        self.fps = fps
        self.window_size = window_size
        self.dataset_name = dataset_name
        self.split = split

        if dataset_name == 't2m':
            self.data_root = data_root
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = max_motion_length
            self.meta_dir = ''

        if dataset_name == 'aist':
            self.data_root = data_root
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = max_motion_length
            self.meta_dir = ''

        elif dataset_name == 'kit':
            self.data_root = data_root
            #'./dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21

            self.max_motion_length = max_motion_length
            self.meta_dir = ''
        
        joints_num = self.joints_num

        mean = np.load(pjoin(self.data_root, 'Mean.npy'))
        std = np.load(pjoin(self.data_root, 'Std.npy'))

        split_file = pjoin(self.data_root, f'{split}.txt')

        self.data = []
        self.lengths = []
        self.id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        for name in tqdm(self.id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

            
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        
        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # if self.split in ["val", "test" , "render"]:
        #     return motion , self.id_list[item]
        
        # batch = {
        #     "motion": motion,
        #     "names" :self.id_list[item],
        # }

        return motion , self.id_list[item]
    
class VQVarLenMotionDataset(data.Dataset):
    def __init__(self, dataset_name, data_root, max_length_seconds = 10, min_length_seconds = 3, fps = 20, split = "train"):
        self.fps = fps
        self.min_length_seconds = min_length_seconds
        self.max_length_seconds = max_length_seconds

        self.min_motion_length = self.fps*min_length_seconds
        self.max_motion_length = self.fps*max_length_seconds
        self.dataset_name = dataset_name
        self.split = split
        self.set_stage(0)

        if dataset_name == 't2m':
            self.data_root = data_root
            self.motion_dir = os.path.join(self.data_root, 'new_joint_vecs')
            self.text_dir = os.path.join(self.data_root, 'texts')
            self.joints_num = 22
#             self.max_motion_length = max_motion_length
            self.meta_dir = ''

        if dataset_name == 'aist':
            self.data_root = data_root
            self.motion_dir = os.path.join(self.data_root, 'new_joint_vecs')
            self.text_dir = os.path.join(self.data_root, 'texts')
            self.joints_num = 22
#             self.max_motion_length = max_motion_length
            self.meta_dir = ''

        elif dataset_name == 'kit':
            self.data_root = data_root
            #'./dataset/KIT-ML'
            self.motion_dir = os.path.join(self.data_root, 'new_joint_vecs')
            self.text_dir = os.path.join(self.data_root, 'texts')
            self.joints_num = 21

#             self.max_motion_length = max_motion_length
            self.meta_dir = ''
        
        joints_num = self.joints_num

        mean = np.load(os.path.join(self.data_root, 'Mean.npy'))
        std = np.load(os.path.join(self.data_root, 'Std.npy'))

        split_file = os.path.join(self.data_root, f'{split}.txt')

        self.data = []
        self.lengths = []
        self.id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        for name in tqdm(self.id_list):
            try:
                motion = np.load(os.path.join(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.min_motion_length:
                    continue
                self.lengths.append(motion.shape[0])
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

            
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def set_stage(self, stage):

        lengths = [self.min_motion_length , self.min_motion_length +20,
                                            self.min_motion_length +40 , 
                                            self.min_motion_length +80 , 
                                            self.min_motion_length +120 , 
                                            self.fps*self.max_length_seconds
                                      ]
        self.max_motion_length = lengths[stage]
        print(f'changing range to: {self.min_motion_length} - {self.max_motion_length}')
        
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        
        motion_len = len(motion)
        
        try:
            self.window_size = np.random.randint(self.min_motion_length , min(motion_len , self.max_motion_length))
        except:
            self.window_size = self.min_motion_length

        
        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]
#         print( motion_len , self.window_size , idx , motion.shape)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

#         if self.split in ["val", "test" , "render"]:
        return motion , self.id_list[item]


#         return motion
    
    

def DATALoader(
            dataset,
            batch_size,
            num_workers = 0,
            shuffle = True,
            collate_fn = None,
           ):

    if collate_fn is None:
        collate_fn= id_collate
    # prob = dataset.compute_sampling_prob()
    # sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(dataset) * 1000, replacement=True)
   
    train_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size,
                                                shuffle=shuffle,
                                                #sampler=sampler,
                                                num_workers=num_workers,
                                                collate_fn=collate_fn,
                                                drop_last = True)

    return train_loader