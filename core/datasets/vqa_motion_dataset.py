import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm




class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, data_root, max_motion_length = 196, window_size = 64, split = "train"):
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

        if self.split in ["val", "test" , "render"]:
            return motion , self.id_list[item]


        return motion
    

def DATALoader(
            dataset,
            batch_size,
            num_workers = 0,
            shuffle = True,
           ):

    # prob = dataset.compute_sampling_prob()
    # sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(dataset) * 1000, replacement=True)
   
    train_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size,
                                                shuffle=shuffle,
                                                #sampler=sampler,
                                                num_workers=num_workers,
                                                #collate_fn=collate_fn,
                                                drop_last = True)

    return train_loader