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

class MotionCollator():
    def __init__(self):
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


class MotionCollatorConditional():
    def __init__(self, dataset_name = "t2m" , clip_model = None , bos = 0 , eos = 2 , pad = 1):
        self.dataset_name = dataset_name
        self.clip_model = clip_model
        self.bos = torch.LongTensor(([bos]))
        self.eos = torch.LongTensor(([eos]))
        self.pad = torch.LongTensor(([pad]))

    def __call__(self, samples):
        

        pad_batch_inputs = []
        pad_batch_mask = []
        condition_batch_masks = []
        motion_lengths = []
        condition_list = []
        names = []
        max_len = max([sample.shape[0] for sample, _,_ in samples])


        for inp,name,condition in samples:
            n = inp.shape[0]
            diff = max_len - n
            mask = torch.BoolTensor([1]*(n+2) + [0]*diff)
            padded = torch.concatenate(
                (self.bos,
                 torch.LongTensor(inp) , 
                 self.eos,
                 torch.ones((diff) , dtype = torch.long)*self.pad ) 
                 )
            pad_batch_inputs.append(padded)
            pad_batch_mask.append(mask)
            motion_lengths.append(n)
            names.append(name)
            if self.dataset_name == "aist":
                music_encoding = condition
                # condition_padded = torch.concatenate(
                #     (torch.ones((1,condition.shape[-1]))*self.bos,
                #      torch.tensor(music_encoding) ,
                #      torch.ones((1,condition.shape[-1]))*self.eos,
                #      torch.ones((diff,condition.shape[-1]))*self.pad,
                     
                #      ))
                # c_mask = torch.BoolTensor([0]+[1]*(n)+ [0] + [0]*diff)
                condition_padded = torch.concatenate(
                    (
                     torch.tensor(music_encoding) ,
                     torch.ones((1,condition.shape[-1]))*self.eos,
                     torch.ones((diff,condition.shape[-1]))*self.pad,
                     
                     ))
                c_mask = torch.BoolTensor([1]*(n) + [0] + [0]*diff)
                condition_list.append(condition_padded)
                condition_batch_masks.append(c_mask)
            elif self.dataset_name in ["t2m" , "kit"]:
                condition_list.append(str(condition))
                condition_batch_masks.append(mask)


        if self.dataset_name in ["t2m" , "kit"]:
            text = clip.tokenize(condition_list, truncate=True).cuda()
            condition_embeddings = self.clip_model.encode_text(text).cpu().float()

        elif self.dataset_name == "aist":
            condition_embeddings = torch.stack(condition_list , 0)

        
        batch = {
            "motion": torch.stack(pad_batch_inputs , 0),
            "motion_lengths": torch.Tensor(motion_lengths),
            "motion_mask" : torch.stack(pad_batch_mask , 0),
            "names" : np.array(names),
            "condition" : condition_embeddings,
            "condition_mask" : torch.stack(condition_batch_masks , 0),

        }

   
        return batch    
    

def simple_collate(samples):
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
    def __init__(self, dataset_name, data_root, max_length_seconds = 10, window_size = 64, fps = 20, split = "train"):
        self.fps = fps

        self.max_length_seconds = max_length_seconds
        self.max_motion_length = self.fps*max_length_seconds
        self.window_size = window_size
        self.dataset_name = dataset_name
        self.split = split

        if dataset_name == 't2m':
            self.data_root = data_root
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.meta_dir = ''

        if dataset_name == 'aist':
            self.data_root = data_root
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self. music_dir = pjoin(self.data_root, 'music')
            self.joints_num = 22
            self.meta_dir = ''

        elif dataset_name == 'kit':
            self.data_root = data_root
            #'./dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
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
                # if motion.shape[0] < self.window_size:
                #     continue
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
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        window_size = min(self.window_size , (motion).shape[0])
        
        idx = random.randint(0, (motion).shape[0] - window_size)

        motion = motion[idx:idx+window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        return motion , self.id_list[item]
    

class VQVarLenMotionDataset(data.Dataset):
    def __init__(self, dataset_name, data_root, mix = False, max_length_seconds = 10, min_length_seconds = 3, fps = 20, split = "train" , num_stages = 6):
        self.fps = fps
        self.min_length_seconds = min_length_seconds
        self.max_length_seconds = max_length_seconds

        self.min_motion_length = self.fps*min_length_seconds
        self.max_motion_length = self.fps*max_length_seconds
        self.dataset_name = dataset_name
        self.split = split
        self.num_stages = num_stages
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
            self.music_paths =  glob(os.path.join(self.data_root, 'music/*.npy'))
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
                # if motion.shape[0] < self.min_motion_length:
                #     # self.id_list.remove(name)
                #     continue
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

        lengths = list(np.array(np.logspace(np.log(self.min_motion_length), np.log(self.fps*self.max_length_seconds), self.num_stages, base=np.exp(1)) + 1 , dtype = np.uint))

        self.max_motion_length = lengths[stage]
        print(f'changing range to: {self.min_motion_length} - {self.max_motion_length}')
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # name = self.id_list[item]
        # motion = np.load(os.path.join(self.motion_dir, name + '.npy'))  
        motion = self.data[item]
        
        
        motion_len = len(motion)
        
        try:
            self.window_size = np.random.randint(self.min_motion_length , min(motion_len , self.max_motion_length))
        except:
            self.window_size = min(motion_len , self.min_motion_length)

        
        idx = random.randint(0, len(motion) - self.window_size)
        # motion = motion[:min(motion_len , self.fps*self.max_length_seconds)]

        motion = motion[idx:idx+self.window_size]
        # print( motion_len , self.window_size , idx , motion.shape)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

      

#         if self.split in ["val", "test" , "render"]:
        return motion , self.id_list[item]

    
class TransMotionDatasetConditional(data.Dataset):
    def __init__(self, dataset_name, data_root, window_size = 60 , datafolder = "joint_indices" ,w_vectorizer = None, max_length_seconds = 60, fps = 20, split = "train" , max_text_len = 20 , bert_style = False):
        self.fps = fps
        self.window_size  =window_size
        self.max_length_seconds = max_length_seconds
        self.max_motion_length = self.fps*max_length_seconds
        self.dataset_name = dataset_name
        self.split = split
        

        if dataset_name == 't2m':
            self.data_root = data_root
            self.motion_dir = os.path.join(self.data_root, datafolder)
            self.text_dir = os.path.join(self.data_root, 'texts')
            self.joints_num = 22
            self.max_text_len = max_text_len
            self.w_vectorizer = w_vectorizer
            self.meta_dir = ''
            self.condition = "text"

        elif dataset_name == 'aist':
            self.data_root = data_root
            self.motion_dir = os.path.join(self.data_root, datafolder)
            self.music_dir =  os.path.join(self.data_root, 'music')
            self.joints_num = 22
            self.meta_dir = ''
            self.condition = "music"

        elif dataset_name == 'kit':
            self.data_root = data_root
            #'./dataset/KIT-ML'
            self.motion_dir = os.path.join(self.data_root, datafolder)
            self.text_dir = os.path.join(self.data_root, 'texts')
            self.joints_num = 21
            self.max_text_len = max_text_len
            self.w_vectorizer = w_vectorizer
            self.meta_dir = ''
            self.condition = "text"
        
        joints_num = self.joints_num

        mean = np.load(os.path.join(self.data_root, 'Mean.npy'))
        std = np.load(os.path.join(self.data_root, 'Std.npy'))

        split_file = os.path.join(self.data_root, f'{split}.txt')


        lengths = []
        self.id_list = []
        new_name_list = []
        data_dict = {}

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                self.id_list.append(line.strip())


        for name in tqdm(self.id_list):
            try:
                motion = np.load(os.path.join(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue
    

                if self.dataset_name in ["t2m" , "kit"]:

                    text_data = []
                    flag = False
                    with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < self.window_size:
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    lengths.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)

                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text':text_data}
                        new_name_list.append(name)
                        lengths.append(len(motion))

                elif self.dataset_name == "aist":

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
    
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):

        data = self.data_dict[self.name_list[item]]
        condition = None


        if self.dataset_name in ["t2m" , "kit"]:
            motion, motion_len, text_list = data['motion'], data['length'], data['text']
            text_data = random.choice(text_list)
            caption= text_data['caption']
            condition = caption

        if self.dataset_name == "aist":
            motion, motion_len, music = data['motion'], data['length'], data['music']
            condition = music


        window_size = min(self.window_size , (motion).shape[0])
        
        idx = random.randint(0, (motion).shape[0] - window_size)

        motion = motion[idx:idx+window_size]


        if self.dataset_name == "aist":
            condition = (condition[idx:idx+self.window_size])


        return motion , self.id_list[item],condition


class VQVarLenMotionDatasetConditional(data.Dataset):
    def __init__(self, dataset_name, data_root, var_len = False , datafolder = "joint_indices" ,w_vectorizer = None, max_length_seconds = 60, min_length_seconds = 3, fps = 20, split = "train" , max_text_len = 20 , bert_style = False):
        self.fps = fps
        self.min_length_seconds = min_length_seconds
        self.max_length_seconds = max_length_seconds

        self.min_motion_length = self.fps*min_length_seconds
        self.max_motion_length = self.fps*max_length_seconds

        
        self.dataset_name = dataset_name
        self.split = split
        self.bert_style = bert_style
        
        self.set_stage(0)

        if dataset_name == 't2m':
            self.data_root = data_root
            self.motion_dir = os.path.join(self.data_root, datafolder)
            self.text_dir = os.path.join(self.data_root, 'texts')
            self.joints_num = 22
            self.max_text_len = max_text_len
            self.w_vectorizer = w_vectorizer
            self.meta_dir = ''
            self.condition = "text"

        elif dataset_name == 'aist':
            self.data_root = data_root
            self.motion_dir = os.path.join(self.data_root, datafolder)
            self.music_dir =  os.path.join(self.data_root, 'music')
            self.joints_num = 22
            self.meta_dir = ''
            self.condition = "music"

        elif dataset_name == 'kit':
            self.data_root = data_root
            #'./dataset/KIT-ML'
            self.motion_dir = os.path.join(self.data_root, datafolder)
            self.text_dir = os.path.join(self.data_root, 'texts')
            self.joints_num = 21
            self.max_text_len = max_text_len
            self.w_vectorizer = w_vectorizer
            self.meta_dir = ''
            self.condition = "text"
        
        joints_num = self.joints_num

        mean = np.load(os.path.join(self.data_root, 'Mean.npy'))
        std = np.load(os.path.join(self.data_root, 'Std.npy'))

        split_file = os.path.join(self.data_root, f'{split}.txt')


        lengths = []
        self.id_list = []
        new_name_list = []
        data_dict = {}

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                self.id_list.append(line.strip())


        for name in tqdm(self.id_list):
            try:
                motion = np.load(os.path.join(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.min_motion_length:
                    continue
    

                if self.dataset_name in ["t2m" , "kit"]:

                    text_data = []
                    flag = False
                    with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < self.min_motion_len:
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    lengths.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)

                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text':text_data}
                        new_name_list.append(name)
                        lengths.append(len(motion))

                elif self.dataset_name == "aist":

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

    def set_stage(self, stage):

        lengths = list(np.array(np.logspace(np.log(self.min_motion_length), np.log(self.fps*self.max_length_seconds), 10, base=np.exp(1)) + 1 , dtype = np.uint))

        self.max_motion_length = lengths[stage]
        print(f'changing range to: {self.min_motion_length} - {self.max_motion_length}')
        
    
    def __len__(self):
        return len(self.data_dict)

    # def text_preprocessing(self , text_list):
    #     text_data = random.choice(text_list)
    #     caption, tokens = text_data['caption'], text_data['tokens']
    #     if len(tokens) < self.max_text_len:
    #         # pad with "unk"
    #         tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
    #         sent_len = len(tokens)
    #         tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
    #     else:
    #         # crop
    #         tokens = tokens[:self.max_text_len]
    #         tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
    #         sent_len = len(tokens)
    #     pos_one_hots = []
    #     word_embeddings = []
    #     for token in tokens:
    #         word_emb, pos_oh = self.w_vectorizer[token]
    #         pos_one_hots.append(pos_oh[None, :])
    #         word_embeddings.append(word_emb[None, :])
    #     pos_one_hots = np.concatenate(pos_one_hots, axis=0)
    #     word_embeddings = np.concatenate(word_embeddings, axis=0)

    #     return word_embeddings, pos_one_hots, caption, sent_len


    def __getitem__(self, item):

        data = self.data_dict[self.name_list[item]]
        condition = None


        if self.dataset_name in ["t2m" , "kit"]:
            motion, motion_len, text_list = data['motion'], data['length'], data['text']
            text_data = random.choice(text_list)
            caption= text_data['caption']
            condition = caption

        if self.dataset_name == "aist":
            motion, motion_len, music = data['motion'], data['length'], data['music']
            condition = music




                    
        try:
            self.window_size = np.random.randint(self.min_motion_length , min(motion_len , self.max_motion_length))
        except:
            self.window_size = self.min_motion_length

        
        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]


        if self.dataset_name == "aist":
            condition = (condition[idx:idx+self.window_size])


        return motion , self.id_list[item],condition


    
    

def DATALoader(
            dataset,
            batch_size,
            num_workers = 0,
            shuffle = True,
            collate_fn = None,
            sampler = None
           ):

    if collate_fn is None:
        collate_fn= simple_collate

    train_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size,
                                                shuffle=shuffle,
                                                sampler=sampler,
                                                num_workers=num_workers,
                                                collate_fn=collate_fn,
                                                drop_last = True)

    return train_loader