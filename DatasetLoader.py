#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import math
import pdb
import glob
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

class meta_loader(Dataset):
    def __init__(self, train_path, train_ext, transform):
        
        ## Read Training Files
        # data/train/id10051/B00000000.jpg
        
        files_ee488b = glob.glob('%s/id*/*.%s'%(train_path,train_ext))
        # files_vggface2_1 = glob.glob(('%s/n000*/*.%s'%(train_path,train_ext)))
        # files_vggface2_2 = glob.glob(('%s/n001*/*.%s'%(train_path,train_ext)))
        # files_vggface2_3 = glob.glob(('%s/n002*/*.%s'%(train_path,train_ext)))
        # files_vggface2 = glob.glob(('%s/n*/*.%s'%(train_path,train_ext)))
        files = files_ee488b

        ## Make a mapping from Class Name to Class Number
        dictkeys = list(set([x.split('/')[-2] for x in files]))
        dictkeys.sort()
        # {id10051: 0, id10052: 1, ...}
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        self.transform  = transform

        self.label_dict = {} #{0: [0,1,...], 1: [126,127,...], 2: [...]}
        self.data_list  = [] #[data/train/id10051/B00000000.jpg, data/train/id10051/B00000001.jpg, ...]
        self.data_label = [] #[0, 0, ..., 1, ..., 2, ...]
        
        for lidx, file in enumerate(files):
            speaker_name = file.split('/')[-2]
            speaker_label = dictkeys[speaker_name];

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = [];

            self.label_dict[speaker_label].append(lidx);
            
            self.data_label.append(speaker_label)
            self.data_list.append(file)

        print('{:d} files from {:d} classes found.'.format(len(self.data_list),len(self.label_dict)))

    def __getitem__(self, indices):
        # indices: batch indices
        feat = []
        for index in indices:
            feat.append(self.transform(Image.open(self.data_list[index])));
        feat = numpy.stack(feat, axis=0) #[len_indices, C, W, H]

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):

        return len(self.data_list)

class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, transform, **kwargs):
        self.test_path  = test_path
        self.data_list  = test_list
        self.transform  = transform

    ## test one sample from the data_list
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.test_path, self.data_list[index]))
        return self.transform(img), self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class meta_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerClass, max_img_per_cls, batch_size):

        self.label_dict         = data_source.label_dict
        self.nPerClass          = nPerClass
        self.max_img_per_cls    = max_img_per_cls;
        self.batch_size         = batch_size;

        self.num_iters          = 0
        
    def __iter__(self):
        
        ## Get a list of identities
        dictkeys = list(self.label_dict.keys()); # [id10051, id10052, ...]
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)] # chop lst w.r.t sz(size) # [len(lst)/sz, sz]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data    = self.label_dict[key] # sample idices for id10051
            numSeg  = round_down(min(len(data),self.max_img_per_cls),self.nPerClass)
            
            rp      = lol(numpy.random.permutation(len(data))[:numSeg],self.nPerClass)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices]) #[[0,2,5,6,9], [1,3,4,7,8]] rp == 2, nPerClass == 5

        ## Data in random order
        mixid           = numpy.random.permutation(len(flattened_label))
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size # index where this batch start
            if flattened_label[ii] not in mixlabel[startbatch:]: # randomly choose diff identities(batch size)
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        batch_indices = [flattened_list[i] for i in mixmap] # for chosen identities get nPerClass samples per each identity #[batch_size, nPerClass]

        self.num_iters = len(batch_indices)

        return iter(batch_indices)
    
    def __len__(self):
        return self.num_iters

def get_data_loader(batch_size, max_img_per_cls, nDataLoaderThread, nPerClass, train_path, train_ext, transform, **kwargs):
    
    train_dataset = meta_loader(train_path, train_ext, transform)

    train_sampler = meta_sampler(train_dataset, nPerClass, max_img_per_cls, batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    
    return train_loader


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst