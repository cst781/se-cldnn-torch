#!/usr/bin/env python
# coding=utf-8
import numpy as np
import scipy
import torch 
import random
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data as tud
import os 
import sys
sys.path.append(os.path.dirname(sys.path[0])+'/tools')
sys.path.append(os.path.dirname(sys.path[0])+'/tools/speech_processing_toolbox/')
import voicetool.base as voicebox
import voicetool.utils as utils
import voicetool.multiworkers as worker
from misc import read_and_config_file

class DataReader(object):
    def __init__(self, file_name, win_len=400, win_inc=100,left_context=0,right_context=0, fft_len=512, window_type='hamming', sample_rate=16000):
        self.left_context = left_context
        self.left_context = left_context
        self.right_context = right_context
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = {
                        'hamming':np.hamming(self.win_len)/1.2607934,
                        'none':np.ones(self.win_len)
                      }[window_type]
        self.file_list = read_and_config_file(file_name, decode=True)

    def extract_feature(self, path):
        path = path['inputs']
        utt_id = path.split('/')[-1]
        
        data = voicebox.audioread(path)
        inputs = voicebox.enframe(data, self.window, self.win_len,self.win_inc)
        inputs = np.fft.rfft(inputs, n=self.fft_len)
        sinputs = utils.splice_feats(np.abs(inputs).astype(np.float32), left=self.left_context, right=self.left_context)
       
        length, dims = sinputs.shape
        sinputs = np.reshape(sinputs, [1, length, dims])
        nsamples = data.shape[0]
        return sinputs, [length], np.angle(inputs), utt_id, nsamples

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])


class Processer(object):

    def __init__(self, win_len=400, win_inc=100,left_context=0,right_context=0, fft_len=512, window_type='hamming'):
        self.left_context = left_context
        self.right_context = right_context
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = {
                        'hamming':np.hamming(self.win_len)/1.2607934,
                        'none':np.ones(self.win_len)
                      }[window_type]
    def process(self, path):

        inputs = voicebox.audioread(path['inputs'])
        inputs = voicebox.enframe(inputs, self.window, self.win_len,self.win_inc)
        inputs = voicebox.fft(inputs, self.fft_len)
        sinputs = utils.splice_feats(inputs, left=self.left_context, right=self.left_context)
        
        labels = voicebox.audioread(path['labels'])
        labels = voicebox.enframe(labels, self.window, self.win_len, self.win_inc)
        labels = voicebox.fft(labels, self.fft_len)
        slabels = utils.splice_feats(labels, left=self.left_context, right=self.left_context)
        
        return sinputs, slabels, sinputs.shape[0]

class TFDataset(Dataset):

    def __init__(self, scp_file_name, processer=Processer()):
        '''
            wave_list: input_wave_path, output_wave_path, duration
            processer: a processer class to handle wave data
        '''
        self.wave_list = read_and_config_file(scp_file_name)
        self.processer = processer
#        max_len = max(int(x['duration']) for x in self.wave_list)
#        bucket_diff = 1
#        num_buckets = max_len // bucket_diff
#        buckets = [[] for _ in range(num_buckets)]
#        for x in self.wave_list:
#            bid = min(int(x['duration'])// bucket_diff, num_buckets -1 )
#            buckets[bid].append(x)

#        sort_fn = lambda x: round(x['duration'], 1)
#        for b in buckets:
#            b.sort(key=sort_fn)
        self.data_list = self.wave_list #[d for b in buckets for d in b]

    def __len__(self):
        return len(self.wave_list)

    def __getitem__(self, index):
        return self.processer.process(self.data_list[index])

class Sampler(tud.sampler.Sampler):
    '''
     
    '''
    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i+batch_size)
                        for i in range(0, it_end, batch_size)]
        self.data_source = data_source
        
    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)


def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = np.array([len(inputs), max_t, inputs[0].shape[1]])
    inputs_mat = np.zeros(shape, np.float32)
    for idx, inp in enumerate(inputs):
        inputs_mat[idx, :inp.shape[0],:] = inp
    return inputs_mat

def collate_fn(data):
    inputs, labels, lens = zip(*data)
    idx = sorted(enumerate(lens), key=lambda x:x[1], reverse=True)
    idx = [x[0] for x in idx]
    lens = [lens[x] for x in idx]
    padded_inputs = zero_pad_concat(inputs)
    padded_labels = zero_pad_concat(labels)
    return torch.from_numpy(padded_inputs[idx]), torch.from_numpy(padded_labels[idx]), torch.from_numpy(np.array(lens))

def make_loader(scp_file_name, batch_size, num_workers=4, processer=Processer()):
    dataset = TFDataset(scp_file_name, processer)
    sampler = Sampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=sampler,
                            collate_fn=collate_fn,
                            drop_last=False
                        )
                            #shuffle=True,
    return loader, None #, dataset
