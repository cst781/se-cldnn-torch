import os
import sys
import torch
import soundfile as sf
import numpy as np
import scipy as sp
import scipy.io as sio
import time
import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp 
import torch.nn as nn
from scipy.signal import fftconvolve as  fftconv
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0])+'/tools/speech_processing_toolbox/')

import voicetool.base as voicebox
import voicetool.utils as utils

np.seterr(divide='ignore', invalid='ignore')
eps = np.finfo(np.float32).eps

def splice_feats(data, left, right):
    """Splice the utterance.
    Args:
        data: Numpy matrix containing the utterance features to be spliced.
             shape [frames, dim]
        left: left contexts.
        right: right contexts.
    Return:
        A numpy array containing the spliced features.
    """
    length, dims = data.shape
    sfeats = []
    # left 
    if left != 0:
        for i in range(left, 0, -1):
            t = data[:length-i]
            for j in range(i):
                t = np.pad(t, ((1, 0), (0, 0)), 'symmetric')
            sfeats.append(t)
    # current 
    sfeats.append(data)
    # right
    if right != 0:
        for i in range(1,right+1):
            t = data[i:]
            for j in range(i):
                t = np.pad(t, ((0, 1 ), (0, 0)), 'symmetric')
            sfeats.append(t)
    return np.concatenate(np.array(sfeats), 1)

def activelev(data):
    #max_val = 1 / np.max(np.abs(data))
    #data = data / np.std(data)
    max_val = (1. + eps) /( np.std(data) + eps)
    data = data * max_val
    return data

def audioread(path):
    wave_data, sr = sf.read(path)
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:,0]
    return wave_data

def convolution( cln_wav , rir_wav):
    """
    single channel add reverb 
    """
    rir_wav = np.array(rir_wav)
    wav_tgt = sp.convolve(cln_wav, rir_wav)
    #wav_tgt = fftconv(cln_wav, rir_wav)
    #max_idx = np.argsort(rir_wav)[-1]
    #wav_tgt = wav_tgt / np.max(np.abs(wav_tgt)) * np.max(np.abs(cln_wav))
    wav_len = len(cln_wav)
    #wav_tgt = wav_tgt[1: wav_len]
    wav_tgt = wav_tgt[:wav_len]
    return wav_tgt

def addnoise(clean_path, noise_path, rir_path, scale, snr, start=None, segement_length=None):
    '''
    if rir is not None, the speech of noisy has reverberation
    and return the clean with reverberation
    else no reverberation
    Args:
        :@param clean_path: the path of a clean wav
        :@param noise_path: the path of a noise wav
        :@param rir_path: the path of a rir for different room size
        :@param start: the start point of the noise wav 
        :@param scale: the scale factor to control volume
        :@param snr:   the snr when add noise
    Return:
        :@param Y: noisy wav
        :@oaram X: clean wav
    '''
    clean = audioread(clean_path)
    noise = audioread(noise_path)
    # cut segment length 
    if (segement_length is not None and start is not None):
        if start == -1:
            length = clean.shape[0]
            stack_length = segement_length - length
            stacked_inputs = clean[:stack_length]
            clean = np.concatenate([clean, stacked_inputs], axis=0)
        elif start > -1:
            clean = clean[start:start+segement_length]
    noise_length = noise.shape[0]
    clean_length = clean.shape[0]
    clean_snr = snr/ 2
    noise_snr = -snr / 2
    clean_weight = 10**(clean_snr/20)
    noise_weight = 10**(noise_snr/20)
    #select the start index for all channel
    
    if clean_length > noise_length:
        start = np.random.randint(clean_length - noise_length)
        noise_selected = np.zeros(clean.shape)
        noise_selected[start:start+noise_length] = noise
    elif clean_length < noise_length:
        start = np.random.randint(noise_length - clean_length)
        noise_selected = noise[start:start+clean_length]
    else:
        noise_selected = noise
    
    noise_n = activelev(noise_selected)
    clean_n = activelev(clean)
    clean_w = clean_n * clean_weight
    noise_w = noise_n * noise_weight
    noisy = clean_w + noise_w
    max_amp = np.max(np.abs([noise_w, clean_w, noisy]))
    
    mix_scale = 1 / max_amp * scale
    X = clean_w * mix_scale
    Y = noisy * mix_scale
    return Y, X


class Processer(object):

    def __init__(
                self, 
                global_vars,
                snr_range=[-5, 20],
                scale=0.9
            ):

        # splice config
        self.left = global_vars['left']
        self.right = global_vars['right']

        # fft config
        self.sample_rate = global_vars['sample_rate']
        self.frame_length = int(global_vars['frame_length']*self.sample_rate/1000)
        self.frame_shift = int(global_vars['frame_shift']*self.sample_rate/1000)
        self.window_type  = global_vars['window_type']
        self.preemphasis = global_vars['preemphasis']
        self.square_root_window = global_vars['square_root_window']
        self.use_log = global_vars['use_log']
        self.use_power = global_vars['use_power']
        self.snr_range = snr_range
        self.window=np.hamming(self.frame_length)
        self.scale = scale

    def process(self, clean_wav_path, noise_wav_path, rir_wave_path, start=-2, segement_length=None, randstate=None):
       
        if randstate is None:
            randstate = np.random.RandomState(len(clean_wav_path))
        #select the SNR range
        if isinstance(self.snr_range, list):
            #snr = (self.snr_range[-1] - self.snr_range[0]) * randstate.ranf() + self.snr_range[0]
            snr = randstate.uniform(self.snr_range[0], self.snr_range[1])#(self.snr_range[-1] - self.snr_range[0]) * randstate.ranf() + self.snr_range[0]
        else:
            snr = snr
        # prepare the scale 
        #t = np.random.randint(-10, 5) / 10 + self.scale
        t = randstate.normal() * 0.5 + self.scale
        lower=0.3
        upper=0.9
        if t < lower or t > upper:
            t = randstate.uniform(lower, upper) 
        scale = t
        inputs, labels = addnoise(clean_wav_path, noise_wav_path, rir_wave_path, scale, snr, start=start, segement_length=segement_length)

        frame_inputs = voicebox.enframe(inputs, self.window, self.frame_length, self.frame_shift)
        fft_len = 2**int(np.log2(self.frame_length)+1)
        fft_inputs = np.fft.rfft(frame_inputs, fft_len)
        inputs = np.abs(fft_inputs)
        inputs = splice_feats(inputs, self.left, self.right)
        frame_labels = voicebox.enframe(labels, self.window, self.frame_length, self.frame_shift)
        fft_labels = np.fft.rfft(frame_labels, fft_len)
        labels = np.abs(fft_labels)
        labels = splice_feats(labels, self.left, self.right) 
        return inputs, labels


def zero_pad_concat(inputs):
    """
    the inputs : multi-channel feature
    """
    max_t = max(inp.shape[0] for inp in inputs)
    shape = np.array([len(inputs), max_t, inputs[0].shape[-1]])
    inputs_mat = np.zeros(shape, np.float32)
    for idx, inp in enumerate(inputs):
        #print(inputs_mat.shape, inp.shape)
        inputs_mat[idx, :inp.shape[0]] = inp
    return inputs_mat

def collate_fn(data):
    inputs, labels, lens = zip(*data)
    idx = sorted(enumerate(lens), key=lambda x :x[1], reverse=True)
    #get the index
    idx = [x[0] for x in idx]
    lens = [lens[x] for x in idx]
    padded_inputs = zero_pad_concat(inputs)
    padded_labels = zero_pad_concat(labels)
    return torch.from_numpy(padded_inputs[idx]), torch.from_numpy(padded_labels[idx]), torch.from_numpy(np.array(lens))

def parse_scp(scp, path_list):
    with open(scp) as fid:
        for line in fid:
            tmp = line.strip().split()
            if len(tmp) == 2 :
                path_list.append({'path': tmp[0], 'duration': float(tmp[1])})
            else:
                path_list.append({'path': tmp[0]})

class TFDataset(Dataset):
    def __init__(
                self,
                clean_scp,
                noise_scp,
                rir_scp,           
                processer,
                use_chunk=False,
                repeat=1,
                chunk_size=4,
                SAMPLE_RATE=16000
            ):
        
        super(TFDataset, self).__init__()
        mgr = mp.Manager()
        self.processer = processer
        self.clean_wav_list = mgr.list()
        self.noise_wav_list = mgr.list()
        self.rir_wav_list = mgr.list()
        self.segement_length = chunk_size * SAMPLE_RATE
        self.index = mgr.list()
        pc_list = []
        # read wav config list 
        p = mp.Process(target=parse_scp, args=(clean_scp, self.clean_wav_list))
        p.start()
        pc_list.append(p)        
        p = mp.Process(target=parse_scp, args=(noise_scp, self.noise_wav_list))
        p.start()
        pc_list.append(p)
        if rir_scp is not None:
            p = mp.Process(target=parse_scp, args=(rir_scp, self.rir_wav_list))
            p.start()
            pc_list.append(p)
        else:
            self.rir_wav_list = None
        for p in pc_list:
            p.join()
        if use_chunk:            
            self._dochuck(SAMPLE_RATE=SAMPLE_RATE)
        else:
            # if not do chunck the idx list is the wav_list idx
            self.index = [idx for idx in range(len(self.clean_wav_list))]
        # repeat index list for more input data
        self.size = len(self.index)
        self.num_states = self.size%10000
        self.randstates = [ np.random.RandomState(x+100) for x in range(self.num_states)]
        self.index *= repeat
        self.size *= repeat
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        randstate = self.randstates[index%self.num_states]
        item = self.index[index]
        noise = self.noise_wav_list[randstate.randint(len(self.noise_wav_list))]['path']
        if self.rir_wav_list is None:
            rir = None
        else:
            rir = self.rir_wav_list[randstate.randint(len(self.rir_wav_list))]['path']
        if isinstance(item, list):
            # if do chunck
            clean, start = item
            inputs, labels = self.processer.process(clean, noise, rir, start=start, segement_length=self.segement_length, randstate=randstate)
        else:
            clean = item
            clean = self.clean_wav_list[clean]['path']
            inputs, labels = self.processer.process(clean, noise, rir, randstate=randstate)
        return inputs, labels, inputs.shape[0]

    def _dochuck(self, SAMPLE_RATE=16000, num_threads=12):
        # mutliproccesing
        def worker(target_list, result_list, start, end, segement_length, SAMPLE_RATE):
            for item in target_list[start:end]:
                path = item['path']
                duration = item['duration']
                length = duration*SAMPLE_RATE
                if length < segement_length:
                    if length * 2 < segement_length:
                        continue
                    result_list.append([path, -1])
                else:
                    sample_index = 0
                    while sample_index + segement_length < length:
                        result_list.append(
                            [path, sample_index])
                        sample_index += segement_length
                    if sample_index != length - 1:
                        result_list.append([
                            path,
                            int(length - segement_length),
                        ])
        pc_list = []
        stride = len(self.clean_wav_list) // num_threads
        if stride < 100:
            p = mp.Process(
                            target=worker,
                            args=(
                                    self.clean_wav_list,
                                    self.index,
                                    0,
                                    len(self.clean_wav_list),
                                    self.segement_length,
                                    SAMPLE_RATE,
                                )
                        )
            p.start()
            pc_list.append(p)
        else: 
            for idx in range(num_threads):
                if idx == num_threads-1:
                    end = len(self.clean_wav_list)
                else:
                    end = (idx+1)*stride
                p = mp.Process(
                                target=worker,
                                args=(
                                    self.clean_wav_list,
                                    self.index,
                                    idx*stride,
                                    end,
                                    self.segement_length,
                                    SAMPLE_RATE,
                                )
                            )
                p.start()
                pc_list.append(p)
        for p in pc_list:
            p.join()

class Sampler(tud.sampler.Sampler):
    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i+batch_size)
                        for i in range(0, it_end, batch_size)]
        self.data_source = data_source
        
    def __iter__(self):
        np.random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)

def make_auto_loader(
                clean_scp, 
                noise_scp, 
                rir_scp=None, 
                batch_size=2, 
                use_chunk=False, 
                repeat=8, 
                SAMPLE_RATE=16000, 
                chunk_size=4, 
                num_threads=12,  
                num_workers=12,  
                processer=None):
    if(processer == None):
        #default config
        global_vars = {}
        global_vars['left'] = 1
        global_vars['right'] = 1
        # fft config
        global_vars['sample_rate'] = SAMPLE_RATE
        global_vars['frame_length'] = 25
        global_vars['frame_shift'] = 6.25
        global_vars['window_type'] = "hamming"
        global_vars['preemphasis'] = 0.0
        global_vars['square_root_window'] =  True
        global_vars['use_log']  = False
        global_vars['use_power'] = False
        processer = Processer(global_vars) 
    dataset = TFDataset(clean_scp,
                        noise_scp,
                        rir_scp,
                        processer,
                        use_chunk=use_chunk,
                        chunk_size=chunk_size,
                        repeat=repeat,
                        SAMPLE_RATE=SAMPLE_RATE)
    num_threads=num_workers
    sampler = Sampler(dataset, batch_size)
    if use_chunk:
        loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_threads,
                sampler=sampler,
                drop_last=False,
            )

    else:
        loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_threads,
                sampler=sampler,
                collate_fn=collate_fn,
                drop_last=True)
                #shuffle=True,
                #pin_memory=True,
    return loader

if __name__ == '__main__':
    clean_list  = '/search/speech/huyanxin/data/data_aishell/fixed_dev.lst'
    rir_list = None #'/home/work_nfs/mtxing/workspace/HW_Seperation/huawei_exp_dataset/data_simu_mix/rir.scp'
    noise_list = '/search/speech/huyanxin/data/musan/train.lst'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    loader = make_auto_loader(
        clean_scp=clean_list,
        noise_scp=noise_list,
        batch_size=16,
        num_workers=9,
        repeat=1,
        use_chunk=True
    )
    stime = time.time()
    print_freq = 100
    for (idx, item) in enumerate(loader):
        inputs = item[0]
        labels = item[1]
        lengths = item[2]
        if (idx+1)%print_freq== 0:
            etime = time.time()
            print((etime - stime)/print_freq, inputs.size())
            stime = etime

