import os
import sys
sys.path.append('./speech_processing_toolbox/')
import voicetool
import voicetool.base as voicebox
import dataset 
import tqdm
import time
import numpy as np
def test_dataset():
    sample_rate = 16000
    processer = dataset.Processer()
    with open('wav.lst') as fid:
        with open('test.scp', 'w') as wfid:
            for line in fid:
                path = line.strip()
                data = voicebox.audioread(path)
                lenth = data.shape[0]
                wfid.writelines(path+' '+path+' '+str(lenth/sample_rate)+'\n')
    data = dataset.TFDataset(processer, './test.scp')
    print(len(data))
    print(data[0])


def test_dataloader():
    sample_rate = 16000
#    with open('wav.lst') as fid:
#        with open('test.scp', 'w') as wfid:
#            for line in fid:
#                path = line.strip()
#                data = voicebox.audioread(path)
#                lenth = data.shape[0]
#                wfid.writelines(path+' '+path+' '+str(lenth/sample_rate)+'\n')

    p = dataset.Processer()
    loader = dataset.make_loader('test.scp', 100, 15)
    print(len(loader))
    for epoch in range(10):
        etime = time.time()
        for x in loader:
            print(x[0].shape)
            print(time.time()-etime)
            etime = time.time()

#test_dataset()
test_dataloader()
