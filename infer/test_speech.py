import torch
import time
import torch.nn
import os
import sys
sys.path.append('../model/')
from infer_cldnn import CLDNN
from infer_sruc import SRUC
from cldnn import CLDNN
from sruc import SRUC
sys.path.append('../tools/')
from misc import reload_for_eval

def test_speed(model, inputs, lens,times=1000, train=True):
    if train:
        model.train()
        stime = time.time()
        for idx in range(times):
            out = model(inputs)[0]
            if train == True:
                model.zero_grad()
                out.mean().backward()
        etime = time.time()
        print('train:') 
    else:
        model.eval()
        with torch.no_grad():
            stime = time.time()
            for idx in range(times):
                out = model(inputs)
            etime = time.time()
        print('eval:') 
    print(etime-stime)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    sruc = SRUC().cuda()
    reload_for_eval(sruc, '../exp/sruc_1_1_0.0005_16k_6_9/',True)
    gruc = CLDNN().cuda()
    reload_for_eval(gruc, '../exp/bigdata_1k6h_cldnn_1_1_0.0008_16k_6_9/',True)
    #example = torch.randn([1,1000,257*3],required_grad=False)
    example = torch.randn([1,500,257*3]).cuda()
    lens=[500,500,500]
    print('test gruc')
    test_speed(gruc, example, lens)
    test_speed(gruc, example, lens,train=False)
    print('test sruc')
    test_speed(sruc, example, lens)
    test_speed(sruc, example, lens, train=False)
