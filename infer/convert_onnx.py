import torch.onnx as tonnx
import os
import sys 
import torch
import onnx
sys.path.append(os.path.dirname(sys.path[0]+'/model'))
sys.path.append('../model/')

from infer_cldnn import CLDNN
from infer_sruc import SRUC
sys.path.append('../tools/')
from misc import reload_for_eval

def torch2onnx(torch_model, inputs, model_path, model_name):
    reload_for_eval(torch_model, model_path, use_cuda=False)
    torch_model.eval()
    torch.onnx.export(torch_model, inputs, model_name)
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))
    #print(torch.abs(onnx_model(inputs)-torch_model(inputs)).sum())
example = torch.randn([1,500,257*3])

torch_cldnn = CLDNN()
cldnn_model_path='../exp/bigdata_1k6h_cldnn_1_1_0.0008_16k_6_9/'
torch2onnx(torch_cldnn, example, cldnn_model_path, 'gruc_1k6h_torch.onnx')

#torch_sruc = SRUC()
#sruc_model_path = '../exp/sruc_1_1_0.0005_16k_6_9/'
#torch2onnx(torch_sruc, example, sruc_model_path, 'sruc_2w_torch.onnx')
