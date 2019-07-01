import torch.onnx as tonnx
import os
import sys 
import torch
sys.path.append(os.path.dirname(sys.path[0]+'/model'))
sys.path.append('../model/')

from sruc import SRUC
from infer_cldnn import CLDNN
from misc import reload_for_eval

model = CLDNN()
model_path='../exp/bigdata_1k6h_cldnn_1_1_0.0008_16k_6_9/'
reload_for_eval(model, model_path, use_cuda=False)
inputs = torch.randn([1,500,257*3])
torch.onnx.export(model, (inputs,), 'gruc_torch.onnx')

import onnx
model = onnx.load('gruc_torch.onnx')

onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
