
import torch
import torch.nn as nn
import numpy as np
from cldnn import CLDNN as model
def test_cldnn():
    test_model = model()
    inputs = torch.rand([10,129,257*3])
    labels = torch.rand([10,129,257*3])
    lens = [129,110,100,90,80,70,60,50,40,30]
    test_model(inputs, lens)
    print(test_model.loss(inputs, labels))
test_cldnn()
