import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]) + '/model')
from show import show_params, show_model
#from sru import infer_SRU as SRU
from sru.infer_sru import SRU 
import torch.nn.functional as F
class Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, use_bias=True, padding='SAME'):
        super(Conv2d, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]), requires_grad=True)
        nn.init.normal_(self.kernel, std=0.05)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self.use_bias = use_bias
        self.padding = padding

    def forward(self, inputs):
        outputs = conv2d(inputs, self.kernel, padding=self.padding, bias=self.bias)
        return outputs

def conv2d(inputs, kernel, padding='SAME', bias=None):
    padding_shape = (
                        3,
                        2,
                        3,
                        2,
                    )
    padded_inputs = F.pad(inputs, padding_shape, 'constant', 0)
    return F.conv2d(padded_inputs, kernel, bias)

class SRUC(nn.Module):

    def __init__(
                self,
                input_dim=257,
                output_dim=257,
                hidden_layers=2,
                hidden_units=512,
                left_context=1,
                right_context=1,
                kernel_size=6,
                kernel_num=9,
                dropout=0.2
        ):
        super(SRUC, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.left_context = left_context
        self.right_context = right_context
        self.kernel_size = kernel_size
        self.kernel_sum = kernel_num

        self.input_layer = nn.Sequential(
                nn.Linear((left_context+1+right_context)*input_dim, hidden_units),
                nn.Tanh()
            )
        
        self.rnn_layer = SRU(
                    input_size=hidden_units,
                    hidden_size=hidden_units,
                    num_layers=self.hidden_layers,
                    dropout=dropout,
                    rescale=True,
                    bidirectional=False,
                    layer_norm=False
            )
        
        self.conv2d_layer = nn.Sequential(
                #nn.Conv2d(in_channels=1,out_channels=kernel_num,kernel_size=(kernel_size, kernel_size), stride=[1,1],padding=(5,5), dilation=(2,2)),
                Conv2d(in_channels=1, out_channels=kernel_num, kernel_size=(kernel_size, kernel_size)),
                nn.Tanh(),
                nn.MaxPool2d(3,stride=1,padding=(1,1))
            )
        
        self.output_layer = nn.Sequential(
                nn.Linear(hidden_units*kernel_num, (left_context+1+right_context)*self.output_dim),
                nn.Sigmoid()
            )
        
        self.loss_func = nn.MSELoss(reduction='sum')
        #self.loss_func = nn.MSELoss()
        #show_model(self)
        #show_params(self)
        #self.apply(self._init_weights)
        #self.flatten_parameters()
    
    def forward(self, inputs):
        #inputs = inputs[0]
        outputs = self.input_layer(inputs)
#        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lens, batch_first=True)
#        outputs, _ = self.rnn_layer(packed_inputs)
#        outputs, lens = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        outputs = torch.transpose(outputs,0,1)
        outputs, _ = self.rnn_layer(outputs)
        outputs = torch.transpose(outputs,0,1)

        # reshape outputs to [batch_size, 1, length, dims]
        outputs = torch.unsqueeze(outputs, 1)
        # conv outputs to [batch_size, channels, length, dims]
        outputs = self.conv2d_layer(outputs)
        # conv outputs to [batch_size, dims, length, channels]
        outputs = torch.transpose(outputs, 1, -1)
        # conv outputs to [batch_size, length, dims, channels]
        outputs = torch.transpose(outputs, 1, 2)
        batch_size, max_len, dims, channels = outputs.size()

        outputs = torch.reshape(outputs, [batch_size, max_len, -1])
        mask = self.output_layer(outputs)
        #outputs = mask
        outputs = mask*inputs
        return outputs[:, :, self.left_context*self.output_dim:(self.left_context+1)*self.output_dim]

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params
    def loss(self, inputs, labels, lengths=None):
        if lengths is None:
            return self.loss_func(inputs, labels) 
        else:
            return self.loss_func(inputs, labels) / lengths.sum()
