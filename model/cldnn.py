import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]) + '/model')
from show import show_params, show_model
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
                        3,
                        3,
                        3,
                    )
    padded_inputs = F.pad(inputs, padding_shape, 'constant', 0)
    return F.conv2d(padded_inputs, kernel, bias)

from modules import Conv2d

class CLDNN(nn.Module):

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
        super(CLDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.left_context = left_context
        self.right_context = right_context
        self.kernel_size = kernel_size
        self.kernel_sum = kernel_num
        super(CLDNN,self).__init__() 

        self.input_layer = nn.Sequential(
                nn.Linear((left_context+1+right_context)*input_dim, hidden_units),
                nn.Tanh()
            )
        
        self.rnn_layer = nn.GRU(
                    input_size=hidden_units,
                    hidden_size=hidden_units,
                    num_layers=hidden_layers,
                    dropout=dropout,
            )
        
        self.conv2d_layer = nn.Sequential(
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
        show_model(self)
        show_params(self)
        #self.apply(self._init_weights)
        #self.flatten_parameters()

    def flatten_parameters(self):
        self.rnn_layer.flatten_parameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for ih in param.chunk(3,0):
                        nn.init.xavier_uniform_(ih)
                elif 'weight_hh' in name:
                    for hh in param.chunk(3,0):
                        nn.init.orthogonal_(hh)
                elif 'bias_ih' in name:
                    nn.init.zeros_(param)

    def forward(self, inputs, lens=None):
        outputs = self.input_layer(inputs)


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
        return outputs, outputs[:, :, self.left_context*self.output_dim:(self.left_context+1)*self.output_dim]

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
if __name__ == '__main__':
    CLDNN()
