import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]) + '/model')
from show import show_params, show_model
import modules
from sru import SRU, SRUCell
import torch.nn.functional as F
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
                target_mode='MSA',
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
        self.target_mode = target_mode

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
                modules.Conv2d(in_channels=1, out_channels=kernel_num, kernel_size=(kernel_size, kernel_size)),
                nn.Tanh(),
                nn.MaxPool2d(3,stride=1,padding=(1,1))
            )
        
        self.output_layer = nn.Sequential(
                nn.Linear(hidden_units*kernel_num, (left_context+1+right_context)*self.output_dim),
                nn.Sigmoid()
            )
        #self.loss_func = nn.MSELoss(reduction='sum')
        #self.loss_func = nn.MSELoss()
        #show_model(self)
        #show_params(self)
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
        if self.target_mode == 'PSA' or self.target_mode == 'MSA':
            outputs = mask*inputs
            return outputs, outputs[:, :, self.left_context*self.output_dim:(self.left_context+1)*self.output_dim]
        elif self.target_mode == 'SPEC' or self.target_mode == 'TCS':
            outputs = mask 
            return outputs, outputs[:, :, self.left_context*self.output_dim:(self.left_context+1)*self.output_dim]
        else:
            outputs = mask 
            return outputs, (mask*inputs)[:, :, self.left_context*self.output_dim:(self.left_context+1)*self.output_dim]

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
    

if __name__ == '__main__':
    net = SRUC(left_context=0, right_context=0)
    import numpy as np
    inputs = torch.randn([10,100,257], requires_grad=False)
    labels = torch.randn([10,100,257], requires_grad=False)
    length = torch.from_numpy(np.array([ x for x in range(100,90)],dtype=np.float32))
    outputs = net(inputs)[0]
    loss = net.loss(outputs, labels, length)
    loss.backward()
