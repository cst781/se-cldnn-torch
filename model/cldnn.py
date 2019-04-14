import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]) + '/model')
from show import show_params, show_model
import modules

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
                    batch_first=True
            )
        
        self.conv2d_layer = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=9,kernel_size=(6,6), stride=[1,1],padding=(5,5), dilation=(2,2)),
                nn.Tanh(),
                nn.MaxPool2d(3,stride=1,padding=(1,1))
            )
        
        self.output_layer = nn.Sequential(
                nn.Linear(hidden_units*kernel_num, (left_context+1+right_context)*self.output_dim),
                nn.Sigmoid()
            )
        
        #self.loss_func = nn.MSELoss(reduction='sum')
        self.loss_func = nn.MSELoss()
        show_model(self)
        show_params(self)

    def forward(self, inputs, lens):
        outputs = self.input_layer(inputs)
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lens, batch_first=True)
        outputs, _ = self.rnn_layer(packed_inputs)
        outputs, lens = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = torch.unsqueeze(outputs, 1)
        outputs = self.conv2d_layer(outputs)
        #print(outputs.size())
        outputs = torch.transpose(outputs, 1, -1)
        outputs = torch.transpose(outputs, 1, 2)
        batch_size, max_len, dims, channels = outputs.size()
        outputs = torch.reshape(outputs, [batch_size, max_len, -1])
        #print(outputs.size())
        mask = self.output_layer(outputs)
        #print(outputs.size())
        #outputs = mask*inputs
        outputs = mask
        return outputs, outputs[:, :, self.left_context*self.output_dim:(self.left_context+1)*self.output_dim]

    def get_params(self, weight_decay):
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
    def loss(self,inputs, labels):
        return self.loss_func(inputs, labels) 
