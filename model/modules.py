import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
class Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, use_bias=True, padding='SAME'):
        super(Conv2d, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]), requires_grad=True)
        nn.init.normal_(self.kernel, std=0.05)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self.use_bias = use_bias
        self.padding = padding

    def forward(self, inputs):
        if self.use_bias == None:
            outputs = conv2d(inputs, self.kernel, padding=self.padding, bias=None)
        else:
            outputs = conv2d(inputs, self.kernel, padding=self.padding, bias=self.bias)
        return outputs

def conv2d(inputs, kernel, padding='SAME', bias=None):
    if padding == 'SAME':
        padding_size = [
                        kernel.size(-1),
                        kernel.size(-1),
                        kernel.size(-2),
                        kernel.size(-2),
                    ]
        if padding_size[1] % 2 ==0:
            padding_size[1] -=1
        if padding_size[3] % 2 ==0:
            padding_size[3] -=1
        padding_shape = (
                        padding_size[0]//2,
                        padding_size[1]//2,
                        padding_size[2]//2,
                        padding_size[3]//2,
                    )
    elif padding == 'VALID':
        padding_shape = ()
    padded_inputs = F.pad(inputs, padding_shape, 'constant', 0)
    return F.conv2d(padded_inputs, kernel, bias)


def Conv1d_Same(inputs, kernel, context=None, bias=None):
    # [batchsize, ch, length
    if context is None:
        padding_shape = (kernel.size(-1)/2, kernel.size(-1)/2)
        padded_inputs = F.pad(inputs, padding_shape, 'constant', 0)
        return F.conv1d(padded_inputs, kernel, bias=bias)     
    else:
        if context[-1] < 0:
            padding_shape = (np.abs(context[0]), )
        else:
            padding_shape = (np.abs(context[0]), np.abs(context[-1]))
        padded_inputs = F.pad(inputs, padding_shape, 'constant', 0)
        return F.conv1d(padded_inputs, kernel, bias)


class MSE(autograd.Function):

    def __init__(self):
        super(MSE, self).__init__()
    
    @staticmethod
    def forward(ctx, inputs, labels, length=None, need_mask=False):

        if need_mask == True and length is not None:
            numpy_length = length.numpy()
            mask = np.zeros([numpy_length.shape[0],  numpy_length[0], inputs.size(-1)])
            for idx in range(numpy_length.shape[0]):
                mask[idx, :numpy_length[idx], :] = 1 
            return F.mse_loss(inputs*mask, labels,reduction='sum')/length.sum()
        else:
            return F.mse_loss(inputs, labels, reduction='sum')/length.sum()

    @staticmethod
    def backward(ctx, grad):
        return grad

if __name__ == '__main__':
    test_conv1d = False
    test_conv2d = False
    test_mse = True
    if test_conv1d:
        context = [-1, 3]
        input_dim=3
        output_dim=4
        kernel_width=context[-1] - context[0] + 1
        inputs = torch.randn(2, input_dim, 1341)
        kernels = nn.Parameter(torch.Tensor(output_dim, input_dim, kernel_width))
    #    print('inputs.size', inputs.size())
    #    print('kernels.size',kernels.size())
    #    print('out.size',Conv1d_Same(inputs, kernels, context=context).size())
    if test_conv2d : 
        inputs = torch.randn(4, 1, 447,100)
        layer = Conv2d(1,15, kernel_size=(3,6))
        outputs = layer(inputs)
        print(inputs.shape, outputs.shape)
    if test_mse:
        mse = MSE()
        inputs = torch.randn([10,100,257], requires_grad=True)
        labels = torch.randn([10,100,257], requires_grad=False)
        length = torch.from_numpy(np.array([ x for x in range(100,90,-1)],dtype=np.float32))
        #loss = F.mse_loss(inputs, labels, reduction='sum')/length.sum()
        loss = mse.apply(inputs, labels, length)
        loss.backward()
