import torch
from torch import nn
from torch.nn import Module
from .build import RNN_REG
from ..base import build_conv_layer, build_norm_layer
from ..common import init_focal_loss_head, init_head_gaussian, init_head_kaiming, init_head_xavier

@RNN_REG.register()
class ConvRNN(Module):
    def __init__(self, input_channel, output_channel, conv_config, activation='tanh', bidirectional=True, num_layers=1, post_net_conv_config=None):
        '''
            input_num: The input feature map number 
            input_channel: the input channel number
            output_channel: the output channel number
        '''
        super().__init__()
        self.layer_num = num_layers
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.bidirectional = bidirectional
        conv_config = conv_config.copy()

        if bidirectional:
            self.direction_num = 2
        else:
            self.direction_num = 1

        if activation == 'tanh':
            self.activation = torch.tanh
            self.weight_init = init_head_xavier
        elif activation == 'relu':
            self.activation = torch.relu
            self.weight_init = init_head_kaiming
        else:
            raise NotImplementedError('Need to implement activation and weight init for activation {}'.format(activation))
        

        for i in range(1, self.layer_num+1):
            conv_config['in_channels'] = input_channel if i==1 else output_channel
            setattr(self, 'layer{}_ih_f'.format(i), build_conv_layer(conv_config))
            conv_config['in_channels'] = output_channel
            setattr(self, 'layer{}_hh_f'.format(i), build_conv_layer(conv_config))
            if self.bidirectional:
                conv_config['in_channels'] = input_channel if i==1 else output_channel
                setattr(self, 'layer{}_ih_b'.format(i), build_conv_layer(conv_config))
                conv_config['in_channels'] = output_channel
                setattr(self, 'layer{}_hh_b'.format(i), build_conv_layer(conv_config))

        self.weight_init(self)

        # This post_net must be construct after the weight init since it needs to be init seperatly
        if post_net_conv_config is not None:
            self.post_net = ReadOutNet(post_net_conv_config)
        else:
            self.post_net = None


    def forward(self, x):
        '''
            x(Tensor(TxNxCxHxW))): the input tensors, first dim is sequence length,
            second dim is batch size
        '''
        insize = x.shape[3:]
        batch_size = x.shape[1]
        hidden_all = x.new_zeros((self.layer_num*self.direction_num, batch_size, self.output_channel, *insize)) 

        input_forward = x
        input_backward = x
        for i in range(0, self.layer_num):
            h_forward = hidden_all[i*self.direction_num]
            output_forward = []
            for t in range(len(input_forward)):
                h_forward = self.activation(getattr(self, 'layer{}_hh_f'.format(i+1))(h_forward) + getattr(self, 'layer{}_ih_f'.format(i+1))(input_forward[t]) )
                output_forward.append(h_forward)
            #print(len(output_forward))
            #print(output_forward[0].shape)
            input_forward = torch.stack(output_forward)
            #print(input_forward.shape)

            if self.bidirectional:
                h_backward = hidden_all[i*self.direction_num+1]
                output_backward = []
                for t in range(len(input_backward)-1, -1, -1):
                    h_backward = self.activation(getattr(self, 'layer{}_hh_b'.format(i+1))(h_backward) + getattr(self, 'layer{}_ih_b'.format(i+1))(input_backward[t]) )
                    output_backward.append(h_backward)
                output_backward.reverse()
                input_backward = torch.stack(output_backward)
        if self.bidirectional:
            if self.post_net is not None:
                return self.post_net(x, input_forward, input_backward)
            else:
                return input_forward, input_backward
        else:
            if self.post_net is not None:
                return self.post_net(x, input_forward)
            else:
                return input_forward

                
class ReadOutNet(nn.Module):
    def __init__(self, conv_config):
        super().__init__()
        self.conv = build_conv_layer(conv_config)
        #init_head_xavier(self)
        init_focal_loss_head(self)

    def forward(self, x, *hidden):
        hidden_sum = sum(hidden)
        T,N,C,H,W = x.shape
        x = x.reshape(-1,C,H,W)
        x = self.conv(x)
        x = x.reshape(T,N,C,H,W)
        #out = torch.sigmoid_(x+hidden_sum)
        # sigmoid should be applied later
        out = x+hidden_sum

        return out
