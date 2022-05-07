import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=32, bias=True, use_xavier=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in + 0 * gc, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + 1 * gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(0.2, True)
        
        # " Similar to [22], we also initialize the last convolution in all s and t subnetworks to zero, so training starts from an identity transform."
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            if use_xavier:
                init.xavier_normal_(layer.weight)
            else:
                nn.init.kaiming_normal_(layer.weight)
            layer.weight.data *= 0.1
            if layer.bias is not None: layer.bias.data.zero_()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))

        return x5

def db_subnet(channel_in, channel_out):
    return DenseBlock(channel_in, channel_out)