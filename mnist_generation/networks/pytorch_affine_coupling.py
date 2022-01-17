import torch
import torch.nn as nn
import numpy as np

class AffineCoupling(nn.Module):
    
    def __init__(self, channels, device):
        super(AffineCoupling, self).__init__()
        self.channels = channels
        assert not self.channels<2, f'Cannot create a coupling layer with fewer than 2 channels. Channels supplied: {self.channels}'

        self.net = nn.Sequential(nn.Linear(self.channels//2, 512), nn.ReLU(), nn.Linear(512,  self.channels))
        self.to(device)

    def forward(self, x:torch.Tensor, log_jac_det=None, rev=False):
        x_a, x_b  = x.chunk(2)
        print(x)
        print(x_a)
        print(x_b)
        log_s, t = self.net(x_b).chunk(2)

        # Previous authors use sigmoid instead of exp here for stability
        # (presumably dy/dx could become too large otherwise)
        s = torch.sigmoid(log_s + 2.)

        if not rev:
            log_jac_det += torch.log(torch.abs(s)).sum()
            y_a = s * x_a + t
            y_b = x_b
        else:
            y_a = (x_a  - t) / s
            y_b = x_b
        
        y = torch.cat([y_a, y_b])
        return y, log_jac_det