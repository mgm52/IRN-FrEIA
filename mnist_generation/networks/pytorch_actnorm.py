import torch
import torch.nn as nn

class ActNorm(nn.Module):
    
    def __init__(self, device):
        super(ActNorm, self).__init__()
        self.log_scale = None
        self.bias = None
        self.to(device)
        
    def forward(self, x:torch.Tensor, log_jac_det=None, rev=False):
        if self.bias is None:
            # Choose scale and bias such that we end up with zero mean and unit variance
            scale = torch.sqrt(((x - x.mean())**2).mean())
            self.log_scale = (-1 * torch.log(scale)).detach().clone()
            self.bias = (-1 * x.mean()).detach().clone()

        if not rev:
            # Apply bias, scale
            x += self.bias
            x *= torch.exp(self.log_scale)
            log_jac_det += self.log_scale.sum()
        else:
            # Inv scale, inv bias
            x *= torch.exp(-1 * self.log_scale)
            x += -1 * self.bias

        return x, log_jac_det