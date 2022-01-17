import torch
import torch.nn as nn

class Permutation(nn.Module):
    
    def __init__(self, channels, device):
        super(Permutation, self).__init__()
        
        self.perm = torch.randperm(channels)
        self.inv_perm = torch.argsort(self.perm)
        
        self.to(device)
    
    def forward(self, x:torch.Tensor, log_jac_det=None, rev=False):
        if not rev:
            return x[self.perm], log_jac_det
        else:
            return x[self.inv_perm], log_jac_det