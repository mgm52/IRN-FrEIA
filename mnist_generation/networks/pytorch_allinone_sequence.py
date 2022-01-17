import torch
import torch.nn as nn
from .pytorch_allinone import AllInOne

class AllInOneSequence(nn.Module):
    
    def __init__(self, num_blocks, device, channels):
        super(AllInOneSequence, self).__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(AllInOne(channels,device))

        self.device = device
        self.to(device)
    
    def forward(self, x:torch.Tensor, log_jac_det=None, rev=False):
        if not rev:
            if log_jac_det is None:
                log_jac_det = torch.tensor(0.0, requires_grad=False, device=self.device)

            for b in self.blocks:
                x, log_jac_det = b(x, log_jac_det=log_jac_det, rev=rev)
        else:
            x = x.clone().detach()
            for b in reversed(self.blocks):
                x, _ = b(x, rev=rev)
        return x, log_jac_det