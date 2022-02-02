import torch
import torch.nn as nn
from .pytorch_actnorm import ActNorm
from .pytorch_permutation import Permutation
from .pytorch_affine_coupling import AffineCoupling

class AllInOne(nn.Module):
    
    def __init__(self, channels, device, subnet_constructor):
        super(AllInOne, self).__init__()
        self.actnorm  = ActNorm(device)
        self.coupling = AffineCoupling(channels, device, subnet_constructor)
        self.permutation  = Permutation(channels, device)
        self.to(device)        

    def forward(self, x:torch.Tensor, log_jac_det=None, rev=False):
        if not rev:
            assert log_jac_det is not None, "no log determinant supplied to AllInOne forward call"
            assert len(x.shape) == 1, "x is not 1-dimensional before being given to actnorm layer"
            x, log_jac_det = self.actnorm(x, log_jac_det=log_jac_det, rev=rev)
            
            assert len(x.shape) == 1, "x is not 1-dimensional before being given to permutation layer"
            x, log_jac_det = self.permutation(x, log_jac_det, rev=rev)

            assert len(x.shape) == 1, "x is not 1-dimensional before being given to coupling layer"
            x, log_jac_det = self.coupling(x, log_jac_det=log_jac_det, rev=rev)
            
        else:
            x, _ = self.coupling(x, rev=rev)
            x, _ = self.permutation(x, rev=rev)
            x, _ = self.actnorm(x, rev=rev)
        
        return x, log_jac_det