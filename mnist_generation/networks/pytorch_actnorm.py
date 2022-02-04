import torch
import torch.nn as nn

class ActNorm(nn.Module):
    
    def __init__(self, input_size, device):
        super(ActNorm, self).__init__()

        # Ensure our starting scale is 1
        self.unsoftplus_scale = nn.Parameter(2 * torch.log(torch.exp(torch.ones(input_size)/2)-1))
        self.beta = 0.5

        self.bias = nn.Parameter(torch.zeros(input_size))
        self.initialized_actnorm = False
        self.to(device)
        
    def forward(self, x:torch.Tensor, log_jac_det=None, rev=False):
        # Data-dependent intitialisation disabled for now as it seems to have negative impact
        if False and not self.initialized_actnorm:
            std_dev = torch.sqrt(((x - x.mean())**2).mean())

            # Choose scale and bias such that we end up with zero mean and unit variance on initial sample
            self.log_scale = nn.Parameter((-1 * torch.log(std_dev)).detach().clone())
            self.bias = nn.Parameter((-1 * x.mean()).detach().clone())
            print("Actnorm using initial scale: e^" + str(self.log_scale))
            print("Actnorm using initial bias: " + str(self.bias.data))
            self.initialized_actnorm = True

        # To ensure scale is mostly positive: scale = softplus(unsoftplus_scale)
        scale = (1/self.beta) * torch.log(1 + torch.exp(self.beta * self.unsoftplus_scale))

        if not rev:
            # Apply bias, scale
            x = x * scale + self.bias
            log_jac_det += torch.sum(torch.log(scale))
        else:
            # Inv scale, inv bias
            x = (x - self.bias) / scale
        return x, log_jac_det