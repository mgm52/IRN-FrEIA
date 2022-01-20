import FrEIA.framework
from FrEIA.modules import coupling_layers
from typing import Callable, Union
import torch

class AffineCouplingOneSidedIRN(coupling_layers._BaseCouplingBlock):
    '''Half of a coupling block following the GLOWCouplingBlock design.  This
    means only one affine transformation on half the inputs.  In the case where
    random permutations or orthogonal transforms are used after every block,
    this is not a restriction and simplifies the design.  '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN"):
        '''
        Additional args in docstring of base class.
        Args:
          subnet_constructor: function or class, with signature
            constructor(dims_in, dims_out).  The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples. One subnetwork will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation)
        # The first 3 channels are always taken as h1 in our IRN coupling block
        # This way, the high res image is taken across the network
        # IRN paper explains that this is because "the shortcut connection is proved to be important in the image scaling tasks"
        self.split_len1 = 3
        self.split_len2 = self.channels - 3
        self.subnet = subnet_constructor(self.split_len1 + self.condition_length, 2 * self.split_len2)


    def forward(self, x, c=[], rev=False, jac=True):
        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)
        x1_c = torch.cat([x1, *c], 1) if self.conditional else x1

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a = self.subnet(x1_c)
        s, t = a[:, :self.split_len2], a[:, self.split_len2:]
        s = self.clamp * self.f_clamp(s)
        j = torch.sum(s, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y2 = (x2 - t) * torch.exp(-s)
            j *= -1
        else:
            y2 = x2 * torch.exp(s) + t

        return (torch.cat((x1, y2), 1),), j