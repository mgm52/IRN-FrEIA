import FrEIA.framework
from FrEIA.modules import coupling_layers
from typing import Callable, Union
import torch

# Performs an enhanced coupling operation (combining additive and affine), splitting off the first 3 channels.
# The additive transformation is what allows the model to alter LR.
class EnhancedCouplingOneSidedIRN(coupling_layers._BaseCouplingBlock):
    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "SIGMOID"):
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

        self.phi = subnet_constructor(self.split_len2, self.split_len1) 
        # idea: try using batchnorm on these two?
        self.rho = subnet_constructor(self.split_len1, self.split_len2)
        self.mu = subnet_constructor(self.split_len1, self.split_len2)


    def forward(self, x, c=[], rev=False, jac=True):
        # self.conditional not supported
        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)

        # INVBLOCK: Y1 ADDITIVE, Y2 AFFINE
        # y1 = x1 + phi(x2)
        # y2 = x2 * exp(p(y1)) + n(y1))

        # REALNVP: Y1 IDENTITY, Y2 AFFINE
        # y1 = x1
        # y2 = x2 * exp(p(y1)_1) + p(y1)_2
        
        j = -1
        if rev:
            rho_x1 = self.rho(x1)
            rho_x1 = self.clamp * self.f_clamp(rho_x1)
            y2 = (x2 - self.mu(x1)) / torch.exp(rho_x1)
            y1 = x1 - self.phi(y2)

            # TODO: Check that this jacobian calcuation is correct
            if jac: j = -1 * torch.sum(rho_x1, dim=tuple(range(1, self.ndims + 1)))
        else:
            # f_clamp is the clamp function, chosen by clamp_activation. in the IRN paper, this is sigmoid.
            # Note: many authors replace exp with sigmoid, to prevent values exploding
            # The IRN doesn't replace exp, but it does buffer it with sigmoid
            y1 = x1 + self.phi(x2)
            rho_y1 = self.rho(y1)
            rho_y1 = self.clamp * self.f_clamp(rho_y1) # note that freia's SIGMOID clamp performs 2*sigmoid - 1 for us.
            y2 = x2 * torch.exp(rho_y1) + self.mu(y1)
            # TODO: Check that this jacobian calcuation is correct (not that we use it...)
            if jac: j = torch.sum(rho_y1, dim=tuple(range(1, self.ndims + 1)))

        return (torch.cat((y1, y2), 1),), j


# Performs an affine coupling operation (realNVP), splitting off the first 3 channels
class AffineCouplingOneSidedIRN(coupling_layers._BaseCouplingBlock):
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

        # INVBLOCK: Y1 ADDITIVE, Y2 AFFINE
        # y1 = x1 + phi(x2)
        # y2 = x2 * exp(p(y1)) + n(y1))

        # REALNVP: Y1 IDENTITY, Y2 AFFINE
        # y1 = x1
        # y2 = x2 * exp(p(y1)_1) + p(y1)_2

        a = self.subnet(x1_c)
        # Note: the reason we take splits of a for s,t may be to avoid having separate networks for s,t
        # --> i.e. instead we can just have one network and use half of it for s, half for t
        s, t = a[:, :self.split_len2], a[:, self.split_len2:]
        s = self.clamp * self.f_clamp(s)
        j = torch.sum(s, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y2 = (x2 - t) * torch.exp(-s)
            j *= -1
        else:
            y2 = x2 * torch.exp(s) + t

        return (torch.cat((x1, y2), 1),), j