from math import floor
import time
from tokenize import Double
from FrEIA.modules.reshapes import HaarDownsampling
from FrEIA.modules.invertible_resnet import ActNorm
from .freia_custom_coupling import AffineCouplingOneSidedIRN, EnhancedCouplingOneSidedIRN
from bicubic_pytorch.core import imresize
from data import mnist8_iterator
import wandb
import FrEIA.framework as ff
import numpy as np
from typing import Iterable, Tuple, List
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dblock import db_subnet
from data2 import DataLoaders

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=200)

class BatchnormSequenceINN(ff.SequenceINN):
    # dims is in (c, w, h) format
    # batch normalisation is only applied on the first bnorm_channels channels of x and z.
    def __init__(self, *dims: int, bnorm_channels = 3, force_tuple_output=False):
        super().__init__(*dims)
        self.bnorm_channels = bnorm_channels
        print(bnorm_channels)
        #self.bn_scale = torch.ones(1, device=device)
        #self.bn_bias = torch.zeros(1, device=device)
    

    # inn x_or_z is in [(n, c, w, h), (n, c, w, h), ...] format
    def forward(self, x_or_z: torch.Tensor, c: Iterable[torch.Tensor] = None,
                rev: bool = False, jac: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        
        iterator = range(len(self.module_list))
        log_det_jac = 0

        if rev:
            iterator = reversed(iterator)

        # Apply batchnorm
        assert len(x_or_z) == 1, "Sequence batchnorm is not compatible with multiple batches"
        #x_or_z[0][:, :self.bnorm_channels], mean, std = self.__standardise_tensor(x_or_z[0][:, :self.bnorm_channels].clone(), 0, 1)

        # TODO: compute jacobian for batchnorm. Can use actnorm as a reference here. Will look something like:
        #jac = (log(std) * x_or_z.shape[1] * np.prod(self.dims_in[1:])).repeat(x_or_z[0].shape[0])

        # This code lifted from freia
        if torch.is_tensor(x_or_z):
            x_or_z = (x_or_z,)
        for i in iterator:
            if self.conditions[i] is None:
                x_or_z, j = self.module_list[i](x_or_z, jac=jac, rev=rev)
            else:
                x_or_z, j = self.module_list[i](x_or_z, c=[c[self.conditions[i]]],
                                                jac=jac, rev=rev)
            log_det_jac = j + log_det_jac

        # Reverse batchnorm using mean & std computed earlier
        if torch.is_tensor(x_or_z[0]):
            x_or_z = x_or_z[0]

        # ISSUE WITH THIS APPROACH:
        # we cannot output both 1. the mean and std of the print-able image, and 2. the mean and std of the y.
        # without the mean and std of the y, our reconstruction will be OFF what it was.
       # y_unadjusted = x_or_z[:, :self.bnorm_channels].clone()
        #x_or_z[:, :self.bnorm_channels], mean_y, std_y = self.__standardise_tensor(y_unadjusted, mean, std)
        
        if torch.is_tensor(x_or_z):
                    # TODO: put jac in here
                    x_or_z = (x_or_z,)

        return x_or_z if self.force_tuple_output else x_or_z[0], log_det_jac

# dims          is in (c, w, h) format
# inn x         is in [(n, c, w, h), (n, c, w, h), ...] format
# inn output    is in [(n, c, w, h), (n, c, w, h), ...] format
def IRN(*dims, ds_count=1, inv_per_ds=1, inv_final_level_extra=0, inv_first_level_extra=0, batchnorm=False):
    # SequenceINN takes dims in (c, w, h) format
    inn = BatchnormSequenceINN(*dims) if batchnorm else ff.SequenceINN(*dims)

    for d in range(ds_count):
        inn.append(HaarDownsampling, order_by_wavelet=True)
        inv_count = inv_per_ds
        if d==0: inv_count += inv_first_level_extra
        elif d==ds_count-1: inv_count += inv_final_level_extra
        for i in range(inv_count):
            inn.append(EnhancedCouplingOneSidedIRN, subnet_constructor=db_subnet)
    return inn.cuda() if device=="cuda" else inn

def standardise_tensor(x: torch.Tensor, new_mean=0, new_std=1):
    #print(f"Taking a tensor with mean {x.mean()} std {x.std()} and giving it mean {new_mean} std {new_std}")

    mean = x.mean()
    std = x.std()
    new_x = new_mean + new_std * (x - mean) / std

    #print(f"Returning with mean {new_x.mean()} std {new_x.std()}\n")
    return new_x, mean, std

class StraightThroughEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, out_fun):
        return out_fun(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quantize_ste(x):
    return StraightThroughEstimator.apply(x, lambda y : (torch.clamp(y, min=0, max=1) * 255.0).round() / 255.0)

def quantize_to_int_ste(x):
    return StraightThroughEstimator.apply(x, lambda y : (torch.clamp(y, min=0, max=1) * 255.0).round().int())

# output is in range [0, 1]
def sample_inn(inn, x: torch.Tensor, batchnorm=False):
    if device=="cuda": x = x.cuda()

    #x = torch.tensor(np.array(x), dtype=torch.float, device=device).reshape(-1, *dataloaders.sample_shape)
    # TODO: move this multiplication logic into the mnist8 data loader
    if x.shape[1]==1: x = x.repeat(1, 3, 1, 1)
    assert x.shape[1] == 3, f"Expected 3 channels, have {x.shape[1]}"
    # x.shape == (n, 3, w1, h1)

    if batchnorm:
        x, mean, std = standardise_tensor(x.clone(), 0, 1)

    y_and_z, jac = inn([x])
    y, z = y_and_z[:, :3], y_and_z[:, 3:]
    y = quantize_ste(y)

    # If y conforms to a standard normal dist, which we will try to force it to, we would have mean_y=0 and std_y=1
    if batchnorm:
        y_printable, mean_y, std_y = standardise_tensor(y.clone(), mean, std)
    else:
        mean_y, std_y = None, None

    ### To simulate real use of the network, I shouldn't refer to x after this point. We must assume that we are reconstructing from y_printable ONLY.
    ### However -> I also extract mean_y and std_y during *training only*.

    # y_and_z.shape == (n, c2, w2, h2)
    # y.shape == (n, 3, w2, h2)
    # z.shape == (n, c2-3, w2, h2)

    z_sample = torch.normal(torch.zeros_like(z), torch.ones_like(z))
    # z_sample.shape == (n, c2-3, w2, h2)

    ### Here, we assume that the y that was extracted in the forward pass has mean 0, std 1.
    ### This may not be true, but we can force it to be true through training.
    if batchnorm: 
        y_unprintable, _, _ = standardise_tensor(y_printable.clone(), 0, 1)
        y_and_z_sample = torch.cat((y_unprintable, z_sample), dim=1)
    else:
        y_and_z_sample = torch.cat((y, z_sample), dim=1)
    # y_and_z_sample.shape == (n, c2, w2, h2)
    x_recon_from_y, _ = inn([y_and_z_sample], rev=True)
    # x_recon_from_y.shape == (n, 3, w1, h1)        

    return x, y, z, x_recon_from_y, mean_y, std_y



# Wavelet:              in: c channels                                    out: 4c channels
# InvBlock:             in: 3 channels, 4c-3 channels                     out: 3 channels, 4c-3 channels
# ...
# (joined to 4c)

# Downscaling Module:   in: c channels                                    out: 4c channels