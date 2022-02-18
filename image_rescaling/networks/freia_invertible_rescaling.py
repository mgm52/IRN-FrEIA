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
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dblock import db_subnet
from data2 import DataLoaders

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=200)

#def subnet(channels_in, channels_out):
    # Usually this is 256 channels
#    return nn.Sequential(nn.Conv2d(channels_in, 256, 3, 1, 1), nn.ReLU(),
#                         nn.Conv2d(256, channels_out, 3, 1, 1))

# dims          is in (c, w, h) format
# inn x         is in [(n, c, w, h), (n, c, w, h), ...] format
# inn output    is in [(n, c, w, h), (n, c, w, h), ...] format
def IRN(*dims, ds_count=1, inv_per_ds=1):
    # SequenceINN takes dims in (c, w, h) format
    inn = ff.SequenceINN(*dims)
    for d in range(ds_count):
        inn.append(HaarDownsampling, order_by_wavelet=True)
        for i in range(0, inv_per_ds):
            inn.append(EnhancedCouplingOneSidedIRN, subnet_constructor=db_subnet)
    return inn.cuda() if device=="cuda" else inn

def sample_inn(inn, x: torch.Tensor, use_test_set=True):
    x = x * 16 # see if moving from 0-1 range to 0-16 range fixes div2k noise

    if device=="cuda": x = x.cuda()

    #x = torch.tensor(np.array(x), dtype=torch.float, device=device).reshape(-1, *dataloaders.sample_shape)
    # TODO: move this multiplication logic into the mnist8 data loader
    if x.shape[1]==1: x = x.repeat(1, 3, 1, 1)
    assert x.shape[1] == 3, f"Expected 3 channels, have {x.shape[1]}"
    # x.shape == (n, 3, w1, h1)

    y_and_z, jac = inn([x])
    y, z = y_and_z[:, :3], y_and_z[:, 3:]
    # y_and_z.shape == (n, c2, w2, h2)
    # y.shape == (n, 3, w2, h2)
    # z.shape == (n, c2-3, w2, h2)

    #x_recon_from_y_and_z, _ = inn([y_and_z], rev=True)

    z_sample = torch.normal(torch.zeros_like(z), torch.ones_like(z))
    # z_sample.shape == (n, c2-3, w2, h2)
    y_and_z_sample = torch.cat((y, z_sample), dim=1)
    # y_and_z_sample.shape == (n, c2, w2, h2)
    x_recon_from_y, _ = inn([y_and_z_sample], rev=True)
    # x_recon_from_y.shape == (n, 3, w1, h1)

    return x, y, z, x_recon_from_y



# Wavelet:              in: c channels                                    out: 4c channels
# InvBlock:             in: 3 channels, 4c-3 channels                     out: 3 channels, 4c-3 channels
# ...
# (joined to 4c)

# Downscaling Module:   in: c channels                                    out: 4c channels