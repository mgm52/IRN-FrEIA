import torch.nn as nn
import torch
import data as data_functions
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(),
                         nn.Linear(512,  dims_out))

def subnet_conv(dims_in, dims_out):
    return nn.Sequential(nn.Conv2d(dims_in, 256, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(256,  dims_out, 3, padding=1))

def inn_fc():
    inn = Ff.SequenceINN(8 * 8)
    for k in range(8):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    return inn

def inn_conv():
    inn = Ff.SequenceINN(8, 8)
    for k in range(8):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_conv, permute_soft=True)
    return inn

def sample_inn_fc_mnist8(inn, num_samples):
    n_dim = (8 * 8)
    # Generate samples approximating the mnist8 distribution, by giving the network samples from a gaussian dist
    for i in range(num_samples):
        z = torch.randn(8 * 8)
        samples, _ = inn(z, rev=True)
        data_functions.see_mnist8(samples.detach().numpy())

def sample_inn_conv_mnist8(inn, num_samples):
    n_dim = (8 * 8)
    # Generate samples approximating the mnist8 distribution, by giving the network samples from a gaussian dist
    for i in range(num_samples):
        z = torch.randn(8 * 8)
        samples, _ = inn(z, rev=True)
        data_functions.see_mnist8(samples.detach().numpy())