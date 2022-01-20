import torch.nn as nn
import torch
import data as data_functions
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(),
                         nn.Linear(512,  dims_out))

def inn_fc():
    inn = Ff.SequenceINN(8 * 8)
    for k in range(4):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    return inn

def sample_inn_fc_mnist8(inn, num_samples):
    # Generate samples approximating the mnist8 distribution, by giving the network samples from a gaussian dist
    for i in range(num_samples):
        z = torch.randn(8 * 8)
        print("Taking sample using z:")
        print(z)
        samples, _ = inn(z, rev=True)
        data_functions.see_mnist8(samples.detach().numpy())