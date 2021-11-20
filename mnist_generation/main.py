# Train a given model
import data as data_functions
import torch

from networks.freia_nets import inn_fc
from networks.freia_nets import inn_conv
from networks.freia_nets import sample_inn_fc_mnist8
from networks.freia_nets import sample_inn_conv_mnist8

import numpy as np

def train_inn_fc_mnist8(inn):
    optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)

    for i in range(2000):
        optimizer.zero_grad()
        data, label = data_functions.sample_mnist8()

        x = torch.Tensor(np.array([data]))
        z, log_jac_det = inn(x)
        loss = 0.5*torch.sum(z**2, 1) - log_jac_det
        loss = loss.mean() / (8*8)

        if i%100==99: print(f'Average loss across parameters: {loss}')
        
        loss.backward()
        optimizer.step()

# Conv version currently broken - need to sort out dims
def train_inn_conv_mnist8(inn):
    optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)
    #n_dim = (1, 8, 8)

    for i in range(2000):
        optimizer.zero_grad()
        data, label = data_functions.sample_mnist8()

        data.resize(1, 1, 8, 8)
        x = torch.Tensor(data)
        z, log_jac_det = inn(x)
        loss = 0.5*torch.sum(z**2, 1) - log_jac_det
        loss = loss.mean() / (8*8)

        if i%100==99: print(f'Average loss across parameters: {loss}')
        
        loss.backward()
        optimizer.step()

def full_inn_fc_mnist8_example():
    inn = inn_fc()
    train_inn_fc_mnist8(inn)
    sample_inn_fc_mnist8(inn, 5)

def full_inn_conv_mnist8_example():
    inn = inn_conv()
    train_inn_conv_mnist8(inn)
    sample_inn_conv_mnist8(inn, 5)

full_inn_conv_mnist8_example()