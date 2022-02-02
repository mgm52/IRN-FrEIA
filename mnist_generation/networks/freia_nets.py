import torch.nn as nn
import torch
import data as data_functions
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(),
                         nn.Linear(512,  dims_out))

def get_inn_fc_freia():
    inn = Ff.SequenceINN(8 * 8)
    for k in range(4):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    return inn