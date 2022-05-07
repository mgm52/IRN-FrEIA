from .pytorch_allinone_sequence import AllInOneSequence
import torch
import torch.nn as nn
import data as data_functions

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(),
                         nn.Linear(512,  dims_out))

def get_inn_fc_pt(device):
    return AllInOneSequence(
        num_blocks=4,
        channels=8*8,
        device=device,
        subnet_constructor=subnet_fc
    )