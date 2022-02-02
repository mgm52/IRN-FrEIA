from .pytorch_allinone_sequence import AllInOneSequence
import torch
import data as data_functions

def get_inn_fc_pt(device):
    return AllInOneSequence(
        num_blocks=4,
        channels=8*8,
        device=device
    )