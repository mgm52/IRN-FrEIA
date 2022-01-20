from .pytorch_allinone_sequence import AllInOneSequence
import torch
import data as data_functions

def inn_fc_pt(device):
    return AllInOneSequence(
        num_blocks=4,
        channels=8*8,
        device=device
    )

def sample_inn_fc_mnist8_pt(inn, num_samples, device):
    # Generate samples approximating the mnist8 distribution, by giving the network samples from a gaussian dist
    for i in range(num_samples):
        z = torch.randn(8 * 8, device=device)
        sample, _ = inn(z, rev=True)
        data_functions.see_mnist8(sample.detach().cpu().numpy())