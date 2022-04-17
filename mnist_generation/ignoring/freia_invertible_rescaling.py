# Okay so.....
from tokenize import Double
import torch
import torch.nn as nn
import data as data_functions
from FrEIA.modules.reshapes import HaarDownsampling
from FrEIA.modules.graph_topology import Split
from FrEIA.modules.coupling_layers import AffineCouplingOneSided
import FrEIA.framework as ff
from freia_custom_coupling import AffineCouplingOneSidedIRN


# Todo: make my subnet look more like their deep conv subnet
def subnet(dims_in, dims_out):
    return nn.Sequential(nn.Conv2d(dims_in, 256, 3, 1, 1), nn.ReLU(),
                         nn.Conv2d(256, dims_out, 3, 1, 1))

# Setting order_by_wavelet=True ensures that the first c channels are downsamples of the original c
# dims_in is in [(c, w, h), (c, w, h), ...] format
# x is in [(n, c, w, h), (n, c, w, h), ...] format
# output is in [(n, c, w, h), (n, c, w, h), ...] format
c = 3
w = 8
h = 8
hds = HaarDownsampling([(c, w, h)], order_by_wavelet=True)

x, label = data_functions.sample_mnist8()

x_t = torch.tensor(x.reshape(1, 1, 8, 8), dtype=torch.float)
x_t = x_t.repeat(1, 3, 1, 1)
print(x_t)

# x_t is now of shape (1, 3, 8, 8)
x_t_hds, jac = hds([x_t])
print(x_t_hds)

# x_t_hds is now of shape (1, 1, 12, 4, 4)
out_shape0 = x_t_hds[0]
out_shape0_batch0 = out_shape0[0]



aco = AffineCouplingOneSided([(12, 4, 4)], subnet_constructor=subnet)
x_t_hds_coupled, jac = aco(x_t_hds)
print(x_t_hds_coupled)


def DownscalingModule(ds_count=1, inv_per_ds=1):
    # SequenceINN takes dims in (c, w, h) format - not in [(c, w, h), (c, w, h), ...] format
    inn = ff.SequenceINN(c, w, h)
    for d in range(ds_count):
        inn.append(HaarDownsampling, order_by_wavelet=True)
        for i in range(0, inv_per_ds):
            inn.append(AffineCouplingOneSidedIRN, subnet_constructor=subnet)
    return inn

inn = DownscalingModule(ds_count=1, inv_per_ds=2)
x_t_hds_coupled_inn, jac = inn([x_t])
print(x_t_hds_coupled_inn)
# x_t_hds_coupled_inn is now of shape (1, 1, c, w, h)

data_functions.see_mnist8(x, size=8)
data_functions.see_mnist8(x_t_hds_coupled_inn[0][0].detach().numpy(), size=x_t_hds_coupled_inn[0][0].shape[-1])

# Wavelet transformation class
    # forward: split x (c channels) into 
    # forward rev: join LR and LD into HR
    

# Coupling class
    # expects 2 input (h1, h2)
    # gives 2 output (h1, h2)


# Wavelet:              in: c channels                                    out: 4c channels
# (split to 3, 4c-3)
# InvBlock:             in: 3 channels, 4c-3 channels                     out: 3 channels, 4c-3 channels
# ...
# (joined to 4c)
# Downscaling Module:   in: c channels                                    out: 4c channels