# Okay so.....
from tokenize import Double
import numpy as np
import torch
import torch.nn as nn
import data as data_functions
from FrEIA.modules.reshapes import HaarDownsampling
from FrEIA.modules.graph_topology import Split
from FrEIA.modules.coupling_layers import AffineCouplingOneSided
import FrEIA.framework as ff
from freia_custom_coupling import AffineCouplingOneSidedIRN

device = "cuda" if torch.cuda.is_available() else "cpu"

# Todo: make my subnet look more like IRN's deep conv subnet
def subnet(channels_in, channels_out):
    return nn.Sequential(nn.Conv2d(channels_in, 256, 3, 1, 1), nn.ReLU(),
                         nn.Conv2d(256, channels_out, 3, 1, 1))

# dims          is in (c, w, h) format
# inn x         is in [(n, c, w, h), (n, c, w, h), ...] format
# inn output    is in [(n, c, w, h), (n, c, w, h), ...] format
def IRN(*dims, ds_count=1, inv_per_ds=1):
    # SequenceINN takes dims in (c, w, h) format
    inn = ff.SequenceINN(*dims)
    for d in range(ds_count):
        inn.append(HaarDownsampling, order_by_wavelet=True)
        for i in range(0, inv_per_ds):
            inn.append(AffineCouplingOneSidedIRN, subnet_constructor=subnet)
    return inn.cuda()

#data_functions.see_mnist8(x, size=8)
#data_functions.see_mnist8(x_t_hds_coupled_inn[0][0].detach().numpy(), size=x_t_hds_coupled_inn[0][0].shape[-1])

def sample_inn(inn, batch_size=1):
    x = data_functions.sample_mnist8_imgs(count=batch_size)
    x = torch.tensor(np.array(x), dtype=torch.float, device=device).reshape(-1, 1, 8, 8)
    # x is now of shape (1, 1, 8, 8)
    x = x.repeat(1, 3, 1, 1)
    # x_t is now of shape (1, 3, 8, 8)

    y_and_z, jac = inn([x])
    # y_and_z is now of shape (1, c, w, h)
    y, z = y_and_z[:, :3], y_and_z[:, 3:]
    # y is now of shape (1, 3, w, h)
    # z is now of shape (1, c-3, w, h)

    #x_recon_from_y_and_z, _ = inn([y_and_z], rev=True)

    # Loss = l1 Loss_Reconstruction + l2 Loss_Guide + l3 Loss_Distribution_Match_Surrogate
    z_sample = torch.normal(torch.zeros_like(z), torch.ones_like(z))
    # z_sample is now of shape (1, c-3, w, h)
    y_and_z_sample = torch.cat((y, z_sample), dim=1)
    # y_and_z_sample is now of shape (1, c, w, h)
    x_recon_from_y, _ = inn([y_and_z_sample], rev=True)

    return x, x_recon_from_y

def train_inn_mnist8(inn, max_iters=10000, target_loss=-1, learning_rate=0.001, batch_size=5):
    optimizer = torch.optim.Adam(inn.parameters(), lr=learning_rate)

    i = 0
    losses = []
    avg_loss = target_loss+1
    while (max_iters==-1 or i < max_iters) and (target_loss==-1 or target_loss<=avg_loss):
        optimizer.zero_grad()
        
        # Todo: iterate through dataset properly (epochs) rather than randomly sampling
        x, x_recon_from_y = sample_inn(inn, batch_size=batch_size)

        loss = torch.abs(x - x_recon_from_y).sum() / batch_size
        losses.append(loss)

        if i%250==249:
            avg_loss = sum(losses) / len(losses)
            #print(y_and_z.shape)
            #print(y.shape)
            #print(z.shape)
            #print(x_recon_from_y_and_z.shape)
            #print(z_sample.shape)
            #print(y_and_z_sample.shape)
            print(x_recon_from_y.shape)
            print(f'Avg loss across parameters, in last 250 samples: {avg_loss}')
            losses = []
        
        if i%5000==4999:
            learning_rate /= 2
            print(f'Halved learning rate from {learning_rate*2} to {learning_rate}')
        
        loss.backward()
        optimizer.step()
        i+=1


inn = IRN(3, 8, 8, ds_count=1, inv_per_ds=2)
train_inn_mnist8(inn, max_iters=6000, target_loss=-1, learning_rate=0.001, batch_size=500)

for i in range(10):
        x, x_recon_from_y = sample_inn(inn)
        data_functions.see_mnist8(x.detach().cpu().numpy())
        data_functions.see_mnist8(x_recon_from_y.detach().cpu().numpy())

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