from math import floor
from tokenize import Double
from FrEIA.modules.reshapes import HaarDownsampling
from .freia_custom_coupling import AffineCouplingOneSidedIRN, EnhancedCouplingOneSidedIRN
from bicubic_pytorch.core import imresize
from data import mnist8_iterator
from loss import calculate_irn_loss
import FrEIA.framework as ff
import numpy as np
import torch
import torch.nn as nn
from data2 import get_div2k_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=200)

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
            inn.append(EnhancedCouplingOneSidedIRN, subnet_constructor=subnet)
    return inn.cuda() if device=="cuda" else inn

def sample_inn(inn, mnist8_iter=None, dataloader=None, size=8, batch_size=1, use_test_set=False):
    if mnist8_iter is None:
        x, _ = next(iter(dataloader))
        c = 3
    else:
        x = mnist8_iter.iterate_mnist8_imgs(count=batch_size, use_test_data=use_test_set)
        c = 1

    x = torch.tensor(np.array(x), dtype=torch.float, device=device).reshape(-1, c, size, size)
    # x is now of shape (n, c, 8, 8)
    if c==1: x = x.repeat(1, 3, 1, 1)
    # x_t is now of shape (n, 3, 8, 8)

    y_and_z, jac = inn([x])
    # y_and_z is now of shape (n, c, w, h)
    y, z = y_and_z[:, :3], y_and_z[:, 3:]
    # y is now of shape (n, 3, w, h)
    # z is now of shape (n, c-3, w, h)

    #x_recon_from_y_and_z, _ = inn([y_and_z], rev=True)

    z_sample = torch.normal(torch.zeros_like(z), torch.ones_like(z))
    # z_sample is now of shape (n, c-3, w, h)
    y_and_z_sample = torch.cat((y, z_sample), dim=1)
    # y_and_z_sample is now of shape (n, c, w, h)
    x_recon_from_y, _ = inn([y_and_z_sample], rev=True)

    return x, y, z, x_recon_from_y

def test_inn(inn,
    mnist8_iter,
    dataloader,
    lambda_recon=1,
    lambda_guide=1,
    lambda_distr=1,
    batch_size=178,
):
    #mnist8_iter = mnist8_iterator()

    with torch.no_grad():
        # todo: use batch_size=-1 instead, then check that it works
        x, y, z, x_recon_from_y = sample_inn(inn, mnist8_iter, dataloader, 8, batch_size=batch_size, use_test_set=True)
        loss_recon, loss_guide, loss_distr, total_loss = calculate_irn_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, batch_size)
    
    print(f'loss_recon={loss_recon}, loss_guide={loss_guide}, loss_distr={loss_distr}')
    print(f'Average loss in test set: {total_loss}')

    return total_loss

def train_inn_mnist8(inn,
    max_batches=10000,
    max_epochs=-1, #TODO: use this
    target_loss=-1,
    learning_rate=0.001,
    batch_size=5,
    lambda_recon=1,
    lambda_guide=1,
    lambda_distr=1,
    use_mnist8=True,
    batches_between_prints=250
):
    optimizer = torch.optim.Adam(inn.parameters(), lr=learning_rate)
    mnist8_iter = mnist8_iterator() if use_mnist8 else None
    dataloader = get_div2k_dataloader(batch_size) if not use_mnist8 else None
    size = 8 if use_mnist8 else 80

    i = 0
    losses = []
    avg_loss = target_loss+1
    while (max_batches==-1 or i < max_batches) and (target_loss==-1 or target_loss<=avg_loss):
        optimizer.zero_grad()
        
        x, y, z, x_recon_from_y = sample_inn(inn, mnist8_iter, dataloader, size, batch_size=batch_size, use_test_set=False)
        loss_recon, loss_guide, loss_distr, total_loss = calculate_irn_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, batch_size)
        
        losses.append(total_loss)

        if i%batches_between_prints==0:
            avg_loss = sum(losses) / len(losses)
            #print(y.shape)
            #print(z.shape)
            #print(x_recon_from_y.shape)
            #print(x_downscaled.shape)
            print(f'loss_recon={loss_recon}, loss_guide={loss_guide}, loss_distr={loss_distr}')
            print(f'Avg loss, in last {batches_between_prints if i > 0 else 1} batches: {avg_loss}')
            print(f'In test dataset:')
            test_inn(inn, mnist8_iter, dataloader, lambda_recon, lambda_guide, lambda_distr)
            print("")
            losses = []
        
        if i%5000==4999:
            learning_rate /= 2
            print(f'Halved learning rate from {learning_rate*2} to {learning_rate}')
        
        total_loss.backward()
        optimizer.step()
        i+=1


# Wavelet:              in: c channels                                    out: 4c channels
# InvBlock:             in: 3 channels, 4c-3 channels                     out: 3 channels, 4c-3 channels
# ...
# (joined to 4c)

# Downscaling Module:   in: c channels                                    out: 4c channels