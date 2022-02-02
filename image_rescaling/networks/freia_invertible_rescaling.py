from math import floor
from tokenize import Double
from FrEIA.modules.reshapes import HaarDownsampling
from .freia_custom_coupling import AffineCouplingOneSidedIRN, EnhancedCouplingOneSidedIRN
from bicubic_pytorch import imresize
from data import mnist8_iterator
import FrEIA.framework as ff
import numpy as np
import torch
import torch.nn as nn

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
    if device=="cuda": return inn.cuda()
    return inn

def sample_inn(inn, mnist8_iter, batch_size=1, use_test_set=False):
    x = mnist8_iter.iterate_mnist8_imgs(count=batch_size, use_test_data=use_test_set)
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

    z_sample = torch.normal(torch.zeros_like(z), torch.ones_like(z))
    # z_sample is now of shape (1, c-3, w, h)
    y_and_z_sample = torch.cat((y, z_sample), dim=1)
    # y_and_z_sample is now of shape (1, c, w, h)
    x_recon_from_y, _ = inn([y_and_z_sample], rev=True)

    return x, y, z, x_recon_from_y

def calculate_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, batch_size):
    # Loss = Loss_Reconstruction + Loss_Guide + Loss_Distribution_Match_Surrogate
    # Purpose of Loss_Reconstruction: accurate upscaling
    loss_recon = lambda_recon * torch.abs(x - x_recon_from_y).sum() / batch_size
    # Purpose of Loss_Guide: sensible downscaling
        # Intuition about using L2 here: the most recognisable downscaled images get the most prominant points correct?
    x_downscaled = imresize(x, sizes=(4, 4))
    loss_guide = lambda_guide * ((x_downscaled - y)**2).sum() / batch_size
    # Purpose of Loss_Distribution_Match_Surrogate:
        # Encouraging the model to always produce things that look like the OG dataset, even when it doesn't know what to do?
        # And encouraging disentanglement (by forcing z to be a normal dist)?
        # Full Loss_Distribution_Match does this by measuring JSD between x and x_reconstructed.
        # Surrogate Loss_Distribution_match does this by measuring CE between z and z_sample.
    # Paper describes this as: -1 * sum [prob(x from dataset) * log2(prob(z in our normal dist))]
    # Because prob(x from dataset) is a constant (I believe?): we have -1 * log2(prob(z in our normal dist))
    # Because surprisal in a standard normal dist is O(x^2): we have z^2
    loss_distr = lambda_distr * (z**2).sum() / batch_size
    
    total_loss = loss_recon + loss_guide + loss_distr

    return loss_recon, loss_guide, loss_distr, total_loss

def test_inn_mnist8(inn,
    lambda_recon=1,
    lambda_guide=1,
    lambda_distr=1,
    batch_size=178
):
    mnist8_iter = mnist8_iterator()

    with torch.no_grad():
        x, y, z, x_recon_from_y = sample_inn(inn, mnist8_iter, batch_size=batch_size, use_test_set=True)
        loss_recon, loss_guide, loss_distr, total_loss = calculate_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, batch_size)
    
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
    lambda_distr=1
):
    optimizer = torch.optim.Adam(inn.parameters(), lr=learning_rate)
    mnist8_iter = mnist8_iterator()

    i = 0
    losses = []
    avg_loss = target_loss+1
    while (max_batches==-1 or i < max_batches) and (target_loss==-1 or target_loss<=avg_loss):
        optimizer.zero_grad()
        
        x, y, z, x_recon_from_y = sample_inn(inn, mnist8_iter, batch_size=batch_size, use_test_set=False)
        loss_recon, loss_guide, loss_distr, total_loss = calculate_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, batch_size)
        
        losses.append(total_loss)

        if i%250==0:
            avg_loss = sum(losses) / len(losses)
            #print(y.shape)
            #print(z.shape)
            #print(x_recon_from_y.shape)
            #print(x_downscaled.shape)
            print(f'loss_recon={loss_recon}, loss_guide={loss_guide}, loss_distr={loss_distr}')
            print(f'Avg loss, in last {250 if i > 0 else 1} batches: {avg_loss}')
            print(f'In test dataset:')
            test_inn_mnist8(inn, lambda_recon, lambda_guide, lambda_distr)
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