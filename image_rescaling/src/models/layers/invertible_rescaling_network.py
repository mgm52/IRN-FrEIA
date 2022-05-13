from math import floor, log2
from tokenize import Double
from FrEIA.modules.reshapes import HaarDownsampling
from FrEIA.modules.invertible_resnet import ActNorm
from .coupling import EnhancedCouplingOneSidedIRN
from utils.bicubic_pytorch.core import imresize
import FrEIA.framework as ff
import numpy as np
from typing import List
import models.layers.straight_through_estimator
from data.make_dataset_compress import jpeg_compress_random
import torch
from utils.utils import standardise_tensor
import models.model_loader
from models.layers.dense_block import db_subnet
from models.layers.batchnorm_seq import BatchnormSequenceINN
from models.layers.straight_through_estimator import quantize_ste, quantize_to_int_ste

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=200)

# dims          is in (c, w, h) format
# inn x         is in [(n, c, w, h), (n, c, w, h), ...] format
# inn output    is in [(n, c, w, h), (n, c, w, h), ...] format
def IRN(*dims, cfg):
    models.model_loader.check_keys(cfg, ["scale", "actnorm", "inv_per_ds", "inv_final_level_extra", "inv_first_level_extra", "batchnorm", "clamp", "clamp_min", "clamp_tightness"])

    ds_count = int(log2(cfg["scale"]))
    assert ds_count == log2(cfg["scale"]), f"IRN scale must be a power of 2 (was given scale={cfg['scale']})"

    # SequenceINN takes dims in (c, w, h) format
    inn = BatchnormSequenceINN(*dims) if cfg["batchnorm"] else ff.SequenceINN(*dims)

    for d in range(ds_count):
        inn.append(HaarDownsampling, order_by_wavelet=True)
        inv_count = cfg["inv_per_ds"]
        if d==0: inv_count += cfg["inv_first_level_extra"]
        elif d==ds_count-1: inv_count += cfg["inv_final_level_extra"]
        for i in range(inv_count):
            inn.append(EnhancedCouplingOneSidedIRN, subnet_constructor=db_subnet, clamp=cfg["clamp"], clamp_min=cfg["clamp_min"], clamp_tightness=cfg["clamp_tightness"])
            if cfg["actnorm"]: inn.append(ActNorm)
    return inn.cuda() if device=="cuda" else inn

def compress_batch(imgs, quality = -1, path=""):
  imgs_cmpr = torch.cat([jpeg_compress_random(img, quality, path).unsqueeze(0) for img in imgs])
  return imgs_cmpr

def compress_ste(x, quality = -1, path=""):
    return models.layers.straight_through_estimator.StraightThroughEstimator.apply(x, lambda y : compress_batch(y, quality, path))

def upscale_irn(irn, scale, ymod: torch.Tensor, zerosample=False):

    z_shape = (ymod.shape[0], 3*scale*scale - 3, ymod.shape[2], ymod.shape[3])
    if zerosample:
        z_sample = torch.zeros(z_shape)
    else:
        z_sample = torch.normal(torch.zeros(z_shape), torch.ones(z_shape))    # z_sample.shape == (n, c2-3, w2, h2)
    # z_sample.shape == (n, c2-3, w2, h2)

    ### Here, we assume that the y that was extracted in the forward pass has mean 0, std 1.
    ### This may not be true, but we can force it to be true through training.
    y_and_z_sample = torch.cat((ymod.cuda(), z_sample.cuda()), dim=1)

    # y_and_z_sample.shape == (n, c2, w2, h2)
    x_recon_from_ymod, _ = irn([y_and_z_sample], rev=True)

    return x_recon_from_ymod

# output is in range [0, 1] batchnorm=False, zerosample=False, sr_mode=False, compress_mode=True
def sample_irn(irn, x: torch.Tensor, cfg):
    if device=="cuda": x = x.cuda()

    #x = torch.tensor(np.array(x), dtype=torch.float, device=device).reshape(-1, *dataloaders.sample_shape)
    # TODO: move this multiplication logic into the mnist8 data loader
    if x.shape[1]==1: x = x.repeat(1, 3, 1, 1)
    assert x.shape[1] == 3, f"Expected 3 channels, have {x.shape[1]}"
    # x.shape == (n, 3, w1, h1)

    if cfg["batchnorm"]:
        x, mean, std = standardise_tensor(x.clone(), 0, 1)

    y_and_z, jac = irn([x])
    y, z = y_and_z[:, :3], y_and_z[:, 3:]
    y_quant = quantize_ste(y)

    # If y conforms to a standard normal dist, which we will try to force it to, we would have mean_y=0 and std_y=1
    if cfg["batchnorm"]:
        y_printable, mean_y, std_y = standardise_tensor(y_quant.clone(), mean, std)
        y_printable = quantize_ste(y_printable)
    else:
        mean_y, std_y = None, None

    ### To simulate real use of the network, I shouldn't refer to x and y after this point. We must assume that we are reconstructing from y_quant ONLY.
    ### However -> I also extract mean_y and std_y during *training only*.

    # y_and_z.shape == (n, c2, w2, h2)
    # y.shape == (n, 3, w2, h2)
    # z.shape == (n, c2-3, w2, h2)

    if cfg["zerosample"]:
        z_sample = torch.zeros_like(z)
    else:
        z_sample = torch.normal(torch.zeros_like(z), torch.ones_like(z))    # z_sample.shape == (n, c2-3, w2, h2)
    # z_sample.shape == (n, c2-3, w2, h2)

    ### Here, we assume that the y that was extracted in the forward pass has mean 0, std 1.
    ### This may not be true, but we can force it to be true through training.
    scale = round(x.shape[-1] / y_and_z.shape[-1])
    if cfg["batchnorm"]: 
        ymod, _, _ = standardise_tensor(y_printable.clone(), 0, 1)
    elif cfg["sr_mode"]:
        x_downscaled = quantize_ste(imresize(x, scale=1.0/scale)) #quantize_ste(imresize(x, scale=1.0/scale))
        ymod = x_downscaled
    elif cfg["compression_mode"]:
        y_compressed = compress_ste(y_quant.cuda(), quality=cfg["compression_quality"])
        ymod = y_compressed.cuda()
    else:
        ymod = y_quant

    x_recon_from_ymod = upscale_irn(irn, scale, ymod, zerosample=cfg["zerosample"])

    # y_and_z_sample.shape == (n, c2, w2, h2)
    # x_recon_from_y.shape == (n, 3, w1, h1)        

    return x, y, z, x_recon_from_ymod, mean_y, std_y, ymod


# output is in range [0, 1]
def multi_sample_irn(irn, x: torch.Tensor, num_applications=1, mid_quantization=True):
    if device=="cuda": x = x.cuda()

    #x = torch.tensor(np.array(x), dtype=torch.float, device=device).reshape(-1, *dataloaders.sample_shape)
    # TODO: move this multiplication logic into the mnist8 data loader
    if x.shape[1]==1: x = x.repeat(1, 3, 1, 1)
    assert x.shape[1] == 3, f"Expected 3 channels, have {x.shape[1]}"
    # x.shape == (n, 3, w1, h1)

    irn_input = x
    for a in range(num_applications):
        y_and_z, jac = irn([irn_input])
        y, z = y_and_z[:, :3], y_and_z[:, 3:]
        scale = round(irn_input.shape[-1] / y_and_z.shape[-1])

        y_quant = quantize_ste(y)
        irn_input = y_quant if mid_quantization else y

    mean_y, std_y = None, None

    ### To simulate real use of the network, I shouldn't refer to x and y after this point. We must assume that we are reconstructing from y_quant ONLY.
    ### However -> I also extract mean_y and std_y during *training only*.

    # y_and_z.shape == (n, c2, w2, h2)
    # y.shape == (n, 3, w2, h2)
    # z.shape == (n, c2-3, w2, h2)

    irn_input = y_quant
    for a in range(num_applications):
        z_shape = (irn_input.shape[0], 3*scale*scale - 3, irn_input.shape[2], irn_input.shape[3])
        z_sample = torch.normal(torch.zeros(z_shape), torch.ones(z_shape))
        # z_sample.shape == (n, c2-3, w2, h2)
        y_and_z_sample = torch.cat((irn_input.cuda(), z_sample.cuda()), dim=1)
        # y_and_z_sample.shape == (n, c2, w2, h2)
        x_recon_from_y, _ = irn([y_and_z_sample], rev=True)
        # x_recon_from_y.shape == (n, 3, w1, h1)    

        x_recon_from_y_quant = quantize_ste(x_recon_from_y)
        irn_input = x_recon_from_y_quant if mid_quantization else x_recon_from_y

    return x, y, z, x_recon_from_y, mean_y, std_y

# Wavelet:              in: c channels                                    out: 4c channels
# InvBlock:             in: 3 channels, 4c-3 channels                     out: 3 channels, 4c-3 channels
# ...
# (joined to 4c)

# Downscaling Module:   in: c channels                                    out: 4c channels