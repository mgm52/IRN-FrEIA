import time
from tracemalloc import start
from networks.invertible_rescaling_network import IRN, sample_irn
from visualize import mnist8_iterator, process_xbit_img, see_multiple_imgs, process_div2k_img
from data import Div2KDataLoaders, DataLoaders
from bicubic_pytorch.core import imresize
import torch
from torchvision.utils import save_image
import torchmetrics
import numpy as np
import wandb
import math
from timeit import default_timer as timer
import glob
import os
import random
from test import rgb_to_y
from loss import calculate_irn_loss
import matplotlib.pyplot as plt

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def test_ychannel():
    diffs = []
    for i in range(5000):
        x = torch.rand(1, 3, 5, 5)

        xy1 = bgr2ycbcr(x.clone().squeeze().permute(1, 2, 0).numpy())
        xy2 = rgb_to_y(x.clone(), bgr=True)

        diffs.append((torch.tensor(xy1) - xy2).abs().mean())
    avg_diff = sum(diffs) / len(diffs)

    print(f"\n Average diff: {avg_diff}")
    assert avg_diff < 0.0000001, f"Y-channel error is larger than expected (average pixel difference {avg_diff})"
