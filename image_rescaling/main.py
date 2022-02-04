# Train a given model
import time
from networks.freia_invertible_rescaling import IRN, train_inn_mnist8, sample_inn
from data import mnist8_iterator, process_xbit_img, see_multiple_imgs, process_div2k_img
from data2 import get_div2k_dataloader
from bicubic_pytorch.core import imresize
import torch
import torchmetrics
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
#from IQA_pytorch import SSIM, DISTS

use_mnist8 = False

size = 8 if use_mnist8 else 80
bits = 4 if use_mnist8 else 8
scaling = 1 if use_mnist8 else 256

# Train the network
inn = IRN(3, size, size, ds_count=1, inv_per_ds=2)
train_inn_mnist8(inn, max_batches=100, max_epochs=-1, target_loss=-1, learning_rate=0.001, batch_size=500,
                 lambda_recon=1, lambda_guide=2, lambda_distr=1, use_mnist8=use_mnist8, batches_between_prints=10)

# Test the network
# mnist8 consists of 16-tone images
mnist8_iter = mnist8_iterator() if use_mnist8 else None
dataloader = get_div2k_dataloader(batch_size=1) if not use_mnist8 else None

psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=16).cuda()
for i in range(5):
    x, y, z, x_recon_from_y = sample_inn(inn, mnist8_iter, dataloader=dataloader, size=size, batch_size=1, use_test_set=True)

    x_downscaled_bc = imresize(x, scale=0.5)
    x_upscaled_bc = imresize(x_downscaled_bc, scale=2)

    if use_mnist8:
        #div2k images are loaded with values in range [0, 1] but mnist8 is in range [0, 16].
        imgs = [process_xbit_img(*im_and_size, bits=bits, scaling=scaling) for im_and_size in [
            (x[0][0], (size, size)),
            (x_downscaled_bc[0][0], (size//2, size//2)),
            (y[0][0], (size//2, size//2)),
            (x[0][0], (size, size)),
            (x_upscaled_bc[0][0], (size, size)),
            (x_recon_from_y[0][0], (size, size)),
        ]]
    else:
        #div2k images are loaded with values in range [0, 1] but mnist8 is in range [0, 16].
        imgs = [process_div2k_img(*im_and_size) for im_and_size in [
            (x[0], (3, size, size)),
            (x_downscaled_bc[0], (3, size//2, size//2)),
            (y[0], (3, size//2, size//2)),
            (x[0], (3, size, size)),
            (x_upscaled_bc[0], (3, size, size)),
            (x_recon_from_y[0], (3, size, size)),
        ]]

    # TODO: change this to compute PSNR on the y channel of images in YCbCr space
    # IRN-down  vs  GT (Bicubic-down)
    psnr_irn_down = psnr_metric(y[0][0], x_downscaled_bc[0][0])
    # Bicubic-down / Bicubic-up  vs  GT (HR)
    psnr_bi_up = psnr_metric(x_upscaled_bc[0][0], x[0][0])
    # IRN-down / IRN-up  vs  GT (HR)
    psnr_irn_up = psnr_metric(x_recon_from_y[0][0], x[0][0])

    see_multiple_imgs(imgs, 2, 3,
        row_titles=[
            "Downscaling task (PSNR)",
            "Downscaling & upscaling task (PSNR)"],
        plot_titles=[
            "HR (-)", "GT [Bi-down] (∞)", "IRN-down (%.2f)" % round(psnr_irn_down.item(), 2),
            "GT [HR] (∞)", "Bi-down & Bi-up (%.2f)" % round(psnr_bi_up.item(), 2), "IRN-down & IRN-up (%.2f)" % round(psnr_irn_up.item(), 2)
        ],
        see=True, save=True,
        filename=f'output/out_{int(time.time())}_{i}'
    )

