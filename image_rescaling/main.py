# Train a given model
import time
from networks.freia_invertible_rescaling import IRN, train_inn_mnist8, sample_inn
from data import mnist8_iterator, process_4bit_img, see_multiple_imgs
from bicubic_pytorch.core import imresize
import torch
import torchmetrics
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
#from IQA_pytorch import SSIM, DISTS

inn = IRN(3, 8, 8, ds_count=1, inv_per_ds=2)
train_inn_mnist8(inn, max_batches=1, max_epochs=-1, target_loss=-1, learning_rate=0.001, batch_size=500,
                 lambda_recon=1, lambda_guide=2, lambda_distr=1)

# mnist8 consists of 16-tone images
mnist8_iter = mnist8_iterator()
psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=16).cuda()
for i in range(10):
    x, y, z, x_recon_from_y = sample_inn(inn, mnist8_iter, batch_size=1, use_test_set=True)

    x_downscaled_bc = imresize(x, sizes=(4, 4))
    x_upscaled_bc = imresize(x_downscaled_bc, sizes=(8, 8))

    imgs = [process_4bit_img(*im_and_size) for im_and_size in [
        (x[0][0], (8, 8)),
        (x_downscaled_bc[0][0], (4, 4)),
        (y[0][0], (4, 4)),
        (x[0][0], (8, 8)),
        (x_upscaled_bc[0][0], (8, 8)),
        (x_recon_from_y[0][0], (8, 8)),
    ]]

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
        see=True, save=False,
        filename=f'output/out_{int(time.time())}_{i}'
    )

