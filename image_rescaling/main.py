# Train a given model
import time
from networks.freia_invertible_rescaling import IRN, train_inn_mnist8, sample_inn
from data import mnist8_iterator, process_4bit_img, save_multiple_imgs
from bicubic_pytorch import imresize

inn = IRN(3, 8, 8, ds_count=1, inv_per_ds=2)
train_inn_mnist8(inn, max_batches=500, max_epochs=-1, target_loss=-1, learning_rate=0.001, batch_size=500,
                 lambda_recon=1, lambda_guide=2, lambda_distr=1)

mnist8_iter = mnist8_iterator()
for i in range(10):
    x, y, z, x_recon_from_y = sample_inn(inn, mnist8_iter, batch_size=1, use_test_set=True)

    x_downscaled = imresize(x, sizes=(4, 4))

    imgs = [
        process_4bit_img(x_downscaled[0][0].detach().cpu().numpy(), (4, 4)),
        process_4bit_img(x[0][0].detach().cpu().numpy(), (8, 8)),
        process_4bit_img(y[0][0].detach().cpu().numpy(), (4, 4)),
        process_4bit_img(x_recon_from_y[0][0].detach().cpu().numpy(), (8, 8)),
    ]

    save_multiple_imgs(imgs, 2, 2, ["Bicubic downscaled", "GT", "IRN-downscaled", "IRN-upscaled"], f'output/out_{int(time.time())}_{i}')