import time
from tracemalloc import start
from torchvision import transforms
from networks.invertible_rescaling_network import IRN, sample_irn, multi_sample_irn
from visualize import mnist8_iterator, process_xbit_img, see_multiple_imgs, process_div2k_img
from data import Div2KDataLoaders, DataLoaders, get_test_dataloader
from bicubic_pytorch.core import imresize
import torch
from torchvision.utils import save_image
import torchmetrics
import numpy as np
import wandb
from network_saving import save_network, load_network
import math
from timeit import default_timer as timer
from networks.invertible_rescaling_network import quantize_ste
import glob
from utils import create_parent_dir
import os
import random
from loss import calculate_irn_loss
from datetime import date
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

def crop_tensor_border(x, border):
    if border==0: return x
    return x[..., border:-border, border:-border]

# input should be size (n, 3, h, w) - in range [0,1]
# output has size (n, 1, h, w) - treat the output as if it has range [0,1]
def rgb_to_y(rgb_imgs, round255=False, plus16=True, bgr=False):
    assert len(rgb_imgs.shape) == 4 and rgb_imgs.shape[1] == 3, "Input must have shape [n, 3, h, w]"
    output = (rgb_imgs[:,2 if bgr else 0,:,:] * 65.481 + rgb_imgs[:,1,:,:] * 128.553 + rgb_imgs[:,0 if bgr else 2,:,:] * 24.966) + (16 if plus16 else 1)
    if round255: output = output.round()
    return output.unsqueeze(dim=1) / 255.0

def get_test_function_irn(lambda_recon, lambda_guide, lambda_distr, metric_crop_border, fast_gpu_testing=False, mean_losses=False, quantize_recon=False):
    def test_function_irn(x, y, z, x_recon_from_y, mean_y, std_y):
        loss_recon, loss_guide, loss_distr, loss_batchnorm, total_loss = calculate_irn_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, mean_y, std_y, batchnorm=False, mean_losses=mean_losses, quantize_recon=quantize_recon)

        test_function_psnr_ssim = get_test_function_psnr_ssim(metric_crop_border, fast_gpu_testing=fast_gpu_testing)
        [psnr_RGB, psnr_Y, ssim_RGB, ssim_Y] = test_function_psnr_ssim(x, y, z, x_recon_from_y, mean_y, std_y)

        return [float(total_loss), psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, float(loss_recon), float(loss_guide), float(loss_distr)]
    return test_function_irn

def get_test_function_psnr_ssim(metric_crop_border, fast_gpu_testing=False):
    # expects x, y, x_recon in range [0,1]
    def test_function_psnr_ssim(x, y, z, x_recon_from_y, mean_y, std_y):
        # Convert to CPU because results differ on CUDA
        test_device = "cuda:0" if fast_gpu_testing else "cpu"

        x = quantize_ste(x).to(test_device)
        y = quantize_ste(y).to(test_device)
        x_recon_from_y = quantize_ste(x_recon_from_y).to(test_device)

        #print(f"Computing for x_recon_from_y with mean {x_recon_from_y.mean()} std {x_recon_from_y.std()} min {x_recon_from_y.min()} max {x_recon_from_y.max()}")
        #print(x_recon_from_y * 255)

        x_recon_from_y_cropped = crop_tensor_border(x_recon_from_y, metric_crop_border)
        x_cropped = crop_tensor_border(x, metric_crop_border)

        x_recon_from_y_cropped_Y = rgb_to_y(x_recon_from_y_cropped)
        x_cropped_Y = rgb_to_y(x_cropped)

        #print(x_recon_from_y_cropped.max())
        #print(x_recon_from_y_cropped_Y.max())
        #print((x_recon_from_y_cropped - x_cropped).mean())
        #print((x_recon_from_y_cropped_Y - x_cropped_Y).mean())

        # Note: PSNR and SSIM score is unaffected by data scale.
        psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1).to(test_device)

        #see_multiple_imgs([x_recon_from_y_cropped, x_cropped], 1, 2, row_titles=[], plot_titles=[], see=True, save=False, smallSize=True)

        psnr_RGB = psnr_metric(x_recon_from_y_cropped, x_cropped)
        psnr_Y   = psnr_metric(x_recon_from_y_cropped_Y, x_cropped_Y)

        

        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1).to(test_device)
        
        ssim_RGB   = ssim_metric(x_recon_from_y_cropped, x_cropped)

        #plt.imshow(x_recon_from_y_cropped_Y[0].permute(1, 2,0))
        #plt.show()
        #plt.imshow(x_cropped_Y[0].permute(1, 2,0))
        #plt.show()

        ssim_Y = ssim_metric(x_recon_from_y_cropped_Y, x_cropped_Y)

        

        return [float(psnr_RGB), float(psnr_Y), float(ssim_RGB), float(ssim_Y)]
    return test_function_psnr_ssim

def get_sample_function_irn(inn):
    return lambda x: sample_irn(inn, x.clone())

def get_multi_sample_function_irn(inn, num_applications, mid_quantization):
    return lambda x: multi_sample_irn(inn, x.clone(), num_applications, mid_quantization)

def get_sample_function_bicub(scale):
    def sample_function_bicub(x):
        x_downscaled = imresize(x, scale=1.0/scale)
        y = x_downscaled

        x_recon_from_y = imresize(y, scale=scale)

        z = None

        mean_y = y.mean()
        std_y = y.std()

        return x, y, z, x_recon_from_y, mean_y, std_y
    return sample_function_bicub

def get_sample_function_imgfolder(path, scale, img_size_divisor=1):

    img_iter = iter(get_test_dataloader(path, img_size_divisor))

    def sample_function_imgfolder(x):
        x_downscaled = imresize(x, scale=1.0/scale)
        y = x_downscaled

        x_recon_from_y = next(img_iter)[0]

        z = None

        mean_y = y.mean()
        std_y = y.std()

        return x, y, z, x_recon_from_y, mean_y, std_y
    return sample_function_imgfolder

def test_rescaling(dataloaders, x_y_z_recon_mean_std_function, test_metrics_function, save_imgs=False, foldername=None):
    with torch.no_grad():
        current_date = date.today().strftime("%Y-%m-%d")
        if not foldername:
            foldername = current_date

        max_test_batches = dataloaders.test_len / dataloaders.test_dataloader.batch_size
        assert(max_test_batches == int(max_test_batches)), f"Test batch size ({dataloaders.test_dataloader.batch_size}) does not fit into test dataset ({dataloaders.test_len})"

        # Get losses across test dataset
        all_test_scores = []
        test_iter = iter(dataloaders.test_dataloader)
        
        hardest_imgs = []
        easiest_imgs = []
        psnry_scores = []

        #sample_function_imgload = get_sample_function_imgfolder("./data/DIV2K/DIV2K_valid_x4recon_IRN-mine-191", 4, 4)

        for test_batch_no in range(int(max_test_batches)):
            # todo: use batch_size=-1 instead, then check that it works
            if (test_batch_no-1) % int(max(2, max_test_batches / 5)) == 0:
                print(f"At test item {test_batch_no}/{int(max_test_batches)}")
                print(f'Average values: {[round(float(t), 4) for t in np.array(all_test_scores).mean(axis=0)]}')

            x_raw, _ = next(test_iter)
            
            x, y, z, x_recon_from_y, mean_y, std_y = x_y_z_recon_mean_std_function(x_raw)
            
            #x_l, y_l, z_l, x_recon_from_y_l, mean_y_l, std_y_l = sample_function_imgload(x_raw)



            #print(f"x_recon_from_y has max value {x_recon_from_y.max()}")
            #save_image(x_recon_from_y, fp=str(test_batch_no)+".png")

            #print(f"Found x_recon_from_y {x_recon_from_y.shape} {x_recon_from_y}")

            
            test_scores = test_metrics_function(x, y, z, x_recon_from_y, mean_y, std_y)

            if test_scores[1] < 25:
                hardest_imgs.append(x.cpu())
            if test_scores[1] > 40:
                easiest_imgs.append(x.cpu())
            psnry_scores.append(test_scores[1])

            if save_imgs and dataloaders.test_dataloader.batch_size==1:
                create_parent_dir(f"output/dataset_test/{foldername}/x")
                save_image(quantize_ste(x_recon_from_y), f"output/dataset_test/{foldername}/{str(test_batch_no).zfill(3)}_{int(time.time())}.png")
                save_image(quantize_ste(y), f"output/dataset_test/{foldername}/{str(test_batch_no).zfill(3)}_LR_{int(time.time())}.png")

            print(f"Img [{test_batch_no}]: acquired test metrics {[round(float(t), 4) for t in test_scores]}")
                        
            #plt.imshow(x[0].permute(1, 2,0))
            #plt.show()


            #if test_scores[-1] > 0.6:
            all_test_scores.append(test_scores)

        #see_multiple_imgs(easiest_imgs, 5, 5, row_titles=[], plot_titles=[], see=True, save=False, filename="dataview", smallSize=True)
        #see_multiple_imgs(hardest_imgs, 5, 5, row_titles=[], plot_titles=[], see=True, save=False, filename="dataview", smallSize=True)
        
        #plt.hist(psnry_scores, bins=10)
        #plt.show()
        
        #plt.hist(psnry_scores, bins=20)
        #plt.show()

        #plt.hist(psnry_scores, bins=35)
        #plt.show()

        #plt.hist(psnry_scores, bins=50)
        #plt.show()

        #plt.hist(psnry_scores, bins=100)
        #plt.show()

        # Calcuate average for each of our scores
        mean_test_scores = np.array(all_test_scores).mean(axis=0)

        print(f"Calculated scores for {len(all_test_scores)} test batches.")
        print(f'Average scores: {[round(float(t), 4) for t in mean_test_scores]}')
        print(f'Min scores: {[min([y[i] for y in all_test_scores]) for i in range(len(all_test_scores[0]))]}')
        print(f'Max scores: {[max([y[i] for y in all_test_scores]) for i in range(len(all_test_scores[0]))]}')

        # Convert from nparray to indvidual elements
        for mts in mean_test_scores:
            yield mts

def test_inn(inn,
    dataloaders: DataLoaders,
    lambda_recon=1,
    lambda_guide=1,
    lambda_distr=1,
    metric_crop_border=4,
    save_imgs=False,
    fast_gpu_testing=False,
    mean_losses=False,
    quantize_recon=False
):
    test_function_irn = get_test_function_irn(lambda_recon, lambda_guide, lambda_distr, metric_crop_border, fast_gpu_testing=fast_gpu_testing, mean_losses=mean_losses, quantize_recon=quantize_recon)
    sample_function_irn = get_sample_function_irn(inn)
    total_loss, psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr = test_rescaling(dataloaders, sample_function_irn, test_function_irn, save_imgs=save_imgs)

    return total_loss, psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr

def test_multi_inn(inn, num_applications, mid_quantization,
    dataloaders: DataLoaders,
    lambda_recon=1,
    lambda_guide=1,
    lambda_distr=1,
    metric_crop_border=4,
    save_imgs=False,
    fast_gpu_testing=False,
    mean_losses=False,
    quantize_recon=False
):
    test_function_irn = get_test_function_irn(lambda_recon, lambda_guide, lambda_distr, metric_crop_border, fast_gpu_testing=fast_gpu_testing, mean_losses=mean_losses, quantize_recon=quantize_recon)
    sample_function_irn = get_multi_sample_function_irn(inn, num_applications, mid_quantization)
    total_loss, psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr = test_rescaling(dataloaders, sample_function_irn, test_function_irn, save_imgs=save_imgs)

    return total_loss, psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr

def test_bicub(dataloaders, scale, metric_crop_border):
    test_function = get_test_function_psnr_ssim(metric_crop_border)
    sample_function = get_sample_function_bicub(scale)
    psnr_RGB, psnr_Y, ssim, ssim_Y = test_rescaling(dataloaders, sample_function, test_function)
    return psnr_RGB, psnr_Y, ssim, ssim_Y

def test_imgfolder(dataloaders, scale, metric_crop_border, path):
    test_function = get_test_function_psnr_ssim(metric_crop_border)
    sample_function = get_sample_function_imgfolder(path, scale, scale)
    psnr_RGB, psnr_Y, ssim, ssim_Y = test_rescaling(dataloaders, sample_function, test_function)
    return psnr_RGB, psnr_Y, ssim, ssim_Y

def get_saved_inn(path):
    config={
            "batch_size": 10,
            "lambda_recon": 100,
            "lambda_guide": 20,
            "lambda_distr": 0.01,
            "initial_learning_rate": 0.001,
            "img_size": 144,
            "scale": 4, # ds_count = log2(scale)
            "inv_per_ds": 2,
            "inv_first_level_extra": 0,
            "inv_final_level_extra": 0,
            "seed": 10,
            "grad_clipping": 2,
            "full_size_test_imgs": True,
            "lr_batch_milestones": [7000, 14000, 21000, 28000],
            "lr_gamma": 0.5,
            "batchnorm": False,
            "actnorm": False,
    }

    inn = IRN(3, config["img_size"], config["img_size"], cfg=config)

    inn, optimizer, epoch, min_training_loss, min_test_loss, max_test_psnr_y = load_network(inn, path, None)

    return inn

if __name__ == '__main__':
    

    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)

    # Load the data (note the training params here are unnecessary)
    scale = 16
    crop = scale

    dataloaders = Div2KDataLoaders(16, 144, full_size_test_imgs=True, test_img_size_divisor=scale)
    
    inn = get_saved_inn("./saved_models/model_1645724973_2772.0_1.66_27g0ujq7.pth")
    #total_loss, psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr = test_inn(inn, dataloaders, 100, 20, 0.01, metric_crop_border=crop, save_imgs=False)
    total_loss, psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr = test_multi_inn(inn, 2, True, dataloaders, 1, 16, 1, metric_crop_border=crop, save_imgs=True)
    
    #psnr_RGB, psnr_Y, ssim, ssim_Y = test_imgfolder(dataloaders, scale, crop, "./data/DIV2K/DIV2K_valid_x4recon_bicubic")

    print(f"(^ total_loss, psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr)\n")
    #print(f"(^ PSNR, PSNR_Y, SSIM, SSIM_Y)\n")