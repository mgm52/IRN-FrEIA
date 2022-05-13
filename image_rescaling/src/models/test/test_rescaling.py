import time
from tracemalloc import start
from torchvision import transforms
from models.layers.invertible_rescaling_network import IRN, sample_irn, multi_sample_irn, upscale_irn
from visualisation.visualise import process_xbit_img, see_multiple_imgs, process_div2k_img
import models.model_loader
from data.dataloaders import DatasetDataLoaders, DataLoaders, get_test_dataloader
from utils.bicubic_pytorch.core import imresize
import torch
from torchvision.utils import save_image
import torchmetrics
import numpy as np
from models.model_loader import load_network
from timeit import default_timer as timer
from models.layers.invertible_rescaling_network import quantize_ste
from utils.utils import create_parent_dir
from utils.utils import rgb_to_y
import random
from models.train.loss_irn import calculate_irn_loss
from datetime import date
import matplotlib.pyplot as plt
from typing import List


plt.rcParams["font.family"] = "serif"

def get_extreme_indices(scores: List[float], n, find_min=False):

  scores = np.array(scores)
  n = min([n, len(scores)])

  top_n_indices = np.argpartition(scores if find_min else -scores, n)
  top_n_indices = top_n_indices[:n]

  top_n_indices = [x for _, x in sorted(zip(scores[top_n_indices], top_n_indices), reverse=True)]

  return top_n_indices


def crop_tensor_border(x, border):
    if border==0: return x
    return x[..., border:-border, border:-border]

def get_test_function_irn(lambda_recon, lambda_guide, lambda_distr, metric_crop_border, cfg, compute_lr_psnr_ssim=False):
    def test_function_irn(x, y, z, x_recon_from_y, mean_y, std_y):
        loss_recon, loss_guide, loss_distr, loss_batchnorm, total_loss = calculate_irn_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, mean_y, std_y, batchnorm=False, mean_losses=cfg["mean_losses"], quantize_recon=cfg["quantize_recon_loss"], y_channel_usage=cfg["y_channel_usage"], y_guide_only=cfg.get("y_guide_only"))

        test_function_psnr_ssim = get_test_function_psnr_ssim(metric_crop_border, fast_gpu_testing=cfg["fast_gpu_testing"], compute_for_lr=compute_lr_psnr_ssim)
        psnr_ssim_scores, psnr_ssim_names = test_function_psnr_ssim(x, y, z, x_recon_from_y, mean_y, std_y)

        return (psnr_ssim_scores + [float(loss_recon), float(loss_guide), float(loss_distr), float(total_loss)],
                psnr_ssim_names + ["Loss (Recon)", "Loss (Guide)", "Loss (Distr)", "Loss (Total)"])
    return test_function_irn

def get_test_function_psnr_ssim(metric_crop_border, fast_gpu_testing=False, compute_for_lr=False):
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
        psnr_RGB = psnr_metric(x_recon_from_y_cropped, x_cropped)
        psnr_Y   = psnr_metric(x_recon_from_y_cropped_Y, x_cropped_Y)        

        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1).to(test_device)
        ssim_RGB   = ssim_metric(x_recon_from_y_cropped, x_cropped)
        ssim_Y = ssim_metric(x_recon_from_y_cropped_Y, x_cropped_Y)

        if not compute_for_lr:
            return ([float(psnr_RGB), float(psnr_Y), float(ssim_RGB), float(ssim_Y)],
                    ["PSNR (RGB)", "PSNR (Y)", "SSIM (RGB)", "SSIM (Y)"])

        x_downscaled = imresize(x, scale=1.0/scale)

        x_downscaled_cropped = crop_tensor_border(x_downscaled, 1)
        y_cropped = crop_tensor_border(y, 1)

        x_downscaled_cropped_Y = rgb_to_y(x_downscaled_cropped)
        y_cropped_Y = rgb_to_y(y_cropped)

        psnr_RGB_LR = psnr_metric(x_downscaled_cropped,   y_cropped)
        psnr_Y_LR   = psnr_metric(x_downscaled_cropped_Y, y_cropped_Y)

        ssim_RGB_LR = ssim_metric(x_downscaled_cropped,   y_cropped)
        ssim_Y_LR   = ssim_metric(x_downscaled_cropped_Y, y_cropped_Y)

        return ([float(psnr_RGB), float(psnr_Y), float(ssim_RGB), float(ssim_Y), float(psnr_RGB_LR), float(psnr_Y_LR), float(ssim_RGB_LR), float(ssim_Y_LR)],
                ["PSNR (RGB) (HR)", "PSNR (Y) (HR)", "SSIM (RGB) (HR)", "SSIM (Y) (HR)", "PSNR (RGB) (LR)", "PSNR (Y) (LR)", "SSIM (RGB) (LR)", "SSIM (Y) (LR)"])
        
    return test_function_psnr_ssim

def get_multi_sample_function_irn(inn, num_applications, mid_quantization):
    return lambda x: multi_sample_irn(inn, x.clone(), num_applications, mid_quantization)
def get_sample_function_irn(inn, cfg):
    return lambda x: sample_irn(inn, x.clone(), cfg)

def get_sample_function_bicub(scale):
    def sample_function_bicub(x):
        x_downscaled = imresize(x, scale=1.0/scale)
        y = x_downscaled

        x_recon_from_y = imresize(y, scale=scale)

        z = None

        mean_y = y.mean()
        std_y = y.std()

        return x, y, z, x_recon_from_y, mean_y, std_y, y
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

        return x, y, z, x_recon_from_y, mean_y, std_y, y
    return sample_function_imgfolder

def get_sample_function_irn_lrimgfolder(irn, lrpath, scale):

    lr_img_iter = iter(get_test_dataloader(lrpath))

    def sample_function_imgfolder(x):
        x_downscaled = (imresize(x, scale=1.0/scale)).cuda()

        y = x_downscaled

        ymod = (next(lr_img_iter)[0]).cuda()

        x_recon_from_y = (upscale_irn(irn, scale, ymod, zerosample=False)).cuda()

        z = None

        mean_y = y.mean()
        std_y = y.std()

        return x.cuda(), y, z, x_recon_from_y, mean_y, std_y, ymod
    return sample_function_imgfolder

def test_rescaling(dataloaders, x_y_z_recon_mean_std_mod_function, test_metrics_function, save_imgs=False, foldername=None, save_extras=False):
    with torch.no_grad():
        current_date = date.today().strftime("%Y-%m-%d")
        if not foldername:
            foldername = f"{current_date}_{int(time.time())}"

        max_test_batches = dataloaders.test_len / dataloaders.test_dataloader.batch_size
        assert(max_test_batches == int(max_test_batches)), f"Test batch size ({dataloaders.test_dataloader.batch_size}) does not fit into test dataset ({dataloaders.test_len})"

        # Get losses across test dataset
        all_test_scores = []
        test_iter = iter(dataloaders.test_dataloader)
        
        psnry_scores = []

        #sample_function_imgload = get_sample_function_imgfolder("./data/DIV2K/DIV2K_valid_x4recon_IRN-mine-191", 4, 4)

        all_scores_msg = ""
        print("Beginning test...")
        for test_batch_no in range(int(max_test_batches)):
            # todo: use batch_size=-1 instead, then check that it works
            if (test_batch_no-1) % int(max(2, max_test_batches / 5)) == 0:
                print(f"At test item {test_batch_no}/{int(max_test_batches)}")
                print(f'Average values: {[round(float(t), 4) for t in np.array(all_test_scores).mean(axis=0)]}')

            x_raw, _ = next(test_iter)
            
            x, y, z, x_recon_from_ymod, mean_y, std_y, ymod = x_y_z_recon_mean_std_mod_function(x_raw)
            
            #x_l, y_l, z_l, x_recon_from_y_l, mean_y_l, std_y_l = sample_function_imgload(x_raw)

            #print(f"x_recon_from_y has max value {x_recon_from_y.max()}")
            #save_image(x_recon_from_y, fp=str(test_batch_no)+".png")

            #print(f"Found x_recon_from_y {x_recon_from_y.shape} {x_recon_from_y}")

            test_scores, test_names = test_metrics_function(x, y, z, x_recon_from_ymod, mean_y, std_y)

            #if test_scores[1] < 25:
            #    hardest_imgs.append(x.cpu())
            #if test_scores[1] > 40:
            #    easiest_imgs.append(x.cpu())
            #psnry_scores.append(test_scores[1])

            if save_imgs and dataloaders.test_dataloader.batch_size==1:
                create_parent_dir(f"../output/test/{foldername}/x")
                save_image(quantize_ste(x_recon_from_ymod), f"../output/test/{foldername}/{str(test_batch_no).zfill(3)}_RECON.png")
                save_image(quantize_ste(y), f"../output/test/{foldername}/{str(test_batch_no).zfill(3)}_LR.png")
                if not torch.equal(y, ymod):
                    save_image(quantize_ste(ymod), f"../output/test/{foldername}/{str(test_batch_no).zfill(3)}_LRMOD.png")

            current_result_msg = f"Img [{test_batch_no}]: test metrics {[round(float(test_scores[t]), 4) for t in range(len(test_scores))]}"
            if not all_scores_msg: all_scores_msg = str(test_names) + "\n"
            all_scores_msg += current_result_msg + "\n"
            print(current_result_msg)
                        
            #plt.imshow(x[0].permute(1, 2,0))
            #plt.show()


            #if test_scores[-1] > 0.6:
            all_test_scores.append(test_scores)
        
        def string_scores(scores, bold_index=-1):
            score_strs = [f'{test_names[tsi]}: {float("%.4g" % scores[tsi])}' for tsi in range(len(test_names))]
            if bold_index > -1:
                score_strs[bold_index] = "\\textbf{" + score_strs[bold_index] + "}"
            return ',\n'.join(score_strs)

        plt.rc('text', usetex=True)
            
        zipped_scores = list(zip(*all_test_scores))

        create_parent_dir(f"../output/test/{foldername}/x")
        create_parent_dir(f"../output/test/{foldername}/metrics/x")

        if save_extras:
            for score_index in range(len(zipped_scores)):
                three_best_i = get_extreme_indices(zipped_scores[score_index], 3, False)
                three_worst_i = get_extreme_indices(zipped_scores[score_index], 3, True)

                best_imgs = [dataloaders.test_dataloader.dataset[x][0] for x in three_best_i]
                worst_imgs = [dataloaders.test_dataloader.dataset[x][0] for x in three_worst_i]

                see_multiple_imgs(best_imgs, 1, 3, row_titles=[], plot_titles=[f"Image [{x}] with scores \n{string_scores(all_test_scores[x], score_index)}" for x in three_best_i], see=False, save=True, filename=f"../output/test/{foldername}/metrics/three_best_score_{test_names[score_index]}", smallSize=True, pdf=True)
                see_multiple_imgs(worst_imgs, 1, 3, row_titles=[], plot_titles=[f"Image [{x}] with scores \n{string_scores(all_test_scores[x], score_index)}" for x in three_worst_i], see=False, save=True, filename=f"../output/test/{foldername}/metrics/three_worst_score_{test_names[score_index]}", smallSize=True, pdf=True)

        plt.rcParams['font.size'] = '16'
        colors = ["#CC2E60", "#B44382", "#B443AF", "#8E43B4", "#493C93", "#37377E", "#013774", "#014B74", "#015F74", "#017374", "#018774", "#019B74"]
        for score_index in range(len(zipped_scores)):
            plt.grid(axis='y', color="0.8", zorder=0)
            plt.hist(zipped_scores[score_index], bins=35, zorder=10, color=colors[(score_index) % len(colors)])
            plt.ylabel("Frequency", labelpad=10)
            plt.xlabel(test_names[score_index])
            
            # ensure y axis is integer
            yint = [int(each) for each in plt.yticks()[0]]
            plt.yticks(yint)

            plt.savefig(f"../output/test/{foldername}/metrics/hist35_{test_names[score_index]}.pdf", bbox_inches='tight')
            plt.close()

        
        #see_multiple_imgs(hardest_imgs, 5, 5, row_titles=[], plot_titles=[], see=True, save=True, filename="dataview-hardest", smallSize=True, pdf=True)
        

        #

        #plt.hist(psnry_scores, bins=100)
        #plt.ylabel("Frequency")
        #plt.xlabel("PSNR (Y)")
        #plt.savefig("../output/test/bicub4x/hist100.pdf")
        #plt.show()

        # Calcuate average for each of our scores
        mean_test_scores = np.array(all_test_scores).mean(axis=0)

        msg = f"Calculated scores for {len(all_test_scores)} test batches in {foldername}.\n\n"
        msg += f'Min scores: \n{string_scores([min([y[i] for y in all_test_scores]) for i in range(len(all_test_scores[0]))])}\n\n'
        msg += f'5 percentile scores: \n{string_scores(np.percentile(all_test_scores, 5, axis=0))}\n\n'
        msg += f'Lower quantile scores: \n{string_scores(np.quantile(all_test_scores, 0.25, axis=0))}\n\n'
        msg += f'Average scores: \n{string_scores(mean_test_scores)}\n\n'
        msg += f'Upper quantile scores: \n{string_scores(np.quantile(all_test_scores, 0.75, axis=0))}\n\n'
        msg += f'95 percentile scores: \n{string_scores(np.percentile(all_test_scores, 95, axis=0))}\n\n'
        msg += f'Max scores: \n{string_scores([max([y[i] for y in all_test_scores]) for i in range(len(all_test_scores[0]))])}\n\n'
        msg += f'Standard devs: \n{string_scores(np.std(all_test_scores, axis=0))}\n'

        create_parent_dir(f"../output/test/{foldername}/x")
        create_parent_dir(f"../output/test/{foldername}/metrics/x")
        f = open(f"../output/test/{foldername}/metrics/results_summary.txt", "w")
        f.write(msg)
        f.close()

        f = open(f"../output/test/{foldername}/metrics/results_all.txt", "w")
        f.write(all_scores_msg)
        f.close()

        print(msg)

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
    foldername=None,
    save_extras=False,
    compute_lr_scores=False,
    cfg={
        "fast_gpu_testing": True,
        "mean_losses": False,
        "quantize_recon": False,
        "zerosample": False,
        "y_channel_usage": 0,
        "sr_mode": False,
        "batchnorm": False,
        "compression_mode": False,
        "compression_quality": -1
    }
):
    test_function_irn = get_test_function_irn(lambda_recon, lambda_guide, lambda_distr, metric_crop_border, cfg, compute_lr_psnr_ssim=compute_lr_scores)
    sample_function_irn = get_sample_function_irn(inn, cfg)
    scores = test_rescaling(dataloaders, sample_function_irn, test_function_irn, save_imgs=save_imgs, foldername=foldername, save_extras=save_extras)

    # unlesss computing lr scores too, returns psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr, total_loss
    return list(scores)

def test_inn_lrimgfolder(inn, lrpath,
    dataloaders: DataLoaders,
    lambda_recon=1,
    lambda_guide=1,
    lambda_distr=1,
    metric_crop_border=4,
    save_imgs=False,
    cfg={
        "fast_gpu_testing": True,
        "mean_losses": False,
        "quantize_recon": False,
        "zerosample": False,
        "y_channel_usage": 0,
        "sr_mode": False,
        "batchnorm": False,
        "compression_mode": False,
        "compression_quality": -1
    }
):
    test_function_irn = get_test_function_irn(lambda_recon, lambda_guide, lambda_distr, metric_crop_border, cfg)
    sample_function_irn = get_sample_function_irn_lrimgfolder(inn, lrpath, scale)
    scores = test_rescaling(dataloaders, sample_function_irn, test_function_irn, save_imgs=save_imgs)

    return list(scores)

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
    scores = test_rescaling(dataloaders, sample_function_irn, test_function_irn, save_imgs=save_imgs)

    return list(scores)

def test_bicub(dataloaders, scale, metric_crop_border, save_imgs=False):
    test_function = get_test_function_psnr_ssim(metric_crop_border)
    sample_function = get_sample_function_bicub(scale)
    psnr_RGB, psnr_Y, ssim, ssim_Y = test_rescaling(dataloaders, sample_function, test_function, save_imgs=save_imgs)
    return psnr_RGB, psnr_Y, ssim, ssim_Y

def test_imgfolder(dataloaders, scale, metric_crop_border, path, foldername=None, save_extras=False):
    test_function = get_test_function_psnr_ssim(metric_crop_border)
    sample_function = get_sample_function_imgfolder(path, scale, scale)
    psnr_RGB, psnr_Y, ssim, ssim_Y = test_rescaling(dataloaders, sample_function, test_function, foldername=foldername, save_extras=save_extras)
    return psnr_RGB, psnr_Y, ssim, ssim_Y

def get_saved_inn(path, cfg_path="clamp/irn_4x_og_2tight.yaml"):
    config=models.model_loader.load_config(cfg_path)

    inn = IRN(3, config["img_size"], config["img_size"], cfg=config)

    inn, optimizer, epoch, min_training_loss, min_test_loss, max_test_psnr_y = load_network(inn, path, None)

    return inn, config

if __name__ == '__main__':
    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)

    # Load the data (note the training params here are unnecessary)
    scale = 4
    crop = scale

    for dset in ["DIV2K"]: #["Set5", "B100", "Set14", "Urban100"]: #"DIV2K"
        if dset=="DIV2K":
            dataloaders = DatasetDataLoaders(16, 144, full_size_test_imgs=True, test_img_size_divisor=scale)
        else:
            dataloaders = DatasetDataLoaders(16, 144, full_size_test_imgs=True, test_img_size_divisor=scale, train_path="", test_path=dset)
        
        name_checkpoint_config = [
            ["IRN (mine)", "IRN-mine_10335_3fiuzzxx.pth", "irn_4x_og.yaml"],
            #["2tightcompress on compress90", "IRN-2T_compress_2ob1a4ml.pth", "compression/irn_4x_og_2tight_compression.yaml"],
            ["IRN-2T", "IRN-2T_9235_2iqhb0ef.pth", "clamp/irn_4x_og_2tight.yaml"],
            ["IRN-Y", "IRN-Y_10045_1bmt7qp4.pth", "y_usage/irn_4x_og_0.5yusage.yaml"],
        ]

        for [n, ch, cf] in name_checkpoint_config:
            inn, cfg = get_saved_inn(f"../models/key_models/{ch}", cf)

            #cfg["compression_mode"] = True
            #cfg["compression_quality"] = 90

            scores = test_inn(
                inn, dataloaders, 1, 16, 1, metric_crop_border=crop, save_imgs=True, cfg=cfg, foldername=n + "_" + dset, save_extras=False, compute_lr_scores=True
            )

    # Best 0.5yusage model
    #inn = get_saved_inn("../models/model_1650181747_10045.0_140161.3_1bmt7qp4.pth")

    # Best 2tight model
    # inn, cfg = get_saved_inn("../models/key_models/IRN-2T_9235_2iqhb0ef.pth")

    # Pretty good 2tight_compress model
    # inn, cfg = get_saved_inn("../models/model_1650813189_9953.0_134821.57_2ob1a4ml.pth")

    # Barely-trained 2tight_compress 50/50 model
    # inn, cfg = get_saved_inn("../models/model_1650834404_9293.0_134821.57_s871r2xu.pth")

    # Enable compression testing
    #cfg["compression_mode"] = True
    #cfg["compression_quality"] = 90

    # cfg["fast_gpu_testing"] = True

    # print(cfg)

    # NOTE: TEST RESULT AFFECTED BY test_inn's DEFAULT CFG
    #total_loss, psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr = test_inn_lrimgfolder(inn, "./data/DIV2K/DIV2K_valid_LR_x4_compress100", dataloaders, 1, 16, 1, metric_crop_border=crop, save_imgs=True)
    #scores = test_inn(
    #    inn, dataloaders, 1, 16, 1, metric_crop_border=crop, save_imgs=True, cfg=cfg, foldername="2tight", save_extras=True, compute_lr_scores=True
    #)
    #total_loss, psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr = test_inn(inn, dataloaders, 1, 16, 1, metric_crop_border=crop, save_imgs=True, fast_gpu_testing=True, y_channel_usage=0.5, sr_mode=True)
    #total_loss, psnr_RGB, psnr_Y, ssim_RGB, ssim_Y, loss_recon, loss_guide, loss_distr = test_multi_inn(inn, 2, True, dataloaders, 1, 16, 1, metric_crop_border=crop, save_imgs=True)
    
    #psnr_RGB, psnr_Y, ssim, ssim_Y = test_imgfolder(dataloaders, scale, crop, "../data/DIV2K/DIV2K_valid_x4recon_bicubic", foldername="bicubic_4x", save_extras=True)

    #test_bicub(dataloaders, scale, crop, save_imgs=False)