# Train a given model
from argparse import ArgumentParser
import time
from tracemalloc import start
from networks.freia_invertible_rescaling import IRN, sample_inn
from data import mnist8_iterator, process_xbit_img, see_multiple_imgs, process_div2k_img
from data2 import Div2KDataLoaders, DataLoaders
from bicubic_pytorch.core import imresize
import config_loader
import torch
from torchvision.utils import save_image
import torchmetrics
import numpy as np
import wandb
from test import test_inn
from network_saving import save_network, load_network
import math
from timeit import default_timer as timer
import glob
import os
import config_loader
import random
from loss import calculate_irn_loss
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
#from IQA_pytorch import SSIM, DISTS

# input should be size (n, 3, h, w) - in range [0,1]
# output has size (n, 1, h, w) - treat the output as if it has range [0,255]
def rgb_to_y(rgb_imgs):
  rgb_imgs *= 255
  output = (rgb_imgs[:,0,:,:] * 65.481 + rgb_imgs[:,1,:,:] * 128.553 + rgb_imgs[:,2,:,:] * 24.966)/255.0 + 1
  return output.unsqueeze(dim=1)

def latest_file_in_folder(folder_path, file_ending=""):
    # Can target specific file extensions by replacing * with e.g. *.pth
    latest_file_path = max(glob.glob(f"{folder_path}/*{file_ending}"), key=os.path.getctime)
    return latest_file_path

def see_irn_example(x, y, z, x_recon_from_y, scale, see=True, save=True, wandb_log=False, wandb_step=-1, name=0, render_grid=True, metric_crop_border=4):
    print("-> About to visualize irn example")
    i = random.randint(0, len(x)-1)

    # Don't border crop the downscaled images
    x_downscaled_bc = imresize(x[i], scale=1.0/scale)
    x_upscaled_bc = imresize(x_downscaled_bc, scale=scale)


    #div2k images are loaded with values in range [0, 1] but mnist8 is in range [0, 16].
    size = x.shape[-1]
    imgs = [process_div2k_img(*im_and_size) for im_and_size in [
        (x[i], x[i].shape),
        (x_downscaled_bc, x_downscaled_bc.shape),
        (y[i], y[i].shape),
        (x[i], x[i].shape),
        (x_upscaled_bc, x_upscaled_bc.shape),
        (x_recon_from_y, [3, x_recon_from_y.shape[-2], x_recon_from_y.shape[-1]]),
    ]]

    # Test the network
    # mnist8 consists of 16-tone images
    # TODO: report y psnr
    x_upscaled_bc_cropped = torch.squeeze(crop_tensor_border(x_upscaled_bc, metric_crop_border))
    x_recon_from_y_cropped = torch.squeeze(crop_tensor_border(x_recon_from_y, metric_crop_border))
    x_cropped = torch.squeeze(crop_tensor_border(x[i], metric_crop_border))

    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1).cuda()
    [psnr_irn_down, psnr_bi_up, psnr_irn_up] = [round(float(psnr_metric(pred_vs_goal[0], pred_vs_goal[1]).item()), 2) for pred_vs_goal in [
        (y[i], x_downscaled_bc), (x_upscaled_bc_cropped, x_cropped), (x_recon_from_y_cropped, x_cropped)]]

    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1).cuda()
    [ssim_irn_down, ssim_bi_up, ssim_irn_up] = [round(float(ssim_metric(torch.unsqueeze(pred_vs_goal[0], dim=0), torch.unsqueeze(pred_vs_goal[1], dim=0)).item()), 4) for pred_vs_goal in [
        (y[i], x_downscaled_bc), (x_upscaled_bc_cropped, x_cropped), (x_recon_from_y_cropped, x_cropped)]]

    if render_grid:
        fig = see_multiple_imgs(imgs, 2, 3,
            row_titles=[
                "Downscaling task (PSNR(RGB)/SSIM)",
                "Downscaling & upscaling task (PSNR/SSIM)"],
            plot_titles=[
                "HR (-)", "GT [Bi-down] (∞)", f"IRN-down ({psnr_irn_down}/{ssim_irn_down})",
                "GT [HR] (∞)", f"Bi-down & Bi-up ({psnr_bi_up}/{ssim_bi_up})", f"IRN-down & IRN-up ({psnr_irn_up}/{ssim_irn_up})"
            ],
            see=see, save=save,
            filename=f'output/out_{int(time.time())}_{i}_{name}'
        )

        if wandb_log:
            wandb.log({f"grid_example": wandb.Image(fig), f"single_example": wandb.Image(imgs[-1])}, commit=False, step=wandb_step)
    else:
        # Warning: see=True does nothing for render_grid=False
        if save:
            file_names = ["HR", "Bi-down", f"IRN-down-{psnr_irn_down}-{ssim_irn_down}", f"Bi-down-Bi-up-{psnr_bi_up}-{ssim_bi_up}", f"IRN-down-IRN-up-{psnr_irn_up}-{ssim_irn_up}"]
            file_imgs = [x[i], x_downscaled_bc, y[i], x_upscaled_bc, x_recon_from_y]
            filename_start = f'output/out_{int(time.time())}_{i}_{name}_'

            for fni in range(len(file_names)):
                save_image(file_imgs[fni] / 1, filename_start + file_names[fni] + ".png")
        if wandb_log:
            table = wandb.Table(columns=["Img name", "Img", "PSNR(RGB)", "SSIM"])
            table.add_data("GT [HR]", wandb.Image(imgs[0]), np.Infinity, 1)
            table.add_data("GT [Bi-down]", wandb.Image(imgs[1]), np.Infinity, 1)
            table.add_data("IRN-down", wandb.Image(imgs[2]), psnr_irn_down, ssim_irn_down)
            table.add_data("Bi-down & Bi-up", wandb.Image(imgs[4]), psnr_bi_up, ssim_bi_up)
            table.add_data("IRN-down & IRN-up", wandb.Image(imgs[5]), psnr_irn_up, ssim_irn_up)
            
            wandb.log({"samples_table":table, f"single_example":wandb.Image(imgs[-1])}, commit=False, step=wandb_step)

    print("-> Finished visualizing.")

def crop_tensor_border(x, border):
    return x[..., border:-border, border:-border]

def train_inn(inn, dataloaders: DataLoaders, cfg, load_checkpoint_path=None, run_name="run"):

    config_loader.check_keys(cfg,
    [
        "max_batches", "max_epochs", "target_loss",
        "epochs_between_tests", "epochs_between_training_log", "epochs_between_samples", "epochs_between_saves",
        "initial_learning_rate", "grad_clipping", "scale",
        "lambda_recon", "lambda_guide", "lambda_distr",
        "mean_losses", "fast_gpu_testing", "quantize_recon_loss",
        "lr_batch_milestones", "lr_gamma"
    ])

    print(f"About to train model: {inn}")

    optimizer = torch.optim.Adam(inn.parameters(), lr=cfg["initial_learning_rate"], weight_decay=0.00001, amsgrad=False)
    
    if(load_checkpoint_path):
        inn, optimizer, epoch, min_training_loss, min_test_loss, max_test_psnr_y = load_network(inn, load_checkpoint_path, optimizer)

        batch_no = math.floor(epoch * dataloaders.train_len / dataloaders.train_dataloader.batch_size)

        all_test_losses = [min_test_loss]
        all_test_psnr_y = [max_test_psnr_y]

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["lr_batch_milestones"], gamma=cfg["lr_gamma"], last_epoch=batch_no)
    else:
        min_training_loss = 99999

        epoch = 0
        batch_no = 0

        all_test_losses = []
        all_test_psnr_y = []

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["lr_batch_milestones"], gamma=cfg["lr_gamma"])

    recent_training_losses = []
    avg_training_loss = cfg["target_loss"]+1

    epoch_prev_test = epoch-1
    epoch_prev_training_log = epoch-1
    epoch_prev_sample = epoch-1
    epoch_prev_save = epoch-1

    train_iter = iter(dataloaders.train_dataloader)

    start_ep = timer()
    time_forward = 0
    time_loss = 0
    time_dataloading = 0
    time_backward = 0
    time_testing_saving_sampling = 0

    while (cfg["max_batches"]==-1 or batch_no < cfg["max_batches"]) and (cfg["target_loss"]==-1 or cfg["target_loss"]<=avg_training_loss) and (cfg["max_epochs"]==-1 or epoch < cfg["max_epochs"]):
        optimizer.zero_grad()
        
        start = timer()
        try:
            x, _ = next(train_iter)
        except StopIteration:
            stop_ep = timer()
            epoch_time = stop_ep - start_ep
            start_ep = stop_ep

            print("---")
            print(f"TOTAL EPOCH TIME: {round(epoch_time, 2)}s")
            print(f"TOTAL FORWARD TIME: {round(time_forward, 2)}s")
            print(f"TOTAL LOSS TIME: {round(time_loss, 2)}s")
            print(f"TOTAL DATALOADING TIME: {round(time_dataloading, 2)}s")
            print(f"TOTAL BACKWARD TIME: {round(time_backward, 2)}s")
            print(f"TOTAL time_testing_saving_sampling TIME: {round(time_testing_saving_sampling, 2)}s")
            print(f"(UNACCOUNTED FOR TIME): {round((epoch_time) - (time_forward + time_loss + time_dataloading + time_backward + time_testing_saving_sampling), 2)}s")
            print("---")

            # Log epoch time, as well as lr to track the scheduler's progress
            wandb.log({"epoch_time_s": epoch_time, "lr": scheduler.get_last_lr()[0], "epoch": epoch}, step=batch_no)

            time_forward = 0
            time_loss = 0
            time_dataloading = 0
            time_backward = 0
            time_testing_saving_sampling = 0

            train_iter = iter(dataloaders.train_dataloader)
            x, _ = next(train_iter)

        stop = timer()
        time_dataloading += stop-start

        start = timer()
        x, y, z, x_recon_from_y, mean_y, std_y = sample_inn(inn, x)
        stop = timer()
        time_forward += stop - start

        start = timer()
        loss_recon, loss_guide, loss_distr, loss_batchnorm, total_loss = calculate_irn_loss(cfg["lambda_recon"], cfg["lambda_guide"], cfg["lambda_distr"], x, y, z, x_recon_from_y, cfg["scale"], mean_y, std_y, mean_losses=cfg["mean_losses"], quantize_recon=cfg["quantize_recon_loss"])
        stop = timer()
        time_loss += stop - start

        if float(total_loss) > avg_training_loss and (len(recent_training_losses) == 0 or float(total_loss) > max(recent_training_losses)):
            print(f"{total_loss} is new highest loss in recent set...")
            x_and_xrecon_and_diff_imgs = list(x.cpu().detach()) + list(x_recon_from_y.cpu().detach()) + list(torch.abs(x_recon_from_y-x).cpu().detach())
            plt_titles = []
            for xi in range(len(x_and_xrecon_and_diff_imgs)):
                img = x_and_xrecon_and_diff_imgs[xi]

                basics = f"min {round(float(img.min()), 2)} \nmax {round(float(img.max()), 2)} \nmean {round(float(img.mean()), 2)} \nsum {round(float(img.sum()), 2)}"
                if xi < x.shape[0]:
                    basics += f" \ndistr mean={float(z[xi].mean())} \ndistr loss={float((z[xi]**2).sum())}"
                plt_titles.append(basics)
            see_multiple_imgs(x_and_xrecon_and_diff_imgs, 3, 16, row_titles=[f"lr, lg, ld, lt: {(float(loss_recon), float(loss_guide), float(loss_distr), float(total_loss))}.\nxmin={x.min()} ymim={y.min()} reconmin={x_recon_from_y.min()}\n\n\n\n"], plot_titles=plt_titles, see=False, save=True, filename=f"{run_name}_{int(time.time())}_{batch_no}_{int(total_loss)}", smallSize=False)

        if batch_no % 5 == 0:
            with torch.no_grad():
                blacksquare = torch.zeros(1, 3, 144, 144)
                blacksquare, bs_y, bs_z, bs_x_recon_from_y, bs_mean_y, bs_std_y = sample_inn(inn, blacksquare)
                bsloss_recon, bsloss_guide, bsloss_distr, bsloss_batchnorm, bstotal_loss = calculate_irn_loss(cfg["lambda_recon"], cfg["lambda_guide"], cfg["lambda_distr"], blacksquare, bs_y, bs_z, bs_x_recon_from_y, cfg["scale"], bs_mean_y, bs_std_y, mean_losses=cfg["mean_losses"], quantize_recon=cfg["quantize_recon_loss"])

                see_multiple_imgs([blacksquare[0].cpu().detach(), imresize(bs_y[0].cpu().detach(), 4), bs_x_recon_from_y[0].cpu().detach()], 1, 3,
                                   row_titles=[f"lr, lg, ld, lt: {(float(bsloss_recon), float(bsloss_guide), float(bsloss_distr), float(bstotal_loss))}. \nreconmin={float(bs_x_recon_from_y.min())}, reconmax={float(bs_x_recon_from_y.max())}\n\n\n\n"],
                                   plot_titles=["x", "y", "x_recon"], see=False, save=True, filename=f"bs_{run_name}_{int(time.time())}_{batch_no}_{int(bstotal_loss)}",
                                   smallSize=False)

        recent_training_losses.append(float(total_loss))

        samples = batch_no * dataloaders.train_dataloader.batch_size
        epoch = float(samples / dataloaders.train_len)

        start = timer()
        if epoch - epoch_prev_training_log >= cfg["epochs_between_training_log"]:
            avg_training_loss = sum(recent_training_losses) / len(recent_training_losses)
            min_training_loss = min([min_training_loss, avg_training_loss])
            
            print(f"At {batch_no} batches (epoch {epoch}): Avg training loss, in last {cfg['epochs_between_training_log'] if epoch>0 else 0} epochs: {avg_training_loss}")
            print(f'loss_recon={loss_recon}, loss_guide={loss_guide}, loss_distr={loss_distr}, loss_batchnorm={loss_batchnorm}')
            recent_training_losses = []
            wandb.log({"train_loss": avg_training_loss, "train_loss_recon": loss_recon, "train_loss_guide": loss_guide, "train_loss_distr": loss_distr, "min_training_loss": min_training_loss, "epoch": epoch}, step=batch_no)
            epoch_prev_training_log = epoch

        if epoch - epoch_prev_sample >= cfg["epochs_between_samples"]:
            print(f"We are sampling at {batch_no+1} batches, {epoch} epochs...")
            with torch.no_grad():
                index_of_sample_image = 4
                x, y, z, x_recon_from_y, mean_y, std_y = sample_inn(inn, dataloaders.test_dataloader.dataset[index_of_sample_image][0].unsqueeze(dim=0))
            see_irn_example(x, y, z, x_recon_from_y, cfg["scale"], see=False, save=True, wandb_log=True, wandb_step=batch_no, name=all_test_losses[-1] if len(all_test_losses)>0 else "-", render_grid=False, metric_crop_border=cfg["scale"])
            #for j in range(2):
            #    see_irn_example(x, y, z, x_recon_from_y, see=False, save=False, name=all_test_losses[-1])
            epoch_prev_sample = epoch
        
        if epoch - epoch_prev_save >= cfg["epochs_between_saves"]:
            print(f"We are saving at {batch_no+1} batches, {epoch} epochs...")
            save_network(inn, optimizer, epoch, min_training_loss, all_test_losses, all_test_psnr_y, f"{round(epoch, 2)}_{round(all_test_losses[-1], 2) if len(all_test_losses)>0 else '-'}_{run_name}")
            print(f"Finished saving.")
            epoch_prev_save = epoch

        if epoch - epoch_prev_test >= cfg["epochs_between_tests"]:
            print(f"At {batch_no} batches (epoch {epoch}):")
            print(f'loss_recon={loss_recon}, loss_guide={loss_guide}, loss_distr={loss_distr}')
            print(f'In test dataset, in last {cfg["epochs_between_tests"] if epoch>0 else 0} epochs:')

            # Crop the border of test images by the resize scale to avoid capturing outliers (this is done in the IRN paper)
            test_loss, test_psnr_rgb, test_psnr_y, test_ssim_RGB, test_ssim_Y, test_lossr, test_lossg, test_lossd = test_inn(inn, dataloaders, cfg["scale"], cfg["lambda_recon"], cfg["lambda_guide"], cfg["lambda_distr"], metric_crop_border=cfg["scale"], fast_gpu_testing=cfg["fast_gpu_testing"], mean_losses=cfg["mean_losses"], quantize_recon=cfg["quantize_recon_loss"])
            all_test_losses.append(test_loss)
            all_test_psnr_y.append(test_psnr_y)
            print("")
    
            # IDEA: also track greyscale PSNR or greyscale loss, as this gives us a sense of how much blame is just on hue
            wandb.log({"test_loss": test_loss, "min_test_loss": min(all_test_losses), "max_test_psnr_y": max(all_test_psnr_y), "test_psnr": test_psnr_rgb, "test_psnr_y": test_psnr_y, "test_ssim": test_ssim_RGB, "test_ssim_y": test_ssim_Y, "test_loss_recon": test_lossr, "test_loss_guide": test_lossg, "test_loss_distr": test_lossd, "epoch": epoch}, step=batch_no)
            epoch_prev_test = epoch
        stop = timer()
        time_testing_saving_sampling += stop-start

        start = timer()
        total_loss.backward()
        stop = timer()
        time_backward += stop - start

        torch.nn.utils.clip_grad_norm_(inn.parameters(), max_norm=cfg["grad_clipping"])
        optimizer.step()
        scheduler.step()
        batch_no+=1
    return min_training_loss, all_test_losses


####    EXECUTION CODE    ####
if __name__ == '__main__':

    #os.environ['CUDA_LAUNCH_BLOCKING'] = str(1) # for debugging purposes
    parser = ArgumentParser()
    parser.add_argument("config", help=".yaml file containing model and train/test options")
    parser.add_argument("-r", "--run_id", dest="run_id", help="id of the wandb run to resume")
    parser.add_argument("-f", "--resume_file", dest="resume_file", help="path to specific saved_model file to resume")
    parser.add_argument("-l", "--resume_latest_run", dest="resume_latest_run", action="store_true", default=False, help="resume latest wandb run")
    args = parser.parse_args()
    print(f"Launched with arguments: {args}")

    configyaml = ""
    if args.config:
        configyaml = config_loader.load_config(args.config)

    resume_latest_run = args.resume_latest_run
    resume_run_id = args.run_id
    resume_file_name = args.resume_file # optional

    wandb.login()
    run = wandb.init(
        project="invertible-rescaling-network",
        dir=".",
        resume=resume_latest_run if resume_latest_run else None,
        id=resume_run_id if resume_run_id else None,
        config=configyaml # this line necessary for running sweeps
    )
    config = wandb.config

    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])


    # Load the data
    dataloaders = Div2KDataLoaders(config.batch_size, config.img_size, full_size_test_imgs=config.full_size_test_imgs, test_img_size_divisor=config.scale)

    # Load the network
    inn = IRN(3, config["img_size"], config["img_size"], cfg=config)

    if resume_file_name:
        load_checkpoint_path = latest_file_in_folder("./saved_models", resume_file_name)
    elif resume_run_id:
        load_checkpoint_path = latest_file_in_folder("./saved_models", f"{run.id}.pth")
    elif resume_latest_run:
        load_checkpoint_path = latest_file_in_folder("./saved_models", ".pth")
    else:
        load_checkpoint_path = ""

    # Train the network
    all_training_losses, all_test_losses = train_inn(inn, dataloaders, cfg=config, load_checkpoint_path=load_checkpoint_path, run_name=run.id)

    #plt.savefig(f'output/test_loss_{int(time.time())}_{int(all_test_losses[-1])}', dpi=100)
    run.finish()