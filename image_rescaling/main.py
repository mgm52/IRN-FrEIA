# Train a given model
import time
from tracemalloc import start
from networks.freia_invertible_rescaling import IRN, sample_inn
from data import mnist8_iterator, process_xbit_img, see_multiple_imgs, process_div2k_img
from data2 import Div2KDataLoaders, DataLoaders
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
        (x_recon_from_y, x_recon_from_y.shape),
    ]]

    # Test the network
    # mnist8 consists of 16-tone images
    # TODO: report y psnr
    x_upscaled_bc_cropped = crop_tensor_border(x_upscaled_bc, metric_crop_border)
    x_recon_from_y_cropped = crop_tensor_border(x_recon_from_y, metric_crop_border)
    x_cropped = crop_tensor_border(x[i], metric_crop_border)

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

def test_inn(inn,
    dataloaders: DataLoaders,
    scale,
    lambda_recon=1,
    lambda_guide=1,
    lambda_distr=1,
    calculate_metrics=False,
    metric_crop_border=4
):
    max_test_batches = dataloaders.test_len / dataloaders.test_dataloader.batch_size
    assert(max_test_batches == int(max_test_batches)), f"Test batch size ({dataloaders.test_dataloader.batch_size}) does not fit into test dataset ({dataloaders.test_len})"

    # Get losses across test dataset
    test_metrics = []
    test_iter = iter(dataloaders.test_dataloader)
    
    for test_batch_no in range(int(max_test_batches)):
        with torch.no_grad():
            # todo: use batch_size=-1 instead, then check that it works
            x, _ = next(test_iter)
            x, y, z, x_recon_from_y = sample_inn(inn, x)
            loss_recon, loss_guide, loss_distr, total_loss = calculate_irn_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, scale)

        if calculate_metrics:
            x_recon_from_y_cropped = crop_tensor_border(x_recon_from_y, metric_crop_border)
            x_cropped = crop_tensor_border(x, metric_crop_border)

            psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1).cuda()
            psnr_RGB = psnr_metric(x_recon_from_y_cropped, x_cropped)
            psnr_Y = psnr_metric(rgb_to_y(x_recon_from_y_cropped/1)*1.0/255.0, rgb_to_y(x_cropped/1)*1.0/255.0)

            ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1).cuda()
            ssim = ssim_metric(x_recon_from_y_cropped, x_cropped)
        else:
            psnr_RGB = -1
            psnr_Y = -1
            ssim = -1

        test_metrics.append([float(total_loss), float(psnr_RGB), float(psnr_Y), float(ssim), float(loss_recon), float(loss_guide), float(loss_distr)])

    # Calcuate average for each of our metrics
    [total_loss, psnr_RGB, psnr_Y, ssim, loss_recon, loss_guide, loss_distr] = np.array(test_metrics).mean(axis=0)

    print(f"Calculated metrics for {len(test_metrics)} test batches.")
    print(f'Average loss_recon={loss_recon}, loss_guide={loss_guide}, loss_distr={loss_distr}')
    print(f'Average total loss in test set: {total_loss}')
    print(f'Average PSNR(RGB)/SSIM in test set: {round(float(psnr_RGB), 4)}/{round(float(ssim), 4)}')

    return total_loss, psnr_RGB, psnr_Y, ssim, loss_recon, loss_guide, loss_distr

def save_network(inn, optimizer, epoch, loss, name):
    save_filename = f"saved_models/model_{int(time.time())}_{name}.pth"

    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": inn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, save_filename)

def load_network(inn, optimizer, path):
    load_filename = path

    print(f"Loading {path}")

    checkpoint = torch.load(path)
    inn.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return inn, optimizer, checkpoint["epoch"], checkpoint["loss"]

def train_inn(inn, dataloaders: DataLoaders,
    max_batches=10000,
    max_epochs=-1,
    target_loss=-1,
    learning_rate=0.001,
    lambda_recon=1,
    lambda_guide=1,
    lambda_distr=1,
    epochs_between_tests=250,
    epochs_between_training_log=250,
    epochs_between_samples=5000,
    epochs_between_saves=5000,
    use_amsgrad=False,
    grad_clipping=10.0,
    scale=2,
    lr_batch_milestones=[100000, 200000, 300000, 400000],
    lr_gamma=0.5,
    load_checkpoint_path="",
    run_name=""
):
    optimizer = torch.optim.Adam(inn.parameters(), lr=learning_rate, weight_decay=0.00001, amsgrad=use_amsgrad)
    
    if(load_checkpoint_path):
        inn, optimizer, epoch, avg_training_loss = load_network(inn, optimizer, load_checkpoint_path)
        batch_no = math.floor(epoch * dataloaders.train_len / dataloaders.train_dataloader.batch_size)

        recent_training_losses = [avg_training_loss]
        all_avg_training_losses = [avg_training_loss]

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_batch_milestones, gamma=lr_gamma, last_epoch=batch_no)
    else:
        epoch = 0
        batch_no = 0

        recent_training_losses = []
        all_avg_training_losses = []
        avg_training_loss = target_loss+1

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_batch_milestones, gamma=lr_gamma)

    epoch_prev_test = epoch-1
    epoch_prev_training_log = epoch-1
    epoch_prev_sample = epoch-1
    epoch_prev_save = epoch-1

    all_test_losses = []
    all_test_psnr_y = []

    train_iter = iter(dataloaders.train_dataloader)

    start_ep = timer()
    time_forward = 0
    time_loss = 0
    time_dataloading = 0
    time_backward = 0
    time_testing_saving_sampling = 0

    while (max_batches==-1 or batch_no < max_batches) and (target_loss==-1 or target_loss<=avg_training_loss) and (max_epochs==-1 or epoch < max_epochs):
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
            wandb.log({"epoch_time_s": epoch_time, "epoch": epoch}, step=batch_no)

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
        x, y, z, x_recon_from_y = sample_inn(inn, x)
        stop = timer()
        time_forward += stop - start

        start = timer()
        loss_recon, loss_guide, loss_distr, total_loss = calculate_irn_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, scale)
        stop = timer()
        time_loss += stop - start

        recent_training_losses.append(float(total_loss))

        samples = batch_no * dataloaders.train_dataloader.batch_size
        epoch = float(samples / dataloaders.train_len)

        start = timer()
        if epoch - epoch_prev_training_log >= epochs_between_training_log:
            avg_training_loss = sum(recent_training_losses) / len(recent_training_losses)
            all_avg_training_losses.append(avg_training_loss)
            print(f"At {batch_no} batches (epoch {epoch}):")
            #print(f'loss_recon={loss_recon}, loss_guide={loss_guide}, loss_distr={loss_distr}')
            #print(f'Avg training loss, in last {epochs_between_training_log if epoch>0 else 0} epochs: {avg_training_loss}')
            recent_training_losses = []
            wandb.log({"train_loss": avg_training_loss, "epoch": epoch}, step=batch_no)
            epoch_prev_training_log = epoch

        if epoch - epoch_prev_sample >= epochs_between_samples:
            print(f"We are sampling at {batch_no+1} batches, {epoch} epochs...")
            with torch.no_grad():
                index_of_sample_image = 4
                x, y, z, x_recon_from_y = sample_inn(inn, dataloaders.test_dataloader.dataset[index_of_sample_image][0].unsqueeze(dim=0))
            see_irn_example(x, y, z, x_recon_from_y, scale, see=False, save=True, wandb_log=True, wandb_step=batch_no, name=all_test_losses[-1] if len(all_test_losses)>0 else "-", render_grid=False, metric_crop_border=scale)
            #for j in range(2):
            #    see_irn_example(x, y, z, x_recon_from_y, see=False, save=False, name=all_test_losses[-1])
            epoch_prev_sample = epoch
        
        if epoch - epoch_prev_save >= epochs_between_saves:
            print(f"We are saving at {batch_no+1} batches, {epoch} epochs...")
            save_network(inn, optimizer, epoch, avg_training_loss, f"{round(epoch, 2)}_{round(all_test_losses[-1], 2) if len(all_test_losses)>0 else '-'}_{run_name}")
            print(f"Finished saving.")
            epoch_prev_save = epoch

        if epoch - epoch_prev_test >= epochs_between_tests:
            print(f"At {batch_no} batches (epoch {epoch}):")
            print(f'loss_recon={loss_recon}, loss_guide={loss_guide}, loss_distr={loss_distr}')
            print(f'In test dataset, in last {epochs_between_tests if epoch>0 else 0} epochs:')

            # Crop the border of test images by the resize scale to avoid capturing outliers (this is done in the IRN paper)
            test_loss, test_psnr_rgb, test_psnr_y, test_ssim, test_lossr, test_lossg, test_lossd = test_inn(inn, dataloaders, scale, lambda_recon, lambda_guide, lambda_distr, calculate_metrics=True, metric_crop_border=scale)
            all_test_losses.append(test_loss)
            all_test_psnr_y.append(test_psnr_y)
            print("")
    
            wandb.log({"test_loss": test_loss, "min_test_loss": min(all_test_losses), "max_test_psnr_y": max(all_test_psnr_y), "test_psnr": test_psnr_rgb, "test_psnr_y": test_psnr_y, "test_ssim": test_ssim, "test_loss_recon": test_lossr, "test_loss_guide": test_lossg, "test_loss_distr": test_lossd, "epoch": epoch}, step=batch_no)
            epoch_prev_test = epoch
        stop = timer()
        time_testing_saving_sampling += stop-start

        start = timer()
        total_loss.backward()
        stop = timer()
        time_backward += stop - start

        torch.nn.utils.clip_grad_norm_(inn.parameters(), max_norm=grad_clipping)
        optimizer.step()
        scheduler.step()
        batch_no+=1
    return all_avg_training_losses, all_test_losses



####    EXECUTION CODE    ####
if __name__ == '__main__':
    resume_latest_run = False
    resume_run_id = "ki9ty9x7"

    wandb.login()
    run = wandb.init(
        project="invertible-rescaling-network",
        dir=".",
        resume=resume_latest_run if resume_latest_run else None,
        id=resume_run_id if resume_run_id else None,
        config={
            "batch_size": 16,
            "lambda_recon": 1,
            "lambda_guide": 16,
            "lambda_distr": 1,
            "initial_learning_rate": 0.0004,
            "img_size": 144,
            "scale": 4, # ds_count = log2(scale)
            "inv_per_ds": 8,
            "inv_first_level_extra": 0,
            "inv_final_level_extra": 0,
            "seed": 10,
            "grad_clipping": 0.1,
            "full_size_test_imgs": True,
            "lr_batch_milestones": [100000, 200000, 300000, 400000],
            "lr_gamma": 0.5
    })
    config = wandb.config

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load the data
    dataloaders = Div2KDataLoaders(config.batch_size, config.img_size, full_size_test_imgs=config.full_size_test_imgs, test_img_size_divisor=config.scale)

    # Load the network
    ds_count = int(np.log2(config.scale))
    assert(ds_count==np.log2(config.scale)), "Scale must be a power of 2."
    inn = IRN(3, config.img_size, config.img_size, ds_count=ds_count, inv_per_ds=config.inv_per_ds, inv_final_level_extra=config.inv_final_level_extra, inv_first_level_extra=config.inv_first_level_extra)

    if resume_latest_run:
        load_checkpoint_path = latest_file_in_folder("./saved_models", ".pth")
    elif resume_run_id:
        load_checkpoint_path = latest_file_in_folder("./saved_models", f"{run.id}.pth")
    else:
        load_checkpoint_path = ""

    # Train the network
    # TODO: replace all these parameters with a "config" param...
    all_training_losses, all_test_losses = train_inn(inn, dataloaders,
                                                    max_batches=-1, max_epochs=-1, target_loss=-1,
                                                    epochs_between_tests=6, epochs_between_training_log=0.05, epochs_between_samples=1, epochs_between_saves=3, 
                                                    learning_rate=config.initial_learning_rate, grad_clipping=config.grad_clipping, scale=config.scale,
                                                    lambda_recon=config.lambda_recon, lambda_guide=config.lambda_guide, lambda_distr=config.lambda_distr,
                                                    lr_batch_milestones=config.lr_batch_milestones, lr_gamma=config.lr_gamma,
                                                    load_checkpoint_path=load_checkpoint_path, run_name=run.id)
    #plt.savefig(f'output/test_loss_{int(time.time())}_{int(all_test_losses[-1])}', dpi=100)

    run.finish()