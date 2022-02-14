# Train a given model
import time
from networks.freia_invertible_rescaling import IRN, sample_inn
from data import mnist8_iterator, process_xbit_img, see_multiple_imgs, process_div2k_img
from data2 import Div2KDataLoaders, DataLoaders
from bicubic_pytorch.core import imresize
import torch
import torchmetrics
import numpy as np
import wandb
import random
from loss import calculate_irn_loss
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
#from IQA_pytorch import SSIM, DISTS

def see_irn_example(x, y, z, x_recon_from_y, see=True, save=True, name=0, index=-1):
    print("-> About to visualize irn example")
    i = random.randint(0, len(x)-1) if index==-1 else index

    x_downscaled_bc = imresize(x[i], scale=0.5)
    x_upscaled_bc = imresize(x_downscaled_bc, scale=2)


    #div2k images are loaded with values in range [0, 1] but mnist8 is in range [0, 16].
    size = x.shape[-1]
    imgs = [process_div2k_img(*im_and_size) for im_and_size in [
        (x[i], (3, size, size)),
        (x_downscaled_bc, (3, size//2, size//2)),
        (y[i], (3, size//2, size//2)),
        (x[i], (3, size, size)),
        (x_upscaled_bc, (3, size, size)),
        (x_recon_from_y[i], (3, size, size)),
    ]]

    # Test the network
    # mnist8 consists of 16-tone images
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=16)
    if torch.cuda.is_available(): psnr_metric = psnr_metric.cuda()

    psnr_irn_down = psnr_metric(y[i], x_downscaled_bc) # IRN-down  vs  GT (Bicubic-down)
    psnr_bi_up = psnr_metric(x_upscaled_bc, x[i]) # Bicubic-down / Bicubic-up  vs  GT (HR)
    psnr_irn_up = psnr_metric(x_recon_from_y[i], x[i]) # IRN-down / IRN-up  vs  GT (HR)

    fig = see_multiple_imgs(imgs, 2, 3,
        row_titles=[
            "Downscaling task (PSNR)",
            "Downscaling & upscaling task (PSNR)"],
        plot_titles=[
            "HR (-)", "GT [Bi-down] (∞)", "IRN-down (%.2f)" % round(psnr_irn_down.item(), 2),
            "GT [HR] (∞)", "Bi-down & Bi-up (%.2f)" % round(psnr_bi_up.item(), 2), "IRN-down & IRN-up (%.2f)" % round(psnr_irn_up.item(), 2)
        ],
        see=see, save=save,
        filename=f'/rds/user/mgm52/hpc-work/invertible-image-rescaling/output/out_{int(time.time())}_{i}'
    )

    #table = wandb.Table(columns=["GT [HR]", "GT [Bi-down]", "IRN-down", "Bi-down & Bi-up", "IRN-down & IRN-up", "IRN-down PSNR", "Bi-down & Bi-up PSNR", "IRN-down & IRN-up PSNR"])
    #table.add_data(wandb.Image(imgs[0]), wandb.Image(imgs[1]), wandb.Image(imgs[2]), wandb.Image(imgs[4]), wandb.Image(imgs[5]), round(psnr_irn_down.item(), 2), round(psnr_bi_up.item(), 2), round(psnr_irn_up.item(), 2))
    #wandb.log({"samples_table":table}, commit=False)

    if False:
        #div2k images are loaded with values in range [0, 1] but mnist8 is in range [0, 16].
        imgs = [process_div2k_img(*im_and_size) for im_and_size in [
            (x_recon_from_y[0][0], (1, size, size)),
            (x_recon_from_y[0][1], (1, size, size)),
            (x_recon_from_y[0][2], (1, size, size))
        ]]

        see_multiple_imgs(imgs, 1, 3,
            row_titles=[
                "R G B"
            ],
            plot_titles=[
                "R", "G", "B"
            ],
            see=see, save=save,
            filename=f'output/outrgb_{int(time.time())}_{i}'
        )
    
    print("-> Finished visualizing.")
    return fig, imgs

def test_inn(inn,
    dataloaders,
    lambda_recon=1,
    lambda_guide=1,
    lambda_distr=1,
    calculate_metrics=False
):
    #mnist8_iter = mnist8_iterator()

    with torch.no_grad():
        # todo: use batch_size=-1 instead, then check that it works
        x, y, z, x_recon_from_y = sample_inn(inn, dataloaders, use_test_set=True)
        loss_recon, loss_guide, loss_distr, total_loss = calculate_irn_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y)
    
    print(f'loss_recon={loss_recon}, loss_guide={loss_guide}, loss_distr={loss_distr}')
    print(f'Average loss in test set: {total_loss}')

    if calculate_metrics:
        psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=16).cuda()
        psnr = psnr_metric(x_recon_from_y, x)

        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=16).cuda()
        ssim = ssim_metric(x_recon_from_y, x)
        print(f'Average PSNR/SSIM in test set: {round(float(psnr), 4)}/{round(float(ssim), 4)}')
    else:
        psnr = -1
        ssim = -1

    return float(total_loss), float(psnr), float(ssim), float(loss_recon), float(loss_guide), float(loss_distr)

def save_network(inn, name):
    save_filename = f"saved_models/model_{int(time.time())}_{name}.pth"
    state_dict = inn.state_dict()
    #for key, param in state_dict.items():
    #    state_dict[key] = param.cpu()
    torch.save(state_dict, save_filename)

def train_inn(inn, dataloaders: DataLoaders,
    max_batches=10000,
    max_epochs=-1,
    target_loss=-1,
    learning_rate=0.001,
    lambda_recon=1,
    lambda_guide=1,
    lambda_distr=1,
    epochs_between_tests=250,
    epochs_between_samples=5000,
    epochs_between_saves=5000,
    use_amsgrad=False,
    use_grad_clipping=False
):
    optimizer = torch.optim.Adam(inn.parameters(), lr=learning_rate, amsgrad=use_amsgrad)

    batch_no = 0
    epoch = 0
    epoch_prev_test = -1
    epoch_prev_sample = -1
    epoch_prev_save = -1

    # TODO: improve logic
    recent_training_losses = []
    all_training_losses = []
    all_test_losses = []
    avg_training_loss = target_loss+1

    while (max_batches==-1 or batch_no < max_batches) and (target_loss==-1 or target_loss<=avg_training_loss) and (max_epochs==-1 or epoch < max_epochs):
        optimizer.zero_grad()
        
        x, y, z, x_recon_from_y = sample_inn(inn, dataloaders, use_test_set=False)
        loss_recon, loss_guide, loss_distr, total_loss = calculate_irn_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y)
        
        recent_training_losses.append(float(total_loss))
        all_training_losses.append(recent_training_losses[-1])

        samples = batch_no * dataloaders.train_dataloader.batch_size
        epoch = float(samples / dataloaders.train_len)

        if epoch - epoch_prev_test >= epochs_between_tests:
            avg_training_loss = sum(recent_training_losses) / len(recent_training_losses)
            #print(y.shape)
            #print(z.shape)
            #print(x_recon_from_y.shape)
            #print(x_downscaled.shape)
            print(f"At {batch_no} batches (epoch {epoch}):")
            print(f'loss_recon={loss_recon}, loss_guide={loss_guide}, loss_distr={loss_distr}')
            print(f'Avg training loss, in last {epochs_between_tests if epoch>0 else 0} epochs: {avg_training_loss}')
            print(f'In test dataset:')
            test_loss, test_psnr, test_ssim, test_lossr, test_lossg, test_lossd = test_inn(inn, dataloaders, lambda_recon, lambda_guide, lambda_distr, calculate_metrics=True)
            all_test_losses.append(test_loss)
            print("")
            recent_training_losses = []
            epoch_prev_test = epoch
        
            if epoch - epoch_prev_sample >= epochs_between_samples:
                print(f"We are sampling at {batch_no+1} batches, {epoch} epochs...")
                with torch.no_grad():
                    x, y, z, x_recon_from_y = sample_inn(inn, dataloaders, use_test_set=True)

                fig, imgs = see_irn_example(x, y, z, x_recon_from_y, see=False, save=False, name=all_test_losses[-1], index=4)
                wandb.log({f"grid_example": wandb.Image(fig), f"single_example": wandb.Image(imgs[-1])}, commit=False, step=samples)
                
                #for j in range(2):
                #    see_irn_example(x, y, z, x_recon_from_y, see=False, save=False, name=all_test_losses[-1])
                epoch_prev_sample = epoch
            
            if epoch - epoch_prev_save >= epochs_between_saves:
                print(f"We are saving at {batch_no+1} batches, {epoch} epochs...")
                save_network(inn, f"{round(epoch, 2)}_{round(all_test_losses[-1], 2)}")

            wandb.log({"train_loss": avg_training_loss, "test_loss": test_loss, "test_psnr": test_psnr, "test_ssim": test_ssim, "test_loss_recon": test_lossr, "test_loss_guide": test_lossg, "test_loss_distr": test_lossd, "epoch": epoch}, step=samples)

        total_loss.backward()
        if use_grad_clipping: torch.nn.utils.clip_grad_norm_(inn.parameters(), max_norm=10.0)
        optimizer.step()
        batch_no+=1
    return all_training_losses, all_test_losses

use_mnist8 = False

wandb.login()


with wandb.init(
    project="invertible-rescaling-network",
    config={
        #"epochs": 10,
        "batch_size": 5,
        "lambda_recon": 100,
        "lambda_guide": 20,
        "lambda_distr": 0.01,
        "initial_learning_rate": 0.002,
        "img_size": 256,
        "ds_count": 1, # rescaling factor = 2^ds_count
        "inv_per_ds": 3,
        "seed": 0,
        "use_grad_clipping": True,
        #"lr": 1e-3,
        #"dropout": random.uniform(0.01, 0.80)
}):
    config = wandb.config

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load the data
    dataloaders = None if use_mnist8 else Div2KDataLoaders(config.batch_size, config.img_size)

    # Train the network
    inn = IRN(3, config.img_size, config.img_size, ds_count=config.ds_count, inv_per_ds=config.inv_per_ds)
    all_training_losses, all_test_losses = train_inn(inn, dataloaders,
                                                    max_batches=-1, max_epochs=60, target_loss=-1,
                                                    epochs_between_tests=1.5, epochs_between_samples=0.1, epochs_between_saves=0.1, 
                                                    learning_rate=config.initial_learning_rate, use_grad_clipping=config.use_grad_clipping,
                                                    lambda_recon=config.lambda_recon, lambda_guide=config.lambda_guide, lambda_distr=config.lambda_distr)
    #plt.savefig(f'output/test_loss_{int(time.time())}_{int(all_test_losses[-1])}', dpi=100)

