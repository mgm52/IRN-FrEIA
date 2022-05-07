import FrEIA.modules as fm
from argparse import ArgumentParser
import time
from models.layers.invertible_rescaling_network import quantize_ste
from tracemalloc import start
from models.layers.invertible_rescaling_network import IRN, sample_irn
from visualisation.visualise import mnist8_iterator, process_xbit_img, see_multiple_imgs, process_div2k_img
from data.dataloaders import Div2KDataLoaders, DataLoaders
from utils.bicubic_pytorch.core import imresize
import models.model_loader
import torch
from torchvision.utils import save_image
import FrEIA.framework as ff
import torchmetrics
import numpy as np
import wandb
from test import test_inn
from models.model_loader import save_network, load_network
from models.layers.dense_block import db_subnet
import math
from timeit import default_timer as timer
from utils.utils import create_parent_dir
import glob
import os
import models.model_loader
import random
from models.train.loss_irn import calculate_irn_loss
from datetime import date
import matplotlib.pyplot as plt

device = "cuda"

def TRN(*dims):
    # SequenceINN takes dims in (c, w, h) format
    inn = ff.SequenceINN(*dims)

    for d in range(8):
        inn.append(fm.AllInOneBlock, subnet_constructor=db_subnet, permute_soft=False)
    return inn.cuda() if device=="cuda" else inn

def get_saved_inn(path):
    config=models.model_loader.load_config("irn_4x_og_0.5yusage.yaml")

    inn = IRN(3, config["img_size"], config["img_size"], cfg=config)

    inn, optimizer, epoch, min_training_loss, min_test_loss, max_test_psnr_y = load_network(inn, path, None)

    return inn


# translator = new model

# dataloader = dataloaders.train
# until max_epochs:
    # x = next(dataloader)
    # x_translate = translator(x)
    # x, y, z, xrecon = irn(x, reverse=true)
    # loss = (x_translate - y).sum()

if __name__ == "__main__":
    size = (16,3,16,16)
    
    configyaml = models.model_loader.load_config("irn_4x_og_0.5yusage.yaml")
    # irn = load model
    inn = get_saved_inn("./models/model_1650181747_10045.0_140161.3_1bmt7qp4.pth")
    inn.eval()

    trn = TRN(3, 144/4, 144/4)

    batchsize=16
    lr=0.00025

    dataloaders = Div2KDataLoaders(batchsize, configyaml["img_size"], full_size_test_imgs=configyaml["full_size_test_imgs"], test_img_size_divisor=configyaml["scale"])
    train_iter = iter(dataloaders.train_dataloader)

    found_start = False

    recon_loss_mode = True

    optimizer = torch.optim.Adam(trn.parameters(), lr=lr, weight_decay=0.00001, amsgrad=False)
    avg_losses = []
    for i in range(5000):
        optimizer.zero_grad()
        try:
            x, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloaders.train_dataloader)
            x, _ = next(train_iter)        
        
        #save_image(quantize_ste(wolf_irn_upscaled), f"output/training/train_sr_assistant/{str(i).zfill(3)}_irn_trn_sr_{int(time.time())}.png")


        x_downscaled = quantize_ste(imresize(x, scale=1.0/4.0)).cuda()
        x_ds_translate, _ = trn(x_downscaled)

        if recon_loss_mode:
            z_sample = torch.normal(torch.zeros(batchsize, 3*4*4 - 3, x_ds_translate.shape[2], x_ds_translate.shape[3]), torch.ones(1, 3*4*4 - 3, x_ds_translate.shape[2], x_ds_translate.shape[3]))

            # NOTE: I am handicapping myself by using an IRN that expects integer values! No need to constrain to int!
            y_and_z_sample = torch.cat((x_ds_translate.cuda(), z_sample.cuda()), dim=1)
            # y_and_z_sample.shape == (n, c2, w2, h2)
            irn_recon, _ = inn([y_and_z_sample], rev=True)

            loss = torch.sum(torch.sqrt((x.cuda() - irn_recon.cuda())**2 + 0.000001)) #F.l1_loss(x, x_recon_quant, reduction="sum")# + torch.abs(torch.std(x, axis=1) - torch.std(x_recon_from_y, axis=1)).mean()
            loss = loss / (x.shape[0])

        else:
            with torch.no_grad():
                x, y, z, x_recon_from_y, mean_y, std_y, ymod = sample_irn(inn, x, {"zerosample": configyaml["zerosample"], "sr_mode": False})
            
            loss = torch.sum(((y - x_ds_translate)**2)) #F.l1_loss(x, x_recon_quant, reduction="sum")# + torch.abs(torch.std(x, axis=1) - torch.std(x_recon_from_y, axis=1)).mean()
            loss = loss / (y.shape[0])
            
        loss.backward()

        optimizer.step()

        avg_losses.append(float(loss))
        if (i+1) % 100 == 0:
            lr /= 2
            print(f"\nlr drop to {lr}\n")
        if (i+1) % (3 if not found_start else 25) == 0:
            avgloss = sum(avg_losses) / len(avg_losses)
            print(f"loss = {avgloss}")
            avg_losses = []
            #create_parent_dir(f"output/training/train_sr_assistant/x")
            #save_image(quantize_ste(x_ds_translate), f"output/training/train_sr_assistant/{str(i).zfill(3)}_trn_{int(time.time())}.png")
            #save_image(quantize_ste(y), f"output/training/train_sr_assistant/{str(i).zfill(3)}_irn_{int(time.time())}.png")

            with torch.no_grad():
                index_of_sample_image = 2

                wolf_hr = dataloaders.test_dataloader.dataset[index_of_sample_image][0].unsqueeze(dim=0).cuda()
                wolf_lr = quantize_ste(imresize(wolf_hr, scale=1.0/4.0)).cuda()

                wolf_trn_lr, _ = trn(wolf_lr)
                z_sample = torch.normal(torch.zeros(1, 3*4*4 - 3, wolf_trn_lr.shape[2], wolf_trn_lr.shape[3]), torch.ones(1, 3*4*4 - 3, wolf_trn_lr.shape[2], wolf_trn_lr.shape[3]))
                # z_sample.shape == (n, c2-3, w2, h2)

                ### Here, we assume that the y that was extracted in the forward pass has mean 0, std 1.
                ### This may not be true, but we can force it to be true through training.

                # NOTE: I am handicapping myself by using an IRN that expects integer values! No need to constrain to int!
                y_and_z_sample = torch.cat((wolf_trn_lr.cuda(), z_sample.cuda()), dim=1)
                # y_and_z_sample.shape == (n, c2, w2, h2)
                wolf_irn_upscaled, _ = inn([y_and_z_sample], rev=True)

                save_image(quantize_ste(wolf_irn_upscaled), f"output/training/train_sr_assistant/{str(i).zfill(3)}_irn_trn_sr_{int(time.time())}.png")

            if avgloss > 3000 and not found_start:
                trn = TRN(3, 144/4, 144/4)
                optimizer = torch.optim.Adam(trn.parameters(), lr=lr, weight_decay=0.00001, amsgrad=False)
            else: found_start = True
            #print("\r loss=%0.3f"%(loss),end="\r")
