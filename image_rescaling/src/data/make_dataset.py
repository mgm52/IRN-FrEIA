import time
from tracemalloc import start
from torchvision import transforms
from models.layers.invertible_rescaling_network import IRN, sample_irn, multi_sample_irn
from visualisation.visualise import process_xbit_img, see_multiple_imgs, process_div2k_img
import models.model_loader
from data.dataloaders import Div2KDataLoaders, DataLoaders, get_test_dataloader
from utils.bicubic_pytorch.core import imresize
import torch
from torchvision.utils import save_image
import torchmetrics
import numpy as np
import wandb
from models.model_loader import save_network, load_network
import math
from timeit import default_timer as timer
from models.layers.invertible_rescaling_network import quantize_ste
import glob
from utils.utils import create_parent_dir
import os
from utils.utils import rgb_to_y
import random
from models.train.loss_irn import calculate_irn_loss
from datetime import date
import matplotlib.pyplot as plt
from models.layers.straight_through_estimator import quantize_ste, quantize_to_int_ste
from PIL import Image
import models.layers.straight_through_estimator
import math
from io import BytesIO
import random
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage


if __name__ == '__main__':

    scale = 4.0

    # Load Div2k
    dataloaders = Div2KDataLoaders(16, 144, full_size_test_imgs=True, test_img_size_divisor=scale)
    test_iter = iter(dataloaders.test_dataloader)

    for i in range(dataloaders.test_len):
        if (i-1) % int(max(2, dataloaders.test_len / 5)) == 0:
            print(f"At image {i}/{int(dataloaders.test_len)}")

        path = f"data/DIV2K/DIV2K_valid_LR_x4/{str(i).zfill(3)}_LR_{int(time.time())}.jpg"
        create_parent_dir(path)

        x_raw, _ = next(test_iter)
        x_downscaled = quantize_ste(imresize(x_raw, scale=1.0/scale))
        save_image(quantize_ste(x_downscaled), path)