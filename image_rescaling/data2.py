import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchsr.datasets import Div2K
from torchsr.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from bicubic_pytorch.core import imresize
import numpy as np

def get_div2k_dataloader(batch_size):
    training_data = Div2K(root="./data", scale=4, split="train", track="bicubic", transform=ToTensor(), download=False)

    print(len(training_data))

    transform = transforms.Compose([transforms.CenterCrop(80),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder("./data/DIV2K/DIV2K_train_LR_x8", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

#images_hr, _ = next(iter(dataloader))
#images_lr = imresize(images_hr, scale=0.5)
#print(images_hr.shape)
#print(images_lr.shape)

