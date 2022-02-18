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

#training_data = Div2K(root="./data", scale=4, split="train", track="bicubic", transform=ToTensor(), download=False)
#testing_data = Div2K(root="./data", scale=2, split="test", track="bicubic", transform=ToTensor(), download=True)

class DataLoaders:
    def __init__(self, train_dataloader, test_dataloader, sample_shape):
        self.train_dataloader: DataLoader = train_dataloader
        self.test_dataloader: DataLoader = test_dataloader
        self.train_len = len(self.train_dataloader.dataset)
        self.test_len = len(self.test_dataloader.dataset)
        self.sample_shape = sample_shape

def Div2KDataLoaders(batch_size, img_size=64, shuffle_training_data=True):
    transform = transforms.Compose([transforms.CenterCrop(img_size), transforms.ToTensor()])
    dataset = datasets.ImageFolder("/rds/user/mgm52/hpc-work/invertible-image-rescaling/data/DIV2K/DIV2K_train_LR_x2", transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_training_data)
    print(f"Loaded {len(dataset)} training images")

    transform = transforms.Compose([transforms.CenterCrop(img_size), transforms.ToTensor()])
    dataset = datasets.ImageFolder("/rds/user/mgm52/hpc-work/invertible-image-rescaling/data/DIV2K/DIV2K_test_LR_x2", transform=transform)
    test_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    print(f"Loaded {len(dataset)} test images")

    return DataLoaders(train_dataloader, test_dataloader, sample_shape=(3, img_size, img_size))    

