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

# NOTE: the paper crops full-sized test images by a border equal to the scaling factor being trained on (e.g. 2x).
# I suppose this is because it believes the border pixels to be less accurate?
def Div2KDataLoaders(batch_size, img_size=64, shuffle_training_data=True, full_size_test_imgs=False):
    #training_data = Div2K(root="./data", scale=4, split="train", track="bicubic", transform=ToTensor(), download=False)
    #testing_data = Div2K(root="./data", scale=8, split="test", track="bicubic", transform=ToTensor(), download=False)

    transform = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder("/rds/user/mgm52/hpc-work/invertible-image-rescaling/data/DIV2K/DIV2K_train_HR", transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_training_data)
    print(f"Loaded {len(dataset)} training images")

    if full_size_test_imgs:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.ImageFolder("/rds/user/mgm52/hpc-work/invertible-image-rescaling/data/DIV2K/DIV2K_valid_HR", transform=transform)
        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        transform = transforms.Compose([transforms.CenterCrop(img_size), transforms.ToTensor()])
        dataset = datasets.ImageFolder("/rds/user/mgm52/hpc-work/invertible-image-rescaling/data/DIV2K/DIV2K_valid_HR", transform=transform)
        test_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    print(f"Loaded {len(dataset)} test images")

    return DataLoaders(train_dataloader, test_dataloader, sample_shape=(3, img_size, img_size))    

