import os
from typing import Any, Callable, Optional
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchsr.datasets import Div2K
from torchsr.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from utils.bicubic_pytorch.core import imresize
import numpy as np
from PIL import Image
import math
from io import BytesIO
import random

class DataLoaders:
    def __init__(self, train_dataloader, test_dataloader, sample_shape):
        self.train_dataloader: DataLoader = train_dataloader
        self.test_dataloader: DataLoader = test_dataloader
        self.train_len = len(self.train_dataloader.dataset)
        self.test_len = len(self.test_dataloader.dataset)
        self.sample_shape = sample_shape

# The default datasets.ImageFolder class doesn't allow you to load a single class of images in a folder with multiple classes
# This fixes that; classname should be the name of the folder within root that you want to load, or None to detect it automatically.
class SingleClassImageFolder(datasets.ImageFolder):
    def __init__(self, root, classname = None, transform = None):
        self.classname = classname
        super().__init__(root, transform)

    def find_classes(self, directory: str):
        print(f"Loading a single class file in {directory}...")
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        if self.classname is None:
            if len(classes) > 1:
                raise FileNotFoundError(f"Multiple class folders found in {directory} but no classname was specified.")
            return classes, {classes[0]: 0}

        if not (self.classname is None):
            if not (self.classname in classes):
                raise FileNotFoundError(f"Couldn't find class folder named '{self.classname}' in {directory}.")
            return [self.classname], {self.classname: 0}

# Crops an image in the centre, in such a way that its height and width is a multiple of divisor
# Expects image to be in shape (..., h, w)
class DivisibleCrop(torch.nn.Module):
    def __init__(self, divisor):
        super().__init__()
        assert(int(divisor)==divisor and divisor>0), "Invalid divisor provided for image size. Please provide a positive int."
        self.divisor = divisor

    def forward(self, img):
        if torch.is_tensor(img):
            h, w = img.shape[-2], img.shape[-1]
        else:
            w, h = img.size

        new_hw = (self.divisor*math.floor(h / self.divisor), self.divisor*math.floor(w / self.divisor))
        return TF.center_crop(img, new_hw)

    def __repr__(self):
        return self.__class__.__name__ + '(divisor={0})'.format(self.divisor)

# Crops an image in the centre, by a given border. Output image is of size (h-border*2, w-border*2).
class BorderCrop(torch.nn.Module):
    def __init__(self, border):
        super().__init__()
        assert(int(border)==border and border>=0), "Invalid border provided for image size. Please provide a non-negative int."
        self.border = border

    def forward(self, img):
        if torch.is_tensor(img):
            h, w = img.shape[-2], img.shape[-1]
        else:
            w, h = img.size

        new_hw = (h - self.border*2, w - self.border*2)
        return TF.center_crop(img, new_hw)

    def __repr__(self):
        return self.__class__.__name__ + '(border={0})'.format(self.border)

def get_test_dataloader(path, test_img_size_divisor=1, img_crop_size=None):
    classname = path[path.rfind("/")+1:]
    direc = path[:path.rfind("/")]

    if img_crop_size is None:
        transform = transforms.Compose([DivisibleCrop(test_img_size_divisor), transforms.ToTensor()])
        dataset = SingleClassImageFolder(direc, classname=classname, transform=transform)
        batch_size = 1
    else:
        transform = transforms.Compose([transforms.CenterCrop(img_crop_size), transforms.ToTensor()])
        dataset = SingleClassImageFolder(direc, classname=classname, transform=transform)
        batch_size = len(dataset)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)


# NOTE: the paper crops full-sized test images by a border equal to the scaling factor being trained on (e.g. 2x).
# I suppose this is because it believes the border pixels to be less accurate?
def Div2KDataLoaders(batch_size, img_size=64, shuffle_training_data=True, full_size_test_imgs=False, test_img_size_divisor=2):
    #training_data = Div2K(root="./data", scale=4, split="train", track="bicubic", transform=ToTensor(), download=False)
    #testing_data = Div2K(root="./data", scale=8, split="test", track="bicubic", transform=ToTensor(), download=False)
    print(f"Loading div2k data with imgsize={img_size}, shuffle={shuffle_training_data}, fullsizetest={full_size_test_imgs}, testsizedivisor={test_img_size_divisor}")
    
    transform = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.ToTensor()
    ])
    dataset = SingleClassImageFolder("./data/DIV2K/DIV2K_train_HR", transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_training_data, num_workers=6)
    print(f"Loaded {len(dataset)} training images")

    test_imgs_path = "./data/DIV2K/DIV2K_valid_HR"
    if full_size_test_imgs:
        test_dataloader = get_test_dataloader(test_imgs_path, test_img_size_divisor)
    else:
        test_dataloader = get_test_dataloader(test_imgs_path, img_crop_size=img_size)
    print(f"Loaded {len(dataset)} test images")

    return DataLoaders(train_dataloader, test_dataloader, sample_shape=(3, img_size, img_size))    

