from isort import file
import torch
import re
import os

def standardise_tensor(x: torch.Tensor, new_mean=0, new_std=1):
    #print(f"Taking a tensor with mean {x.mean()} std {x.std()} and giving it mean {new_mean} std {new_std}")

    mean = x.mean()
    std = x.std()
    new_x = new_mean + new_std * (x - mean) / std

    #print(f"Returning with mean {new_x.mean()} std {new_x.std()}\n")
    return new_x, mean, std

def extract_directory(filepath):
  return re.sub(r"/[^/]*$", "", filepath)

def create_parent_dir(filepath):
    if "/" in filepath:
        dirpath = extract_directory(filepath)
        if dirpath and not os.path.isdir(dirpath): os.mkdir(dirpath)
