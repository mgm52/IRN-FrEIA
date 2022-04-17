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


# input should be size (n, 3, h, w) - in range [0,1]
# output has size (n, 1, h, w) - treat the output as if it has range [0,1]
# (in reality, output has range [0.062745, 0.92149])
def rgb_to_y(rgb_imgs, round255=False, plus16=True, bgr=False):
    assert len(rgb_imgs.shape) == 4 and rgb_imgs.shape[1] == 3, "Input must have shape [n, 3, h, w]"
    output = (rgb_imgs[:,2 if bgr else 0,:,:] * 65.481 + rgb_imgs[:,1,:,:] * 128.553 + rgb_imgs[:,0 if bgr else 2,:,:] * 24.966) + (16 if plus16 else 1)
    if round255: output = output.round()
    return output.unsqueeze(dim=1) / 255.0
