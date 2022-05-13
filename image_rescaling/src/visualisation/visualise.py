# Get dataset
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
from utils.utils import create_parent_dir
import os
import textwrap

def process_div2k_img(data, shape, verbose=False):
    im = np.array(data.detach().cpu().numpy())
    assert np.prod(shape) == np.size(im), f'process_xbit_img given a shape ({shape}) that doesnt match element count {np.size(im)} - expected {np.prod(shape)} elements instead'

    im.resize(*shape)
    # Convert from (c, w, h) format to (w, h, c)
    im = np.moveaxis(im, 0, -1)

    #im = im * scaling

    if verbose:
        print(f"For shape {shape}:")
        print(f'Max : {np.amax(im)}')
        print(f'Med : {np.median(im)}')
        print(f'Min : {np.amin(im)}')

    im = np.clip(im, 0, 1)

    return im

def process_xbit_img(data, shape, clip_values=True, floor_values=True, bits=4, scaling=1):
    im = np.array(data.detach().cpu().numpy())
    assert np.prod(shape) == np.size(im), f'process_xbit_img given a shape ({shape}) that doesnt match element count {np.size(im)} - expected {np.prod(shape)} elements instead'

    im.resize(*shape)

    im = im * scaling

    print(f"For shape {shape}:")
    print(f'Max : {np.amax(im)}')
    print(f'Med : {np.median(im)}')
    print(f'Min : {np.amin(im)}')

    max_color = pow(2, bits)
    if clip_values: im = np.clip(im, 0, max_color)
    if floor_values: im = np.floor(im)

    return im

def see_multiple_imgs(imgs, rows, cols, row_titles=[], plot_titles=[], see=True, save=False, filename="out", smallSize=False, pdf=False, bottomtitle=False, wrap_text=False):
    if len(imgs)==0: return None
    assert rows*cols >= len(imgs), f'Cannot print {len(imgs)} images on a {rows}x{cols} grid'
    
    f, axes = plt.subplots(figsize=(3*cols, 3*rows) , nrows=rows, ncols=1, sharey=True) 
    f.set_dpi(200)

    if rows > 1:
        for row_num, row_ax in enumerate(axes, start=1):
            # Add title to row
            if row_num-1<len(row_titles): row_ax.set_title(row_titles[row_num-1] + "\n", fontsize=14, loc="left")
            row_ax.axis('off')
    elif rows==1 and len(row_titles)==1:
        plt.title(row_titles[0], y=-0.01 if bottomtitle else 0)

    maximgsize = max(imgs[0].shape)

    for i in range(1, rows*cols + 1):
        # Add subplot to index i-1 within a rows*cols grid
        ax = f.add_subplot(rows,cols,i)
        if i-1<len(plot_titles): ax.set_title(textwrap.fill(plot_titles[i-1], 30) if wrap_text else plot_titles[i-1], fontsize=12, loc="left") #fontsize=int(9.0 + maximgsize * 2.0/100.0)
        if i-1<len(imgs) and not (imgs[i-1] is None):
            if len(imgs[i-1].shape) == 4 and imgs[i-1].shape[0] == 1:
                imgs[i-1] = imgs[i-1][0]
            if imgs[i-1].shape[0] == 3:
                imgs[i-1] = imgs[i-1].permute(1, 2, 0)
            ax.imshow(torch.clamp(imgs[i-1].cpu(), 0, 1))
        ax.axis('off')

    plt.tight_layout()
    if smallSize:
        f.set_size_inches(cols * maximgsize * 0.5/256.0, rows * maximgsize * 0.6/256.0)
    else:
        f.set_size_inches(cols * maximgsize * 7.5/256.0, rows * maximgsize * 9/256.0)

    if rows == 1: axes.axis("off")
    plt.axis("off")

    if(save):
        create_parent_dir(filename)
        plt.savefig(filename + ('.pdf' if pdf else '.png'), dpi=200, bbox_inches='tight')
    if(see): plt.show()
    if(not see): plt.close()

    return f