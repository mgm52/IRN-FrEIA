from re import S
import numpy as np
import matplotlib.pyplot as plt
import random
import data.data as D
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits

def see_moons(data, labels=1):
    plt.scatter(data[:, 0][labels==1], data[:, 1][labels==1], c="b")
    plt.scatter(data[:, 0][labels==0], data[:, 1][labels==0], c="r")
    plt.show()

def see_example_moons():
    data, labels = D.sample_mmoons()
    print(f'Moon points: {data}')
    print(f'Moon labels: {labels}')
    see_moons(data, labels)

def see_mnist8(data, label=-1, clip_values=True, floor_values=True, save=False, filename="out"):
    if label > -1: print(f'Digit label: {label}')

    im = np.array(data.detach().cpu().numpy())
    size = int(np.sqrt(im.size))
    assert size==np.sqrt(im.size), f'see_mnist8 only accepts square images - given {im.size}, wanted {size*size}'

    im = process_4bit_img(data, (size, size), clip_values=clip_values, floor_values=floor_values)
    see_img(im, save=save, filename=filename)

def process_4bit_img(data, shape, clip_values=True, floor_values=True):
    im = np.array(data.detach().cpu().numpy())
    assert np.prod(shape) == np.size(im), f'see_4bit_img given a shape ({shape}) that doesnt match element count {np.size(im)} - expected {np.prod(shape)} elements instead'

    im.resize(*shape)

    print(f'For {shape} image:')
    print(f'Max {np.amax(im)}')
    print(f'Med {np.median(im)}')
    print(f'Min {np.amin(im)}')

    if clip_values: im = np.clip(im, 0, 16)
    if floor_values: im = np.floor(im)

    return im

def see_img(im, see=True, save=False, filename="out"):
    plt.imshow(im)
    plt.axis("off")

    if(save): plt.savefig(filename + '.png')
    if(see): plt.show()
    if(not see): plt.close()

def see_example_mnist8():
    data, label = D.sample_mnist8()
    see_mnist8(data, label)