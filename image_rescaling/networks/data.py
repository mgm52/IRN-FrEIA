# Get dataset
from tkinter import CENTER
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits


mnist8_data, mnist8_labels = load_digits(return_X_y=True)
def sample_mnist8():
    i = random.randint(0, len(mnist8_data)-1)
    data = np.array(mnist8_data[i])
    label = mnist8_labels[i]

    data.resize(8*8)
    return data, label

def sample_mnist8_imgs(count=1, use_test_data=False):
    test_limit = np.ceil(len(mnist8_data) * 0.9)
    samples = []
    for j in range(count):
        if not use_test_data: i = random.randint(0, test_limit)
        else: i = random.randint(test_limit, len(mnist8_data)-1)

        data = np.array(mnist8_data[i])
        samples.append(data)
    return samples

class mnist8_iterator:

    def __init__(self, shuffle_data=True):
        self.dataset = mnist8_data
        if shuffle_data: np.random.shuffle(self.dataset)

        self.current_train_epoch = -1
        self.prev_train_index = -1
        self.prev_test_index = -1
    
    def iterate_mnist8_imgs(self, count=1, use_test_data=False):
        test_limit = np.ceil(len(self.dataset) * 0.9)
        samples = []
        for j in range(count):
            if not use_test_data:
                self.prev_train_index = (self.prev_train_index + 1) % test_limit
                if self.prev_train_index==0:
                    self.current_train_epoch += 1
                    #if self.current_train_epoch % 10 == 0: print(f'Starting epoch {self.current_train_epoch} over data')
                i = self.prev_train_index
            else:
                self.prev_test_index = (self.prev_test_index + 1) % (len(self.dataset) - test_limit)
                i = test_limit + self.prev_test_index

            data = np.array(self.dataset[int(i)])
            samples.append(data)
        return samples

def see_mnist8(data, label=-1, clip_values=True, floor_values=True):
    if label > -1: print(f'Digit label: {label}')

    im = np.array(data)
    size = int(np.sqrt(im.size))
    assert size==np.sqrt(im.size), f'see_mnist8 only accepts square images - given {im.size}, wanted {size*size}'

    im = process_4bit_img(data, (size, size), clip_values=clip_values, floor_values=floor_values)
    see_img(im)

def process_4bit_img(data, shape, clip_values=True, floor_values=True):
    im = np.array(data)
    assert np.prod(shape) == np.size(im), f'see_4bit_img given a shape ({shape}) that doesnt match element count {np.size(im)} - expected {np.prod(shape)} elements instead'

    im.resize(*shape)

    print(f'Max in {shape} image: {np.amax(im)}')
    print(f'Median in {shape} image: {np.median(im)}')
    print(f'Min in {shape} image: {np.amin(im)}')

    if clip_values: im = np.clip(im, 0, 16)
    if floor_values: im = np.floor(im)

    return im

def see_img(im):
    plt.imshow(im)
    plt.show()

def see_multiple_imgs(imgs, title=""):
    f, axes = plt.subplots(1, len(imgs))
    for i in range(len(imgs)):
        axes[i].imshow(imgs[i])
    plt.title(title, loc="right")
    plt.show()

def see_example_mnist8():
    data, label = sample_mnist8()
    see_mnist8(data, label)