# Get dataset
from re import S
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits

def sample_mmoons():
    data, labels = make_moons(n_samples=100, noise=0.05)
    return data, labels

def see_moons(data, labels=1):
    plt.scatter(data[:, 0][labels==1], data[:, 1][labels==1], c="b")
    plt.scatter(data[:, 0][labels==0], data[:, 1][labels==0], c="r")
    plt.show()

def see_example_moons():
    data, labels = sample_mmoons()
    print(f'Moon points: {data}')
    print(f'Moon labels: {labels}')
    see_moons(data, labels)

mnist8_data, mnist8_labels = load_digits(return_X_y=True)
def sample_mnist8():
    i = random.randint(0, len(mnist8_data)-1)
    data = np.array(mnist8_data[i])
    label = mnist8_labels[i]

    data.resize(8*8)
    return data, label

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
        if count == -1:
            if not use_test_data:
                # Return all training data
                self.current_train_epoch += 1
                return self.dataset[:int(test_limit)]
            else:
                # Return all test data
                return self.dataset[int(test_limit):]
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
    data, label = sample_mnist8()
    see_mnist8(data, label)