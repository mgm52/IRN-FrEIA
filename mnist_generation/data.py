# Get dataset
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

def see_mnist8(data, label=0):
    #print(f'Digit label: {label}')

    im = np.array(data, dtype=np.uint8)
    im.resize(8, 8)

    print(f'Max in sample: {np.amax(im)}')
    print(f'Median in sample: {np.median(im)}')
    print(f'Min in sample: {np.amin(im)}')

    plt.imshow(im)
    plt.show()

def see_example_mnist8():
    data, label = sample_mnist8()
    see_mnist8(data, label)