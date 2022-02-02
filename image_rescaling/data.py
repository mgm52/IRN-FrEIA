# Get dataset
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
    im = np.array(data.detach().cpu().numpy())
    assert np.prod(shape) == np.size(im), f'see_4bit_img given a shape ({shape}) that doesnt match element count {np.size(im)} - expected {np.prod(shape)} elements instead'

    im.resize(*shape)

    print(f'Max in {shape} image: {np.amax(im)}')
    print(f'Median in {shape} image: {np.median(im)}')
    print(f'Min in {shape} image: {np.amin(im)}')

    if clip_values: im = np.clip(im, 0, 16)
    if floor_values: im = np.floor(im)

    return im

def see_img(im, see=True, save=False, filename="out"):
    plt.imshow(im)

    if(save): plt.savefig(filename + '.png')
    if(see): plt.show()
    if(not see): plt.close()

def see_multiple_imgs(imgs, rows, cols, row_titles=[], plot_titles=[], see=True, save=False, filename="out"):
    assert rows*cols >= len(imgs), f'Cannot print {len(imgs)} images on a {rows}x{cols} grid'
    
    f, axes = plt.subplots(figsize=(3*cols, 3*rows) , nrows=rows, ncols=1, sharey=True) 

    for row_num, row_ax in enumerate(axes, start=1):
        # Add title to row
        if row_num-1<len(row_titles): row_ax.set_title(row_titles[row_num-1] + "\n", fontsize=14, loc="left")
        row_ax.axis('off')

    for i in range(1, rows*cols + 1):
        # Add subplot to index i-1 within a rows*cols grid
        ax = f.add_subplot(rows,cols,i)
        if i-1<len(plot_titles): ax.set_title(plot_titles[i-1], fontsize=11, loc="left")
        if i-1<len(imgs) and not (imgs[i-1] is None): ax.imshow(imgs[i-1])
        ax.axis('off')

    plt.tight_layout()
    f.set_size_inches(cols * 2.5, rows * 3)

    if(save): plt.savefig(filename + '.png', dpi=100)
    if(see): plt.show()
    if(not see): plt.close()

def see_example_mnist8():
    data, label = sample_mnist8()
    see_mnist8(data, label)