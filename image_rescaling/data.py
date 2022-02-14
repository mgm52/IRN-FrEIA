# Get dataset
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits

class mnist8_iterator:
    def __init__(self, shuffle_data=True):
        self.dataset, _ = load_digits(return_X_y=True)
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

    max_color = 16
    im = np.clip(im, 0, max_color)
    #im = np.floor(im)
    im = im / max_color

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

def see_multiple_imgs(imgs, rows, cols, row_titles=[], plot_titles=[], see=True, save=False, filename="out"):
    assert rows*cols >= len(imgs), f'Cannot print {len(imgs)} images on a {rows}x{cols} grid'
    
    f, axes = plt.subplots(figsize=(3*cols, 3*rows) , nrows=rows, ncols=1, sharey=True) 
    f.set_dpi(200)

    if rows > 1:
        for row_num, row_ax in enumerate(axes, start=1):
            # Add title to row
            if row_num-1<len(row_titles): row_ax.set_title(row_titles[row_num-1] + "\n", fontsize=14, loc="left")
            row_ax.axis('off')

    maximgsize = max(imgs[0].shape)

    for i in range(1, rows*cols + 1):
        # Add subplot to index i-1 within a rows*cols grid
        ax = f.add_subplot(rows,cols,i)
        if i-1<len(plot_titles): ax.set_title(plot_titles[i-1], fontsize=int(9.0 + maximgsize * 2.0/100.0), loc="left")
        if i-1<len(imgs) and not (imgs[i-1] is None): ax.imshow(imgs[i-1])
        ax.axis('off')

    plt.tight_layout()
    f.set_size_inches(cols * maximgsize * 7.5/256.0, rows * maximgsize * 9/256.0)

    if(save): plt.savefig(filename + '.png', dpi=200)
    if(see): plt.show()
    if(not see): plt.close()

    return f