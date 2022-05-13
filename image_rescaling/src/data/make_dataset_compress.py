import time
from tracemalloc import start
from torchvision import transforms
from dataloaders import DatasetDataLoaders, DataLoaders, get_test_dataloader
from utils.bicubic_pytorch.core import imresize
import torch
from timeit import default_timer as timer
from models.layers.invertible_rescaling_network import quantize_ste
from utils.utils import create_parent_dir
import random
from models.layers.straight_through_estimator import quantize_ste, quantize_to_int_ste
from PIL import Image
import models.layers.straight_through_estimator
from io import BytesIO
import random
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

def jpeg_compress_random(image_t, quality = -1, save_path=""):

    image = ToPILImage()(image_t)
    outputIoStream = BytesIO()

    if quality < 0:
        quality = random.randrange(10, 100)

    print(f"Saving with quality {quality}")
    image.save(outputIoStream, "JPEG", quality=int(quality), optimize=True)
    if save_path: image.save(save_path, "JPEG", quality=int(quality), optimize=True)

    outputIoStream.seek(0)
    return (ToTensor()(Image.open(outputIoStream))).cuda()

def compress_batch(imgs, quality = -1, path=""):
  imgs_cmpr = torch.cat([jpeg_compress_random(img, quality, path).unsqueeze(0) for img in imgs])
  return imgs_cmpr

def compress_ste(x, quality = -1, path=""):
    return models.layers.straight_through_estimator.StraightThroughEstimator.apply(x, lambda y : compress_batch(y, quality, path))

if __name__ == '__main__':

    scale = 4.0
    quality = 100

    # load div2k
    dataloaders = DatasetDataLoaders(16, 144, full_size_test_imgs=True, test_img_size_divisor=scale)
    test_iter = iter(dataloaders.test_dataloader)

    for i in range(dataloaders.test_len):
        if (i-1) % int(max(2, dataloaders.test_len / 5)) == 0:
            print(f"At image {i}/{int(dataloaders.test_len)}")

        path = f"data/DIV2K/DIV2K_valid_LR_x4_compress{quality}/{str(i).zfill(3)}_LRMOD_{int(time.time())}.jpg"
        create_parent_dir(path)

        x_raw, _ = next(test_iter)

        x_downscaled = quantize_ste(imresize(x_raw, scale=1.0/scale))
        x_compressed = compress_ste(x_downscaled, quality, path)

        #save_image(quantize_ste(x_compressed), path)