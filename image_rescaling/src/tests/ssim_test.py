import cv2
import torch
import torchmetrics
import numpy as np

def ssim_irn(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim_irn(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_irn(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_irn(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

##### CODE ABOVE THIS LINE IS IMPORTED FROM IRN (Xiao et al) FOR THE SAKE OF UNIT TESTING #####

def ssim_test(minv=0, maxv=255, x=None, y=None):
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=maxv-minv)

    x_test = x * (maxv-minv) + minv
    y_test = y * (maxv-minv) + minv

    ssim_xy = ssim_metric(x_test,y_test)
    print(f"My SSIM RGB: {ssim_xy}")

    ssims = []
    for i in range(0, x_test.shape[0]):
        ssims.append(ssim_metric(x_test[i].reshape(1, 3, 15, 15), y_test[i].reshape(1, 3, 15, 15)))
    print(f"My summed SSIM RGB over samples: {sum(ssims) / len(ssims)}")

    ssims = []
    for i in range(0, x_test.shape[1]):
        ssims.append(ssim_metric(x_test[:, i, ...].reshape(x_test.shape[0], 1, x_test.shape[-2], x_test.shape[-1]),
                                 y_test[:, i, ...].reshape(y_test.shape[0], 1, y_test.shape[-2], y_test.shape[-1])))
    print(f"My summed SSIM RGB over channels: {sum(ssims) / len(ssims)}")

    x_test = x_test[0, 0, ...]
    y_test = y_test[0, 0, ...]

    ssim_xy_single = ssim_metric(x_test.reshape(1, 1, 15, 15), y_test.reshape(1, 1, 15, 15))
    ssim_irn_xy_single = calculate_ssim(255 * (x_test.numpy() - minv) / (maxv-minv), 255 * (y_test.numpy() - minv) / (maxv-minv))

    print(f"My SSIM single-channel: {ssim_xy_single}")
    print(f"IRN's SSIM single-channel: {ssim_irn_xy_single}")
    ssim_err = (ssim_xy_single - ssim_irn_xy_single).abs()
    assert ssim_err < 0.000001, f"SSIM error value too large: got {ssim_err}"

    print(f"ERROR: {ssim_err}")


def test_ssim_ranges():
    #torch.manual_seed(10)
    #random.seed(10)
    #np.random.seed(10)

    for i in range(10):
        x = torch.rand(20, 3, 15, 15)
        y = torch.rand(20, 3, 15, 15)

        print("FOR SSIM IN [0,1]:")
        ssim_test(0, 1, x, y)
        print("\nFOR SSIM IN [0,255]:")
        ssim_test(0, 255, x, y)

if __name__ == '__main__':
    test_ssim_ranges()