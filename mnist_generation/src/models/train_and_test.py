# Train a given model
from cgi import test
import data as data_functions
import torch
import math
import numpy as np
from models.layers.freia_nets import get_inn_fc_freia
from models.layers.pytorch_nets import get_inn_fc_pt
from torchmetrics.image.fid import FrechetInceptionDistance
from data import mnist8_iterator

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_inn_fc_mnist8(inn, mnist8_iter, min_loss=0.8, max_iters=10000, pytorch_mode=False):
    #for name, param in inn.named_parameters():
    #    if param.requires_grad: print(name)

    optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)

    i = 0
    avg_loss = min_loss+1
    losses = []
    while avg_loss > min_loss and i < max_iters:
        optimizer.zero_grad()
        data = mnist8_iter.iterate_mnist8_imgs(count=1, use_test_data=False)
        data = data[0].reshape(64)

        if pytorch_mode:
            x = torch.tensor(data, dtype=torch.float32, device=device)
        else:
            x = torch.Tensor(np.array([data]))

        z, log_det_jac = inn(x)
        loss = 0.5*torch.sum(z**2) - log_det_jac

        loss = loss.mean() / (8*8)
        losses.append(loss)

        if i%250==249:
            avg_loss = sum(losses) / len(losses)
           # if pytorch_mode:
                #print(len(z))
            #else:
                #print(z.shape)
            print(f'Avg loss across parameters, in last 250 samples: {avg_loss}')
            losses = []
        
        loss.backward()
        optimizer.step()
        i+=1

def see_mnist8_results(inn, num_samples, device):
    for i in range(num_samples):
        z = torch.randn(8 * 8, device=device)
        sample, _ = inn(z, rev=True)
        data_functions.see_mnist8(sample)

# Lower fid = real & fake data are more similar
def compute_mnist8_fid_score(inn, device):
    # Acquire "real" data
    # Go from mnist8_iter format (179, 64) to ncwh format (179, 3, 8, 8)
    test_data = torch.tensor(mnist8_iter.iterate_mnist8_imgs(-1, True), dtype=torch.uint8).reshape(179, 1, 8, 8)
    test_data = test_data.repeat(1, 3, 1, 1)

    # Acquire "fake" data
    # Go from mnist8_iter format (179, 64) to ncwh format (179, 3, 8, 8)
    sampled_data = []
    for i in range(len(test_data)):
        z = torch.randn(8 * 8, device=device)
        sample, _ = inn(z, rev=True)
        sampled_data.append(sample.detach().cpu())
    sampled_data = torch.stack(sampled_data).to(torch.uint8).reshape(179, 1, 8, 8)
    sampled_data = sampled_data.repeat(1, 3, 1, 1)
    assert sampled_data.shape == test_data.shape, "Generated image sample is not of the same shape as the test data"

    # Compute FID
    fid = FrechetInceptionDistance(feature=64)
    fid.update(test_data, real=True)
    fid.update(sampled_data, real=False)
    fid_val = fid.compute()

    return fid_val

def log_standard_normal_pdf(x):
    return -0.5 * torch.sum(math.log(2 * math.pi) + (x**2))

# Higher likelihood = better
def compute_log_likelihood(inn, device, pytorch_mode):
    # Acquire test data in mnist8_iter format (179, 64)
    test_data = torch.tensor(mnist8_iter.iterate_mnist8_imgs(-1, True), device=device)

    # Acquire "fake" data
    # Go from mnist8_iter format (179, 64) to ncwh format (179, 3, 8, 8)
    log_likelihoods = []
    for i in range(len(test_data)):
        x = torch.tensor(test_data[i], dtype=torch.float32, device=device)
        if not pytorch_mode: x = x.reshape(1, 64)

        z, log_jac_det = inn(x)
        log_likelihood = log_standard_normal_pdf(z).item() + log_jac_det.item()
        log_likelihoods.append(log_likelihood)

    return sum(log_likelihoods) / len(log_likelihoods)

####    EXECUTION CODE    ####
if __name__ == '__main__':

    mnist8_iter = mnist8_iterator()

    inn_freia = get_inn_fc_freia()
    inn_pt = get_inn_fc_pt(device=device)

    to_train=5000
    while True:
        print("---------- STARTING TRAINING LOOP -------------")
        print(to_train)
        print("Training freia")
        train_inn_fc_mnist8(inn_freia, mnist8_iter, -1, to_train, False)
        print("Training pt")
        train_inn_fc_mnist8(inn_pt, mnist8_iter, -1, to_train, True)

        print("--- Viewing freia examples ---")
        see_mnist8_results(inn_freia, 5, device="cpu")
        print("--- Viewing pt examples ---")
        see_mnist8_results(inn_pt, 5, device=device)

        # Higher likelihood better
        ll_freia = compute_log_likelihood(inn_freia, device="cpu", pytorch_mode=False)
        ll_pt = compute_log_likelihood(inn_pt, device=device, pytorch_mode=True)

        print(f'FrEIA log likelihood: {ll_freia}')
        print(f'Pytorch log likelihood: {ll_pt}')

        # Lower fid better
        fid_freia = compute_mnist8_fid_score(inn_freia, device="cpu")
        fid_pt = compute_mnist8_fid_score(inn_pt, device=device)

        print(f'FrEIA FID: {fid_freia.item()}')
        print(f'Pytorch FID: {fid_pt.item()}')

        to_train *= 2