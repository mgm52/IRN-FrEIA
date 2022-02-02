# Train a given model
from cgi import test
import data as data_functions
import torch
import numpy as np
from networks.freia_nets import get_inn_fc_freia
from networks.pytorch_nets import get_inn_fc_pt
from torchmetrics.image.fid import FrechetInceptionDistance
from data import mnist8_iterator

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_inn_fc_mnist8(inn, mnist8_iter, min_loss=0.8, max_iters=10000, pytorch_mode=False):
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
            z, log_jac_det = inn(x)
            loss = 0.5*torch.sum(z**2) - log_jac_det

        else:
            x = torch.Tensor(np.array([data]))
            z, log_jac_det = inn(x)
            loss = 0.5*torch.sum(z**2) - log_jac_det

        loss = loss.mean() / (8*8)
        losses.append(loss)

        if i%250==249:
            avg_loss = sum(losses) / len(losses)
            if pytorch_mode:
                print(len(z))
            else:
                print(z.shape)
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

# Takes input x shape (1, 64) or possibly (64)
# Gives output z shape (64)
def full_inn_fc_mnist8_example(mnist8_iter):
    inn = get_inn_fc_freia()
    train_inn_fc_mnist8(inn, mnist8_iter, 0.8, 10, False)
    see_mnist8_results(inn, 0)
    return inn

# Takes input x shape (64)
# Gives output z shape (1, 64)
def full_inn_fc_mnist8_pt_example(mnist8_iter):
    inn = get_inn_fc_pt(device=device)
    train_inn_fc_mnist8(inn, mnist8_iter, 0.5, 20000, True)
    #see_mnist8_results(inn, 5)
    return inn

mnist8_iter = mnist8_iterator()

inn_freia = get_inn_fc_freia()
train_inn_fc_mnist8(inn_freia, mnist8_iter, -1, 1000, False)

inn_pt = get_inn_fc_pt(device=device)
train_inn_fc_mnist8(inn_pt, mnist8_iter, -1, 1000, True)

see_mnist8_results(inn_freia, 5, device="cpu")
see_mnist8_results(inn_pt, 5, device=device)

# Lower fid better
fid_freia = compute_mnist8_fid_score(inn_freia, device="cpu")
fid_pt = compute_mnist8_fid_score(inn_pt, device=device)

print(fid_freia)
print(fid_pt)