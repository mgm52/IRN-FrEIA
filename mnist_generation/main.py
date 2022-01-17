# Train a given model
import data as data_functions
import torch

from networks.freia_nets import inn_fc
from networks.freia_nets import inn_conv
from networks.freia_nets import sample_inn_fc_mnist8

from networks.pytorch_nets import inn_fc_pt
from networks.pytorch_nets import sample_inn_fc_mnist8_pt

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_inn_fc_mnist8(inn, min_loss=0.8, max_iters=10000, pytorch_mode=False):
    optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)

    i = 0
    avg_loss = min_loss+1
    losses = []
    while avg_loss > min_loss and i < max_iters:
        optimizer.zero_grad()
        data, label = data_functions.sample_mnist8()

        if pytorch_mode:
            # c format
            data = data.reshape(64)
            x = torch.tensor(data, dtype=torch.float32, device=device)
            z, log_jac_det = inn(x)

            loss = 0.5*torch.sum(z[0]**2) - log_jac_det

            #print(f'FINALLY ended up with loss of {loss}')
        else:
            x = torch.Tensor(np.array([data]))
            z, log_jac_det = inn(x)
            loss = 0.5*torch.sum(z**2, 1) - log_jac_det

        
        loss = loss.mean() / (8*8)

        if loss<0.0:
            print("Achieved loss of " + str(loss))
            #print(z)

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

# Takes input x shape (1, 64) or possibly (64)
# Gives output z shape (64)
def full_inn_fc_mnist8_example():
    inn = inn_fc()
    train_inn_fc_mnist8(inn, 0.8, 100000, False)
    sample_inn_fc_mnist8(inn, 10)

# Takes input x shape (64)
# Gives output z shape (1, 64)
def full_inn_fc_mnist8_pt_example():
    inn = inn_fc_pt(device=device)
    train_inn_fc_mnist8(inn, -999.15, 20000, True)
    sample_inn_fc_mnist8_pt(inn, 5, device=device)

full_inn_fc_mnist8_pt_example()