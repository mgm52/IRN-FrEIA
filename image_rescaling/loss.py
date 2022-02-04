import torch
from bicubic_pytorch.core import imresize

def calculate_irn_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, batch_size):
    # Purpose of Loss_Reconstruction: accurate upscaling
    loss_recon = torch.abs(x - x_recon_from_y).sum() / batch_size

    # Purpose of Loss_Guide: sensible downscaling
        # Intuition about using L2 here: the most recognisable downscaled images get the most prominant points correct?
        # --> for this reason L2 should be better at reducing PSNR than L1
    x_downscaled = imresize(x, scale=0.5)
    loss_guide = ((x_downscaled - y)**2).sum() / batch_size

    # Purpose of Loss_Distribution_Match_Surrogate:
        # Encouraging the model to always produce things that look like the OG dataset, even when it doesn't know what to do?
        # And encouraging disentanglement? (by forcing z to be a normal dist)
        # Full Loss_Distribution_Match does this by measuring JSD between x and x_reconstructed.
        # Surrogate Loss_Distribution_match does this by measuring CE between z and z_sample.
    # Paper describes this as: -1 * sum [prob(x from dataset) * log2(prob(z in our normal dist))]
    # Because prob(x from dataset) is a constant: we have -1 * log2(prob(z in our normal dist))
    # Because surprisal in a standard normal dist is O(x^2): we have z^2
    loss_distr = (z**2).sum() / batch_size
    
    # Total_Loss = lr * Loss_Reconstruction + lg * Loss_Guide + ld * Loss_Distribution_Match_Surrogate
    total_loss = lambda_recon*loss_recon + lambda_guide*loss_guide + lambda_distr*loss_distr

    return loss_recon, loss_guide, loss_distr, total_loss