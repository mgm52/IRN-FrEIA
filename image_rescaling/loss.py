import torch
import torch.nn.functional as F
from bicubic_pytorch.core import imresize
from networks.freia_invertible_rescaling import quantize_ste
from utils import rgb_to_y

# Expects x to be in range [0, 1], y to be roughly in range [0, 1].
# y_channel_usage in range [0, 1]
def calculate_irn_loss(lambda_recon, lambda_guide, lambda_distr, x, y, z, x_recon_from_y, scale, mean_y, std_y, batchnorm=False, mean_losses=False, quantize_recon=False, y_channel_usage=0):
    # Purpose of Loss_Reconstruction: accurate upscaling
    # Might make sense to quantize x_recon because this means the model has more freedom - more "valid" outputs
    
    if quantize_recon: x_recon_quant = quantize_ste(x_recon_from_y)
    else: x_recon_quant = x_recon_from_y
    loss_recon = torch.sum(torch.sqrt((x_recon_quant - x)**2 + 0.000001)) #F.l1_loss(x, x_recon_quant, reduction="sum")# + torch.abs(torch.std(x, axis=1) - torch.std(x_recon_from_y, axis=1)).mean()
    loss_recon = loss_recon / (x.numel() if mean_losses else x.shape[0])

    if y_channel_usage != 0:
        loss_recon_ych = torch.sum(torch.sqrt((rgb_to_y(x_recon_quant) - rgb_to_y(x))**2 + 0.000001)) #F.l1_loss(x, x_recon_quant, reduction="sum")# + torch.abs(torch.std(x, axis=1) - torch.std(x_recon_from_y, axis=1)).mean()
        loss_recon_ych = loss_recon / (x.numel() if mean_losses else x.shape[0])
        # 0.858745 = (0.92149 - 0.062745) = max value of y_channel difference
        loss_recon_ych = loss_recon * (1/0.858745 if mean_losses else 3/0.858745) # correct range to be [0, 1]
        loss_recon = (1 - y_channel_usage) * loss_recon + y_channel_usage * loss_recon_ych

    # Purpose of Loss_Guide: sensible downscaling
        # Intuition about using L2 here: the most recognisable downscaled images get the most prominant points correct?
        # --> for this reason L2 should be better at reducing PSNR than L1
        # [Param] Funnily enough, previous work shows that L1 often produces higher PSNR. See table 1 https://arxiv.org/pdf/1511.08861.pdf
        # [Param] What you need is probably a smoothed L1 (see discussion in .md file)
    if lambda_guide != 0:
        x_downscaled = imresize(x, scale=1.0/scale) #quantize_ste(imresize(x, scale=1.0/scale))
        loss_guide = F.mse_loss(x_downscaled, y, reduction="sum")
        loss_guide = loss_guide / (y.numel() if mean_losses else y.shape[0])

        if y_channel_usage != 0:
            loss_guide_ych = F.mse_loss(rgb_to_y(x_downscaled), rgb_to_y(y), reduction="sum")
            loss_guide_ych = loss_guide / (y.numel() if mean_losses else y.shape[0])
            # 0.737443 = (0.92149 - 0.062745)**2 = max value of y_channel squared difference
            loss_guide_ych = loss_guide * (1/0.737443 if mean_losses else 3/0.737443) # correct range to be [0, 1]
            loss_guide = (1 - y_channel_usage) * loss_guide + y_channel_usage * loss_guide_ych
    else:
        loss_guide = 0

    # Purpose of Loss_Distribution_Match_Surrogate:
        # Encouraging the model to always produce things that look like the OG dataset, even when it doesn't know what to do?
        # And encouraging disentanglement? (by forcing z to be a normal dist)
        # Full Loss_Distribution_Match does this by measuring JSD between x and x_reconstructed.
        # Surrogate Loss_Distribution_match does this by measuring CE between z and z_sample.
    # Paper describes this as: -1 * sum [prob(x from dataset) * log2(prob(z in our normal dist))]
    # Because prob(x from dataset) is a constant: we have -1 * log2(prob(z in our normal dist))
    # Because surprisal in a standard normal dist is O(x^2): we have z^2
    if lambda_distr != 0:
        loss_distr = (z**2).sum()
        loss_distr = loss_distr / (z.numel() if mean_losses else z.shape[0])
    else:
        loss_distr = 0

    if batchnorm: loss_batchnorm = torch.abs(mean_y) + torch.abs(1 - std_y)
    else: loss_batchnorm = 0
    
    # Total_Loss = lr * Loss_Reconstruction + lg * Loss_Guide + ld * Loss_Distribution_Match_Surrogate
    loss_recon *= lambda_recon
    loss_guide *= lambda_guide
    loss_distr *= lambda_distr
    if batchnorm: loss_batchnorm *= 1000

    total_loss = loss_recon + loss_guide + loss_distr + loss_batchnorm
    #total_loss = torch.mean((0.5 - x_recon_from_y)**2)

    return loss_recon, loss_guide, loss_distr, loss_batchnorm, total_loss