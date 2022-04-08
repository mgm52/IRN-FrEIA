import torch

class StraightThroughEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, out_fun):
        return out_fun(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def quantize_ste(x):
    return StraightThroughEstimator.apply(x, lambda y : (torch.clamp(y, min=0, max=1) * 255.0).round() / 255.0)

def quantize_to_int_ste(x):
    return StraightThroughEstimator.apply(x, lambda y : (torch.clamp(y, min=0, max=1) * 255.0).round().int())