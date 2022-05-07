from math import floor, log2
import time
from tokenize import Double
from FrEIA.modules.reshapes import HaarDownsampling
from FrEIA.modules.invertible_resnet import ActNorm
from .coupling import AffineCouplingOneSidedIRN, EnhancedCouplingOneSidedIRN
from utils.bicubic_pytorch.core import imresize
import wandb
import FrEIA.framework as ff
import numpy as np
from typing import Iterable, Tuple, List
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import models.model_loader
from networks.dense_block import db_subnet
from data.dataloaders import DataLoaders
from models.layers.straight_through_estimator import quantize_ste, quantize_to_int_ste

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=200)

class BatchnormSequenceINN(ff.SequenceINN):
    # dims is in (c, w, h) format
    # batch normalisation is only applied on the first bnorm_channels channels of x and z.
    def __init__(self, *dims: int, bnorm_channels = 3, force_tuple_output=False):
        super().__init__(*dims)
        self.bnorm_channels = bnorm_channels
        print(bnorm_channels)
        #self.bn_scale = torch.ones(1, device=device)
        #self.bn_bias = torch.zeros(1, device=device)
    

    # inn x_or_z is in [(n, c, w, h), (n, c, w, h), ...] format
    def forward(self, x_or_z: torch.Tensor, c: Iterable[torch.Tensor] = None,
                rev: bool = False, jac: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        
        iterator = range(len(self.module_list))
        log_det_jac = 0

        if rev:
            iterator = reversed(iterator)

        # Apply batchnorm
        assert len(x_or_z) == 1, "Sequence batchnorm is not compatible with multiple batches"
        #x_or_z[0][:, :self.bnorm_channels], mean, std = self.__standardise_tensor(x_or_z[0][:, :self.bnorm_channels].clone(), 0, 1)

        # TODO: compute jacobian for batchnorm. Can use actnorm as a reference here. Will look something like:
        #jac = (log(std) * x_or_z.shape[1] * np.prod(self.dims_in[1:])).repeat(x_or_z[0].shape[0])

        # This code lifted from freia
        if torch.is_tensor(x_or_z):
            x_or_z = (x_or_z,)
        for i in iterator:
            if self.conditions[i] is None:
                x_or_z, j = self.module_list[i](x_or_z, jac=jac, rev=rev)
            else:
                x_or_z, j = self.module_list[i](x_or_z, c=[c[self.conditions[i]]],
                                                jac=jac, rev=rev)
            log_det_jac = j + log_det_jac

        # Reverse batchnorm using mean & std computed earlier
        if torch.is_tensor(x_or_z[0]):
            x_or_z = x_or_z[0]

        # ISSUE WITH THIS APPROACH:
        # we cannot output both 1. the mean and std of the print-able image, and 2. the mean and std of the y.
        # without the mean and std of the y, our reconstruction will be OFF what it was.
       # y_unadjusted = x_or_z[:, :self.bnorm_channels].clone()
        #x_or_z[:, :self.bnorm_channels], mean_y, std_y = self.__standardise_tensor(y_unadjusted, mean, std)
        
        if torch.is_tensor(x_or_z):
                    # TODO: put jac in here
                    x_or_z = (x_or_z,)

        return x_or_z if self.force_tuple_output else x_or_z[0], log_det_jac
        