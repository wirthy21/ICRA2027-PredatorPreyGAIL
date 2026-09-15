import sys, os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.ModularNetworks import Attention

"""
References:
(1) https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy
(2) https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
"""

class RBF(nn.Module):
    """
    Radial Basis Function Kernel, to compute pairwise similarities between samples.

    Input: Concatenated samples of encoded expert and policy batches
    Output: Kernel matrix
    """
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        # Changed to buffer, necessary to have easier device handling 
        self.register_buffer("bandwidth_multipliers", mul_factor ** (torch.arange(n_kernels) - n_kernels // 2))
        self.bandwidth = bandwidth
        
    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            # average of off-diagonal distances
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return torch.as_tensor(self.bandwidth, device=L2_distances.device, dtype=L2_distances.dtype)

    def forward(self, X):
        # pairwise squared Euclidean distances
        L2_distances = torch.cdist(X, X) ** 2
        # compute multi-kernel RBF and sum across kernels
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)



class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy, to determine how different two distributions are.

    Input: sampled window of state-action pairs from either expert or policy trajectories
    Output: MMD loss value
    """
    def __init__(self, encoder, role, kernel=RBF()):
        super().__init__()
        self.encoder = encoder
        self.role = role
        self.kernel = kernel

    def encode_transitions(self, tensor):
        states = tensor[..., :-1] # states only for encoder
        _, transitions = self.encoder(states) # get transition features
        features = transitions.reshape(-1, transitions.size(-1))
        return features

    def forward(self, expert_batch, generative_batch):

        # encode both batches
        X = self.encode_transitions(expert_batch)
        Y = self.encode_transitions(generative_batch)

        # Compute kernel matrix on concatenated samples
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean() # kernel among expert samples
        XY = K[:X_size, X_size:].mean() # kernel between expert and generative samples
        YY = K[X_size:, X_size:].mean() # kernel among generative samples

        # compute maximum mean discrepancy (see reference (1))
        mmd_loss = XX - 2 * XY + YY
        return mmd_loss