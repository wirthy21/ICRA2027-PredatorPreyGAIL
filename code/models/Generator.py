import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ModularNetworks import PairwiseInteraction, Attention


"""
References:
Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)
"""


class ModularPolicy(nn.Module):
    """
    ModularPolicy = PairwiseInteraction + Attention
    Input: states tensor [agents, neigh, features]
    Output: aggregated action for current agent
    """
    def __init__(self, features=4):
        super(ModularPolicy, self).__init__()

        self.pairwise = PairwiseInteraction(features)
        self.attention = Attention(features)

    def forward(self, states, deterministic=False):
        mu, sigma = self.pairwise(states)

        weights_logit = self.attention(states)
        weights = torch.softmax(weights_logit, dim=1) # normalize over neighbors dimension

        # deterministic is only for pretraining with BC, had problems in training stabilization when output was stochastic 
        # GAIL training always stochastic
        if deterministic:
            scaled_action = torch.sigmoid(mu) # no sampling, directly use mu
            action = (scaled_action * weights).sum(dim=1) # weighted sum over neighbors
            return action, weights # action = [0,1]
        else:
            eps = torch.randn_like(mu) # sample from standard normal (initially used numpy's Normal(mu, sigma), but with vmap for batch_env this caused issues)
            action = mu + sigma * eps
            scaled_action = torch.sigmoid(action)
            action = (scaled_action * weights).sum(dim=1) # weighted sum over neighbors
            return action, weights # action = [0,1]
        
# https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
    def set_parameters(self, init=True):
        # Initialize all parameters
        if init is True:
            for layer in self.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()