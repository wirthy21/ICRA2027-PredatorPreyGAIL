import torch
import torch.nn as nn
import torch.nn.functional as F

"""
References:
Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)
Heras et al. (2019) - Deep attention networks reveal the rules of collective motion in zebrafish (https://doi.org/10.1371/journal.pcbi.1007354)
"""


class PairwiseInteraction(nn.Module):
    """
    Input: states processed of shape [(flag +) dx, dy, vx, vy]
    Output: mu & sigma to sample actions from a normal distribution
    """
    def __init__(self, features=4):
        super().__init__()
        self.fc1 = nn.Linear(features, 128) # features count can vary, if prey-only (4 states) or pred-prey (flag + 4 states)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)  # mu, var_logit

    def forward(self, states):
        # ReLU like Wu et al. (2025)
        params = F.relu(self.fc1(states))
        params = F.relu(self.fc2(params))
        params = F.relu(self.fc3(params))
        params = self.fc4(params)

        mu, var_logit = params[..., :1], params[..., 1:]
        var = F.softplus(var_logit) + 1e-6 # enforce positivity, had problems with div by zero
        sigma = var.sqrt() # standard deviation
        return mu, sigma


class Attention(nn.Module):
    """
    Input: states processed of shape [(flag +) dx, dy, vx, vy]
    Output: unnormalized attention scores, to weight actions
    """
    def __init__(self, features=4):
        super().__init__()
        self.fc1 = nn.Linear(features, 128) # features count can vary, if prey-only (4 states) or pred-prey (flag + 4 states)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1) # weight

    def forward(self, states):
        # ReLU like Wu et al. (2025)
        weight = F.relu(self.fc1(states))
        weight = F.relu(self.fc2(weight))
        weight = F.relu(self.fc3(weight))
        weight = self.fc4(weight)
        return weight