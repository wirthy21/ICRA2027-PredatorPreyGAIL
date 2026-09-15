import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.encoder_utils import *
from utils.train_utils import compute_wasserstein_loss, gradient_penalty

"""
References:
Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)
Wu et al. (2025) - CBIL: Collective Behavior Imitation Learning for Fish from Real Videos (https://doi.org/10.48550/arXiv.2504.00234)

Initial structure derived from CBIL GitHub repository:
https://github.com/littlecobber/CBIL
"""


class Discriminator(nn.Module):
    """
    Input: sampled window of state-action pairs from either expert or policy trajectories
    Output: Wasserstein score matrix of agent-wise transitions [batch, frames-1, agents]
    """
    def __init__(self, encoder, role, z_dim=32):
        super(Discriminator, self).__init__()
        self.encoder = encoder # encodes clips into transition features
        self.role = role
        self.z_dim = z_dim # latent dimension
        self.input_dim = 2 * z_dim # to fit transition feature dimension

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def encoder_forward(self, tensor):
        states = tensor[..., :-1] # states only for encoder
        _, trans = self.encoder(states) # get transition features
        batch, frames_minus_one, agent, feat_dim = trans.shape
        feats = trans.reshape(batch * frames_minus_one * agent, feat_dim) # flatten for MLP
        return feats, (batch, frames_minus_one, agent)

    def forward(self, tensor):
        features, shape = self.encoder_forward(tensor)
        batch, frames_minus_one, agent = shape

        # pass through MLP to get wasserstein scores
        params = torch.relu(self.fc1(features))
        params = torch.relu(self.fc2(params))
        params = torch.relu(self.fc3(params))
        params = self.fc4(params).squeeze(-1)

        # reshape back to score matrix
        matrix = params.view(batch, frames_minus_one, agent) # frames-1 because of transitions in between frames
        return matrix

    def update(self, expert_batch, policy_batch, optim_dis, lambda_gp,
               noise=0, generation=None, num_generations=None):

        # add noise to inputs for discriminator regularization
        # helps with GAIL balancing, noise regularizes the discriminator performance in early training stages
        if noise > 0.0:
            # noise only until half of training, with linear decay
            noise_until = 0.5 * num_generations
            decay = 1.0 - (generation / noise_until)
            decay = max(0.0, decay)
            noise_term = noise * decay

            # clone to keep original batches unchanged
            expert_batch = expert_batch.clone()
            policy_batch = policy_batch.clone()

            # noise only on states (actions are left unchanged)
            expert_batch[..., :-1] += torch.randn_like(expert_batch[..., :-1]) * noise_term
            policy_batch[..., :-1] += torch.randn_like(policy_batch[..., :-1]) * noise_term

        # discriminator forward pass, for expert and policy batches
        exp_scores = self.forward(expert_batch)
        gen_scores = self.forward(policy_batch)

        # gradient penalty
        grad_penalty = gradient_penalty(self, expert_batch, policy_batch)

        # wasserstein loss with gradient penalty
        loss, loss_gp = compute_wasserstein_loss(exp_scores, gen_scores, lambda_gp, grad_penalty)

        # optimization step
        optim_dis.zero_grad()
        loss_gp.backward()
        optim_dis.step()

        return {
            "dis_loss": round(loss.item(), 4),
            "dis_loss_gp": round(loss_gp.item(), 4),
            "grad_penalty": round(grad_penalty.item(), 4),
            "expert_score_mean": round(exp_scores.mean().item(), 4),
            "policy_score_mean": round(gen_scores.mean().item(), 4),
        }

    # https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
    def set_parameters(self, init=True):
        # Initialize all parameters
        if init is True:
            for layer in self.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
