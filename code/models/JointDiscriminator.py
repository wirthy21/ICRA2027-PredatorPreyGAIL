"""
References:
Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)
Wu et al. (2025) - CBIL: Collective Behavior Imitation Learning for Fish from Real Videos (https://doi.org/10.48550/arXiv.2504.00234)

Initial structure derived from CBIL GitHub repository:
https://github.com/littlecobber/CBIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.encoder_utils import *
from utils.train_utils import compute_wasserstein_loss, gradient_penalty, gradient_penalty_joint, alive_mask

class JointDiscriminator(nn.Module):
    """
    The joint discriminator receives concatenated latent transition features
    from both predator and prey encoders.
    
    Input: predator and prey tensors
    Output: Wasserstein score matrix
    """
    def __init__(self, pred_encoder, prey_encoder, z_dim=32, hidden_dim=128):
        super(JointDiscriminator, self).__init__()

        # seperate encoders for pred and prey
        self.pred_encoder = pred_encoder  # from Stage 1
        self.prey_encoder = prey_encoder  # from Stage 1

        # latent dimension
        self.z_dim = z_dim
        
        # each encoder outputs a transition feature of size 2*z_dim due to concatenation
        # joint input = concat of both pooled features → 4*z_dim
        self.input_dim = 4 * z_dim

        # MLP identical in depth to the individual discriminator,
        # but operating on the joint feature
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # single Wasserstein score per (batch, frame)


    def encode_role(self, pred_tensor, prey_tensor):
        # extract states only 
        pred_states = pred_tensor[..., :-1]
        prey_states = prey_tensor[..., :-1]

        # encode each role with its own encoder
        _, pred_trans = self.pred_encoder(pred_states)  # [B, T-1, pred_agents, 2*z]
        _, prey_trans = self.prey_encoder(prey_states)  # [B, T-1, prey_agents, 2*z]
        batch, frames_minus_one = pred_trans.shape[:2] # grabbing B and T-1 here -- only unpack these 2 values

        # since we need to combine predator and prey (which have different agent counts) into single vector, we do pooling
        # mean pooling collapses the agent dimension — per-agent detail is averaged away, not preserved
        m_pred = alive_mask(pred_tensor).unsqueeze(-1)  # [B, T-1, pred_agents, 1]
        m_prey = alive_mask(prey_tensor).unsqueeze(-1)  # [B, T-1, prey_agents, 1]
        
        pred_pooled = (pred_trans * m_pred).sum(dim=2) / m_pred.sum(dim=2).clamp_min(1.0)
        prey_pooled = (prey_trans * m_prey).sum(dim=2) / m_prey.sum(dim=2).clamp_min(1.0)

        # create joint representation
        joint = torch.cat([pred_pooled, prey_pooled], dim=-1)  # [B, T-1, 4*z]

        # flatten for MLP
        feats = joint.reshape(batch * frames_minus_one, 4 * self.z_dim)
        return feats, (batch, frames_minus_one)


    def forward(self, pred_tensor, prey_tensor):
        # encode both roles and get joint pooled features
        features, shape = self.encode_role(pred_tensor, prey_tensor)
        batch, frames_minus_one = shape
        
        # pass through MLP to get wasserstein scores
        params = torch.relu(self.fc1(features))
        params = torch.relu(self.fc2(params))
        params = torch.relu(self.fc3(params))
        params = self.fc4(params).squeeze(-1)

        # reshape back to [B, T-1]
        matrix = params.view(batch, frames_minus_one) # frames-1 because of transitions in between frames
        return matrix


    # training step
    def update(self, expert_pred_batch, expert_prey_batch,
           policy_pred_batch, policy_prey_batch,
           optim_dis, lambda_gp,
           noise=0, generation=None, num_generations=None):

        # add noise to inputs for discriminator regularization
        if noise > 0.0:

            # noise only until half of training, with linear decay
            noise_until = 0.5 * num_generations
            decay = 1.0 - (generation / noise_until)
            decay = max(0.0, decay)
            noise_term = noise * decay

            # clone to keep original batches unchanged
            expert_pred_batch = expert_pred_batch.clone()
            expert_prey_batch = expert_prey_batch.clone()
            policy_pred_batch = policy_pred_batch.clone()
            policy_prey_batch = policy_prey_batch.clone()

            # noise only on states (actions are left unchanged)
            expert_pred_batch[..., :-1] += torch.randn_like(expert_pred_batch[..., :-1]) * noise_term
            expert_prey_batch[..., :-1] += torch.randn_like(expert_prey_batch[..., :-1]) * noise_term
            policy_pred_batch[..., :-1] += torch.randn_like(policy_pred_batch[..., :-1]) * noise_term
            policy_prey_batch[..., :-1] += torch.randn_like(policy_prey_batch[..., :-1]) * noise_term

        # discriminator forward pass for expert and policy
        exp_scores = self.forward(expert_pred_batch, expert_prey_batch)
        gen_scores = self.forward(policy_pred_batch, policy_prey_batch)

        # gradient penalty
        grad_penalty = gradient_penalty_joint(self, 
                                          expert_pred_batch, expert_prey_batch,
                                          policy_pred_batch, policy_prey_batch)

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
    # weight reset
    def set_parameters(self, init=True):
        if init:
            for layer in self.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

                    

