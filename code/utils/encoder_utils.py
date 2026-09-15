import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.ModularNetworks import Attention


"""
References:
Wu et al. (2025) - CBIL: Collective Behavior Imitation Learning for Fish from Real Videos (https://doi.org/10.48550/arXiv.2504.00234)
Bardes et al (2022) - VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning (https://arxiv.org/abs/2105.04906)

Architecture inspired by:
https://github.com/littlecobber/CBIL/blob/main/AdvancedVAE/basic_videovae.py
https://www.geeksforgeeks.org/nlp/encoder-decoder-models/
https://github.com/AlexanderFabisch/vtae/blob/master/trajectory_vae.py

VigRec:
https://github.com/facebookresearch/vicreg
https://medium.com/@ttleseuldace/paper-review-vicreg-for-self-supervised-learning-a8f7cfc849cb
https://github.com/augustwester/vicreg/blob/main/model.py
"""


class NeighborPooling(nn.Module):
    """
    Pools neighbor-wise features for each focal agent into a single embedding
    Structure is similar to policy networks, uses modular networks with original attention/aggregation network
    Only, structural difference is the single output neuron, instead of mu and sigma

    Input: tensor of shape (batch*frames, agents, neighbors, features), without action feature
    Output: pooled embeddings (..., agents, embd_dim)
    """
    def __init__(self, features=4, embd_dim=32):
        super().__init__()

        input_dim = features * 2 # necessary due to masking

        # Neighbor embedding, maps each neighbor feature vector to embd_dim
        self.embed = nn.Sequential(nn.Linear(input_dim, 64),
                                   nn.LeakyReLU(0.1),
                                   nn.Linear(64, embd_dim),
                                   nn.LeakyReLU(0.1))

        # same attention class as in ModularNetworks.py
        self.attention = Attention(input_dim)

    def forward(self, states, neigh_mask=None, feat_mask=None):
        # feature masking for augmentation
        if feat_mask is None:
            feat_mask = torch.ones_like(states) # keep all features

        # apply feature mask
        states = states * feat_mask

        # concatenate masked features
        cat_states = torch.cat([states, feat_mask], dim=-1)

        # compute neigh embeddings and attention weights
        embed = self.embed(cat_states)
        weights_logit = self.attention(cat_states)

        # if neighbor mask is given, prevent from receiving attention weights
        if neigh_mask is not None:
            weights_logit = weights_logit.masked_fill(neigh_mask == 0, float("-inf"))

        # normalize weights across neigh dim
        weights = torch.softmax(weights_logit, dim=2)

        # weighted sum pooling
        pooled = (embed * weights).sum(dim=2)

        return pooled
    

class AgentEmbedding(nn.Module):
    """
    Maps pooled neighbor embedding to a latent state z for each focal agent.

    Input: pooled neighbord embeddings
    Output: latent agent state z
    """
    def __init__(self, embd_dim=32, z=32):
        super().__init__()

        # MLP to map pooled embedding to latent state
        self.embed = nn.Sequential(nn.Linear(embd_dim, 64),
                                   nn.LeakyReLU(0.1),
                                   nn.Linear(64, z))

    def forward(self, pooled_embd):
        embed = self.embed(pooled_embd)
        return embed


class TransitionEncoder(nn.Module):
    """
    Final Encoder, combines neighbor pooling and agent embedding to map states to latent states z

    Input: states, neigh_mask, feat_mask
    Output: z_state (batch, frames, agents, z), transition_feature (batch, frames-1, agents, 2*z)
    """

    def __init__(self, features=4, embd_dim=32, z=32):
        super().__init__()
        self.z = z
        self.neigh_pooling = NeighborPooling(features=features, embd_dim=embd_dim)
        self.agent_pooling = AgentEmbedding(embd_dim=embd_dim, z=z)

    def forward(self, states, neigh_mask=None, feat_mask=None):

        # flatten batch and frames to process all frames in one forward pass
        batch, frames, agents, neigh, features = states.shape
        flat = states.reshape(batch * frames, agents, neigh, features)

        # reshape neigh_mask to match flattened layout
        if neigh_mask is not None:
            neigh_mask = neigh_mask.reshape(batch * frames, agents, neigh, 1)

        # reshape feat_mask to match flattened layout
        if feat_mask is not None:
            feat_mask = feat_mask.reshape(batch * frames, agents, neigh, features)

        # neighbor pooling
        pooled_embd = self.neigh_pooling(flat, neigh_mask=neigh_mask, feat_mask=feat_mask)
        pooled = pooled_embd.view(batch, frames, agents, -1)

        # agent embedding to latent state z
        z_state = self.agent_pooling(pooled)
        
        # compute transition features
        z_t   = z_state[:, :-1]
        z_tp1 = z_state[:,  1:]

        # latent state difference
        dz = z_tp1 - z_t

        # concatenate z_t and dz to form transition feature
        transition_feature = torch.cat([z_t, dz], dim=-1)
        return z_state, transition_feature
    


class TrajectoryAugmentation(nn.Module):
    """
    Generates two augmented "views" of the same trajectory states for VICReg
    Augmentations: Gaussian noise, neighbor dropout, feature dropout

    Input: states
    Output: augmented states, neigh_mask, feat_mask
    """

    def __init__(self, noise_std=0.01, neigh_drop=0.10, feat_drop=0.05):
        super().__init__()
        self.noise_std = noise_std
        self.neigh_drop = neigh_drop
        self.feat_drop = feat_drop

    def forward(self, states):
        batch, frames, agents, neigh, features = states.shape
        device = states.device

        # create masks initialized with ones (keep all)
        neigh_mask = torch.ones((batch, frames, agents, neigh, 1), device=device, dtype=states.dtype)
        feat_mask  = torch.ones((batch, frames, agents, neigh, features), device=device, dtype=states.dtype)

        # drop neighbors with probability neigh_drop
        if self.neigh_drop > 0:
            neigh_mask = (torch.rand(batch, frames, agents, neigh, 1, device=device) > self.neigh_drop).float()

        # drop features with probability feat_drop
        if self.feat_drop > 0:
            feat_mask = (torch.rand(batch, frames, agents, 1, features, device=device) > self.feat_drop).float()
            feat_mask = feat_mask.expand(batch, frames, agents, neigh, features)

        # edge case: always keep flag feature (prey-only)
        if features == 5:
            feat_mask[..., 0] = 1.0

        # add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(states) * self.noise_std
            if features == 5: # skip flag
                noise[..., 0] = 0.0
            states = states + noise

        return states, neigh_mask, feat_mask


class VicRegProjector(nn.Module):
    """
    MLP projector for VICReg
    Maps transition features to a space where VICReg loss is applied
    """

    def __init__(self, input_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128))
        
    def forward(self, states):
        return self.net(states)


# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
def off_diagonal(tensor):
    """
    Returns all off-diagonal elements of a square matrix as a flat vector
    """
    dim = tensor.size(0)
    mask = ~torch.eye(dim, dtype=torch.bool, device=tensor.device)
    return tensor[mask]


def vicreg_loss(z1, z2, sim_coeff=25.0, std_coeff=15.0, cov_coeff=5.0, eps=1e-4):
    """
    VICReg loss between two batches of representations z1 and z2

    Inputs: augmented states representations z1, z2
    Outputs: vicreg loss, logs dict
    """

    # invariance loss
    sim_loss = torch.mean((z1 - z2) ** 2)

    # center batches
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    # variance regularization, std loss
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

    # covariance regularization, cov loss
    batch, dim = z1.shape
    cov_z1 = (z1.T @ z1) / (batch - 1)
    cov_z2 = (z2.T @ z2) / (batch - 1)
    cov_loss = (off_diagonal(cov_z1).pow(2).sum() / dim) + (off_diagonal(cov_z2).pow(2).sum() / dim)

    # total loss, weighted sum
    loss = sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss

    logs = {"sim": sim_loss.detach(),
            "std": std_loss.detach(),
            "cov": cov_loss.detach(),
            "std_mean": 0.5 * (std_z1.mean().detach() + std_z2.mean().detach())}
    
    return loss, logs


def sample_data(data, batch_size=10, window_len=10):
    """
    Samples a random batch of trajectories from tensor
    """
    if data.dim() == 5: # expert tensor 
        idx = torch.randint(0, data.size(0), (batch_size,), device=data.device)
        return data[idx]
    
    if data.dim() == 4: # generative tensor
        trajectory_len = data.size(0)
        start = torch.randint(0, trajectory_len - window_len + 1, (batch_size,), device=data.device)
        return torch.stack([data[s:s + window_len] for s in start.tolist()], dim=0) # generate same shape as expert batch
    



def train_encoder(encoder, projector, aug, exp_tensor, epochs, optimizer, role="prey"):
    """
    Self-supervised training of encoder with VICReg
    """

    device = next(encoder.parameters()).device

    total_losses = []
    invar_losses   = []
    var_losses   = []
    cov_losses   = []

    for epoch in range(1, epochs + 1):
        encoder.train()
        projector.train()

        # sample expert batch
        expert_batch = sample_data(exp_tensor, batch_size=10, window_len=10).to(device)

        # exclude action feature
        states = expert_batch[..., :-1]

        # generate two augmented views for VICReg
        x1, neigh_mask1, feat_mask1 = aug(states)
        x2, neigh_mask2, feat_mask2 = aug(states)

        # encode both views, train the encoder to produce consistent transition representations
        z_state1, trans1 = encoder(x1, neigh_mask=neigh_mask1, feat_mask=feat_mask1)
        z_state2, trans2 = encoder(x2, neigh_mask=neigh_mask2, feat_mask=feat_mask2)
        
        # reshape transition features for Projector
        r1 = trans1.reshape(-1, trans1.size(-1))
        r2 = trans2.reshape(-1, trans2.size(-1))

        # apply projector
        y1 = projector(r1)
        y2 = projector(r2)

        # VICReg loss
        loss, logs = vicreg_loss(y1, y2)

        # store losses for visualization
        total_losses.append(loss.item())
        invar_losses.append(logs["sim"].item())
        var_losses.append(logs["std"].item())
        cov_losses.append(logs["cov"].item())

        # optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping, necessary for stable training
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(projector.parameters()), 1.0)
        optimizer.step()

        # logging every 10 epochs
        if epoch % 10 == 0:
            print(f"epoch {epoch:03d}: loss={loss.item():.6f} "
                f"sim={logs['sim'].item():.4f} std={logs['std'].item():.4f} "
                f"cov={logs['cov'].item():.4f} std_mean={logs['std_mean'].item():.3f}")


    # plot loss components 
    epochs = list(range(1, epochs + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, invar_losses, label="invariance")
    plt.plot(epochs, var_losses, label="variance")
    plt.plot(epochs, cov_losses, label="covariance")
    plt.xlabel("epoch")
    plt.ylabel("loss value")
    plt.title(f"[{role.upper()}] VICReg loss components over training")
    plt.legend()
    plt.tight_layout()
    plt.show()
            
    