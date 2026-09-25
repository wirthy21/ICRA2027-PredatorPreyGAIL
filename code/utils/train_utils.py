"""
train_utils.py:
In this file are the core training functions central to the pipeline:
ES-based policy optimization, discriminator/task reward shaping, behavior-cloning pretraining,
gradient penalties, and expert-vs-generated evaluation metrics. 

References:
Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)

ES Clipping idea derived from:
Liu et al (2019) - Trust Region Evolution Strategies (https://doi.org/10.1609/aaai.v33i01.33014352)

ES Structure: Wu et al. (2025) - p.7 Algorithm 2
Sinkhorn Loss: https://www.kernel-operations.io/geomloss/
Gradient Penalty: https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead/

"""

import copy
import torch
import numpy as np
from torch import nn
from torch import autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.eval_utils import *
from utils.vec_sim_utils import *
from utils.encoder_utils import *
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.dataset_utils import pad_expert_tensors, pad_rollout_tensors

def alive_mask(tensor):
    """
    Compute a per-agent alive mask for transition-level rewards.

    Given a trajectory tensor with an "active" channel (1 = real agent/neighbor,
    0 = padded), returns a float mask indicating which agents are alive at each
    transition step.
    """
    # active channel at index -2; drop last timestep to align with transition
    return (tensor[..., 0, -2] > 0.5)[:, :-1].to(tensor.dtype)


def gradient_estimate(theta, rewards_norm, epsilons, sigma, lr, num_perturbations, rel_clip=0.01, theta_norm_ref=None):
    """
    Estimates an ES gradient step from mirrored perturbations

    Args:
        theta: 1D parameter vector to update.
        rewards_norm: Normalized rewards for each perturbation (same length as epsilons).
        epsilons: List of perturbation vectors (same length as rewards_norm).
        sigma: Standard deviation used for generating perturbations.
        lr: Learning rate.
        num_perturbations: Number of perturbations used (len(epsilons)).
        rel_clip: Maximum allowed update norm as a fraction of the reference norm.
        theta_norm_ref: Reference norm for clipping.

    Returns:
        theta_new: Updated parameter vector.
        metrics: Dict with norms and clipping info:
            - theta_norm
            - delta_raw_norm
            - delta_norm
            - max_delta_norm
            - clip_ratio
    """
    # ES gradient estimate
    grad = torch.zeros_like(theta)
    for eps, reward in zip(epsilons, rewards_norm):
        grad += eps * reward

    # raw ES update
    delta_raw = (lr / (2 * sigma**2 * num_perturbations)) * grad  

    # normalize update with clipping
    delta_raw_norm = delta_raw.norm().clamp_min(1e-12)        
    theta_norm = theta.norm().clamp_min(1e-12)

    # choose clipping basis
    if theta_norm_ref is not None:
        clip_basis = theta.new_tensor(theta_norm_ref)
    else:
        clip_basis = theta_norm
        
    # compute maximum allowed update norm
    max_delta_norm = rel_clip * clip_basis
    max_delta_norm = torch.maximum(max_delta_norm, theta.new_tensor(1e-12))

    # compute clipping ratio and apply
    clip_ratio = (max_delta_norm / delta_raw_norm).clamp(max=1.0)
    delta = delta_raw * clip_ratio 

    # apply clipped update
    theta_new = theta + delta

    return theta_new, {"theta_norm": float(theta_norm.item()),
                        "delta_raw_norm": float(delta_raw_norm.item()),
                        "delta_norm": float(delta.norm().item()),
                        "max_delta_norm": float(max_delta_norm.item()),
                        "clip_ratio": float(clip_ratio.item())}


def discriminator_reward(discriminator, gen_tensor, mode="mean", lambda_mode=None,
                         pred_tensor=None, prey_tensor=None):
    """
    Compute reward signals from a discriminator

    Supports:
      - Single-role discriminator: D(gen_tensor)
      - Predator-vs-prey discriminator: D(pred_tensor, gen_tensor)
      - Joint predator–prey discriminator: D(pred_tensor, prey_tensor)

    Args:
        discriminator: Discriminator module.
        gen_tensor: Generated trajectory tensor (B, T, agents, neigh, feat).
        mode: Reward mode ("mean", "avoid", "attack").
        lambda_mode: Weight for task-specific reward (avoid/attack) when combining
            with discriminator reward. If None, only task reward is returned.
        pred_tensor: Optional predator tensor for predator-involving discriminators.
        prey_tensor: Optional prey tensor for joint predator–prey discriminator.

    Returns:
        Depending on mode and lambda_mode:
          - mean: dis_reward
          - avoid/attack, lambda_mode=None: task_reward
          - avoid/attack, lambda_mode!=None: (combined_reward, dis_reward, task_reward)
    """
    # ----- discriminator forward pass -----
    if prey_tensor is not None:
        # joint predator–prey discriminator
        matrix = discriminator(pred_tensor, prey_tensor)
    elif pred_tensor is not None:
        # predator vs generated prey discriminator
        matrix = discriminator(pred_tensor, gen_tensor)
    else:
        # single-role discriminator
        matrix = discriminator(gen_tensor)

    # ----- compute discriminator reward, padding-aware -----
    # joint discriminator: [B, T-1]
    # single-role: [B, T-1, agents]
    if matrix.dim() == 2:
        dis_reward = matrix.mean(dim=1)
    else:
        alive = alive_mask(gen_tensor)
        dis_reward = (matrix * alive).sum(dim=(1, 2)) / alive.sum(dim=(1, 2)).clamp_min(1.0)

    # ----- mean discriminator reward only -----
    if mode == "mean":
        return dis_reward
    
    # ----- avoid mode: prey staying away from predator -----
    if mode == "avoid":
        
        # compute euclidean distances, in prey tensor first term is flag
        dx = gen_tensor[..., 1]
        dy = gen_tensor[..., 2]
        dist = torch.sqrt(dx**2 + dy**2) + 1e-8

        # distance to predator
        pred_dist = dist[:, :, :, 0] # (B, T, agents)

        # only active agents contribute to average
        alive = (gen_tensor[..., 0, -2] > 0.5).to(pred_dist.dtype)
        avoid_reward = (pred_dist * alive).sum(dim=(1, 2)) / alive.sum(dim=(1, 2)).clamp_min(1.0)

        if lambda_mode is None:
            return avoid_reward

        # avoid and discriminator reward
        reward = dis_reward + lambda_mode * avoid_reward
        return reward, dis_reward, avoid_reward


    # ----- attack mode: predator approaching nearest prey -----
    if mode == "attack":
        dx = gen_tensor[..., 0]
        dy = gen_tensor[..., 1]
        dist = torch.sqrt(dx**2 + dy**2) + 1e-8

        # mask out padded neighbors when searching for nearest prey
        active = gen_tensor[..., -2].bool()
        dist = torch.where(active, dist, dist.new_tensor(float("inf")))

        nearest_prey_dist = dist.min(dim=-1).values
        attack_reward = (-nearest_prey_dist).mean(dim=(1, 2))
        
        if lambda_mode is None:
            return attack_reward

        # combined attack + discriminator reward
        reward = dis_reward + lambda_mode * attack_reward
        return reward, dis_reward, attack_reward


def optimize_es(role, module, mode,
                discriminator, lr, 
                sigma, num_perturbations, 
                pred_policy=None, prey_policy=None,
                init_pos=None, device="cuda",
                settings_batch_env=None,
                n_prey=32,
                theta_norm_ref=None):
    """
    Perform one ES update step on a selected module (pairwise or attention) of a policy.

    Supports:
      - Prey or predator policies.
      - Single-role or joint predator–prey discriminators.
      - Optional padding to a fixed prey count and reference-norm clipping.

    Args:
        role: "prey" or "pred".
        module: "pairwise" or "attention".
        mode: Dict with keys {"mode", "lambda"} for discriminator_reward.
        discriminator: Discriminator module (single-role or joint).
        lr: Learning rate for ES.
        sigma: Perturbation std.
        num_perturbations: Number of perturbation pairs (positive/negative).
        pred_policy: Predator policy (required if role == "pred" or using joint disc).
        prey_policy: Prey policy (required if role == "prey" or using joint disc).
        init_pos: Initial positions for rollouts.
        device: Torch device.
        settings_batch_env: Environment settings for batched rollouts.
        n_prey: Number of prey to simulate before padding.
        theta_norm_ref: Optional fixed reference norm for gradient clipping.

    Returns:
        metrics: Dict with diagnostic metrics (diff_mean, diff_std, delta_norm, etc.).
    """
    # select network to optimize
    if role == "prey":
        network = prey_policy.pairwise if module == 'pairwise' else prey_policy.attention
    else:
        network = pred_policy.pairwise if module == 'pairwise' else pred_policy.attention

    theta = nn.utils.parameters_to_vector(network.parameters())

    # run perturbed rollouts
    pred_rollouts, prey_rollouts, epsilons = apply_perturbations(prey_policy, pred_policy, init_pos,
                                                role=role, module=module, device=device,
                                                sigma=sigma, num_perturbations=num_perturbations,
                                                settings_batch_env=settings_batch_env, n_prey=n_prey)

    # pad to max_prey=32 format
    # ensures shapes match what the encoder/discriminator expect
    pred_rollouts, prey_rollouts = pad_rollout_tensors(pred_rollouts, prey_rollouts, max_prey=32)

    # compute discriminator rewards
    is_joint = hasattr(discriminator, 'pred_encoder')

    if is_joint:
        # joint predator–prey discriminator
        role_tensor = pred_rollouts if role == "pred" else prey_rollouts
        dis_reward = discriminator_reward(discriminator, role_tensor,
                                          mode=mode["mode"], lambda_mode=mode["lambda"],
                                          pred_tensor=pred_rollouts, prey_tensor=prey_rollouts)
    elif role == "prey":
        # single-role prey discriminator
        dis_reward = discriminator_reward(discriminator, prey_rollouts,
                                          mode=mode["mode"], lambda_mode=mode["lambda"])
    else:
        # single-role predator discriminator
        dis_reward = discriminator_reward(discriminator, pred_rollouts,
                                          mode=mode["mode"], lambda_mode=mode["lambda"])

    # split rewards into positive and negative perturbations
    reward = dis_reward[0] if isinstance(dis_reward, tuple) else dis_reward
    reward_pos = reward[:num_perturbations]
    reward_neg = reward[num_perturbations:]

    # reward difference per perturbation pair
    diffs = (reward_pos - reward_neg).detach()

    # rank normalization of rewards
    ranks = torch.argsort(torch.argsort(diffs)).float()
    ranks_norm = (ranks - ranks.mean()) / (ranks.std() + 1e-8)

    # es parameter update with clipping
    theta_est, grad_metrics = gradient_estimate(theta, ranks_norm, epsilons, sigma, lr, num_perturbations,
                                                theta_norm_ref=theta_norm_ref)  

    # if std is too small, skip update
    if diffs.std(unbiased=False) < 1e-6:
        theta_est = theta

    # write updated parameters back to network
    nn.utils.vector_to_parameters(theta_est, network.parameters())
    
    return {"diff_mean": round(diffs.mean().item(), 6),
            "diff_std": round(diffs.std(unbiased=False).item(), 6),
            "delta_norm": round((theta_est - theta).norm().item(), 6),
            "clip_ratio": round(grad_metrics["clip_ratio"], 6),
            "delta_raw_norm": round(grad_metrics["delta_raw_norm"], 6),
            "max_delta_norm": round(grad_metrics["max_delta_norm"], 6),
            "avoid/attack reward": round(dis_reward[2].mean().item(), 6) if isinstance(dis_reward, tuple) else None}


def pretrain_policy(policy, expert_data, role=None,
                     batch_size=256, epochs=250, 
                     lr=1e-3, deterministic=True, 
                     patience=10, device='cuda'):
    """
    Pretrain a policy via behavior cloning on expert actions.

    Filters out samples corresponding to padded agents.

    Args:
        policy: Policy network (prey or predator).
        expert_data: Expert trajectories (B, T, agents, neigh, feat).
        role: "prey" or "pred" (used in logs).
        batch_size: Training batch size.
        epochs: Maximum number of training epochs.
        lr: Learning rate for Adam.
        deterministic: If True, use deterministic forward pass (mu only).
        patience: Early stopping patience (epochs without val improvement).
        device: Torch device.

    Returns:
        policy: Pretrained policy network.
    """
    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # remove the last dimension of window length
    # action calculation on transitions only, therefore last dimension has action = 0
    expert_data = expert_data[:, :-1]

    # flatten expert data so each sample is (neigh, features)
    n, frames, agents, neigh, features = expert_data.shape
    expert_data = expert_data.reshape(n * frames * agents, neigh, features)
    
    # split states and actions
    states  = expert_data[..., :-1]
    actions = expert_data[:, 0, -1]

    # drop padded agents
    # this avoids learning "no neighbors → action = 0"  
    keep = (states[..., -1].max(dim=-1).values > 0.5)
    print(f"[{role.upper()}] BC: keeping {keep.sum().item():,} / {keep.numel():,} "
          f"samples ({100*keep.float().mean():.1f}%)")
    states, actions = states[keep], actions[keep]

    # create dataset, apply train/val split
    dataset = TensorDataset(states, actions)
    val_size = int(0.2 * len(dataset))  # 80/20 split
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # prepare data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    bad_epochs = 0
    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):

        # training
        policy.train()
        epoch_train_loss = 0.0
        train_count = 0

        for states, actions in train_loader:
            states = states.to(device=device)
            exp_actions = actions.to(device=device)

            # policy forward pass
            # uses deterministic = True, so policy is trained on mu for stable training
            est_actions, weights = policy.forward(states, deterministic=deterministic)
            est_actions = est_actions.squeeze(-1)
            
            # compute MSE loss
            loss = F.mse_loss(est_actions, exp_actions)

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track avg epoch loss (weighted by batch size)
            bs = exp_actions.size(0)
            epoch_train_loss += loss.item() * bs
            train_count += bs
            
        epoch_train_loss = epoch_train_loss / max(1, train_count)
        train_losses.append(float(epoch_train_loss))

        # validation
        policy.eval()
        epoch_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device=device)
                exp_actions = actions.to(device=device)

                # policy forward pass
                # uses deterministic = True, so policy is trained on mu for stable training
                est_actions, weights = policy.forward(states, deterministic=deterministic)
                est_actions = est_actions.squeeze(-1)
                
                loss = F.mse_loss(est_actions, exp_actions)

                # track avg epoch loss (weighted by batch size)
                bs = exp_actions.size(0)
                epoch_val_loss += loss.item() * bs
                val_count += bs

        epoch_val_loss = epoch_val_loss / max(1, val_count)
        val_losses.append(float(epoch_val_loss))

        # early stopping
        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            best_state = copy.deepcopy(policy.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        # print logs every 25 epochs
        if epoch % 25 == 0:
            print(f"[{role.upper()}] Epoch {epoch}/{epochs} | train_loss:{epoch_train_loss:.6f} | val_loss:{epoch_val_loss:.6f}")

        if bad_epochs >= patience:
            print(f"[{role.upper()}] Early stopping at epoch {epoch}.")
            break

    # plot loss curves
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"[{role.upper()}] Pretrain Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # load best model 
    policy.load_state_dict(best_state)
    return policy


def calculate_metrics(pred_policy=None, prey_policy=None, init_pool=None, 
                      pred_encoder=None, prey_encoder=None,
                      exp_pred_tensor=None, exp_prey_tensor=None,
                      pred_mmd_loss=None, prey_mmd_loss=None,
                      sinkhorn_loss=None, device=None, env_settings=None,
                      n_prey=32, n_episodes=5):
    
    """
    Compute evaluation metrics for generated trajectories vs. expert trajectories

    Args:
        pred_policy: Predator policy (optional; if None, only prey metrics are computed).
        prey_policy: Prey policy.
        init_pool: Initial position pool for the environment.
        pred_encoder: Predator encoder.
        prey_encoder: Prey encoder.
        exp_pred_tensor: Expert predator trajectories.
        exp_prey_tensor: Expert prey trajectories.
        pred_mmd_loss: MMD loss module for predators.
        prey_mmd_loss: MMD loss module for prey.
        sinkhorn_loss_fn: Sinkhorn loss module.
        device: Torch device.
        env_settings: Environment settings tuple for run_env_vectorized.
        n_prey: Number of prey to simulate per episode.
        n_episodes: Number of independent episodes to generate.

    Returns:
        metrics: Dict with prey/predator MMD and Sinkhorn means and stds.
    """
    # generate pred if pred_policy is given
    n_pred = 1 if pred_policy is not None else 0

    mmd_list = []
    sinkhorn_list = []

    for _episode in range(n_episodes):
        
        # generate trajectories with current policies
        gen_pred_tensor, gen_prey_tensor = run_env_vectorized(prey_policy=prey_policy, 
                                                          pred_policy=pred_policy, 
                                                          n_prey=n_prey, n_pred=n_pred, 
                                                          step_size=env_settings[4],
                                                          max_steps=200,
                                                          prey_speed=env_settings[2],
                                                          pred_speed=env_settings[3],
                                                          max_speed_norm=env_settings[7],
                                                          area_width=env_settings[1],
                                                          area_height=env_settings[0],
                                                          max_turn=env_settings[5],
                                                          init_pool=init_pool)
        gen_pred_tensor, gen_prey_tensor = pad_rollout_tensors(
            gen_pred_tensor, gen_prey_tensor, max_prey=32)

        # Monte Carlo estimate over 100 batches
        for i in range(100 // n_episodes):
            
            # sample prey windows
            expert_prey_batch, _ = sample_data(exp_prey_tensor, batch_size=10, window_len=10)
            expert_prey_batch = expert_prey_batch.to(device)
            generative_prey_batch, _ = sample_data(gen_prey_tensor, batch_size=10, window_len=10)
            generative_prey_batch = generative_prey_batch.to(device)

            # compute MMD metric for prey
            with torch.no_grad():
                mmd_prey_metric = prey_mmd_loss.forward(expert_prey_batch, generative_prey_batch)

            # compute Sinkhorn on prey transition embeddings
            _, trans_exp_prey = prey_encoder(expert_prey_batch[..., :-1])
            _, trans_gen_prey = prey_encoder(generative_prey_batch[..., :-1])
            batch, frames, agents, dim = trans_exp_prey.shape
            prey_x = trans_exp_prey.reshape(batch * frames, agents, dim)
            prey_y = trans_gen_prey.reshape(batch * frames, agents, dim)
            sinkhorn_prey = sinkhorn_loss(prey_x, prey_y)

            if n_pred > 0:
                # sample pred windows
                expert_pred_batch, _ = sample_data(exp_pred_tensor, batch_size=20, window_len=10)
                expert_pred_batch = expert_pred_batch.to(device)
                generative_pred_batch, _ = sample_data(gen_pred_tensor, batch_size=20, window_len=10)
                generative_pred_batch = generative_pred_batch.to(device)

                # compute MMD metric for pred
                with torch.no_grad():
                    mmd_pred_metric = pred_mmd_loss.forward(expert_pred_batch, generative_pred_batch)

                # compute Sinkhorn on pred transition embeddings
                _, trans_exp_pred = pred_encoder(expert_pred_batch[..., :-1])
                _, trans_gen_pred = pred_encoder(generative_pred_batch[..., :-1])
                batch, frames, agents, dim = trans_exp_pred.shape
                pred_x = trans_exp_pred.reshape(batch * frames, agents, dim)
                pred_y = trans_gen_pred.reshape(batch * frames, agents, dim)
                sinkhorn_pred = sinkhorn_loss(pred_x, pred_y)

                mmd_list.append((mmd_prey_metric.item(), mmd_pred_metric.item()))
                sinkhorn_list.append((sinkhorn_prey.mean().item(), sinkhorn_pred.mean().item()))
            else:
                mmd_list.append((mmd_prey_metric.item(), None))
                sinkhorn_list.append((sinkhorn_prey.mean().item(), None))

    # aggregate prey mmd metrics
    mmd_prey_mean = np.mean([mmd[0] for mmd in mmd_list])
    mmd_prey_std  = np.std([mmd[0] for mmd in mmd_list], ddof=1)

    # aggregate prey sinkhorn metrics
    sinkhorn_prey_mean = np.mean([s[0] for s in sinkhorn_list])
    sinkhorn_prey_std  = np.std([s[0] for s in sinkhorn_list], ddof=1)

    if n_pred > 0:
        # aggregate pred mmd metrics
        mmd_pred_mean = np.mean([mmd[1] for mmd in mmd_list])
        mmd_pred_std  = np.std([mmd[1] for mmd in mmd_list], ddof=1)

        # aggregate pred sinkhorn metrics
        sinkhorn_pred_mean = np.mean([s[1] for s in sinkhorn_list])
        sinkhorn_pred_std  = np.std([s[1] for s in sinkhorn_list], ddof=1)
    else:
        mmd_pred_mean = None
        mmd_pred_std = None
        sinkhorn_pred_mean = None
        sinkhorn_pred_std = None

    return {"mmd_prey_mean": mmd_prey_mean,
            "mmd_prey_std": mmd_prey_std,
            "mmd_pred_mean": mmd_pred_mean,
            "mmd_pred_std": mmd_pred_std,
            "sinkhorn_prey_mean": sinkhorn_prey_mean,
            "sinkhorn_prey_std": sinkhorn_prey_std,
            "sinkhorn_pred_mean": sinkhorn_pred_mean,
            "sinkhorn_pred_std": sinkhorn_pred_std}


# https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead/
def gradient_penalty(discriminator, expert_traj, generated_traj):
    """
    Wasserstein GAIL gradient penalty
    Enforces the discriminator to be 1-Lipschitz by penalizing the gradient norm

    Args:
        discriminator: Discriminator D(x) that takes a single trajectory tensor.
        expert_traj: Expert trajectories.
        generated_traj: Generated (policy) trajectories.

    Returns:
        gp: Gradient penalty term
    """
    batch_size = expert_traj.size(0)

    # random weight term for interpolation between expert and generated data
    eps_shape = [batch_size] + [1] * (expert_traj.dim() - 1)
    eps = torch.rand(*eps_shape, device=expert_traj.device)
    
    # interpolation between expert data and generated data
    interpolation = eps * expert_traj + (1 - eps) * generated_traj
    interpolation.requires_grad_(True)
    
    # get logits for interpolated images
    interp_logits = discriminator(interpolation)
    grad_outputs = torch.ones_like(interp_logits)
    
    # compute gradients
    gradients = autograd.grad(outputs=interp_logits,
                inputs=interpolation,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True)[0]
    
    # compute and return gradient norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)


def gradient_penalty_joint(discriminator, expert_pred, expert_prey, policy_pred, policy_prey):
    """
    Wasserstein GAIL gradient penalty for joint discriminator
    
    Args:
        discriminator: Joint discriminator D(pred, prey).
        expert_pred: Expert predator trajectories.
        expert_prey: Expert prey trajectories.
        policy_pred: Generated (policy) predator trajectories.
        policy_prey: Generated (policy) prey trajectories.

    Returns:
        gp: Gradient penalty term
    """
    batch_size = expert_pred.size(0)

    # same alpha for both branches — single interpolation point in joint space
    eps_shape = [batch_size] + [1] * (expert_pred.dim() - 1)
    eps = torch.rand(*eps_shape, device=expert_pred.device)

    # interpolate pred and prey separately but with the same eps
    interp_pred = eps * expert_pred + (1 - eps) * policy_pred
    interp_prey = eps * expert_prey + (1 - eps) * policy_prey

    interp_pred.requires_grad_(True)
    interp_prey.requires_grad_(True)

    # joint forward pass with both interpolated tensors
    interp_logits = discriminator(interp_pred, interp_prey)
    grad_outputs = torch.ones_like(interp_logits)

    # compute gradients w.r.t. both inputs
    gradients = autograd.grad(
        outputs=interp_logits,
        inputs=[interp_pred, interp_prey],
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )

    # flatten and concatenate both gradient tensors, then compute joint norm
    grad_pred = gradients[0].view(batch_size, -1)
    grad_prey = gradients[1].view(batch_size, -1)
    joint_grad = torch.cat([grad_pred, grad_prey], dim=1)

    grad_norm = joint_grad.norm(2, dim=1)
    return torch.mean((grad_norm - 1) ** 2)


def compute_wasserstein_loss(expert_scores, policy_scores, lambda_gp, gp):
    """
    Compute the Wasserstein discriminator loss with gradient penalty

    Input: expert_scores, policy_scores, lambda_gp, gp
    Output: loss, loss with gradient penalty
    """
    loss = policy_scores.mean() - expert_scores.mean()
    loss_gp = loss + lambda_gp * gp
    return loss, loss_gp


def sliding_window(tensor, window_size=10):
    """
    Create overlapping windows from a tensor

    Input: tensor of shape (frames, agents, neigh, feat)
    Output: tensor of shape (num_windows, window_size, agents, neigh, feat)
    """
    sequences = []
    # iterate over the tensor to create windows
    for start in range(0, tensor.size(0) - window_size + 1):
        end = start + window_size
        sequences.append(tensor[start:end])
    return torch.stack(sequences)
