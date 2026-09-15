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


"""
References:
Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)

ES Clipping idea derived from:
Liu et al (2019) - Trust Region Evolution Strategies (https://doi.org/10.1609/aaai.v33i01.33014352)


ES Structure: Wu et al. (2025) - p.7 Algorithm 2
Sinkhorn Loss: https://www.kernel-operations.io/geomloss/
Gradient Penalty: https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead/
"""


def gradient_estimate(theta, rewards_norm, epsilons, sigma, lr, num_perturbations, rel_clip=0.01):
    """
    Estimates an ES gradient step from mirrored perturbations

    Input: parameter vector, normalized rewards, perturbations, training settings
    Output: updated parameter vector, gradient metrics
    """

    # compute gradient estimate
    grad = torch.zeros_like(theta)
    for eps, reward in zip(epsilons, rewards_norm):
        grad += eps * reward

    # raw ES update
    delta_raw = (lr / (2 * sigma**2 * num_perturbations)) * grad  

    # normalize update with clipping
    delta_raw_norm = delta_raw.norm().clamp_min(1e-12)        
    theta_norm = theta.norm().clamp_min(1e-12)

    # compute maximum allowed update norm
    max_delta_norm = rel_clip * theta_norm
    max_delta_norm = torch.maximum(max_delta_norm, theta.new_tensor(1e-12))

    # compute clipping ratio and apply
    clip_ratio = (max_delta_norm / delta_raw_norm).clamp(max=1.0)
    delta = delta_raw * clip_ratio 

    # apply update
    theta_new = theta + delta

    return theta_new, {"theta_norm": float(theta_norm.item()),
                        "delta_raw_norm": float(delta_raw_norm.item()),
                        "delta_norm": float(delta.norm().item()),
                        "max_delta_norm": float(max_delta_norm.item()),
                        "clip_ratio": float(clip_ratio.item())}


def discriminator_reward(discriminator, gen_tensor, mode="mean", lambda_mode=None):
    """
    Computes rewards from discriminator outputs
    Can be mean reward, avoid predator reward, or attack prey reward and their combinations

    Input: discriminator, generated trajectory tensor, reward mode, lambda weights
    Output: computed rewards
    """

    # get discriminator output matrix
    matrix = discriminator(gen_tensor)

    # compute mean discriminator reward
    dis_reward = matrix.mean(dim=(1, 2))

    # mean discriminator reward only
    if mode == "mean":
        return (dis_reward)
    
    
    # avoid mode only
    if mode == "avoid" and lambda_mode is None:
        # compute euclidean distances, in prey tensor first term is flag
        dx = gen_tensor[..., 1]
        dy = gen_tensor[..., 2]
        dist = torch.sqrt(dx**2 + dy**2) + 1e-8

        # avoid reward based on distance to predator
        pred_dist = dist[:, :, :, 0]
        avoid_reward = pred_dist.mean(dim=(1, 2))
        return (avoid_reward)


    # avoid + discriminator reward
    if mode == "avoid" and lambda_mode is not None:
        # compute euclidean distances, in prey tensor first term is flag
        dx = gen_tensor[..., 1]
        dy = gen_tensor[..., 2]
        dist = torch.sqrt(dx**2 + dy**2) + 1e-8

        # avoid reward based on distance to predator
        pred_dist = dist[:, :, :, 0]
        avoid_reward = pred_dist.mean(dim=(1, 2))

        # compute combined rewards
        reward = dis_reward + lambda_mode * avoid_reward
        return (reward, dis_reward, avoid_reward)
    

    # attack mode only
    if mode == "attack" and lambda_mode is None:
        # compute euclidean distances
        dx = gen_tensor[..., 0]
        dy = gen_tensor[..., 1]
        dist = torch.sqrt(dx**2 + dy**2) + 1e-8

        # attack reward based on nearest prey
        nearest_prey_dist = dist.min(dim=-1).values
        attack_reward = (-nearest_prey_dist).mean(dim=(1, 2))
        return (attack_reward)
    

    # attack + discriminator reward
    if mode == "attack" and lambda_mode is not None:
        # compute euclidean distances
        dx = gen_tensor[..., 0]
        dy = gen_tensor[..., 1]
        dist = torch.sqrt(dx**2 + dy**2) + 1e-8

        # attack reward based on nearest prey
        nearest_prey_dist = dist.min(dim=-1).values
        attack_reward = (-nearest_prey_dist).mean(dim=(1, 2))

        # compute combined rewards
        reward = dis_reward + lambda_mode * attack_reward
        return (reward, dis_reward, attack_reward)


def optimize_es(role, module, mode,
                discriminator, lr, 
                sigma, num_perturbations, 
                pred_policy=None, prey_policy=None,
                init_pos=None, device="cuda",
                settings_batch_env=None):
    
    """
    Runs a ES update step on the selected module of the policy network (PIN or AN)

    Input: policy, discriminator, training settings
    Output: updated policy, training metrics
    """

    # select network to optimize
    if role == "prey":
        network = prey_policy.pairwise if module == 'pairwise' else prey_policy.attention
    else:
        network = pred_policy.pairwise if module == 'pairwise' else pred_policy.attention

    # convert parameters to an single vector
    theta = nn.utils.parameters_to_vector(network.parameters())

    # run rollouts with +eps and -eps perturbations
    pred_rollouts, prey_rollouts, epsilons = apply_perturbations(prey_policy, pred_policy, init_pos,
                                role=role, module=module, device=device,
                                sigma=sigma, num_perturbations=num_perturbations,
                                settings_batch_env=settings_batch_env)
    
    # compute rewards from discriminator
    if role == "prey":
        dis_reward = discriminator_reward(discriminator, prey_rollouts, mode=mode["mode"], lambda_mode=mode["lambda"])
    else:
        dis_reward = discriminator_reward(discriminator, pred_rollouts, mode=mode["mode"], lambda_mode=mode["lambda"])

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
    theta_est, grad_metrics = gradient_estimate(theta, ranks_norm, epsilons, sigma, lr, num_perturbations)

    # if std is too small, do not update (random walk)
    if diffs.std(unbiased=False) < 1e-6:
        theta_est = theta

    # write updated parameters back to network
    nn.utils.vector_to_parameters(theta_est, network.parameters())
    
    # return metrics, useful for stabilization and debugging
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
    Pretraining of policy network with behavior cloning on actions from expert data

    Input: policy, expert data, training settings
    Output: pretrained policy
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

    # create tensor dataset, apply train/val split
    dataset = TensorDataset(states, actions)
    val_size = int(0.2 * len(dataset))  # 80/20 split
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # prepare data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    bad_epochs = 0
    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training
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

        # Validation
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

    # load best model state
    policy.load_state_dict(best_state)
    return policy


def calculate_metrics(pred_policy=None, prey_policy=None, init_pool=None, 
                      pred_encoder=None, prey_encoder=None,
                      exp_pred_tensor=None, exp_prey_tensor=None,
                      pred_mmd_loss=None, prey_mmd_loss=None,
                      sinkhorn_loss=None, device=None, env_settings=None):
    
    """
    Compute evaluation metrics for generated trajectories vs. expert trajectories

    Input: policies, encoders, expert tensors, loss functions, device, env settings
    Output: performance metrics, MMD and Sinkhorn approximation
    """

    # generate pred if pred_policy is given
    n_pred = 1 if pred_policy is not None else 0

    # generate trajectories with current policies
    gen_pred_tensor, gen_prey_tensor = run_env_vectorized(prey_policy=prey_policy, 
                                                          pred_policy=pred_policy, 
                                                          n_prey=32, n_pred=n_pred, 
                                                          step_size=env_settings[4],
                                                          max_steps=200,
                                                          prey_speed=env_settings[2],
                                                          pred_speed=env_settings[3],
                                                          area_width=env_settings[1],
                                                          area_height=env_settings[0],
                                                          max_turn=env_settings[5],
                                                          init_pool=init_pool)
    
    mmd_list = []
    sinkhorn_list = []

    # Monte Carlo estimate over 100 batches
    for i in range(100):
        # sample prey windows
        expert_prey_batch = sample_data(exp_prey_tensor, batch_size=10, window_len=10).to(device)
        generative_prey_batch = sample_data(gen_prey_tensor, batch_size=10, window_len=10).to(device)

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
            expert_pred_batch = sample_data(exp_pred_tensor, batch_size=20, window_len=10).to(device)
            generative_pred_batch = sample_data(gen_pred_tensor, batch_size=20, window_len=10).to(device)

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

    Input: discriminator, expert & generated trajectories
    Output: gradient penalty
    """
    batch_size = expert_traj.size(0)

    # random weight term for interpolation between expert and generated data
    eps_shape = [batch_size] + [1] * (expert_traj.dim() - 1)
    eps = torch.rand(*eps_shape, device=expert_traj.device)
    
    # Interpolation between expert data and generated data
    interpolation = eps * expert_traj + (1 - eps) * generated_traj
    interpolation.requires_grad_(True)
    
    # get logits for interpolated images
    interp_logits = discriminator(interpolation)
    grad_outputs = torch.ones_like(interp_logits)
    
    # Compute gradients
    gradients = autograd.grad(outputs=interp_logits,
                inputs=interpolation,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True)[0]
    
    # Compute and return gradient Norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
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
    # Iterate over the tensor to create windows
    for start in range(0, tensor.size(0) - window_size + 1):
        end = start + window_size
        sequences.append(tensor[start:end])
    return torch.stack(sequences)
