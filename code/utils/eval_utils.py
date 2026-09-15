import sys, os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from utils.encoder_utils import *

"""
References:
Sinkhorn: https://www.kernel-operations.io/geomloss/
MMD: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py

Polarization & Angular Momentum: Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)
Degree of Sparsity: Li et al. (2023) - Predator–prey survival pressure is sufficient to evolve swarming behaviors (https://doi.org/10.48550/arXiv.2308.12624)
Escape Alignment: Bartashevich et al. (2024) - Collective anti-predator escape manoeuvres through optimal attack and avoidance strategies (https://doi.org/10.1038/s42003-024-07267-2)

"""

def get_expert_values(exp_pred_tensor=None, exp_prey_tensor=None, 
                      prey_mmd_loss=None, pred_mmd_loss=None, 
                      prey_encoder=None, pred_encoder=None, 
                      sinkhorn_loss=None):

    """
    Calculates target values for the MMD and Sinkhorn metrics using expert data.
    The targets are estimated by repeatedly sampling random trajectory windows from the expert dataset and comparing expert samples against each other.
    Monte Carlo estimation over 500 iterations.

    Input: expert tensors, loss functions, encoders
    Output: means and stds for MMD and Sinkhorn expert values for prey and predator
    """

    prey_mmd_list = []
    pred_mmd_list = []

    prey_sinkhorn_list = []
    pred_sinkhorn_list = []

    for i in tqdm(range(500)):
        # sample expert data
        expert_prey_batch1 = sample_data(exp_prey_tensor, batch_size=10, window_len=10)
        expert_prey_batch2 = sample_data(exp_prey_tensor, batch_size=10, window_len=10)

        # compute prey MMD
        prey_mmd = prey_mmd_loss(expert_prey_batch1, expert_prey_batch2).item()
        prey_mmd_list.append(prey_mmd)

        # compute prey Sinkhorn on transition feature
        _, trans_exp_prey = prey_encoder(expert_prey_batch1[..., :-1])
        _, trans_gen_prey = prey_encoder(expert_prey_batch2[..., :-1])
        batch, frames, agents, dim = trans_exp_prey.shape
        prey_x = trans_exp_prey.reshape(batch * frames, agents, dim)
        prey_y = trans_gen_prey.reshape(batch * frames, agents, dim)
        sinkhorn_prey = sinkhorn_loss(prey_x, prey_y)
        prey_sinkhorn_list.append(sinkhorn_prey.mean().item())

        # if predator data is provided compute MMD and Sinkhorn
        if exp_pred_tensor is not None:
            # sample expert data
            expert_pred_batch1 = sample_data(exp_pred_tensor, batch_size=10, window_len=10)
            expert_pred_batch2 = sample_data(exp_pred_tensor, batch_size=10, window_len=10)

            # compute pred MMD
            pred_mmd = pred_mmd_loss(expert_pred_batch1, expert_pred_batch2).item()
            pred_mmd_list.append(pred_mmd)

            # compute pred Sinkhorn on transition feature
            _, trans_exp_pred = pred_encoder(expert_pred_batch1[..., :-1])
            _, trans_gen_pred = pred_encoder(expert_pred_batch2[..., :-1])
            batch, frames, agents, dim = trans_exp_pred.shape
            pred_x = trans_exp_pred.reshape(batch * frames, agents, dim)
            pred_y = trans_gen_pred.reshape(batch * frames, agents, dim)
            sinkhorn_pred = sinkhorn_loss(pred_x, pred_y)
            pred_sinkhorn_list.append(sinkhorn_pred.mean().item())

    # print prey results
    print(f"\nExpert Prey MMD: {np.mean(prey_mmd_list)} ± {np.std(prey_mmd_list)}")
    print(f"Expert Prey Sinkhorn: {np.mean(prey_sinkhorn_list)} ± {np.std(prey_sinkhorn_list)}")
    
    if exp_pred_tensor is not None:
        # print pred results
        print(f"\nExpert Pred MMD: {np.mean(pred_mmd_list)} ± {np.std(pred_mmd_list)}")
        print(f"Expert Pred Sinkhorn: {np.mean(pred_sinkhorn_list)} ± {np.std(pred_sinkhorn_list)}")

    # compute means and stds
    mmd_means = (np.mean(prey_mmd_list), np.mean(pred_mmd_list) if exp_pred_tensor is not None else None)
    mmd_stds = (np.std(prey_mmd_list), np.std(pred_mmd_list) if exp_pred_tensor is not None else None)

    # compute means and stds
    sinkhorn_means = (np.mean(prey_sinkhorn_list), np.mean(pred_sinkhorn_list) if exp_pred_tensor is not None else None)
    sinkhorn_stds = (np.std(prey_sinkhorn_list), np.std(pred_sinkhorn_list) if exp_pred_tensor is not None else None)
    
    # return results
    return mmd_means, mmd_stds, sinkhorn_means, sinkhorn_stds



def plot_train_metrics(disc_metrics, dis_balance_factor, role="prey", save_dir=None):
    """
    Plots the training discriminator metrics over generations

    Input: discriminator metrics, balance factor, role, save directory
    """

    # prepare dataframe
    disc_dicts = [x[0] if role == "prey" else x[1] for x in disc_metrics]
    disc_df = pd.DataFrame(disc_dicts)
    disc_df["gen"] = (disc_df.index // dis_balance_factor) + 1
    disc_mean = disc_df.groupby("gen").mean(numeric_only=True).sort_index()

    # generations
    gens = np.arange(len(disc_mean))

    # compute score differences and absolute scores
    score_diffs = (disc_mean["expert_score_mean"] - disc_mean["policy_score_mean"]).abs().to_numpy()
    abs_scores = disc_mean[["expert_score_mean", "policy_score_mean"]].to_numpy()

    # plot absolute scores
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    axes[0].plot(gens, abs_scores[:, 0], label="Expert Score")
    axes[0].plot(gens, abs_scores[:, 1], label="Policy Score")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"{role.upper()} Absolute Scores over Generations")
    axes[0].legend()
    axes[0].grid(True)

    # plot score differences
    axes[1].plot(gens, score_diffs, color="orange")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Score Difference")
    axes[1].set_title(f"{role.upper()} Score Difference over Generations")
    axes[1].grid(True)

    plt.tight_layout()

    # save plot
    plot_path = save_dir / f"disc_{role}_metrics.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_eval_metrics(metrics_list, 
                      mmd_means=None, sinkhorn_means=None, 
                      mmd_stds=None, sinkhorn_stds=None,
                      role="prey", save_dir=None,
                      eval_steps=5, max_steps=1500):
    
    """
    Plots Sinkhorn and MMD metrics over evaluation generations, comparing against expert target value

    Input: Sinkhorn and MMD metrics, role, save directory, eval steps, max steps
    """
    
    # calculate length of x-axis
    xs = np.arange(len(metrics_list)) * eval_steps

    if role == "prey":
        # extract prey metrics
        mmd_mean = np.array([m["mmd_prey_mean"] for m in metrics_list], dtype=float)
        mmd_std  = np.array([m["mmd_prey_std"]  for m in metrics_list], dtype=float)
        sk_mean  = np.array([m["sinkhorn_prey_mean"] for m in metrics_list], dtype=float)
        sk_std   = np.array([m["sinkhorn_prey_std"]  for m in metrics_list], dtype=float)
        exp_mmd_mean = mmd_means[0]
        exp_mmd_std = mmd_stds[0]
        exp_sinkhorn_mean = sinkhorn_means[0]
        exp_sinkhorn_std = sinkhorn_stds[0]
    else:
        # extract pred metrics
        mmd_mean = np.array([m["mmd_pred_mean"] for m in metrics_list], dtype=float)
        mmd_std  = np.array([m["mmd_pred_std"]  for m in metrics_list], dtype=float)
        sk_mean  = np.array([m["sinkhorn_pred_mean"] for m in metrics_list], dtype=float)
        sk_std   = np.array([m["sinkhorn_pred_std"]  for m in metrics_list], dtype=float)
        exp_mmd_mean = mmd_means[1]
        exp_mmd_std = mmd_stds[1]
        exp_sinkhorn_mean = sinkhorn_means[1]
        exp_sinkhorn_std = sinkhorn_stds[1]

    # plot MMD metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    axes[0].plot(xs, mmd_mean, label=f"{role.upper()} MMD")
    axes[0].fill_between(xs, mmd_mean - mmd_std, mmd_mean + mmd_std, alpha=0.2)
    axes[0].axhline(exp_mmd_mean, color="red", linewidth=2, label="Expert MMD Mean ± Std")
    axes[0].fill_between(xs, exp_mmd_mean - exp_mmd_std, exp_mmd_mean + exp_mmd_std, alpha=0.2)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("MMD")
    axes[0].set_title(f"{role.upper()} MMD (Mean ± Std)")
    axes[0].set_xlim(0, max_steps)
    axes[0].grid(True)

    # plot Sinkhorn metrics
    axes[1].plot(xs, sk_mean, label=f"{role.upper()} Sinkhorn")
    axes[1].fill_between(xs, sk_mean - sk_std, sk_mean + sk_std, alpha=0.2)
    axes[1].axhline(exp_sinkhorn_mean, color="red", linewidth=2, label="Expert Sinkhorn Mean ± Std")
    axes[1].fill_between(xs, exp_sinkhorn_mean - exp_sinkhorn_std, exp_sinkhorn_mean + exp_sinkhorn_std, alpha=0.2)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Sinkhorn")
    axes[1].set_title(f"{role.upper()} Sinkhorn (Mean ± Std)")
    axes[1].set_xlim(0, max_steps)
    axes[1].grid(True)

    plt.tight_layout()

    # save plot
    filename = f"eval_{role}_metrics.png"
    plot_path = save_dir / filename
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_es_metrics(policy_metrics, role="prey", save_dir=None):
    """
    Plots gradient step size and reward-diff std over generations

    Input: policy metrics, role, save directory
    """

    # generations and metrics
    gens = np.arange(len(policy_metrics))
    pms = [pm[role] for pm in policy_metrics]

    # get gradient step size for modular networks
    pin_delta_norm = np.array([pm[0].get("delta_norm", np.nan) for pm in pms], dtype=float)
    an_delta_norm  = np.array([pm[1].get("delta_norm", np.nan) for pm in pms], dtype=float)

    # get reward-diff std for modular networks
    pin_diff_std = np.array([pm[0].get("diff_std", np.nan) for pm in pms], dtype=float)
    an_diff_std  = np.array([pm[1].get("diff_std", np.nan) for pm in pms], dtype=float)

    # plot gradient step size
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    axes[0].plot(gens, pin_delta_norm, label="PIN delta_norm")
    axes[0].plot(gens, an_delta_norm, label="AN delta_norm")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("delta_norm")
    axes[0].set_title(f"{role.upper()} ES step size per generation")
    axes[0].legend()
    axes[0].grid(True)

    # plot reward-diff std
    axes[1].plot(gens, pin_diff_std, label="PIN diff_std")
    axes[1].plot(gens, an_diff_std, label="AN diff_std")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("diff_std")
    axes[1].set_title(f"{role.upper()} Reward-diff std per generation")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    # save plot
    plot_path = save_dir / f"es_{role}_policy_metrics.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.show()


def compute_polarization(vx, vy):
    """
    Computes the polarization score

    Input: velocities in x and y directions
    Output: polarization score
    """

    # stack velocities and normalize
    stacked_vs = np.stack([vx, vy], axis=1)
    norms = np.linalg.norm(stacked_vs, axis=1, keepdims=True)
    norm_vs = stacked_vs / norms

    prey_vs = norm_vs[1:] # prey_only
    mean_vs = prey_vs.mean(axis=0) # mean velocity vector
    polarization_score = np.linalg.norm(mean_vs) # polarization score
    return polarization_score


def compute_angular_momentum(x, y, vx, vy):
    """
    Computes the angular momentum score

    Input: x and y coordinates, velocities in x and y directions
    Output: angular momentum score
    """

    # stack positions and velocities
    positions = np.stack([x, y], axis=1)
    velocities = np.stack([vx, vy], axis=1)

    # prey-only
    prey_pos = positions[1:]
    prey_vel = velocities[1:]

    # center of mass,angular momentum around it
    center = prey_pos.mean(axis=0)
    rel_pos = prey_pos - center

    # normalize velocities
    prey_speed  = np.linalg.norm(prey_vel, axis=1, keepdims=True)
    prey_velocity = prey_vel / (prey_speed + 1e-8)

    angular_momentum = rel_pos[:, 0] * prey_velocity[:, 1] - rel_pos[:, 1] * prey_velocity[:, 0] # cross product z-component
    return np.abs(angular_momentum.mean())


def degree_of_sparsity(xs, ys):
    """
    Computes the degree of sparsity (nearest-neighbor distance)

    Input: x and y coordinates
    Output: mean nearest neighbor distance
    """

    # stack positions
    positions = np.stack([xs, ys], axis=1)

    # calculate pairwise squared distances
    diff = positions[:, None, :] - positions[None, :, :]
    dist_squared = np.sum(diff * diff, axis=-1)

    # exclude self-distances
    np.fill_diagonal(dist_squared, np.inf)

    # nearest-neighbor distances
    nn_dists = np.sqrt(np.min(dist_squared, axis=1))
    return float(nn_dists.mean())


def distance_to_predator(xs, ys):
    """
    Computes the distance to predator

    Input: x and y coordinates
    Output: distance to predator
    """

    # separate predator and prey positions
    predator_pos = np.array([xs[0], ys[0]])
    prey_pos = np.stack([xs[1:], ys[1:]], axis=1)

    # compute center of prey and distance to predator
    center = prey_pos.mean(axis=0)
    return float(np.linalg.norm(center - predator_pos))


def pred_distance_to_nearest_prey(xs, ys):
    """
    Computes predator distance to nearest prey

    Input: x and y coordinates
    Output: nearest prey distance to predator
    """

    # extract predator and prey positions
    pred_pos = np.array([xs[0], ys[0]], dtype=np.float32)
    prey_pos = np.stack([xs[1:], ys[1:]], axis=1).astype(np.float32)

    # compute distances
    dists = np.linalg.norm(prey_pos - pred_pos[None, :], axis=1)
    return float(np.min(dists))


def escape_alignment(xs, ys, vxs, vys):
    """
    Computes the escape alignment

    Input: x and y coordinates, velocities in x and y directions
    Output: escape alignment
    """

    # separate predator and prey positions and velocities
    pred_pos = np.array([xs[0], ys[0]])
    prey_pos = np.stack([xs[1:], ys[1:]], axis=1)
    prey_vel = np.stack([vxs[1:], vys[1:]], axis=1)

    # compute escape directions and normalize
    escape_dir = prey_pos - pred_pos
    escape_dir_norm = np.linalg.norm(escape_dir, axis=1, keepdims=True)
    escape_dir = escape_dir / (escape_dir_norm + 1e-8)

    # normalize prey velocities
    vel_norm = np.linalg.norm(prey_vel, axis=1, keepdims=True)
    prey_vel_normed = prey_vel / (vel_norm + 1e-8)

    # compute alignment
    alignment = np.mean(np.sum(prey_vel_normed * escape_dir, axis=1))
    return float(alignment)
