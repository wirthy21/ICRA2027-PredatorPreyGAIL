import torch
import numpy as np
from math import *
from numpy.linalg import *
from torch.func import functional_call, vmap
from collections import OrderedDict


"""
References:
Env Structure: https://github.com/hossein-haeri/couzin_swarm_model/blob/master/swarm_pray_predator.py
Wall Enforcement: https://github.com/mdodsworth/pyglet-boids/blob/master/boids/boid.py

Intro to vec envs: https://www.youtube.com/watch?v=Kv5HRTFuo6M
Really informative article: https://medium.com/data-science/vectorize-and-parallelize-rl-environments-with-jax-q-learning-at-the-speed-of-light-49d07373adf5

vmap: https://docs.pytorch.org/docs/stable/generated/torch.vmap.html
"""


def velocity_from_theta(theta, speed):
    """
    Gets velocity vector from (updated) heading angle and speed

    Input: theta, speed
    Output: velocity vector
    """

    vx = torch.cos(theta) * speed
    vy = torch.sin(theta) * speed
    return torch.stack([vx, vy], dim=-1)


def apply_turnrate(theta, action, max_turn):
    """
    Updates heading angle based on action and maximum turn rate
    
    Input: theta, action, max_turn
    Output: updated theta
    """

    # maps normalized action [0,1] to turn rate [-max_turn, max_turn]
    dtheta = (action - 0.5) * 2.0 * max_turn
    theta = theta + dtheta # update theta
    return (theta + torch.pi) % (2*torch.pi) - torch.pi # wrap to [-pi, pi]


def enforce_walls(pos, theta, area_width, area_height):
    """
    Check and handle collisions with the environment boundaries
    tensor-based on batches
    """

    # check for collisions with walls
    bounced_x = (pos[..., 0] < 0) | (pos[..., 0] > area_width)
    bounced_y = (pos[..., 1] < 0) | (pos[..., 1] > area_height)

    # clamp positions to be within bounds
    pos[..., 0] = pos[..., 0].clamp(0.0, float(area_width))
    pos[..., 1] = pos[..., 1].clamp(0.0, float(area_height))

    # reflect heading angles upon collision
    theta = torch.where(bounced_x, torch.pi - theta, theta)
    theta = torch.where(bounced_y, -theta, theta)

    # wrap angles to [-pi, pi]
    theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi
    return pos, theta


def get_state_tensors(prey_log_step, pred_log_step, n_pred=1, 
                      area_width=50, area_height=50, 
                      max_speed_norm=5, neigh_idx=None):
    
    """
    Converts logs to expert feature tensors, tensor-based version

    Input: pred and prey logs, n_pred, area size, max speed for normalization
    Output: predator & prey tensor and metrics dict
    """

    # combine predator and prey logs
    device = prey_log_step.device
    combined = torch.cat([pred_log_step, prey_log_step], dim=1)
    batch, n_agents, _ = combined.shape
    n_neigh = n_agents - 1

    # extract positions, velocities, and directions
    xs, ys   = combined[..., 0], combined[..., 1]
    vxs, vys = combined[..., 2], combined[..., 3]
    cos_t, sin_t = combined[..., 4], combined[..., 5]

    # scale positions to [0,1]
    xs_scaled = xs / float(area_width)
    ys_scaled = ys / float(area_height)

    # compute pairwise distances
    dx = xs_scaled.unsqueeze(1) - xs_scaled.unsqueeze(2)
    dy = ys_scaled.unsqueeze(1) - ys_scaled.unsqueeze(2)

    # compute relative velocities in the agent's heading direction
    rel_vx = cos_t.unsqueeze(2) * vxs.unsqueeze(1) + sin_t.unsqueeze(2) * vys.unsqueeze(1)
    rel_vy = -sin_t.unsqueeze(2) * vxs.unsqueeze(1) + cos_t.unsqueeze(2) * vys.unsqueeze(1)

    # normalize relative velocities (same as expert)
    rel_vx = torch.clamp(rel_vx, -max_speed_norm, max_speed_norm) / max_speed_norm
    rel_vy = torch.clamp(rel_vy, -max_speed_norm, max_speed_norm) / max_speed_norm

    # stack features
    features = torch.stack([dx, dy, rel_vx, rel_vy], dim=-1)

    # gather neighbor features based on neigh_idx
    gather_idx = neigh_idx.view(1, n_agents, n_neigh, 1).expand(batch, n_agents, n_neigh, 4).to(device)
    neigh = features.gather(dim=2, index=gather_idx)

    # split into predator and prey tensors
    pred_tensor = neigh[:, :n_pred]
    prey_tensor = neigh[:, n_pred:]

    if n_pred > 0:
        # create mask for predator neighbors in prey tensor
        mask_pred_neigh = (neigh_idx < n_pred).to(device)
        mask_pred_neigh = mask_pred_neigh.view(1, n_agents, n_neigh, 1)
        prey_mask = mask_pred_neigh[:, n_pred:]
        prey_mask = prey_mask.expand(batch, -1, -1, -1).to(prey_tensor.dtype)

        # concatenate mask to prey tensor
        prey_tensor = torch.cat([prey_mask, prey_tensor], dim=-1)

    return pred_tensor, prey_tensor



def init_positions_in_env(init_pool, area_width=50, area_height=50, device="cuda"):
    """
    Initializes agents from a given pool of states

    Input: init_pool, area size, device
    Output: positions, headings
    """

    # sample random initial state
    steps, agents, coordinates = init_pool.shape
    idx = torch.randint(steps, (), device=device)
    sample = init_pool[idx].to(device)

    # get positions and headings
    positions = sample[:, :2]
    theta = sample[:, 2]

    # center the swarm in the environment
    center_env = torch.tensor([area_width * 0.5, area_height * 0.5], dtype=positions.dtype, device=device).view(1, 2)
    center_pos = positions.mean(dim=0, keepdim=True)
    positions = positions + (center_env - center_pos)

    return positions.to(torch.float32), theta.to(torch.float32)


def run_env_vectorized(prey_policy=None, pred_policy=None, 
                       n_prey=32, n_pred=1, 
                       step_size=0.5, max_steps=100, 
                       deterministic=False,
                       prey_speed=5, pred_speed=5, 
                       area_width=50, area_height=50, max_turn=np.pi, 
                       init_pool=None, device="cuda"):
    
    """
    This version was keep during the development of vectorized batch env.
    Runs env with given policies for prey and predator each, vectorized.
    To generated longer trajectories with single policies.
    """

    n_agents = n_prey + n_pred
    n_neigh = n_agents - 1

    # initialize positions and headings
    positions, theta = init_positions_in_env(init_pool, area_width=area_width, area_height=area_height, device=device)
    
    # initialize speeds
    speed = torch.full((n_agents,), float(prey_speed), dtype=torch.float32, device=device)
    if n_pred > 0:
        speed[:n_pred] = float(pred_speed) # set predator speeds

    if n_pred > 0:
        # initialize trajectory tensors
        prey_traj = torch.empty((max_steps, n_prey, n_neigh, 6), dtype=torch.float32, device=device)
        pred_traj = torch.empty((max_steps, n_pred, n_neigh, 5), dtype=torch.float32, device=device)
    else:
        # initialize prey-only trajectory tensor
        prey_traj = torch.empty((max_steps, n_prey, n_neigh, 5), dtype=torch.float32, device=device)
        pred_traj = None

    # neighbor indices for state tensor construction
    idx = torch.arange(n_agents, device=device)
    neigh_idx = idx.repeat(n_agents, 1)
    neigh_idx = neigh_idx[~torch.eye(n_agents, dtype=torch.bool, device=device)].view(n_agents, n_agents - 1)


    t = 0
    with torch.inference_mode():
        while t < max_steps:
            # compute velocities from headings and speeds
            vel = velocity_from_theta(theta, speed)

            # construct prey logs for current time step
            prey_pos_now = positions[n_pred:]
            prey_vel_now = vel[n_pred:] 
            prey_dir = prey_vel_now / (torch.linalg.norm(prey_vel_now, dim=-1, keepdim=True) + 1e-12)
            prey_log_t = torch.cat([prey_pos_now, prey_vel_now, prey_dir], dim=-1)

            if n_pred > 0:
                # construct predator log for current time step
                pred_pos_now = positions[:n_pred] 
                pred_vel_now = vel[:n_pred] 
                pred_dir = pred_vel_now / (torch.linalg.norm(pred_vel_now, dim=-1, keepdim=True) + 1e-12)
                predator_log_t = torch.cat([pred_pos_now, pred_vel_now, pred_dir], dim=-1) 
            else:
                # empty predator log for prey-only case
                predator_log_t = torch.empty((0, 6), dtype=torch.float32, device=device)

            # get state tensors for predators and prey
            pred_states, prey_states = get_state_tensors(prey_log_t.unsqueeze(0), predator_log_t.unsqueeze(0),
                                                         area_width=area_width, area_height=area_height,
                                                         n_pred=n_pred, max_speed_norm=5, neigh_idx=neigh_idx)
            pred_states = pred_states[0]
            prey_states = prey_states[0]
            
            if n_pred > 0:
                # forward pass through both policies
                pred_actions, pred_weights = pred_policy.forward(pred_states, deterministic=deterministic) 
                prey_actions, prey_weights = prey_policy.forward(prey_states, deterministic=deterministic)

                # log predator trajectories
                pred_traj[t, :, :, :4] = pred_states
                pred_traj[t, :, :, 4:] = pred_actions.unsqueeze(1).expand(-1, n_neigh, -1)
                
                # log prey trajectories
                prey_traj[t, :, :, :5] = prey_states
                prey_traj[t, :, :, 5:] = prey_actions.unsqueeze(1).expand(-1, n_neigh, -1)
            else:
                # prey-only case forward pass
                prey_in = prey_states              
                prey_actions, prey_weights = prey_policy.forward(prey_in, deterministic=deterministic)  # (n_prey,1)

                # log prey trajectories
                prey_traj[t, :, :, :4] = prey_states
                prey_traj[t, :, :, 4:] = prey_actions.unsqueeze(1).expand(-1, n_neigh, -1)

            # update headings based on actions
            theta[n_pred:] = apply_turnrate(theta[n_pred:], prey_actions.squeeze(-1), max_turn)
            if n_pred > 0:
                # update predator headings
                theta[:n_pred] = apply_turnrate(theta[:n_pred], pred_actions.squeeze(-1), max_turn)

            # update positions with wall enforcement
            vel = velocity_from_theta(theta, speed)
            positions, theta = enforce_walls(positions, theta, area_width, area_height)
            positions = positions + vel * float(step_size)

            t += 1

    # trim trajectories to actual length
    prey_tensor = prey_traj[:t]
    pred_tensor = pred_traj[:t] if n_pred > 0 else None
    return pred_tensor, prey_tensor



def init_positions(init_pool, batch=32, area_width=50, area_height=50, mode="dual", device="cuda"):
    """
    Initializes agents from a given pool of states, batch version
    Important that pos & neg perturbations get same init!

    Input: init_pool, area size, device
    Output: positions, headings
    """

    # sample random initial states for the batch
    steps, agents, coordinates = init_pool.shape
    idx = torch.randint(steps, (batch,), device=device)
    sample = init_pool[idx].to(device)

    # get positions and headings
    positions = sample[:, :, :2].clone()
    theta = sample[:, :, 2].clone()

    # center the swarm in the environment
    center_env = torch.tensor([area_width * 0.5, area_height * 0.5], dtype=positions.dtype, device=device).view(1, 1, 2)
    center_pos = positions.mean(dim=1, keepdim=True)
    positions = positions + (center_env - center_pos)

    # repeat for mirrirored perturbations
    if mode == "dual":
        positions = positions.repeat(2, 1, 1) 
        theta = theta.repeat(2, 1)  

    return (positions, theta)



def policy_perturbation(pred_policy, prey_policy, 
                        role="prey", module="pairwise", 
                        sigma=0.1, num_perturbations=32,
                        device="cuda"):
    
    """
    Creates positive and negative perturbations of policy parameters

    Input: policies, role, module, sigma, num_perturbations, device
    Output: perturbed parameter dicts, epsilons
    """
    
    # select policy based on role
    policy = prey_policy if role == "prey" else pred_policy
    base_state_dict = policy.state_dict()

    # determine module to perturbated
    prefix = "pairwise." if module == "pairwise" else "attention."

    # extract parameter keys for perturbation
    param_keys = [k for k, v in base_state_dict.items()
                    if k.startswith(prefix) and torch.is_tensor(v) and v.is_floating_point()]

    # prepare base state dict on device
    base = OrderedDict()
    for k, v in base_state_dict.items():
        if torch.is_tensor(v):
            base[k] = v.detach().to(device)

    pos_list = []
    neg_list  = []
    epsilons = []

    # generate perturbations
    for _ in range(num_perturbations):
        pos = base.copy()
        neg = base.copy()

        eps_vec_parts = []
        for k in param_keys:
            # sample perturbation
            eps = torch.randn_like(base[k])

            # create positive and negative perturbed parameters
            pos[k] = base[k] + sigma * eps
            neg[k] = base[k] - sigma * eps
            eps_vec_parts.append(eps.reshape(-1))

        # concatenate epsilon vector
        eps_vec = torch.cat(eps_vec_parts, dim=0)

        pos_list.append(pos)
        neg_list.append(neg)
        epsilons.append(eps_vec)

    # combine positive and negative perturbations
    pert_list_all = pos_list + neg_list

    return pert_list_all, epsilons


def batch_policy_forward(policy, states, pert_list, deterministic=False):
    """
    Avoids for-loops by batching parameter perturbations

    Input: policy, states, pert_list, deterministic
    Output: actions, weights
    """

    # stack each parameter tensor across perturbations
    keys = pert_list[0].keys()
    params_batched = OrderedDict((k, torch.stack([p[k] for p in pert_list], dim=0)) for k in keys)

    # functional_call runs a module with an explicit parameter dict
    def vmap_function(params, x):
        return functional_call(policy, params, (x,), kwargs={"deterministic": deterministic})

    # vmap over (params, states) with different randomness per perturbation (stochastic policies)
    return vmap(vmap_function, in_dims=(0, 0), randomness="different")(params_batched, states)



def run_batch_env(prey_policy=None, pred_policy=None, 
                    n_prey=32, n_pred=1, 
                    step_size=0.5, batch=32,
                    max_steps=100, deterministic=False,
                    prey_speed=5, pred_speed=5, 
                    area_width=50, area_height=50, max_turn=np.pi,
                    init_pos=None, pert_list=None, role="prey", device="cuda"):
    """
    Runs env vectorized with batch processed policy parameter perturbations
    Applied during ES update
    """

    n_agents = n_prey + n_pred
    n_neigh = n_agents - 1

    # initialize positions and headings
    positions = init_pos[0].to(device)
    theta = init_pos[1].to(device)
    speed = torch.full((batch, n_agents), float(prey_speed), dtype=torch.float32, device=device)

    if n_pred > 0:
        speed[:, :n_pred] = float(pred_speed) # set predator speeds

        # initialize trajectory tensors
        prey_traj = torch.empty((batch, max_steps, n_prey, n_neigh, 6), dtype=torch.float32, device=device)
        pred_traj = torch.empty((batch, max_steps, n_pred, n_neigh, 5), dtype=torch.float32, device=device)
    else:
        # initialize prey-only trajectory tensor
        prey_traj = torch.empty((batch, max_steps, n_prey, n_neigh, 5), dtype=torch.float32, device=device)
        pred_traj = None

    # neighbor indices for state tensor construction
    idx = torch.arange(n_agents, device=device)
    neigh_idx = idx.repeat(n_agents, 1)
    neigh_idx = neigh_idx[~torch.eye(n_agents, dtype=torch.bool, device=device)].view(n_agents, n_agents - 1)

    t = 0
    with torch.inference_mode():
        while t < max_steps:
            # compute velocities from headings and speeds
            vel = velocity_from_theta(theta, speed)

            # construct prey logs for current time step
            prey_pos_now = positions[:, n_pred:]
            prey_vel_now = vel[:, n_pred:]
            prey_dir = prey_vel_now / (torch.linalg.norm(prey_vel_now, dim=-1, keepdim=True) + 1e-12)
            prey_log_t = torch.cat([prey_pos_now, prey_vel_now, prey_dir], dim=-1)

            if n_pred > 0:
                # construct predator log for current time step
                pred_pos_now = positions[:, :n_pred]
                pred_vel_now = vel[:, :n_pred]
                pred_dir = pred_vel_now / (torch.linalg.norm(pred_vel_now, dim=-1, keepdim=True) + 1e-12)
                predator_log_t = torch.cat([pred_pos_now, pred_vel_now, pred_dir], dim=-1)
            else:
                # empty predator log for prey-only case
                predator_log_t = torch.empty((batch, 0, 6), dtype=torch.float32, device=device)

            # get state tensors for predators and prey
            pred_states, prey_states = get_state_tensors(prey_log_t, predator_log_t,
                                                         area_width=area_width, area_height=area_height,
                                                         n_pred=n_pred, max_speed_norm=5, neigh_idx=neigh_idx)

            if n_pred > 0:
                # keep env-wise shape
                pred_in_env = pred_states.view(batch, n_pred, n_neigh, 4)  
                prey_in_env = prey_states.view(batch, n_prey, n_neigh, 5)  

                if role == "pred":
                    # pred actions with batched perturbations
                    pred_actions, pred_weights = batch_policy_forward(pred_policy, pred_in_env, pert_list, deterministic=deterministic)
                else:
                    # pred actions without perturbations, necessary, because only one module is perturbed at a time
                    pred_actions, pred_weights = pred_policy.forward(pred_in_env.view(batch * n_pred, n_neigh, 4), deterministic=deterministic)
                    pred_actions = pred_actions.view(batch, n_pred, 1)

                if role == "prey":
                    # prey actions with batched perturbations
                    prey_actions, prey_weights = batch_policy_forward(prey_policy, prey_in_env, pert_list, deterministic=deterministic)
                else:
                    # prey actions without perturbations
                    prey_actions, prey_weights = prey_policy.forward(prey_in_env.view(batch * n_prey, n_neigh, 5), deterministic=deterministic)
                    prey_actions = prey_actions.view(batch, n_prey, 1)

                # log predator trajectories
                pred_traj[:, t, :, :, :4] = pred_states
                pred_traj[:, t, :, :, 4:] = pred_actions.unsqueeze(3).expand(-1, -1, n_neigh, -1)

                # log prey trajectories
                prey_traj[:, t, :, :, :5] = prey_states
                prey_traj[:, t, :, :, 5:] = prey_actions.unsqueeze(3).expand(-1, -1, n_neigh, -1)

            else:
                # prey-only case
                prey_in_env = prey_states.view(batch, n_prey, n_neigh, 4)

                if role == "prey" and pert_list is not None:
                    # prey actions with batched perturbations
                    prey_actions, prey_weights = batch_policy_forward(prey_policy, prey_in_env, pert_list, deterministic=deterministic)
                else:
                    # prey actions without perturbations
                    prey_actions, prey_weights = prey_policy.forward(prey_in_env.view(batch * n_prey, n_neigh, 4), deterministic=deterministic)
                    prey_actions = prey_actions.view(batch, n_prey, 1)

                # log prey trajectories
                prey_traj[:, t, :, :, :4] = prey_states
                prey_traj[:, t, :, :, 4:] = prey_actions.unsqueeze(3).expand(-1, -1, n_neigh, -1)

            # update headings based on actions
            theta[:, n_pred:] = apply_turnrate(theta[:, n_pred:], prey_actions.squeeze(-1), max_turn)
            if n_pred > 0:
                theta[:, :n_pred] = apply_turnrate(theta[:, :n_pred], pred_actions.squeeze(-1), max_turn)

            # update positions with wall enforcement
            vel = velocity_from_theta(theta, speed)
            positions, theta = enforce_walls(positions, theta, area_width, area_height)
            positions = positions + vel * float(step_size)

            t += 1

    # trim trajectories to actual length
    prey_tensor = prey_traj[:, :t]
    pred_tensor = pred_traj[:, :t] if n_pred > 0 else None
    return pred_tensor, prey_tensor


def apply_perturbations(prey_policy, pred_policy, init_pos, 
                        role, module, device,
                        sigma, num_perturbations,
                        settings_batch_env):
    
    """
    Applies policy perturbations and runs batch env with them

    Input: policies, init pos, role, module, device,
           sigma, num_perturbations, batch env settings
    Output: predator & prey rollouts, epsilons
    """
    
    # create perturbations
    pert_list, epsilons = policy_perturbation(pred_policy, prey_policy,
                                            role=role, module=module,
                                            sigma=sigma, num_perturbations=num_perturbations,
                                            device=device)
    
    n_pred = 1 if pred_policy is not None else 0

    # run batch env with perturbations
    pred_rollouts, prey_rollouts = run_batch_env(prey_policy=prey_policy, 
                                                 pred_policy=pred_policy,
                                                 n_pred=n_pred,
                                                 step_size=settings_batch_env[4],
                                                 batch=2*num_perturbations, 
                                                 max_steps=settings_batch_env[6],
                                                 prey_speed=settings_batch_env[2],
                                                 pred_speed=settings_batch_env[3],
                                                 area_width=settings_batch_env[1],
                                                 area_height=settings_batch_env[0],
                                                 max_turn=settings_batch_env[5],
                                                 init_pos=init_pos, 
                                                 pert_list=pert_list, 
                                                 role=role)
    
    return pred_rollouts, prey_rollouts, epsilons