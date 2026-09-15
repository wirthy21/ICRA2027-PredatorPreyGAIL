import pylab
import torch
import numpy as np
from copy import deepcopy
from numpy.linalg import *
from utils.eval_utils import *
import matplotlib.pyplot as plt
from utils.eval_utils import compute_polarization, compute_angular_momentum, degree_of_sparsity, distance_to_predator, escape_alignment


"""
References:
Env Structure: https://github.com/hossein-haeri/couzin_swarm_model/blob/master/swarm_pray_predator.py
Wall Enforcement: https://github.com/mdodsworth/pyglet-boids/blob/master/boids/boid.py
"""


class Agent:
    def __init__(self, agent_id, speed, area_width, area_height):
        # Initialize agent with random position
        self.id = agent_id
        self.speed = float(speed)
        self.pos = np.array([np.random.uniform(0, area_width),
                             np.random.uniform(0, area_height)], dtype=np.float64)

        # Initialize a random heading angle, derive the velocity vector from it
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.vel = np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float64) * speed


    def update_position(self, step_size):
        # position update, using the current velocity and a step size parameter
        self.pos += self.vel * step_size


def apply_turnrate_on_theta(agent, action, speed, max_turn):
    """
    Update the agents heading angle and velocity based on the policy action
    """

    # maps normalized action [0,1] to turn rate [-max_turn, max_turn]
    dtheta = (action - 0.5) * 2.0 * max_turn

    # update heading and wrap angle back to [-pi, pi]
    agent.theta = (agent.theta + dtheta + np.pi) % (2*np.pi) - np.pi

    # update velocity vector based on new heading while maintaining speed
    agent.vel = np.array([np.cos(agent.theta), np.sin(agent.theta)]) * speed


def enforce_walls(agent, area_width, area_height):
    """
    Check and handle collisions with the environment boundaries
    """
    bounced = False

    # left and right walls
    if agent.pos[0] < 0:
        agent.pos[0] = 0
        agent.theta = np.pi - agent.theta
        bounced = True
    elif agent.pos[0] > area_width:
        agent.pos[0] = area_width
        agent.theta = np.pi - agent.theta
        bounced = True

    # bottom and top walls
    if agent.pos[1] < 0:
        agent.pos[1] = 0
        agent.theta = -agent.theta
        bounced = True
    elif agent.pos[1] > area_height:
        agent.pos[1] = area_height
        agent.theta = -agent.theta
        bounced = True

    if bounced:
        # if bounced, update velocity vector based on new heading while maintaining speed
        agent.theta = (agent.theta + np.pi) % (2*np.pi) - np.pi
        speed = float(norm(agent.vel)) 
        agent.vel = np.array([np.cos(agent.theta), np.sin(agent.theta)], dtype=np.float64) * speed


def get_state_tensors(prey_log_step, pred_log_step, n_pred=1, 
                      area_width=50, area_height=50, max_speed_norm=5,
                      mask=None):
    
    """
    Converts logs to expert feature tensors (derived from Wu et al. 2025)

    Input: pred and prey logs, n_pred, area size, max speed for normalization
    Output: predator & prey tensor and metrics dict
    """
    
    # combine predator and prey logs
    combined = np.vstack([pred_log_step, prey_log_step])
    n_agents = combined.shape[0]

    # extract positions, velocities, and directions
    xs, ys  = combined[:, 0], combined[:, 1]
    vxs, vys = combined[:, 2], combined[:, 3]
    dir_x, dir_y = combined[:, 4], combined[:, 5]

    # scale positions to [0,1]
    xs_scaled = xs / float(area_width)
    ys_scaled = ys / float(area_height)

    # compute pairwise distances
    dx = xs_scaled[None, :] - xs_scaled[:, None]
    dy = ys_scaled[None, :] - ys_scaled[:, None]

    # compute relative velocities in the agent's heading direction
    cos_t = dir_x.astype(np.float32)
    sin_t = dir_y.astype(np.float32)
    rel_vx = cos_t[:, None] * vxs[None, :] + sin_t[:, None] * vys[None, :]
    rel_vy = -sin_t[:, None] * vxs[None, :] + cos_t[:, None] * vys[None, :]

    # normalize relative velocities (same as expert)
    rel_vx = np.clip(rel_vx, -max_speed_norm, max_speed_norm) / max_speed_norm
    rel_vy = np.clip(rel_vy, -max_speed_norm, max_speed_norm) / max_speed_norm

    # stack features and apply mask to exclude self-features
    features = np.stack([dx, dy, rel_vx, rel_vy], axis=-1).astype(np.float32) # no theta needed
    neigh = features[mask].reshape(n_agents, n_agents-1, 4)
    tensor = torch.from_numpy(neigh)

    # split predator and prey tensors
    pred_tensor = tensor[:n_pred]
    prey_tensor = tensor[n_pred:]

    if n_pred > 0:
        # add flag to prey tensor
        agents, neighs, _ = prey_tensor.shape
        flag = torch.zeros((agents, neighs, 1), dtype=prey_tensor.dtype, device=prey_tensor.device)
        flag[:, :n_pred, 0] = 1
        prey_tensor = torch.cat([flag, prey_tensor], dim=-1)

    # compute swarm metrics
    polarization = compute_polarization(vxs, vys)
    angular_momentum_val = compute_angular_momentum(xs, ys, vxs, vys)
    sparsity = degree_of_sparsity(xs, ys)
    dist_pred = distance_to_predator(xs, ys)
    dist_nearest_prey = pred_distance_to_nearest_prey(xs, ys)
    escape_align = escape_alignment(xs, ys, vxs, vys)

    # store metrics
    metrics = {"polarization": polarization,
               "angular_momentum": angular_momentum_val,
               "degree_of_sparsity": sparsity,
               "distance_to_predator": dist_pred,
               "distance_nearest_prey": dist_nearest_prey,
               "escape_alignment": escape_align,
               "xs": xs_scaled,
               "ys": ys_scaled,
               "dx": dx,
               "dy": dy,
               "vxs": vxs,
               "vys": vys,
               "theta": np.arctan2(dir_y, dir_x),
               "features": features}


    return pred_tensor, prey_tensor, metrics


def apply_init_pool(init_pool, pred, prey, area_width=50, area_height=50, experiment=False):
    """
    Initializes agents from a given pool of states
    Experiment mode: needed for trajectory prediction expertiment
    """
    n_pred = len(pred)
    n_prey = len(prey)

    if experiment is False:
        # sample a random state from the pool
        steps, agents, coordinates = init_pool.shape
        sample = init_pool[torch.randint(steps, (1,)).item(), :agents].clone()

        # get positions and headings
        positions = sample[:, :2]
        thetas = sample[:, 2]

        # center the swarm in the environment
        center_env = positions.new_tensor([area_width * 0.5, area_height * 0.5])
        center_pos = positions.mean(dim=0)
        positions = positions + (center_env - center_pos)

    else:
        # use the provided initial pool directly (no sampling)
        agents, coordinates = init_pool.shape
        positions = init_pool[:, :2] * area_height # scale positions
        thetas = init_pool[:, 2]

    # apply positions and headings to predator
    for i in range(n_pred):
        agent = pred[i]
        agent.pos = positions[i].detach().cpu().numpy().astype(np.float64)
        agent.theta = float(thetas[i].item())
        agent.vel = np.array([np.cos(agent.theta), np.sin(agent.theta)], dtype=np.float64) * agent.speed

    # apply positions and headings to prey
    for i in range(n_prey):
        j = n_pred + i
        agent = prey[i]
        agent.pos = positions[j].detach().cpu().numpy().astype(np.float64)
        agent.theta = float(thetas[j].item())
        agent.vel = np.array([np.cos(agent.theta), np.sin(agent.theta)], dtype=np.float64) * agent.speed



def run_env_simulation(prey_policy=None, pred_policy=None, 
                       n_prey=32, n_pred=1, step_size=0.5,
                       max_steps=100, deterministic=False,
                       prey_speed=5, pred_speed=5, 
                       area_width=50, area_height=50, 
                       max_turn=np.pi, visualization='off', 
                       init_pool=None, experiment=False):
    
    """
    Runs env with given policies for prey and predator each.
    Primary purpose is visualization and data generation for analysis.
    """

    # deepcopy policies to avoid problems with device handling
    prey_policy = deepcopy(prey_policy).to('cpu')
    pred_policy = deepcopy(pred_policy).to('cpu') if pred_policy is not None else None

    # initialize agents
    prey = [Agent(i, prey_speed, area_width, area_height) for i in range(n_prey)]
    pred = [Agent(i, pred_speed, area_width, area_height) for i in range(n_pred)]

    # apply initial pool if provided
    if init_pool is not None and experiment is False:
        apply_init_pool(init_pool, pred, prey, area_width=area_width, area_height=area_height)
    
    # apply initial pool in experiment mode (without sampling)
    if init_pool is not None and experiment is True:
        apply_init_pool(init_pool, pred, prey, area_width=area_width, area_height=area_height, experiment=True)

    # prepare tensors to log trajectories
    n_agents = n_prey + n_pred
    neigh = n_agents - 1
    prey_traj = torch.empty((max_steps, n_prey, neigh, 6), dtype=torch.float32) if n_pred > 0 else torch.empty((max_steps, n_prey, neigh, 5), dtype=torch.float32)
    pred_traj = torch.empty((max_steps, n_pred, neigh, 5), dtype=torch.float32) if n_pred > 0 else None

    # visualization setup
    if visualization == 'on':
        prey_pos_vis = np.zeros((n_prey, 2), dtype=np.float32)
        prey_vel_vis = np.zeros((n_prey, 2), dtype=np.float32)
        pred_pos_vis = np.zeros((n_pred, 2), dtype=np.float32) if n_pred > 0 else None
        pred_vel_vis = np.zeros((n_pred, 2), dtype=np.float32) if n_pred > 0 else None
        fig, ax = plt.subplots()

    # create mask to exclude self-features, done here to avoid repeated creation in the loop
    mask = ~np.eye(n_agents, dtype=bool)
    metrics_list = []
    weights_list = []
    t = 0

    while t < max_steps:
        # log current positions and velocities for prey
        prey_pos_now = np.asarray([a.pos for a in prey], dtype=np.float32)
        prey_vel_now = np.asarray([a.vel for a in prey], dtype=np.float32) 
        prey_dir = prey_vel_now / (np.linalg.norm(prey_vel_now, axis=1, keepdims=True) + 1e-12)
        prey_log_t = np.concatenate([prey_pos_now, prey_vel_now, prey_dir], axis=1)

        if n_pred > 0:
            # log current positions and velocities for predator
            pred_pos_now = np.asarray([a.pos for a in pred], dtype=np.float32)
            pred_vel_now = np.asarray([a.vel for a in pred], dtype=np.float32)
            pred_dir = pred_vel_now / (np.linalg.norm(pred_vel_now, axis=1, keepdims=True) + 1e-12)
            predator_log_t = np.concatenate([pred_pos_now, pred_vel_now, pred_dir], axis=1)
        else:
            # empty predator log
            predator_log_t = np.empty((0, 6), dtype=np.float32)

        # visualization update
        if visualization == 'on':
            prey_pos_vis[:, :] = prey_pos_now
            prey_vel_vis[:, :] = prey_vel_now

            if n_pred > 0:
                pred_pos_vis[:, :] = pred_pos_now
                pred_vel_vis[:, :] = pred_vel_now

            # plot update of black prey arrows
            ax.clear()
            pylab.quiver(prey_pos_vis[:, 0], prey_pos_vis[:, 1],
                         prey_vel_vis[:, 0], prey_vel_vis[:, 1],
                         scale=120,
                         width=0.01,
                         headwidth=3,
                         headlength=3,
                         headaxislength=3)

            if n_pred > 0:
                # plot update of red predator arrow
                pylab.quiver(pred_pos_vis[:, 0], pred_pos_vis[:, 1],
                             pred_vel_vis[:, 0], pred_vel_vis[:, 1],
                             color="#FF0000",
                             scale=120,
                             width=0.01,
                             headwidth=3,
                             headlength=3,
                             headaxislength=3)
                
            ax.set_aspect('equal', 'box')
            ax.set_xlim(0, area_width)
            ax.set_ylim(0, area_height)

            plt.pause(0.00000001)

        # get state tensors and metrics from logs
        pred_states, prey_states, metrics = get_state_tensors(prey_log_t, predator_log_t, n_pred=n_pred,
                                                              area_width=area_width, area_height=area_height, 
                                                              mask=mask)
        
        metrics_list.append(metrics)

        # Policy forward pass to get actions
        if n_pred > 0:
            with torch.inference_mode():
                # predator policy forward
                pred_actions, pred_weights = pred_policy.forward(pred_states, deterministic=deterministic)
                pred_traj[t, :, :, :4] = pred_states
                pred_traj[t, :, :, 4:] = pred_actions.unsqueeze(1).expand(-1, neigh, -1)

            with torch.inference_mode():
                # prey policy forward
                prey_actions, prey_weights = prey_policy.forward(prey_states, deterministic=deterministic)
                prey_traj[t, :, :, :5] = prey_states
                prey_traj[t, :, :, 5:] = prey_actions.unsqueeze(1).expand(-1, neigh, -1)

            # store weights and indices for analysis of swarm composition
            weights = (pred_weights.detach().cpu().numpy(), prey_weights.detach().cpu().numpy())
            weights_idx = {"pred": {"self": 0, "neighbors": [idx for idx in range(n_agents) if idx != 0]},
                           "prey": {"self": list(range(n_pred, n_agents)), "neighbors": [[idx for idx in range(n_agents) if idx != i] for i in list(range(n_pred, n_agents))]}}

        else:
            with torch.inference_mode():
                # prey-only case policy forward
                prey_actions, prey_weights = prey_policy.forward(prey_states, deterministic=deterministic)
                prey_traj[t, :, :, :4] = prey_states
                prey_traj[t, :, :, 4:] = prey_actions.unsqueeze(1).expand(-1, neigh, -1)
            
            # store weights and indices for analysis of swarm composition
            weights = (None, prey_weights.detach().cpu().numpy())
            weights_idx = {"pred": None,
                           "prey": {"self": list(range(n_pred, n_agents)), "neighbors": [[idx for idx in range(n_agents) if idx != i] for i in list(range(n_pred, n_agents))]}}

        weights_list.append({"weights": weights, "weights_idx": weights_idx}) # log weights

        # apply actions to prey agents
        prey_actions = prey_actions.squeeze(-1).detach().cpu().numpy()
        for i, agent in enumerate(prey):
            apply_turnrate_on_theta(agent, prey_actions[i], prey_speed, max_turn)

        if n_pred > 0:
            # apply actions to predator agents
            pred_actions = pred_actions.squeeze(-1).detach().cpu().numpy()
            for i, predator in enumerate(pred):
                apply_turnrate_on_theta(predator, pred_actions[i], pred_speed, max_turn)

        for agent in prey:
            # wall enforcement and position update for prey
            enforce_walls(agent, area_width, area_height)
            agent.update_position(step_size=step_size)
            

        if n_pred > 0:
            for predator in pred:
                # wall enforcement and position update for predator
                enforce_walls(predator, area_width, area_height)
                predator.update_position(step_size=step_size)
                

        t += 1 # increment time step

        # trim trajectories to actual length
        prey_tensor = prey_traj[:t]
        pred_tensor = pred_traj[:t] if n_pred > 0 else None

    return pred_tensor, prey_tensor, (metrics_list, weights_list)