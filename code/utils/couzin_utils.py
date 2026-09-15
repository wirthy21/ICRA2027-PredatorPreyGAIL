import pylab
import torch
import numpy as np
from math import *
from numpy.linalg import *
from utils.eval_utils import *
import matplotlib.pyplot as plt
from utils.eval_utils import compute_polarization, compute_angular_momentum, degree_of_sparsity, distance_to_predator, escape_alignment

"""
References:
Env: https://github.com/hossein-haeri/couzin_swarm_model/blob/master/swarm_pray_predator.py
Changed to 2D only, added logging tensors
"""

class Agent:
    def __init__(self, agent_id, speed, area_width, area_height):
        # Initialize agent with random position
        self.id = agent_id
        self.pos = np.array([np.random.uniform(0, area_width),
                             np.random.uniform(0, area_height)])

        # Initialize agent with random velocity
        self.vel = np.random.uniform(-1, 1, 2)
        self.is_alive = 1 # alive status
        self.vel = self.vel / norm(self.vel) * speed

    def update_position(self, delta_t):
        # Update position based on velocity and time step
        if self.is_alive:
            self.pos = self.pos + self.vel * delta_t


def rotation_matrix_about(z, theta):
    """
    Rotation matrix in 2D about z-axis by angle theta
    """

    z = float(z)
    sgn = np.sign(z)
    if sgn == 0.0 or theta == 0.0:
        return np.eye(2, dtype=np.float64)

    ang = sgn * theta
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def enforce_walls(agent, area_width, area_height):
    """
    Check and handle collisions with the environment boundaries
    """

    # left and right walls
    if agent.pos[0] < 0:
        agent.pos[0] = 0
        agent.vel[0] *= -1
    elif agent.pos[0] > area_width:
        agent.pos[0] = area_width
        agent.vel[0] *= -1

    # bottom and top walls
    if agent.pos[1] < 0:
        agent.pos[1] = 0
        agent.vel[1] *= -1
    elif agent.pos[1] > area_height:
        agent.pos[1] = area_height
        agent.vel[1] *= -1


def wrap_to_pi(a):
    """
    Wrap angle to [-pi, pi]
    """
    return (a + np.pi) % (2*np.pi) - np.pi


def run_couzin_simulation(
    visualization='on', n=32, max_steps=100, dt=0.5, free_offset=10, number_of_sharks=1, alpha=0.1,
    r_r=2, r_o=10, r_a=40, r_thr=30, r_lethal=1, field_of_view=3*np.pi/2, field_of_view_shark=2*np.pi,
    theta_dot_max=0.5, theta_dot_max_shark=0.5, constant_speed=5, shark_speed=5,
    area_width=50, area_height=50):

    """
    Runs env using Couzin model as control.
    Primary purpose is clean and consistent data generation for expert datasets.
    """

    # init agents
    swarm = [Agent(i, constant_speed, area_width, area_height) for i in range(n)]
    sharks = [Agent(i, shark_speed, area_width, area_height) for i in range(number_of_sharks)]

    # init logging arrays
    swarm_pos = np.zeros((n, 2))
    swarm_vel = np.zeros((n, 2))

    # shark logging arrays
    sharks_pos = np.zeros((number_of_sharks, 2))
    sharks_vel = np.zeros((number_of_sharks, 2))

    t = 0

    # Figure
    if visualization == 'on':
        fig, ax = plt.subplots()

    # initialize logs
    prey_log = np.zeros((max_steps, n, 6))
    predator_log = np.zeros((max_steps, number_of_sharks, 6))
    pred_tensor_list = []
    prey_tensor_list = []
    metrics_list = []
    action_list = []

    # precompute max turn angles
    max_turn_prey  = float(theta_dot_max * dt) + 1e-12
    max_turn_shark = float(theta_dot_max_shark * dt) + 1e-12

    while t < max_steps:
        # log current prey state
        for i, agent in enumerate(swarm):
            swarm_pos[i, :] = agent.pos
            swarm_vel[i, :] = agent.vel

            # direction vector
            vel_norm = norm(agent.vel[0:2])
            if vel_norm > 1e-12:
                dir_xy = agent.vel[0:2] / vel_norm
            else:
                dir_xy = np.zeros(2)

            # store prey logs
            prey_log[t, i, 0:2] = agent.pos[0:2]  
            prey_log[t, i, 2:4] = agent.vel[0:2]   
            prey_log[t, i, 4:6] = dir_xy  

        if number_of_sharks > 0:
            # log current shark state
            for i, shark in enumerate(sharks):
                sharks_pos[i, :] = shark.pos
                sharks_vel[i, :] = shark.vel / (norm(shark.vel) + 1e-12) / 80 * area_width

                # direction vector
                vel_norm_s = norm(shark.vel[0:2])
                if vel_norm_s > 1e-12:
                    dir_xy_s = shark.vel[0:2] / vel_norm_s
                else:
                    dir_xy_s = np.zeros(2)

                # predator logs
                predator_log[t, i, 0:2] = shark.pos[0:2] 
                predator_log[t, i, 2:4] = shark.vel[0:2]    
                predator_log[t, i, 4:6] = dir_xy_s      

        # store previous thetas for action calculation
        prey_theta_prev = np.array([np.arctan2(a.vel[1], a.vel[0]) for a in swarm], dtype=np.float32)
        if number_of_sharks > 0:
            shark_theta_prev = np.array([np.arctan2(s.vel[1], s.vel[0]) for s in sharks], dtype=np.float32)
        else:
            shark_theta_prev = None

        # visualization of agents
        if visualization == 'on':
            ax.clear()

            # plot prey as black arrows
            pylab.quiver(swarm_pos[:, 0], swarm_pos[:, 1],
                         swarm_vel[:, 0], swarm_vel[:, 1],
                         scale=120,
                         width=0.01,
                         headwidth=3,
                         headlength=3,
                         headaxislength=3)

            if number_of_sharks > 0:
                # plot sharks as red arrows
                pylab.quiver(sharks_pos[:, 0], sharks_pos[:, 1],
                             sharks_vel[:, 0], sharks_vel[:, 1],
                             color="#FF0000",
                             scale=15,
                             width=0.01,
                             headwidth=3,
                             headlength=3,
                             headaxislength=3)

            ax.set_aspect('equal', 'box')
            ax.set_xlim(0, area_width)
            ax.set_ylim(0, area_height)

            plt.pause(0.00000001)

        # Prey update, based on Couzin's model
        for agent in swarm:
            d = np.zeros(2)
            d_social = np.zeros(2)
            d_r = np.zeros(2)
            d_o = np.zeros(2)
            d_a = np.zeros(2)
            d_thr = np.zeros(2)

            if agent.is_alive:
                # social interactions (prey–prey)
                for neighbor in swarm:
                    if agent.id == neighbor.id or not neighbor.is_alive:
                        continue

                    # compute distance
                    r_vec = neighbor.pos - agent.pos
                    dist = norm(r_vec)
                    if dist >= r_a:
                        continue

                    # normalize vectors
                    r_normalized = r_vec / (dist + 1e-12)
                    agent_vel_normalized = agent.vel / (norm(agent.vel) + 1e-12)

                    # field of view
                    dot_val = float(np.dot(r_normalized, agent_vel_normalized)) 
                    dot_val = np.clip(dot_val, -1.0, 1.0) 
                    if acos(dot_val) >= field_of_view / 2:
                        continue

                    # zones of interaction
                    if dist < r_r:
                        d_r -= r_normalized
                    elif dist < r_o:
                        d_o += neighbor.vel / (norm(neighbor.vel) + 1e-12)
                    elif dist < r_a:
                        d_a += r_normalized

                # determine social direction
                if norm(d_r) != 0:
                    d_social = d_r
                elif norm(d_a) != 0 and norm(d_o) != 0:
                    d_social = (d_o + d_a) / 2
                elif norm(d_o) != 0:
                    d_social = d_o
                elif norm(d_a) != 0:
                    d_social = d_a

                # threat avoidance (prey–predator)
                if number_of_sharks > 0:
                    for each_shark in sharks:
                        diff = each_shark.pos - agent.pos
                        dist = norm(diff)
                        if dist <= r_thr:
                            d_thr -= diff / (dist ** 2 + 1e-12)

                # combine directions
                if norm(d_social) != 0:
                    d = alpha * d_social / (norm(d_social) + 1e-12)
                if norm(d_thr) != 0:
                    d += (1 - alpha) / 2 * d_thr / (norm(d_thr) + 1e-12)

                # steering
                if norm(d) != 0:
                    z = np.cross(d / (norm(d) + 1e-12), agent.vel / (norm(agent.vel) + 1e-12))
                    angle_between = asin(min(1.0, abs(z)))

                    # apply max turn rate
                    if angle_between >= theta_dot_max * dt:
                        rot = rotation_matrix_about(z, theta_dot_max * dt)
                        agent.vel = np.asarray(np.asmatrix(agent.vel) * rot)[0]
                    else:
                        agent.vel = d / (norm(d) + 1e-12) * constant_speed

        # predator update
        if number_of_sharks > 0:
            for each_shark in sharks:
                d = np.zeros(2)
                closest_dist = np.inf

                for prey in swarm:
                    # find closest prey within field of view
                    if not prey.is_alive:
                        continue

                    # compute distance
                    r_vec = prey.pos - each_shark.pos
                    dist = norm(r_vec)
                    if dist < closest_dist:
                        direction = r_vec / (dist + 1e-12) # field of view shark

                        # check field of view
                        shark_dir = each_shark.vel / (norm(each_shark.vel) + 1e-12)
                        dot_val = float(np.dot(direction, shark_dir))
                        dot_val = np.clip(dot_val, -1.0, 1.0)
                        if acos(dot_val) < field_of_view_shark / 2:
                            closest_dist = dist
                            d = direction

                if norm(d) != 0:
                    # steering towards closest prey
                    z = np.cross(d / (norm(d) + 1e-12), each_shark.vel / (norm(each_shark.vel) + 1e-12))
                    angle_between = asin(min(1.0, abs(z)))

                    # apply max turn rate
                    if angle_between >= theta_dot_max_shark * dt:
                        rot = rotation_matrix_about(z, theta_dot_max_shark * dt)
                        each_shark.vel = np.asarray(np.asmatrix(each_shark.vel) * rot)[0]
                    else:
                        each_shark.vel = d / (norm(d) + 1e-12) * shark_speed

        # calculate prey actions change in heading
        prey_theta_new = np.array([np.arctan2(a.vel[1], a.vel[0]) for a in swarm], dtype=np.float32)
        dtheta_prey = wrap_to_pi(prey_theta_new - prey_theta_prev)
        prey_actions = (dtheta_prey / (2.0 * max_turn_prey)) + 0.5
        prey_actions = np.clip(prey_actions, 0.0, 1.0).astype(np.float32)
        
        if number_of_sharks > 0:
            # calculate shark actions change in heading
            shark_theta_new = np.array([np.arctan2(s.vel[1], s.vel[0]) for s in sharks], dtype=np.float32)
            dtheta_shark = wrap_to_pi(shark_theta_new - shark_theta_prev)
            shark_actions = (dtheta_shark / (2.0 * max_turn_shark)) + 0.5
            shark_actions = np.clip(shark_actions, 0.0, 1.0)
        else:
            shark_actions = None

        # get state tensors
        pred_tensor, prey_tensor, metrics = get_state_tensors(prey_log[t], predator_log[t],
                                                              area_width=area_width, area_height=area_height,
                                                              constant_speed=constant_speed, shark_speed=shark_speed,
                                                              number_of_sharks=number_of_sharks)

        # store tensors and metrics
        pred_tensor_list.append(pred_tensor)
        prey_tensor_list.append(prey_tensor)
        metrics_list.append(metrics)
        action_list.append({"prey": prey_actions, "pred": shark_actions})

        for agent in swarm:
            # wall enforcement and position update for prey
            enforce_walls(agent, area_width, area_height)
            agent.update_position(dt)

        if number_of_sharks > 0:
            # wall enforcement and position update for predator
            for shark in sharks:
                enforce_walls(shark, area_width, area_height)
                shark.update_position(dt)

        t += 1

    if number_of_sharks > 0:
        # stack tensors
        final_pred_tensor = torch.stack(pred_tensor_list, dim=0)
        stacked_prey_tensor = torch.stack(prey_tensor_list, dim=0)

        # add shark flag to prey tensor
        frames, agents, neigh, feat = stacked_prey_tensor.shape
        flag = torch.zeros((frames, agents, neigh, 1), dtype=stacked_prey_tensor.dtype, device=stacked_prey_tensor.device)
        flag[:, :, :number_of_sharks, 0] = 1
        final_prey_tensor = torch.cat([flag, stacked_prey_tensor], dim=-1)
    else:
        # prey-only case
        final_pred_tensor = None
        final_prey_tensor = torch.stack(prey_tensor_list, dim=0)

    if number_of_sharks > 0:
        # get shark logs for init pool
        combined = np.concatenate([predator_log[:t], prey_log[:t]], axis=1).astype(np.float32)
        xs = combined[..., 0].astype(np.float32)
        ys = combined[..., 1].astype(np.float32)
        dirx = combined[..., 4].astype(np.float32)
        diry = combined[..., 5].astype(np.float32)
    else:
        # get prey logs for init pool
        xs = prey_log[..., 0].astype(np.float32)
        ys = prey_log[..., 1].astype(np.float32)
        dirx = prey_log[:t, ..., 4].astype(np.float32)
        diry = prey_log[:t, ..., 5].astype(np.float32)

    theta = np.arctan2(diry, dirx).astype(np.float32)

    # stack initial positions and orientations and create init pool
    init_pool = np.stack([xs, ys, theta], axis=-1).astype(np.float32)
    init_pool = init_pool[50:] # cut 50 off due to random init
    init_pool = torch.from_numpy(init_pool).to(torch.float32)

    return final_pred_tensor, final_prey_tensor, metrics_list, action_list, init_pool



def get_state_tensors(prey_log_step, shark_log_step, 
                      area_width=50, area_height=50, 
                      constant_speed=5, shark_speed=5, 
                      number_of_sharks=1):
    
    """
    Converts logs to expert feature tensors (derived from Wu et al. 2025)

    Input: pred and prey logs, n_pred, area size, max speed for normalization
    Output: predator & prey tensor and metrics dict
    """

    # combine predator and prey logs
    combined = np.vstack([shark_log_step, prey_log_step])
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

    # compute heading angles and scale to [0,1]
    thetas = np.arctan2(dir_y, dir_x).astype(np.float32)
    theta_scaled = (thetas + np.pi) / (2.0 * np.pi) # [0,1]
    theta_scaled = np.clip(theta_scaled, 0.0, 1.0).astype(np.float32)
    theta_mat = np.tile(theta_scaled[:, None], (1, n_agents))

    # normalize relative velocities (same as expert)
    max_speed_norm = max(constant_speed, shark_speed)
    rel_vx = np.clip(rel_vx, -max_speed_norm, max_speed_norm) / max_speed_norm
    rel_vy = np.clip(rel_vy, -max_speed_norm, max_speed_norm) / max_speed_norm

    # stack features and apply mask to exclude self-features
    features = np.stack([dx, dy, rel_vx, rel_vy, theta_mat], axis=-1).astype(np.float32)
    mask = ~np.eye(n_agents, dtype=bool)
    neigh = features[mask].reshape(n_agents, n_agents - 1, 5)

    # split predator and prey tensors
    pred_tensor = torch.from_numpy(neigh[:number_of_sharks])
    prey_tensor = torch.from_numpy(neigh[number_of_sharks:])

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
               "features": features}

    return pred_tensor, prey_tensor, metrics