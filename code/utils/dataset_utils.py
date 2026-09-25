"""
dataset_utils.py:
This file converts Label-Studio keypoint annotations and video detections into padded and
windowed expert tensors for training and evaluating imitation policies.

References:
Data Labeling - https://labelstud.io/
Hungarian Algorithm - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)
"""

import math
import torch
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


def find_valid_windows(filtered_frames, window_len=10, total_detections=33):
    """
    Finds continuous windows where all detections are present.

    Args:
        filtered_frames: list of detection dicts with 'frame' and 'track_id'
        window_len: minimum length of a valid window
        total_detections: expected number of distinct track IDs per frame

    Returns:
        List of dicts: {"start_frame", "end_frame", "length", "ids"}
    """
    # map frame -> set of track IDs
    ids_by_frame = defaultdict(set)
    
    for d in filtered_frames:
        frame = int(d["frame"])
        track_id = int(d["track_id"])
        ids_by_frame[frame].add(track_id)
        
    frames = sorted(ids_by_frame.keys())

    episodes = []
    i = 0

    while i < len(frames):
        frame = frames[i]
        current_ids = ids_by_frame[frame]

        # skip frames w/o expected number of directions
        if len(current_ids) != total_detections:
            i += 1
            continue

        start = frame
        end = frame

        # extend the window while frames are consecutive and IDs match
        while i + 1 < len(frames):
            next_frame = frames[i + 1]
            if next_frame != end + 1 or ids_by_frame[next_frame] != current_ids:
                break
            i += 1
            end = next_frame

        length = end - start + 1
        
        # keep only the windows long enough
        if length >= window_len:
            episodes.append({"start_frame": start,
                             "end_frame": end,
                             "length": length,
                             "ids": sorted(current_ids)})

        i += 1

    return episodes


def extract_windows(episodes, window_len=10):
    """
    Splits episodes into smaller sliding windows

    Args:
        episodes: List of dicts with "start_frame", "end_frame", "length", "ids".
        window_len: Length of each extracted window.

    Returns:
        List of window dicts with "start_frame", "end_frame", "length", "ids".
    """
    windows = []

    for ep in episodes:
        if ep["length"] < window_len:
            continue

        # number of possible windows in this episode
        num_windows = ep["length"] - window_len + 1

        for offset in range(num_windows):
            start = ep["start_frame"] + offset
            end = start + window_len - 1
            
            windows.append({
                "start_frame": start,
                "end_frame": end,
                "length": window_len,
                "ids": ep["ids"],
            })

    return windows


def get_expert_features(frame, width, height, max_speed=10):    
    """
    Convert detections in a single frame to expert feature tensors.

    Based on Wu et al. 2025.

    Args:
        frame: List of detection dicts with keys:
               'x', 'y', 'vx', 'vy', 'angle', 'label', 'track_id'.
        width, height: Frame dimensions for normalizing positions.
        max_speed: Max speed used to clip and normalize relative velocities.

    Returns:
        pred_tensor: (1, N-1, 4) torch tensor for the predator.
        prey_tensor: (N-1, N-1, 4) torch tensor for the prey.
        xs, ys, thetas: Lists of normalized x, y, and heading angles.
    """
    # sort so predator (label == "1") comes first, then by track_id
    frame = sorted(frame, key=lambda d: (d["label"] != "1", int(d["track_id"])))

    # positions
    xs = np.array([d["x"] for d in frame], dtype=float)
    ys = np.array([d["y"] for d in frame], dtype=float)

    # clip to frame bounds and normalize to [0, 1]
    xs = np.clip(xs, 0, width) / width
    ys = np.clip(ys, 0, height) / height

    # velocities
    vxs = np.array([d["vx"] for d in frame], dtype=float)
    vys = np.array([d["vy"] for d in frame], dtype=float)
    
    # extract heading angle, scale to [0, 1]
    thetas = np.array([d["angle"] for d in frame], dtype=float)

    # pairwise position differences in normalized coordinates
    dx = xs[None, :] - xs[:, None]
    dy = ys[None, :] - ys[:, None]

    # compute relative velocities in the agent's heading direction
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    rel_vx = cos_t[:, None] * vxs[None, :] + sin_t[:, None] * vys[None, :]
    rel_vy = -sin_t[:, None] * vxs[None, :] + cos_t[:, None] * vys[None, :]

    # clip and scale relative velocities to [-1, 1]
    rel_vx = np.clip(rel_vx, -max_speed, max_speed) / max_speed
    rel_vy = np.clip(rel_vy, -max_speed, max_speed) / max_speed

    # build feature tensor
    features = np.stack([dx, dy, rel_vx, rel_vy], axis=-1)

    # remove self-interactions
    n = features.shape[0]
    mask = ~np.eye(n, dtype=bool)
    neigh = features[mask].reshape(n, n - 1, 4)

    # first agent is predator, rest are prey
    pred_tensor = torch.from_numpy(neigh[0]).unsqueeze(0)
    prey_tensor = torch.from_numpy(neigh[1:])

    return pred_tensor, prey_tensor, xs.tolist(), ys.tolist(), thetas.tolist()


def get_expert_tensors(filtered_frames, extracted_windows, width, height, max_speed=10, window_size=5):
    """
    Build expert tensors for all extracted windows.

    Args:
        filtered_frames: List of detection dicts for all frames.
        extracted_windows: List of window dicts with "start_frame".
        width, height: Frame dimensions for normalization.
        max_speed: Max speed for velocity normalization.
        window_size: Number of frames per window.

    Returns:
        pred_tensor: (num_windows, window_size, 1, num_neighbors, 5)
        prey_tensor: (num_windows, window_size, num_prey, num_neighbors, 5)
        coordinates: (num_windows, window_size, num_agents, 3)  # [x, y, theta]
    """
    # return empty tensors if no windows extracted
    if len(extracted_windows) == 0:
        return torch.empty(0), torch.empty(0)
    
    # group detections by frame
    dets_by_frame = defaultdict(list)
    for det in filtered_frames:
        dets_by_frame[int(det["frame"])].append(det)
    
    start_frames = [window['start_frame'] for window in extracted_windows]
    
    pred_windows = []
    prey_windows = []
    window_coordinates = []

    # build tensors for each window
    for idx, start in enumerate(start_frames):
        window_detections = []
        
        # collect detections for the window
        for frame in range(start, start + window_size):
            dets = dets_by_frame[int(frame)]
            window_detections.append(dets)

        preds = []
        preys = []
        thetas = []
        frame_coordinates = []

        # convert each frames detections to tensors
        for dets in window_detections:
            pred_tensor, prey_tensor, xs, ys, theta = get_expert_features(dets, width, height, max_speed)

            preds.append(pred_tensor)
            preys.append(prey_tensor)

            theta_tensor = torch.tensor(theta, dtype=torch.float32)
            thetas.append(theta_tensor)
            
            # store coordinates for init_pool
            xy = torch.from_numpy(np.stack([xs, ys, theta], axis=-1)).float()
            frame_coordinates.append(xy)

        # compute delta theta for the window
        theta_stacked = torch.stack(thetas, dim=0)
        dtheta = theta_stacked[1:] - theta_stacked[:-1]
        dtheta_pi = (dtheta + np.pi) % (2 * np.pi) - np.pi
        dtheta_scaled = (dtheta_pi + np.pi) / (2 * np.pi) # scale to [0, 1]

        preds_out = []
        preys_out = []

        for t in range(window_size):
            if t < window_size - 1:
                # broadcast delta theta to predator tensor
                pred_dt = dtheta_scaled[t, 0].view(1, 1, 1).repeat(1, preds[t].shape[1], 1)

                # broadcast delta theta to prey tensor
                prey_dt = dtheta_scaled[t, 1:].view(-1, 1, 1).repeat(1, preys[t].shape[1], 1)
            else:
                # Imporant: last frame has no delta theta, fill with zeros
                # Action only relevant for pretraining (correctly handled there)
                # GAIL trainings on transition embeddings using states-only, last action therefore no problem (better than dropping)
                pred_dt = torch.zeros((1, preds[t].shape[1], 1), dtype=torch.float32)
                prey_dt = torch.zeros((preys[t].shape[0], preys[t].shape[1], 1), dtype=torch.float32)

            # concatenate as last feature channel
            preds_out.append(torch.cat([preds[t].float(), pred_dt], dim=-1))  
            preys_out.append(torch.cat([preys[t].float(), prey_dt], dim=-1))   

        # stack frames inside the window
        pred_windows.append(torch.stack(preds_out, dim=0))              
        prey_windows.append(torch.stack(preys_out, dim=0))                
        window_coordinates.append(torch.stack(frame_coordinates, dim=0))   

    # stack all windows
    pred_tensor = torch.stack(pred_windows, dim=0)
    prey_tensor = torch.stack(prey_windows, dim=0)  
    coordinates = torch.stack(window_coordinates, dim=0) 

    return pred_tensor, prey_tensor, coordinates


def scale_data(data):
    """
    Convert Label-Studio JSON annotations to predator & prey point arrays.

    Args:
        data: JSON dict with "annotations" -> "result" containing keypoints.

    Returns:
        pred_arr: (1, 2) array with predator [x, y].
        prey_arr: (N, 2) array with prey [x, y].
    """
    prey_pts = []
    pred_pts = None

    # extract points from annotations
    result = data["annotations"][0]["result"]

    for r in result:
        width, height = r["original_width"], r["original_height"]
        value = r["value"]

        # convert percentage coordinates to pixels
        x = (value["x"] / 100.0) * width
        y = (value["y"] / 100.0) * height

        # get label (default to prey)
        labels = value.get("keypointlabels", [])
        label = labels[0] if labels else "Prey"
        
        if label == "Predator":
            pred_pts = (x, y)
        else:
            prey_pts.append((x, y))

    # convert to numpy arrays
    pred_arr = np.array([pred_pts])
    prey_arr = np.array(prey_pts)
    return pred_arr, prey_arr


def hungarian_assign(point_seq):
    """
    Track identities across frames using the Hungarian algorithm.

    Args:
        point_seq: List of (N, 2) arrays of points per frame.

    Returns:
        ordered: (T, N, 2) array of points with consistent ordering over time.
    """
    ordered = [point_seq[0]]
    prev = point_seq[0]

    # for each time step, assign points to previous points using Hungarian algorithm
    for t in range(1, len(point_seq)):
        current_point = point_seq[t]

        # pairwise distances between previous and current points
        distance = np.linalg.norm(prev[:, None, :] - current_point[None, :, :], axis=2)

        # Hungarian assignment
        _, assigned_current_indices = linear_sum_assignment(distance)

        # reorder current points to match previous identities
        current_order = current_point[assigned_current_indices]
        ordered.append(current_order)
        prev = current_order

    # stack ordered points into array
    return np.stack(ordered, axis=0)  # (T, N, 2)


def get_velocity(positions):
    """
    Compute frame-to-frame velocity from positions.

    Assumes every second frame was labeled, so divides by 2.

    Args:
        positions: (T, N, 2) array of positions.

    Returns:
        velocities: (T-1, N, 2) array of velocities.
    """
    velocities = []
    
    for i in range(1, len(positions)):
        # position between consecutive frames
        velo = positions[i] - positions[i - 1]
        velocity = velo / 2 # every second frame got labeled
        velocities.append(velocity)
        
    return np.array(velocities)


def get_records(pred_ordered, prey_ordered, pred_velocities, prey_velocities):
    """
    Build records from positions and velocities in video-pipeline format.

    Args:
        pred_ordered: (T, 1, 2) array of predator positions.
        prey_ordered: (T, N_prey, 2) array of prey positions.
        pred_velocities: (T, 1, 2) array of predator velocities.
        prey_velocities: (T, N_prey, 2) array of prey velocities.

    Returns:
        records: List of dicts with frame, track_id, label, x, y, vx, vy, speed, angle.
    """
    records = []
    pred_vel = pred_velocities.shape[0]
    prey_num = prey_ordered.shape[1]

    for step in range(pred_vel):
        x  = float(pred_ordered[step, 0, 0])
        y  = float(pred_ordered[step, 0, 1])
        vx = float(pred_velocities[step, 0, 0])
        vy = float(pred_velocities[step, 0, 1])

        # predator record
        records.append({"frame": step,
                        "track_id": 1,
                        "label": "1",
                        "conf": 1.0,
                        "x": x, 
                        "y": y,
                        "vx": vx, 
                        "vy": vy,
                        "speed": float(math.hypot(vx, vy)),
                        "angle": float(math.atan2(vy, vx))})

        # prey records
        for i in range(prey_num):
            x  = float(prey_ordered[step, i, 0])
            y  = float(prey_ordered[step, i, 1])
            vx = float(prey_velocities[step, i, 0])
            vy = float(prey_velocities[step, i, 1])

            records.append({"frame": step,
                            "track_id": i + 1,
                            "label": "2",
                            "conf": 1.0,
                            "x": x, 
                            "y": y,
                            "vx": vx, 
                            "vy": vy,
                            "speed": float(math.hypot(vx, vy)),
                            "angle": float(math.atan2(vy, vx))})

    return records


def get_hl_expert_tensors(records, max_speed):
    """
    Convert hand-labeled records into expert tensors.

    Args:
        records: List of detection dicts with "frame", "x", "y", "vx", "vy", etc.
        max_speed: Max speed for velocity normalization.

    Returns:
        pred_tensor: (T, 1, N-1, F+1) predator tensor (last channel = scaled dtheta/action).
        prey_tensor: (T, N-1, N-1, F+1) prey tensor.
    """
    preds = []
    preys = []
    thetas = []

    # process frames in order
    frame_ids = sorted({rec["frame"] for rec in records})
    
    for frame_idx in frame_ids:    
        # get records for the frame
        frame = [rec for rec in records if rec["frame"] == frame_idx]
        if not frame:
            continue

        # convert frame records into expert tensors
        pred_tensor, prey_tensor, xs, ys, theta = get_expert_features(frame, width=2160, height=2160, max_speed=max_speed)
        preds.append(pred_tensor)
        preys.append(prey_tensor)
        thetas.append(torch.tensor(theta, dtype=torch.float32))

    # return empty if nothing collected
    if len(preds) == 0:
        return torch.empty(0), torch.empty(0)

    # compute delta theta across frames
    theta_stacked = torch.stack(thetas, dim=0)
    dtheta = theta_stacked[1:] - theta_stacked[:-1]
    # wrap to [-pi, pi] and scale to [0, 1]
    dtheta_pi = (dtheta + np.pi) % (2 * np.pi) - np.pi
    dtheta_scaled = (dtheta_pi + np.pi) / (2 * np.pi)

    preds_out = []
    preys_out = []
    size = len(preds)

    for t in range(size):
        if t < size - 1:
            # delta theta for predator
            pred_dt = dtheta_scaled[t, 0].view(1, 1, 1).repeat(1, preds[t].shape[1], 1)

            # delta theta for prey
            prey_dt = dtheta_scaled[t, 1:].view(-1, 1, 1).repeat(1, preys[t].shape[1], 1)
        else:
            # last frame has no delta theta (use zeros)
            pred_dt = torch.zeros((1, preds[t].shape[1], 1), dtype=torch.float32)
            prey_dt = torch.zeros((preys[t].shape[0], preys[t].shape[1], 1), dtype=torch.float32)

        # append as last feature channel
        preds_out.append(torch.cat([preds[t].float(), pred_dt], dim=-1))
        preys_out.append(torch.cat([preys[t].float(), prey_dt], dim=-1))

    # stack frames into tensors
    pred_tensor = torch.stack(preds_out, dim=0)
    prey_tensor = torch.stack(preys_out, dim=0)

    return pred_tensor, prey_tensor


def extract_tensor_windows(tensor, window_len=10):
    """
    Split a time-series tensor into sliding windows.

    Args:
        tensor: (T, ...) tensor.
        window_len: Length of each window.

    Returns:
        (num_windows, window_len, ...) tensor of windows.
    """
    T = tensor.shape[0]
    windows = [
        tensor[start:start + window_len]
        for start in range(T - window_len + 1)
    ]
    return torch.stack(windows, dim=0)


def pad_expert_tensors(pred_tensor, prey_tensor, max_prey=32):
    """
    Pad expert tensors to max_prey and insert an 'active' mask channel
    before the last column (so the true action label stays at index -1).

    Args:
        pred_tensor: (T, 1, N, F) predator tensor.
        prey_tensor: (T, N, N, F) prey tensor.
        max_prey: Target number of prey agents.

    Returns:
        pred_tensor_padded: (T, 1, max_prey, F+1)
        prey_tensor_padded: (T, max_prey, max_prey, F+1)
    """
    n_prey = pred_tensor.shape[-2]
    
    if n_prey > max_prey:
        raise ValueError(f"n_prey={n_prey} exceeds max_prey={max_prey}")
    
    pad_n = max_prey - n_prey

    # predator: insert active channel before last, then pad neighbor axis
    active = torch.ones(pred_tensor.shape[:-1] + (1,), dtype=pred_tensor.dtype)
    pred_tensor = torch.cat([pred_tensor[..., :-1], active, pred_tensor[..., -1:]], dim=-1)
    
    if pad_n > 0:
        shape = list(pred_tensor.shape)
        shape[-2] = pad_n
        pred_tensor = torch.cat([pred_tensor, torch.zeros(shape, dtype=pred_tensor.dtype)], dim=-2)

    # prey: insert active channel before last, then pad neighbor and agent axes
    active = torch.ones(prey_tensor.shape[:-1] + (1,), dtype=prey_tensor.dtype)
    prey_tensor = torch.cat([prey_tensor[..., :-1], active, prey_tensor[..., -1:]], dim=-1)
    
    if pad_n > 0:
        # pad neighbor axis (dim=-2)
        neigh_shape = list(prey_tensor.shape)
        neigh_shape[-2] = pad_n
        prey_tensor = torch.cat([prey_tensor, torch.zeros(neigh_shape, dtype=prey_tensor.dtype)], dim=-2)
        # pad agent axis (dim=-3)
        agent_shape = list(prey_tensor.shape)
        agent_shape[-3] = pad_n
        prey_tensor = torch.cat([prey_tensor, torch.zeros(agent_shape, dtype=prey_tensor.dtype)], dim=-3)

    return pred_tensor, prey_tensor


def pad_rollout_tensors(pred_tensor, prey_tensor, max_prey=32):
    """
    Pad rollout tensors that already have an 'active' channel.
    Only pads neighbor/agent axes with zeros; does not insert a new active column.

    Args:
        pred_tensor: (T, 1, N, F) predator tensor (already has active channel).
        prey_tensor: (T, N, N, F) prey tensor (already has active channel).
        max_prey: Target number of prey agents.

    Returns:
        pred_tensor_padded: (T, 1, max_prey, F)
        prey_tensor_padded: (T, max_prey, max_prey, F)
    """
    n_prey = pred_tensor.shape[-2]
    if n_prey > max_prey:
        raise ValueError(f"n_prey={n_prey} exceeds max_prey={max_prey}")
    
    pad_n = max_prey - n_prey
    if pad_n == 0:
        return pred_tensor, prey_tensor

    # pad predator neighbor axis
    shape = list(pred_tensor.shape); shape[-2] = pad_n
    pred_tensor = torch.cat([pred_tensor, torch.zeros(shape, dtype=pred_tensor.dtype, device=pred_tensor.device)], dim=-2)

    # pad prey neighbor axis
    neigh_shape = list(prey_tensor.shape); neigh_shape[-2] = pad_n
    prey_tensor = torch.cat([prey_tensor, torch.zeros(neigh_shape, dtype=prey_tensor.dtype, device=prey_tensor.device)], dim=-2)

    # pad prey agent axis
    agent_shape = list(prey_tensor.shape); agent_shape[-3] = pad_n
    prey_tensor = torch.cat([prey_tensor, torch.zeros(agent_shape, dtype=prey_tensor.dtype, device=prey_tensor.device)], dim=-3)

    return pred_tensor, prey_tensor
