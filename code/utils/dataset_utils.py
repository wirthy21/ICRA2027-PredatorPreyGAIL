import os
import cv2
import math
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from deep_sort_realtime.deepsort_tracker import DeepSort

"""
1. Part - Functions for dataset processing from videos
2. Part - Functions for processing hand-labeled trajectory data

References:
YOLO:            https://docs.ultralytics.com/usage/python/
DeepSORT:        https://pypi.org/project/deep-sort-realtime/
DeepSORT GitHub: https://github.com/nwojke/deep_sort
Data Labeling:   https://labelstud.io/
Hungarian Alg.:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)
"""

##### Video Processing Functions #####

def process_frame(cap, model, tracker, frame_idx, device="cpu"):
    """
    Processes a video to obtain track records

    Input: video, yolo model, deepsort tracker, frame index, device
    Output: dict of track records for the frame
    """
    _, frame = cap.read()
    height, width = frame.shape[:2]

    # run yolo on the frame
    result = model(frame, verbose=False, device=device)[0]

    xywh = result.boxes.xywh.cpu().numpy() # bounding boxes (x_center, y_center, width, height)
    confs = result.boxes.conf.cpu().numpy() # confidence per detection
    cls_ids = result.boxes.cls.cpu().numpy().astype(int) # class id per detection

    # necessary format for DeepSORT
    raw_detections = list(zip(xywh, confs, cls_ids))

    # update tracks, with detections and the current frame
    tracks = tracker.update_tracks(raw_detections, frame=frame)

    records = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        # track mean contains kalman filter states (x, y, w, h, vx, vy)
        x_raw = track.mean[0]
        y_raw = track.mean[1]

        # clip coordinates for frame bounds (necessary due to kalman filter predictions)
        x = float(np.clip(x_raw, 0, width))
        y = float(np.clip(y_raw, 0, height))

        records.append({"frame":    int(frame_idx),
                        "track_id": int(track.track_id),
                        "label":    str(track.det_class),
                        "conf":     track.det_conf,
                        "x":        x,
                        "y":        y,
                        "vx":       float(track.mean[4]),
                        "vy":       float(track.mean[5]),
                        "speed":    abs(float(math.hypot(track.mean[4], track.mean[5]))),
                        "angle":    float(math.atan2(track.mean[5], track.mean[4]))})
        
    return records


def filter_frames(total_frames):
    """
    Filters frames to keep only one predator and all prey detections

    Input: records of all frames
    Output: filtered records with only best predator and all prey detections, max speed for env settings
    """

    # use only necessary detections of Pred Head and Prey {0: 'Predator', 1: 'Predator Head', 2: 'Prey', 3: 'Prey Head'}
    pred_prey_frames = [frame for frame in total_frames if frame['label'] in ('1', '2')] #Pred Head 1, Prey 2

    # filter detections with None confidence
    filtered_conf = [frame for frame in pred_prey_frames if frame['conf'] is not None]

    # drop multiple pred detections, keep only the one with highest confidence
    best_pred_label = {}
    preys = []
    for data in filtered_conf:
        frame = data['frame']
        if data['label'] == '1':
            if frame not in best_pred_label or data['conf'] > best_pred_label[frame]['conf']:
                best_pred_label[frame] = data
        else:
            preys.append(data)

    # combine best pred and all preys
    best_pred_prey_frames = list(best_pred_label.values()) + preys

    # compute max speed among all detections
    all_speeds = [det["speed"] for det in best_pred_prey_frames]
    max_speed = max(all_speeds)

    return best_pred_prey_frames, max_speed


def find_valid_windows(filtered_frames, window_len=10, total_detections=33):
    """
    Finds continuous windows where all detections are present

    Input: records of all filtered frames, window length, total number of detections expected
    Output: continuous windows where all detections are present
    """

    # get all track ids per frame
    ids_by_frame = defaultdict(set)
    for d in filtered_frames:
        ids_by_frame[int(d["frame"])].add(int(d["track_id"]))

    frames = sorted(ids_by_frame.keys())

    episodes = []
    i = 0

    while i < len(frames):
        frame = frames[i]
        inital_ids = ids_by_frame[frame]

        # skip if number of detections is wrong
        if len(inital_ids) != total_detections:
            i += 1
            continue

        start = frame
        end = frame

        # extend the episode when frames are consecutive and have same ids
        while i + 1 < len(frames):
            next_frame = frames[i + 1]
            if next_frame != end + 1:
                break
            if ids_by_frame[next_frame] != inital_ids:
                break
            i += 1
            end = next_frame

        length = end - start + 1
        
        # keep valid windows only
        if length >= window_len:
            episodes.append({"start_frame": start,
                             "end_frame": end,
                             "length": length,
                             "ids": sorted(inital_ids)})

        i += 1

    return episodes


def extract_windows(episodes, window_len=10):
    """
    Splits episodes into smaller sliding windows

    Input: valid episodes, window length
    Output: extracted windows of given length from episodes
    """
    windows = []

    for episode in episodes:
        clip_len = episode["length"]
        if clip_len < window_len:
            continue

        clip_start = episode["start_frame"]
        ids = episode["ids"]

        # number of windows that can be extracted from the episode
        num_windows = clip_len - window_len + 1
        for offset in range(num_windows):
            window_start = clip_start + offset
            window_end = window_start + window_len - 1

            windows.append({"start_frame": window_start,
                            "end_frame": window_end,
                            "length": window_len,
                            "ids": ids})

    return windows



def get_expert_features(frame, width, height, max_speed=10):    
    """
    Converts detections in a frame to expert feature tensors (derived from Wu et al. 2025)

    Input: frame detections, frame width, frame height, max speed for normalization
    Output: predator & prey tensor and x, y, theta for init_pool
    """

    # Sort frames so that predator head is always first
    frame = sorted(frame, key=lambda det: (det['label'] != '1', int(det['track_id']))) # sort so that Pred Head is always first

    # extract positions
    xs = np.array([det['x'] for det in frame])
    ys = np.array([det['y'] for det in frame])

    # clip to frame bounds (necessary due to kalman filter predictions)
    clipped_xs = np.clip(xs, 0, width)
    clipped_ys = np.clip(ys, 0, height)

    # scale positions to [0, 1]
    scaled_xs = clipped_xs / width
    scaled_ys = clipped_ys / height

    # extract velocities
    vxs = np.array([det['vx'] for det in frame])
    vys = np.array([det['vy'] for det in frame])

    # extract heading angle, scale to [0, 1]
    thetas = np.array([det['angle'] for det in frame])

    # compute pairwise distances
    dx = scaled_xs[None, :] - scaled_xs[:, None] # [-1, 1]
    dy = scaled_ys[None, :] - scaled_ys[:, None] # [-1, 1]

    # compute relative velocities in the agent's heading direction
    cos_t = np.cos(thetas)                        
    sin_t = np.sin(thetas)
    rel_vx = cos_t[:, None] * vxs[None, :] + sin_t[:, None] * vys[None, :]
    rel_vy = -sin_t[:, None] * vxs[None, :] + cos_t[:, None] * vys[None, :]

    # clip and scale relative velocities to [-1, 1], clipping necessary due to detection jumps
    scaled_rel_vx = np.clip(rel_vx, -max_speed, max_speed) / max_speed
    scaled_rel_vy = np.clip(rel_vy, -max_speed, max_speed) / max_speed

    # build feature tensor
    features = np.stack([dx, dy, scaled_rel_vx, scaled_rel_vy], axis=-1)

    # remove self-interactions
    n = scaled_xs.shape[0]
    mask = ~np.eye(n, dtype=bool)
    neigh = features[mask].reshape(n, n-1, 4)

    # separate predator and prey tensors
    pred_tensor = torch.from_numpy(neigh[0]).unsqueeze(0)
    prey_tensor = torch.from_numpy(neigh[1:])

    return pred_tensor, prey_tensor, scaled_xs.tolist(), scaled_ys.tolist(), thetas.tolist()


def get_expert_tensors(filtered_frames, extracted_windows, width, height, max_speed=10, window_size=5):
    """
    Builts expert tensor for all extracted windows

    Input: frame detections, valid windows, frame width, frame height, max speed for normalization, window size
    Output: predator & prey tensor window, and coordinates for init_pool
    """

    # return empty tensors if no windows extracted
    if len(extracted_windows) == 0:
        return torch.empty(0), torch.empty(0)
    
    # group detections by frame number
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



##### Hand-labeled Trajectories Functions #####


def scale_data(data):
    """
    Convert annotations from JSON format into predator & prey point arrays

    Input: JSON data
    Output: predator & prey point arrays
    """
    prey_pts = []
    pred_pts = None

    # extract points from annotations (Label-Studio structure)
    result = data["annotations"][0]["result"]

    for r in result:
        width, height = r["original_width"], r["original_height"]

        # coordinates are stored as percentage values, convert to pixel values
        value = r["value"]
        x = (value["x"] / 100.0) * width
        y = (value["y"] / 100.0) * height

        # get label for the point
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
    Identity tracking across frames using Hungarian algorithm

    Input: predator & prey point arrays
    Output: ordered predator & prey point arrays
    """
    ordered = [point_seq[0]]
    prev = point_seq[0]

    # for each time step, assign points to previous points using Hungarian algorithm
    for t in range(1, len(point_seq)):
        current_point = point_seq[t]

        # pairwise distance matrix
        distance = np.linalg.norm(prev[:, None, :] - current_point[None, :, :], axis=2)

        # best matching with Hungarian algorithm
        _, assigned_current_indices = linear_sum_assignment(distance)

        # reorder current points based on assignment
        current_order = current_point[assigned_current_indices]
        ordered.append(current_order)
        prev = current_order

    # stack ordered points into array
    ordered = np.stack(ordered, axis=0)  # (T, N, 2)
    return ordered


def get_velocity(positions):
    """
    Computes frame-to-frame velocity from positions

    Input: positions from ordered predator & prey point arrays
    Output: velocities
    """
    velocities = []
    for i in range(1, len(positions)):
        # position between consecutive frames
        velo = positions[i] - positions[i - 1]
        velocity = velo / 2 #every second frame got labeled
        velocities.append(velocity)
    return np.array(velocities)


def get_records(pred_ordered, prey_ordered, pred_velocities, prey_velocities):
    """
    Builds records from positions and velocities, to fit video pipeline format

    Input: positions and velocities of predator & prey
    Output: records list
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
    Convert hand-labeled records into expert tensors

    Input: records, max speed for normalization
    Output: predator & prey expert tensors
    """
    preds = []
    preys = []
    thetas = []

    # Iterate frames in correct order
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

        theta_tensor = torch.tensor(theta, dtype=torch.float32)
        thetas.append(theta_tensor)

    # return empty if nothing collected
    if len(preds) == 0:
        return torch.empty(0), torch.empty(0)

    # compute delta theta across frames
    theta_stacked = torch.stack(thetas, dim=0)
    dtheta = theta_stacked[1:] - theta_stacked[:-1]
    dtheta_pi = (dtheta + np.pi) % (2 * np.pi) - np.pi
    dtheta_scaled = (dtheta_pi + np.pi) / (2 * np.pi)  # scale to [0, 1]

    preds_out = []
    preys_out = []
    size = len(preds)

    for t in range(size):
        if t < size - 1:
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

    # stack frames into tensors
    pred_tensor = torch.stack(preds_out, dim=0)
    prey_tensor = torch.stack(preys_out, dim=0)

    return pred_tensor, prey_tensor


def extract_tensor_windows(tensor, window_len=10):
    """
    Splits a tensor along time dimension into sliding windows

    Input: pred & prey tensor
    Output: stacked windows
    """
    clip_len = tensor.shape[0]
    windows = []
    for start in range(clip_len - window_len + 1):
        windows.append(tensor[start:start + window_len])

    return torch.stack(windows, dim=0)
