"""Reusable trajectory analyses for the ICRA 2027 predator--prey paper.

This module is deliberately isolated from the Jannik and Nicole repositories.
Those repositories are imported/read as sources, but never written to.

Geometry convention
-------------------
All canonical trajectories use a mathematical world frame (positive y points
up).  A focal animal's local frame is ``+x = forward, +y = left``.  Nicole's
image coordinates (positive y down) are converted by ``y -> height - y`` and
``heading -> -heading`` at the adapter boundary.  Nicole's learned-policy
simulator already uses the mathematical convention and is not transformed.

Time convention
---------------
Every trajectory may carry an explicit ``step_duration``.  The biological
pipeline uses 15 effective samples per second because both tracking sources
retain every second frame of 30-fps video.  Couzin time remains simulation
time and is never labelled as biological seconds.

Module layout
-------------
The first part retains the original NumPy loading and analysis helpers for
backward compatibility.  The later tensor workflow performs the complete
multi-case paper analysis, followed by plotting utilities and deterministic
sanity checks.  Section comments mark these responsibilities so the file can
be read from top to bottom without following every helper immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import importlib
import inspect
import os
import pickle
import sys
import types
import warnings

import numpy as np
import torch
from tqdm import tqdm


REQUIRED_COLUMNS = (
    "source", "condition", "clip_id", "frame", "time_step", "agent_id",
    "role", "x", "y", "heading",
)
OPTIONAL_COLUMNS = (
    "vx", "vy", "fps", "timestamp", "policy_id", "step_duration",
    "sampling_stride", "rollout_seed", "simulation_seed", "wall_mode",
)
FLOAT_COLUMNS = {
    "x", "y", "heading", "vx", "vy", "fps", "timestamp",
    "step_duration", "sampling_stride", "rollout_seed", "simulation_seed",
}


class ExpertDataUnavailable(RuntimeError):
    """Raised when a scientifically adequate continuous expert source is absent."""


@dataclass(frozen=True)
class TrajectoryTable:
    """Lightweight columnar canonical per-agent/per-frame trajectory table."""

    columns: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        missing = [name for name in REQUIRED_COLUMNS if name not in self.columns]
        if missing:
            raise ValueError(f"Missing canonical columns: {missing}")
        lengths = {len(np.asarray(value)) for value in self.columns.values()}
        if len(lengths) != 1:
            raise ValueError("All trajectory columns must have equal length")
        n = next(iter(lengths), 0)
        normalized = {k: np.asarray(v) for k, v in self.columns.items()}
        for name in OPTIONAL_COLUMNS:
            if name not in normalized:
                if name in FLOAT_COLUMNS:
                    normalized[name] = np.full(n, np.nan, dtype=float)
                else:
                    normalized[name] = np.full(n, "", dtype=object)
        object.__setattr__(self, "columns", normalized)

    def __len__(self) -> int:
        return len(self.columns["x"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    def subset(self, mask: np.ndarray) -> "TrajectoryTable":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError("Subset mask has the wrong shape")
        return TrajectoryTable({k: v[mask] for k, v in self.columns.items()})

    def to_records(self) -> list[dict[str, Any]]:
        return [{k: _python_scalar(v[i]) for k, v in self.columns.items()} for i in range(len(self))]


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def to_canonical_trajectory(records: Iterable[Mapping[str, Any]]) -> TrajectoryTable:
    """Validate records and return the canonical columnar representation."""

    rows = list(records)
    if not rows:
        return TrajectoryTable({
            **{name: np.asarray([], dtype=object) for name in REQUIRED_COLUMNS},
            **{name: np.asarray([], dtype=float if name in FLOAT_COLUMNS else object)
               for name in OPTIONAL_COLUMNS},
        })
    missing = [name for name in REQUIRED_COLUMNS if any(name not in row for row in rows)]
    if missing:
        raise ValueError(f"Records are missing required fields: {sorted(set(missing))}")
    names = list(dict.fromkeys(REQUIRED_COLUMNS + OPTIONAL_COLUMNS + tuple(k for r in rows for k in r)))
    columns: dict[str, np.ndarray] = {}
    for name in names:
        default: Any = np.nan if name in FLOAT_COLUMNS else ""
        columns[name] = np.asarray([row.get(name, default) for row in rows])
    for name in ("frame", "time_step"):
        columns[name] = columns[name].astype(float)
    for name in FLOAT_COLUMNS & set(columns):
        columns[name] = columns[name].astype(float)
    columns["heading"] = wrap_angle(columns["heading"])
    roles = set(columns["role"].astype(str))
    if not roles <= {"predator", "prey"}:
        raise ValueError(f"Unknown roles: {sorted(roles - {'predator', 'prey'})}")
    return TrajectoryTable(columns)


def concatenate_trajectory_tables(tables: Sequence[TrajectoryTable]) -> TrajectoryTable:
    """Concatenate canonical tables while preserving the union of metadata columns."""

    tables = [table for table in tables if len(table)]
    if not tables:
        return to_canonical_trajectory([])
    return to_canonical_trajectory(row for table in tables for row in table.to_records())


def with_constant_metadata(table: TrajectoryTable, **metadata: Any) -> TrajectoryTable:
    """Return a copy with constant per-row metadata, leaving the source table unchanged."""

    records = table.to_records()
    for record in records:
        record.update(metadata)
    return to_canonical_trajectory(records)


def initial_state_pool_from_trajectory(
    table: TrajectoryTable, *, expected_n_prey: int,
) -> torch.Tensor:
    """Extract one absolute ``[x, y, heading]`` state per independent clip."""

    states: list[torch.Tensor] = []
    seen: set[tuple[str, str]] = set()
    for indices in frame_groups(table):
        key = (str(table["source"][indices[0]]), str(table["clip_id"][indices[0]]))
        if key in seen:
            continue
        seen.add(key)
        roles = table["role"][indices].astype(str)
        pred = indices[roles == "predator"]
        prey = indices[roles == "prey"]
        if len(pred) != 1 or len(prey) != expected_n_prey:
            raise ValueError(f"{key}: expected one predator and {expected_n_prey} prey")
        ordered = np.concatenate((pred, prey))
        values = np.stack(
            (table["x"][ordered], table["y"][ordered], table["heading"][ordered]), axis=-1,
        )
        states.append(torch.as_tensor(values, dtype=torch.float32))
    if not states:
        raise ValueError("No complete initial states were available")
    return torch.stack(states)


# ---------------------------------------------------------------------------
# Geometry helpers

# These small NumPy helpers define the coordinate, angle, and frame-grouping
# conventions used by the original loaders.  Keeping them together makes the
# transformations applied at source boundaries explicit.


def wrap_angle(angle: Any) -> Any:
    """Wrap angle(s) to [-pi, pi]."""

    arr = np.asarray(angle, dtype=float)
    wrapped = (arr + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def angular_difference(angle_after: Any, angle_before: Any) -> Any:
    """Return wrapped ``angle_after - angle_before``."""

    return wrap_angle(np.asarray(angle_after) - np.asarray(angle_before))


def euclidean_distance(a: Any, b: Any) -> Any:
    """Euclidean distance along the final array axis."""

    result = np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=-1)
    return float(result) if np.ndim(result) == 0 else result


def prey_centroid(prey_positions: np.ndarray) -> np.ndarray:
    """Centroid of active prey positions."""

    positions = np.asarray(prey_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("prey_positions must have shape (N, 2), N > 0")
    return positions.mean(axis=0)


def predator_to_nearest_prey_distance(predator_position: Any, prey_positions: np.ndarray) -> float:
    """Distance from the single predator to the closest active prey."""

    positions = np.asarray(prey_positions, dtype=float)
    if len(positions) == 0:
        return float("nan")
    return float(np.min(euclidean_distance(positions, np.asarray(predator_position))))


def focal_frame_transform(displacement: Any, focal_heading: Any) -> np.ndarray:
    """Rotate world displacement into ``+x forward, +y left`` focal axes."""

    delta = np.asarray(displacement, dtype=float)
    theta = np.asarray(focal_heading, dtype=float)
    dx, dy = delta[..., 0], delta[..., 1]
    c, s = np.cos(theta), np.sin(theta)
    return np.stack((c * dx + s * dy, -s * dx + c * dy), axis=-1)


def vector_angle(vector: Any) -> Any:
    """Bearing angle of 2-D vector(s) in the canonical world frame."""

    vector = np.asarray(vector, dtype=float)
    result = np.arctan2(vector[..., 1], vector[..., 0])
    return float(result) if result.ndim == 0 else result


def away_from_predator_direction(prey_position: Any, predator_position: Any) -> Any:
    """Direction from predator toward prey."""

    return vector_angle(np.asarray(prey_position, dtype=float) - np.asarray(predator_position, dtype=float))


def heading_alignment(heading: Any, target_direction: Any) -> Any:
    """Cosine alignment: +1 aligned, 0 orthogonal, -1 opposite."""

    result = np.cos(np.asarray(heading, dtype=float) - np.asarray(target_direction, dtype=float))
    return float(result) if result.ndim == 0 else result


def active_prey_filter(table: TrajectoryTable) -> np.ndarray:
    """Return the canonical prey mask; inactive detections should be omitted by adapters."""

    return table["role"].astype(str) == "prey"


def frame_groups(table: TrajectoryTable) -> list[np.ndarray]:
    """Indices grouped by source, clip, and ordered source timestep."""

    groups: dict[tuple[str, str, float], list[int]] = {}
    for i, key in enumerate(zip(table["source"].astype(str), table["clip_id"].astype(str), table["time_step"])):
        groups.setdefault(key, []).append(i)
    return [np.asarray(groups[key], dtype=int) for key in sorted(groups, key=lambda x: (x[0], x[1], x[2]))]


# ---------------------------------------------------------------------------
# Data adapters and repository-isolated rollout bridge

# Source repositories are treated as read-only dependencies.  Imports are
# isolated because both projects use generic package names such as ``models``
# and ``utils`` that would otherwise remain in ``sys.modules``.


def _is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with open(_windows_binary_path(path), "rb") as handle:
            return handle.read(100).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _windows_binary_path(path: Path) -> str:
    """Support Nicole's deeply nested data paths on Windows."""

    resolved = str(path.resolve())
    return "\\\\?\\" + resolved if os.name == "nt" and not resolved.startswith("\\\\?\\") else resolved


def _records_from_pickle_object(obj: Any) -> list[dict[str, Any]]:
    if hasattr(obj, "to_dict"):
        try:
            return list(obj.to_dict("records"))
        except TypeError:
            pass
    if isinstance(obj, Mapping):
        if obj and all(hasattr(v, "__len__") and not isinstance(v, (str, bytes)) for v in obj.values()):
            lengths = {len(v) for v in obj.values()}
            if len(lengths) == 1:
                return [{k: v[i] for k, v in obj.items()} for i in range(next(iter(lengths)))]
        for key in ("records", "data", "frames", "trajectories"):
            if key in obj:
                return _records_from_pickle_object(obj[key])
    if isinstance(obj, (list, tuple)):
        rows: list[dict[str, Any]] = []
        for item in obj:
            if isinstance(item, Mapping) and not any(k in item for k in ("records", "data", "frames")):
                rows.append(dict(item))
            else:
                rows.extend(_records_from_pickle_object(item))
        return rows
    raise ValueError(f"Unsupported expert pickle object: {type(obj).__name__}")


def _pick(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    lowered = {str(k).lower(): k for k in row}
    for name in names:
        if name.lower() in lowered:
            return row[lowered[name.lower()]]
    return default


def _infer_role(value: Any, role_map: Mapping[str, str] | None = None) -> str:
    text = str(value).lower()
    if role_map is not None and text in {str(k).lower(): v for k, v in role_map.items()}:
        mapped = {str(k).lower(): str(v).lower() for k, v in role_map.items()}[text]
        if mapped not in {"predator", "prey"}:
            raise ValueError(f"Invalid mapped role {mapped!r} for label {value!r}")
        return mapped
    if "pred" in text or text in {"0", "p", "hunter"}:
        return "predator"
    if "prey" in text or "fish" in text or text in {"1", "f"}:
        return "prey"
    raise ValueError(f"Cannot infer predator/prey role from label {value!r}")


def load_expert_32prey(
    source: str | Path,
    *,
    condition: str | None = None,
    fps: float | None = None,
    image_y_down: bool = True,
    coordinate_height: float = 2160.0,
    require_complete_frames: bool = True,
    role_map: Mapping[str, str] | None = None,
    n_prey: int = 32,
) -> TrajectoryTable:
    """Load continuous Nicole/Jannik records into the canonical representation.

    Expected records contain global ``frame, track_id, label, x, y`` and either
    ``angle``/``heading`` or ``vx, vy``.  A directory is searched recursively
    for continuous ``*.pkl`` files; 10-frame window tensors are intentionally
    rejected because global geometry/provenance cannot be reconstructed safely.
    """

    root = Path(source)
    if root.is_dir():
        candidates = sorted(root.rglob("*.pkl"))
        candidates = [p for p in candidates if "window" not in str(p).lower() and "max_speed" not in p.name.lower()]
    elif root.is_file():
        candidates = [root]
    else:
        raise ExpertDataUnavailable(
            f"NOT AVAILABLE / INSUFFICIENT EXPERT TRAJECTORY DATA: {root} does not exist. "
            f"Provide the configured continuous {n_prey}-prey source; window tensors are not substituted."
        )
    if not candidates:
        raise ExpertDataUnavailable(
            "NOT AVAILABLE / INSUFFICIENT EXPERT TRAJECTORY DATA: no continuous pickle files found."
        )
    pointers = [p for p in candidates if _is_lfs_pointer(p)]
    usable = [p for p in candidates if p not in pointers]
    if not usable:
        raise ExpertDataUnavailable(
            "NOT AVAILABLE / INSUFFICIENT EXPERT TRAJECTORY DATA: configured continuous files are absent or "
            "Git-LFS pointers only. Hydrate the configured source outside this analysis workflow."
        )

    canonical_rows: list[dict[str, Any]] = []
    for path in usable:
        with open(_windows_binary_path(path), "rb") as handle:
            raw_rows = _records_from_pickle_object(pickle.load(handle))
        clip_id = path.name.removesuffix(".pkl")
        path_condition = "attack" if "attack" in str(path).lower() else "interaction" if "interaction" in str(path).lower() else "unknown"
        prepared: list[tuple[int, Mapping[str, Any], str, float]] = []
        for order, row in enumerate(raw_rows):
            role = _infer_role(_pick(row, ("label", "role", "class", "agent_type")), role_map)
            frame = float(_pick(row, ("frame", "frame_id", "frame_idx", "time_step"), order))
            prepared.append((order, row, role, frame))
        if require_complete_frames:
            counts: dict[float, list[int]] = {}
            for _, _, role, frame in prepared:
                count = counts.setdefault(frame, [0, 0])
                count[0 if role == "predator" else 1] += 1
            eligible_frames = {frame for frame, count in counts.items() if count == [1, n_prey]}
            prepared = [item for item in prepared if item[3] in eligible_frames]
        for order, row, role, frame in prepared:
            x = float(_pick(row, ("x", "center_x", "x_center")))
            y_raw = float(_pick(row, ("y", "center_y", "y_center")))
            vx_raw = _pick(row, ("vx", "velocity_x"), np.nan)
            vy_raw = _pick(row, ("vy", "velocity_y"), np.nan)
            heading_raw = _pick(row, ("angle", "heading", "theta"), None)
            if heading_raw is None:
                if not np.isfinite(float(vx_raw)) or not np.isfinite(float(vy_raw)):
                    raise ValueError(f"No heading or finite velocity in {path.name}, frame {frame}")
                heading_raw = np.arctan2(float(vy_raw), float(vx_raw))
            y = coordinate_height - y_raw if image_y_down else y_raw
            heading = -float(heading_raw) if image_y_down else float(heading_raw)
            vx = float(vx_raw)
            vy = -float(vy_raw) if image_y_down else float(vy_raw)
            row_fps = _pick(row, ("fps", "frame_rate"), fps if fps is not None else np.nan)
            timestamp = _pick(row, ("timestamp", "time"), np.nan)
            canonical_rows.append({
                "source": "expert", "condition": path_condition, "clip_id": clip_id,
                "frame": frame, "time_step": frame,
                "agent_id": str(_pick(row, ("track_id", "agent_id", "id"))), "role": role,
                "x": x, "y": y, "heading": wrap_angle(heading), "vx": vx, "vy": vy,
                "fps": float(row_fps) if row_fps is not None else np.nan,
                "timestamp": float(timestamp) if timestamp is not None else np.nan,
            })
    table = to_canonical_trajectory(canonical_rows)
    if condition is not None and condition != "all":
        table = table.subset(table["condition"].astype(str) == condition)
    if require_complete_frames:
        keep = np.zeros(len(table), dtype=bool)
        for indices in frame_groups(table):
            roles = table["role"][indices].astype(str)
            if np.sum(roles == "predator") == 1 and np.sum(roles == "prey") == n_prey:
                keep[indices] = True
        dropped = int(len(table) - keep.sum())
        if dropped:
            warnings.warn(f"Dropped {dropped} expert rows from incomplete frames; no threshold was imputed.")
        table = table.subset(keep)
    if len(table) == 0:
        raise ExpertDataUnavailable(
            f"Continuous files were readable, but no complete one-predator/{n_prey}-prey frames remained."
        )
    return table


def load_expert_trajectories(source: str | Path, *, n_prey: int, **kwargs: Any) -> TrajectoryTable:
    """Prey-count-generic entry point; the historical loader remains compatible."""

    return load_expert_32prey(source, n_prey=n_prey, **kwargs)


def _load_torch_tensor(path: Path) -> torch.Tensor:
    """Load a tensor artifact on CPU, including long Windows paths."""

    binary_path = _windows_binary_path(path)
    if not os.path.isfile(binary_path):
        raise ExpertDataUnavailable(f"Usable pairwise tensor file not found: {path}")
    try:
        with open(binary_path, "rb") as handle:
            is_pointer = handle.read(100).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        is_pointer = True
    if is_pointer:
        raise ExpertDataUnavailable(f"Pairwise tensor is an unavailable Git-LFS pointer: {path}")
    try:
        value = torch.load(binary_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument
        value = torch.load(binary_path, map_location="cpu")
    if not torch.is_tensor(value):
        raise ValueError(f"Expected one tensor in {path.name}, got {type(value).__name__}")
    return value.detach().cpu()


def _is_binary_channel(values: torch.Tensor, *, atol: float = 1e-6) -> bool:
    finite = values[torch.isfinite(values)]
    return bool(len(finite) and torch.all((finite.abs() <= atol) | ((finite - 1.0).abs() <= atol)))


def _pairwise_tensor_layout(
    predator_windows: torch.Tensor, prey_windows: torch.Tensor,
) -> dict[str, Any]:
    """Validate saved pairwise windows and locate mask/flag channels.

    The hand-labelled files store ``[dx, dy, rel_vx, rel_vy, active,
    theta_norm]`` for the predator. Prey files additionally prepend the flag
    that identifies neighbor slot zero as the predator.
    """

    if predator_windows.ndim != 5 or predator_windows.shape[2] != 1:
        raise ValueError("Predator windows must have shape [W,L,1,M,F]")
    if prey_windows.ndim != 5 or prey_windows.shape[2] != prey_windows.shape[3]:
        raise ValueError("Prey windows must have shape [W,L,M,M,F]")
    if predator_windows.shape[:2] != prey_windows.shape[:2]:
        raise ValueError("Predator and prey windows must share W and L")
    if predator_windows.shape[3] != prey_windows.shape[2]:
        raise ValueError("Predator and prey tensors must share the padded prey width")
    if predator_windows.shape[-1] < 5 or prey_windows.shape[-1] < 5:
        raise ValueError("Pairwise tensors require at least five feature channels")

    pred_active = (-2 if predator_windows.shape[-1] >= 6
                   and _is_binary_channel(predator_windows[..., -2]) else None)
    prey_active = (-2 if prey_windows.shape[-1] >= 6
                   and _is_binary_channel(prey_windows[..., -2]) else None)
    prey_flag = None
    if prey_windows.shape[-1] >= 7 and _is_binary_channel(prey_windows[..., 0]):
        flag = prey_windows[:, 0, :, :, 0]
        expected = torch.zeros_like(flag); expected[:, :, 0] = 1.0
        active_rows = (prey_windows[:, 0, :, :, prey_active].amax(dim=-1) > 0.5
                       if prey_active is not None
                       else torch.ones_like(flag[:, :, 0], dtype=torch.bool))
        if torch.allclose(flag[active_rows], expected[active_rows], atol=1e-6, rtol=0):
            prey_flag = 0

    prey_offset = 1 if prey_flag is not None else 0
    return {
        "pred_position": (0, 1), "pred_velocity": (2, 3),
        "pred_theta": predator_windows.shape[-1] - 1, "pred_active": pred_active,
        "prey_position": (prey_offset, prey_offset + 1),
        "prey_velocity": (prey_offset + 2, prey_offset + 3),
        "prey_theta": prey_windows.shape[-1] - 1, "prey_active": prey_active,
        "prey_flag": prey_flag, "window_length": int(predator_windows.shape[1]),
        "max_prey": int(predator_windows.shape[3]),
    }


def _active_prey_count(
    predator_windows: torch.Tensor, prey_windows: torch.Tensor, layout: Mapping[str, Any],
) -> torch.Tensor:
    """Return the true (unpadded) group size of every saved window."""

    if layout["pred_active"] is not None:
        return (predator_windows[:, 0, 0, :, layout["pred_active"]] > 0.5).sum(dim=-1)
    if layout["prey_active"] is not None:
        active = prey_windows[:, 0, :, :, layout["prey_active"]]
        return (active.amax(dim=-1) > 0.5).sum(dim=-1)
    return torch.full((predator_windows.shape[0],), layout["max_prey"], dtype=torch.long)


def _stitch_pairwise_windows(
    predator_windows: torch.Tensor, prey_windows: torch.Tensor, selected_indices: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Undo stride-one window extraction and recover independent sequences."""

    indices = selected_indices.to(dtype=torch.long, device="cpu").flatten()
    if not len(indices):
        return []
    chosen = predator_windows.index_select(0, indices)
    overlap = (torch.empty(0, dtype=torch.bool) if len(chosen) == 1 else
               (chosen[:-1, 1:] == chosen[1:, :-1]).reshape(len(chosen) - 1, -1).all(dim=1))
    starts = [0] + (torch.nonzero(~overlap, as_tuple=False).flatten() + 1).tolist()
    stops = starts[1:] + [len(indices)]
    clips: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start, stop in zip(starts, stops):
        source_indices = indices[start:stop]
        first = int(source_indices[0])
        pred_parts, prey_parts = [predator_windows[first]], [prey_windows[first]]
        if len(source_indices) > 1:
            tail = source_indices[1:]
            pred_parts.append(predator_windows.index_select(0, tail)[:, -1])
            prey_parts.append(prey_windows.index_select(0, tail)[:, -1])
        clips.append((torch.cat(pred_parts, dim=0), torch.cat(prey_parts, dim=0)))
    return clips


def _wrap_torch_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def _decode_theta_norm(value: torch.Tensor) -> torch.Tensor:
    """Decode [0,1] angles while also accepting already-radian tensors."""

    finite = value[torch.isfinite(value)]
    if len(finite) and bool(finite.min() >= -1e-5) and bool(finite.max() <= 1.0 + 1e-5):
        return _wrap_torch_angle(value * (2.0 * torch.pi) - torch.pi)
    return _wrap_torch_angle(value)


def _relative_prey_headings_from_velocity(
    predator_velocity: torch.Tensor, prey_velocity: torch.Tensor, *, eps: float = 1e-8,
) -> torch.Tensor:
    """Recover each prey focal-frame angle relative to the predator frame."""

    time, n_prey = predator_velocity.shape[:2]
    result = predator_velocity.new_full((time, n_prey), torch.nan)
    for focal in range(n_prey):
        other = [index for index in range(n_prey) if index != focal]
        slots = [index + 1 if index < focal else index for index in other]
        pred_view = predator_velocity[:, other]
        prey_view = prey_velocity[:, focal, slots]
        valid = (pred_view.norm(dim=-1) > eps) & (prey_view.norm(dim=-1) > eps)
        difference = _wrap_torch_angle(
            torch.atan2(pred_view[..., 1], pred_view[..., 0])
            - torch.atan2(prey_view[..., 1], prey_view[..., 0])
        )
        cosine = torch.where(valid, difference.cos(), 0.0).sum(dim=1)
        sine = torch.where(valid, difference.sin(), 0.0).sum(dim=1)
        result[:, focal] = torch.where(valid.any(dim=1), torch.atan2(sine, cosine), torch.nan)
    return result


def _integrate_heading(theta_norm: torch.Tensor) -> torch.Tensor:
    heading = theta_norm.new_zeros(len(theta_norm))
    if len(theta_norm) > 1:
        heading[1:] = torch.cumsum(_decode_theta_norm(theta_norm[:-1]), dim=0)
    return _wrap_torch_angle(heading)


def _estimate_global_predator_heading(
    predator_to_prey: torch.Tensor,
    predator_frame_velocity: torch.Tensor,
    theta_norm: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Align focal-frame velocities with common-axis relative-position changes."""

    fallback = _integrate_heading(theta_norm)
    if len(predator_to_prey) < 2:
        return fallback
    local = predator_frame_velocity[:-1] - predator_frame_velocity[:-1].mean(dim=1, keepdim=True)
    world = torch.diff(predator_to_prey, dim=0)
    world = world - world.mean(dim=1, keepdim=True)
    dot = (local * world).sum(dim=(1, 2))
    cross = (local[..., 0] * world[..., 1] - local[..., 1] * world[..., 0]).sum(dim=1)
    strength = local.square().sum(dim=(1, 2)) * world.square().sum(dim=(1, 2))
    estimate = torch.atan2(cross, dot)
    heading = fallback.clone()
    heading[:-1] = torch.where(strength > eps, estimate, heading[:-1])
    if len(heading) > 1:
        heading[-1] = _wrap_torch_angle(heading[-2] + _decode_theta_norm(theta_norm[-2]))
    return _wrap_torch_angle(heading)


def _pairwise_clip_to_records(
    predator: torch.Tensor,
    prey: torch.Tensor,
    layout: Mapping[str, Any],
    *,
    n_prey: int,
    clip_id: str,
    condition: str,
    coordinate_scale: float,
    effective_fps: float,
    sampling_stride: int,
) -> tuple[list[dict[str, Any]], str]:
    """Create a metric-equivalent canonical clip from pairwise interactions."""

    pred_position = predator[:, 0, :n_prey, list(layout["pred_position"])].float()
    prey_pred_position = prey[:, :n_prey, 0, list(layout["prey_position"])].float()
    pred_velocity = predator[:, 0, :n_prey, list(layout["pred_velocity"])].float()
    prey_velocity = prey[:, :n_prey, :n_prey, :][..., list(layout["prey_velocity"])].float()

    magnitude = pred_position.norm(dim=-1).mean().clamp_min(1e-8)
    reciprocal_error = (pred_position + prey_pred_position).norm(dim=-1).mean() / magnitude
    coordinate_mode = "common" if bool(reciprocal_error < 1e-4) else "local"
    theta_norm = predator[:, 0, 0, layout["pred_theta"]].float()
    if coordinate_mode == "local":
        relative_heading = _wrap_torch_angle(
            torch.atan2(-pred_position[..., 1], -pred_position[..., 0])
            - torch.atan2(prey_pred_position[..., 1], prey_pred_position[..., 0])
        )
        predator_heading = _integrate_heading(theta_norm)
        cosine, sine = predator_heading.cos(), predator_heading.sin()
        x, y = pred_position[..., 0], pred_position[..., 1]
        positions = torch.stack(
            (cosine[:, None] * x - sine[:, None] * y,
             sine[:, None] * x + cosine[:, None] * y), dim=-1,
        )
    else:
        relative_heading = _relative_prey_headings_from_velocity(pred_velocity, prey_velocity)
        predator_heading = _estimate_global_predator_heading(pred_position, pred_velocity, theta_norm)
        positions = pred_position

    prey_heading = _wrap_torch_angle(predator_heading[:, None] + relative_heading)
    if torch.isnan(prey_heading).any():
        missing = torch.isnan(prey_heading)
        prey_turn = _decode_theta_norm(prey[:, :n_prey, 0, layout["prey_theta"]].float())
        for time in range(1, len(prey_heading)):
            fallback = _wrap_torch_angle(prey_heading[time - 1] + prey_turn[time - 1])
            prey_heading[time] = torch.where(missing[time], fallback, prey_heading[time])
        prey_heading = torch.nan_to_num(prey_heading, nan=0.0)

    positions = positions * float(coordinate_scale)
    rows: list[dict[str, Any]] = []
    for time in range(len(predator)):
        rows.append({
            "source": "expert", "condition": condition, "clip_id": clip_id,
            "frame": time, "time_step": time, "agent_id": "predator", "role": "predator",
            "x": 0.0, "y": 0.0, "heading": float(predator_heading[time]), "policy_id": "",
            "fps": effective_fps, "timestamp": time / effective_fps,
            "step_duration": 1.0 / effective_fps, "sampling_stride": sampling_stride,
        })
        for agent in range(n_prey):
            rows.append({
                "source": "expert", "condition": condition, "clip_id": clip_id,
                "frame": time, "time_step": time, "agent_id": f"prey_{agent:02d}",
                "role": "prey", "x": float(positions[time, agent, 0]),
                "y": float(positions[time, agent, 1]),
                "heading": float(prey_heading[time, agent]), "policy_id": "",
                "fps": effective_fps, "timestamp": time / effective_fps,
                "step_duration": 1.0 / effective_fps, "sampling_stride": sampling_stride,
            })
    return rows, coordinate_mode


def load_biological_window_trajectories(
    window_root: str | Path,
    *,
    n_prey: int,
    condition: str = "interaction",
    coordinate_scale: float = 2160.0,
    effective_fps: float = 15.0,
    sampling_stride: int = 2,
) -> TrajectoryTable:
    """Load hand-labelled pairwise windows as de-duplicated canonical clips.

    The active mask separates padded 16- and 32-prey samples. Exact overlap is
    then used to stitch stride-one windows before metric calculation, preventing
    the repeated frames from inflating sample sizes and confidence intervals.
    """

    root = Path(window_root)
    if root.name != "windows":
        candidate = root / "expert_tensors" / "windows"
        if candidate.is_dir():
            root = candidate
    if condition not in {"interaction", "attack", "all"}:
        raise ValueError("condition must be 'interaction', 'attack', or 'all'")
    if n_prey < 2:
        raise ValueError("n_prey must be at least two")
    if effective_fps <= 0 or sampling_stride < 1:
        raise ValueError("effective_fps must be positive and sampling_stride >= 1")
    if condition == "all":
        folder = root / "10 windows"
        pred_pattern, prey_pattern = "pred_tensors_hl_w*_n*.pkl", "prey_tensors_hl_w*_n*.pkl"
    else:
        folder = root / "10 windows (split by attack or interaction -- used for calculating speed)"
        pred_pattern = f"pred_tensors_hl_{condition}_w*_n*.pkl"
        prey_pattern = f"prey_tensors_hl_{condition}_w*_n*.pkl"
    try:
        names = os.listdir(_windows_binary_path(folder))
    except OSError as exc:
        raise ExpertDataUnavailable(f"Pairwise tensor directory is unavailable: {folder}") from exc
    pred_paths = sorted(folder / name for name in names if Path(name).match(pred_pattern))
    prey_paths = sorted(folder / name for name in names if Path(name).match(prey_pattern))
    if len(pred_paths) != 1 or len(prey_paths) != 1:
        raise ExpertDataUnavailable(
            f"Expected one predator/prey hand-labelled tensor pair in {folder}; "
            f"found {len(pred_paths)} and {len(prey_paths)}."
        )

    predator_windows = _load_torch_tensor(pred_paths[0])
    prey_windows = _load_torch_tensor(prey_paths[0])
    layout = _pairwise_tensor_layout(predator_windows, prey_windows)
    counts = _active_prey_count(predator_windows, prey_windows, layout)
    selected = torch.nonzero(counts == n_prey, as_tuple=False).flatten()
    if not len(selected):
        available = sorted(set(int(value) for value in counts.tolist()))
        raise ExpertDataUnavailable(
            f"No {n_prey}-prey windows in {folder}; available group sizes are {available}."
        )

    if layout["pred_active"] is not None:
        active = predator_windows.index_select(0, selected)[..., layout["pred_active"]] > 0.5
        expected = torch.zeros_like(active); expected[..., :n_prey] = True
        if not torch.equal(active, expected):
            raise ValueError("Active predator-neighbor slots are not a stable leading prey block")
    if layout["prey_active"] is not None:
        prey_mask = prey_windows[..., layout["prey_active"]]
        active_agents = (prey_mask.index_select(0, selected).amax(dim=-1) > 0.5)
        expected_agents = torch.zeros_like(active_agents); expected_agents[..., :n_prey] = True
        if not torch.equal(active_agents, expected_agents):
            raise ValueError("Active focal-prey slots are not a stable leading block")

    clips = _stitch_pairwise_windows(predator_windows, prey_windows, selected)
    rows: list[dict[str, Any]] = []
    modes: set[str] = set()
    for clip_index, (predator, prey_tensor) in enumerate(clips):
        clip_rows, mode = _pairwise_clip_to_records(
            predator, prey_tensor, layout, n_prey=n_prey,
            clip_id=f"hand_{condition}_{n_prey}_{clip_index:03d}", condition=condition,
            coordinate_scale=coordinate_scale,
            effective_fps=effective_fps, sampling_stride=sampling_stride,
        )
        rows.extend(clip_rows); modes.add(mode)
    table = to_canonical_trajectory(rows)
    if len(modes) > 1:
        warnings.warn(f"Mixed pairwise coordinate conventions detected: {sorted(modes)}")
    return table


def _load_policy_source_modules(
    config: Mapping[str, Any], source_root: str | Path, *, include_simulator: bool = False,
) -> tuple[Any, Any, Any | None]:
    """Import a policy repository without leaking its generic ``models``/``utils`` packages."""

    root = Path(source_root).resolve()
    code_dir = (root / config.get("code_subdir", "notebooks")).resolve()
    if not code_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {code_dir}")

    package_names = ("models", "utils")
    saved_modules = {
        name: module for name, module in tuple(sys.modules.items())
        if any(name == package or name.startswith(f"{package}.") for package in package_names)
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    code_dir_text = str(code_dir)
    sys.path.insert(0, code_dir_text)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keep source repositories untouched
    previous_eval_module = sys.modules.get("utils.eval_utils")
    if include_simulator:
        eval_stub = types.ModuleType("utils.eval_utils")
        eval_stub.compute_polarization = lambda vx, vy: float(np.hypot(np.mean(vx), np.mean(vy)))
        eval_stub.compute_angular_momentum = lambda *args: float("nan")
        eval_stub.degree_of_sparsity = lambda *args: float("nan")
        eval_stub.distance_to_predator = lambda *args: float("nan")
        eval_stub.escape_alignment = lambda *args: float("nan")
        eval_stub.pred_distance_to_nearest_prey = lambda *args: float("nan")
        sys.modules["utils.eval_utils"] = eval_stub
    try:
        torch = importlib.import_module("torch")
        ModularPolicy = importlib.import_module("models.Generator").ModularPolicy
        sim_utils = importlib.import_module("utils.sim_utils") if include_simulator else None
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        if sys.path and sys.path[0] == code_dir_text:
            sys.path.pop(0)
        else:
            try:
                sys.path.remove(code_dir_text)
            except ValueError:
                pass
        for name in tuple(sys.modules):
            if any(name == package or name.startswith(f"{package}.") for package in package_names):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if previous_eval_module is not None:
            sys.modules["utils.eval_utils"] = previous_eval_module
    return torch, ModularPolicy, sim_utils


def load_policy_pair(config: Mapping[str, Any], source_root: str | Path) -> tuple[Any, Any]:
    """Load one compatible predator/prey checkpoint pair from its source repository."""

    if not config.get("available", True):
        raise FileNotFoundError(config.get("note", "Selected policy is not available"))
    root = Path(source_root).resolve()
    prey_path = (root / config["prey_checkpoint"]).resolve()
    pred_path = (root / config["pred_checkpoint"]).resolve()
    for path in (prey_path, pred_path):
        if not path.is_file() or _is_lfs_pointer(path):
            raise FileNotFoundError(f"Usable policy checkpoint not found: {path}")
    torch, ModularPolicy, _ = _load_policy_source_modules(config, root)
    prey_policy = ModularPolicy(features=int(config.get("prey_features", 5))).to("cpu")
    pred_policy = ModularPolicy(features=int(config.get("pred_features", 4))).to("cpu")

    def load_state(path: Path) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    prey_policy.load_state_dict(load_state(prey_path), strict=True)
    pred_policy.load_state_dict(load_state(pred_path), strict=True)
    prey_policy.eval()
    pred_policy.eval()
    return prey_policy, pred_policy


def load_jannik_policy_pair(config: Mapping[str, Any], jannik_root: str | Path) -> tuple[Any, Any]:
    """Backward-compatible alias for older notebooks."""

    return load_policy_pair(config, jannik_root)


def generate_policy_rollouts(
    config: Mapping[str, Any],
    source_root: str | Path,
    *,
    n_rollouts: int,
    rollout_steps: int,
    seed: int,
    init_pool_path: str | Path | None = None,
    n_prey: int = 32,
    init_pool_tensor: Any | None = None,
    rollout_seeds: Sequence[int] | None = None,
    condition: str = "unknown",
    effective_fps: float | None = None,
    sampling_stride: int = 1,
    wall_mode: str = "post_step_reflect",
) -> TrajectoryTable:
    """Run the selected policy repository's simulator and export its trajectories.

    The function calls the simulator shipped with the selected policy source and
    explicitly passes the training-time environment and state normalization.
    """

    if n_rollouts < 1 or rollout_steps < 2:
        raise ValueError("n_rollouts >= 1 and rollout_steps >= 2 are required")
    if init_pool_path is not None and init_pool_tensor is not None:
        raise ValueError("Pass either init_pool_path or init_pool_tensor, not both")
    if effective_fps is not None and effective_fps <= 0:
        raise ValueError("effective_fps must be positive")
    if sampling_stride < 1:
        raise ValueError("sampling_stride must be at least one")
    if wall_mode not in {"legacy", "post_step_reflect"}:
        raise ValueError("wall_mode must be 'legacy' or 'post_step_reflect'")
    root = Path(source_root).resolve()
    prey_path = (root / config["prey_checkpoint"]).resolve()
    pred_path = (root / config["pred_checkpoint"]).resolve()
    for path in (prey_path, pred_path):
        if not path.is_file() or _is_lfs_pointer(path):
            raise FileNotFoundError(f"Usable policy checkpoint not found: {path}")
    torch, ModularPolicy, sim_utils = _load_policy_source_modules(config, root, include_simulator=True)

    prey_policy = ModularPolicy(features=int(config.get("prey_features", 5))).to("cpu")
    pred_policy = ModularPolicy(features=int(config.get("pred_features", 4))).to("cpu")

    def load_state(path: Path) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    prey_policy.load_state_dict(load_state(prey_path), strict=True)
    pred_policy.load_state_dict(load_state(pred_path), strict=True)
    prey_policy.eval()
    pred_policy.eval()
    init_pool = init_pool_tensor
    if init_pool_path is not None:
        pool_path = Path(init_pool_path)
        if not pool_path.is_file() or _is_lfs_pointer(pool_path):
            raise FileNotFoundError(f"Usable expert-derived initialization pool not found: {pool_path}")
        try:
            init_pool = torch.load(pool_path, map_location="cpu", weights_only=True)
        except (TypeError, RuntimeError):
            init_pool = torch.load(pool_path, map_location="cpu")
    if init_pool is not None:
        init_pool = torch.as_tensor(init_pool, dtype=torch.float32, device="cpu")
        if tuple(init_pool.shape[1:]) != (n_prey + 1, 3):
            raise ValueError(
                f"Expected {n_prey}-prey pool shape (steps, {n_prey + 1}, 3), got {tuple(init_pool.shape)}"
            )

    env = dict(config.get("environment", {}))
    required_env = ("area_width", "area_height", "prey_speed", "pred_speed", "step_size", "max_turn")
    missing = [key for key in required_env if key not in env]
    if missing:
        raise ValueError(f"Policy environment configuration is missing {missing}")
    if init_pool is not None:
        if len(init_pool) == 0 or not bool(torch.isfinite(init_pool).all()):
            raise ValueError("The initialization pool must be non-empty and finite")
        positions = init_pool[..., :2]
        x_valid = (positions[..., 0] >= 0) & (positions[..., 0] <= float(env["area_width"]))
        y_valid = (positions[..., 1] >= 0) & (positions[..., 1] <= float(env["area_height"]))
        if not bool((x_valid & y_valid).all()):
            raise ValueError("The initialization pool contains positions outside the declared arena")
    records: list[dict[str, Any]] = []
    rollout_label = str(config.get("label", config.get("id", "Policy rollouts")))
    replicate_seeds = tuple(int(value) for value in (rollout_seeds or (seed,)))
    if not replicate_seeds or len(set(replicate_seeds)) != len(replicate_seeds):
        raise ValueError("rollout_seeds must be non-empty and unique")
    iterator = ((replicate_seed, rollout_id) for replicate_seed in replicate_seeds
                for rollout_id in range(n_rollouts))
    for replicate_seed, rollout_id in tqdm(
        iterator, total=len(replicate_seeds) * n_rollouts, desc=rollout_label, leave=False,
    ):
        rollout_seed = replicate_seed + rollout_id
        np.random.seed(rollout_seed)
        torch.manual_seed(rollout_seed)
        # ``sim_utils.apply_init_pool(..., experiment=False)`` recentres the sampled
        # configuration.  A valid expert frame can consequently be shifted beyond
        # an arena boundary.  Sample here, reproducibly, and use the simulator's
        # fixed-configuration path so the expert's absolute geometry is preserved.
        rollout_init = init_pool
        fixed_initial_configuration = False
        if init_pool is not None:
            pool_index = int(torch.randint(len(init_pool), (1,)).item())
            rollout_init = init_pool[pool_index].clone()
            # The upstream fixed-config path scales both coordinates by area_height.
            rollout_init[:, :2] /= float(env["area_height"])
            fixed_initial_configuration = True
        simulation_kwargs = dict(
            prey_policy=prey_policy, pred_policy=pred_policy, n_prey=n_prey, n_pred=1,
            max_steps=int(rollout_steps), deterministic=bool(config.get("deterministic", False)),
            visualization="off", init_pool=rollout_init, experiment=fixed_initial_configuration,
            step_size=float(env["step_size"]), prey_speed=float(env["prey_speed"]),
            pred_speed=float(env["pred_speed"]), area_width=float(env["area_width"]),
            area_height=float(env["area_height"]), max_turn=float(env["max_turn"]),
        )
        if "max_speed_norm" in inspect.signature(sim_utils.run_env_simulation).parameters:
            simulation_kwargs["max_speed_norm"] = float(env.get("max_speed_norm", 5.0))
        original_enforce = original_update = None
        if wall_mode == "post_step_reflect":
            original_enforce = sim_utils.enforce_walls
            original_update = sim_utils.Agent.update_position
            sim_utils.enforce_walls = lambda *_args, **_kwargs: None

            def update_then_reflect(agent: Any, step_size: float) -> None:
                """Mirror ``Agent.update_position(step_size=...)`` and reflect afterwards."""
                original_update(agent, step_size)
                original_enforce(agent, float(env["area_width"]), float(env["area_height"]))

            sim_utils.Agent.update_position = update_then_reflect
        try:
            with torch.inference_mode():
                _, _, logged = sim_utils.run_env_simulation(**simulation_kwargs)
        finally:
            if original_enforce is not None and original_update is not None:
                sim_utils.enforce_walls = original_enforce
                sim_utils.Agent.update_position = original_update
        metrics_list = logged[0]
        for step, metrics in enumerate(metrics_list):
            xs = np.asarray(metrics["xs"], dtype=float) * float(env["area_width"])
            ys = np.asarray(metrics["ys"], dtype=float) * float(env["area_height"])
            if wall_mode == "post_step_reflect" and (
                np.any((xs < -1e-6) | (xs > float(env["area_width"]) + 1e-6))
                or np.any((ys < -1e-6) | (ys > float(env["area_height"]) + 1e-6))
            ):
                raise AssertionError("Corrected policy rollout left the declared arena")
            headings = np.asarray(metrics["theta"], dtype=float)
            vxs, vys = np.asarray(metrics["vxs"], dtype=float), np.asarray(metrics["vys"], dtype=float)
            for agent in range(n_prey + 1):
                role = "predator" if agent == 0 else "prey"
                records.append({
                    "source": "policy", "condition": condition,
                    "clip_id": f"seed_{replicate_seed}_rollout_{rollout_id:04d}",
                    "frame": step, "time_step": step, "agent_id": f"{role}_{agent if agent == 0 else agent - 1}",
                    "role": role, "x": xs[agent], "y": ys[agent], "heading": headings[agent],
                    "vx": vxs[agent], "vy": vys[agent],
                    "fps": effective_fps if effective_fps is not None else np.nan,
                    "timestamp": step / effective_fps if effective_fps is not None else np.nan,
                    "step_duration": 1.0 / effective_fps if effective_fps is not None else float(env["step_size"]),
                    "sampling_stride": sampling_stride, "rollout_seed": replicate_seed,
                    "simulation_seed": rollout_seed,
                    "wall_mode": wall_mode,
                    "policy_id": str(config.get("label", config.get("id", "selected_policy"))),
                })
    return to_canonical_trajectory(records)


def generate_couzin_expert_rollouts(
    source_root: str | Path,
    *,
    n_prey: int,
    n_rollouts: int,
    rollout_steps: int,
    seed: int,
    area_width: float = 50.0,
    area_height: float = 50.0,
    dt: float = 0.5,
    alpha: float = 0.1,
    theta_dot_max: float = 0.5,
    theta_dot_max_shark: float | None = None,
    constant_speed: float = 5.0,
    shark_speed: float = 5.0,
    rollout_seeds: Sequence[int] | None = None,
    wall_mode: str = "post_step_reflect",
) -> TrajectoryTable:
    """Generate Couzin-model expert sequences using the existing simulator."""

    if n_prey < 2 or n_rollouts < 1 or rollout_steps < 2:
        raise ValueError("n_prey >= 2, n_rollouts >= 1 and rollout_steps >= 2 are required")
    if dt <= 0 or alpha < 0 or theta_dot_max <= 0:
        raise ValueError("dt/theta_dot_max must be positive and alpha non-negative")
    if wall_mode not in {"legacy", "post_step_reflect"}:
        raise ValueError("wall_mode must be 'legacy' or 'post_step_reflect'")
    theta_dot_max_shark = theta_dot_max if theta_dot_max_shark is None else theta_dot_max_shark
    root = Path(source_root).resolve()
    package_names = ("utils",)
    saved_modules = {
        name: module for name, module in tuple(sys.modules.items())
        if any(name == package or name.startswith(f"{package}.") for package in package_names)
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    root_text = str(root)
    sys.path.insert(0, root_text)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    # couzin_utils imports a heavyweight evaluation module only to log legacy
    # diagnostics.  Provide the small functions it needs; the paper metrics are
    # recomputed from positions/headings below.
    eval_stub = types.ModuleType("utils.eval_utils")
    eval_stub.compute_polarization = lambda vx, vy: float(np.hypot(np.mean(vx), np.mean(vy)))
    eval_stub.compute_angular_momentum = lambda *args: float("nan")
    eval_stub.degree_of_sparsity = lambda *args: float("nan")
    eval_stub.distance_to_predator = lambda *args: float("nan")
    eval_stub.escape_alignment = lambda *args: float("nan")
    eval_stub.pred_distance_to_nearest_prey = lambda *args: float("nan")
    previous_eval_module = sys.modules.get("utils.eval_utils")
    sys.modules["utils.eval_utils"] = eval_stub
    try:
        couzin_utils = importlib.import_module("utils.couzin_utils")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(root_text)
        for name in tuple(sys.modules):
            if name == "utils" or name.startswith("utils."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if previous_eval_module is not None:
            sys.modules["utils.eval_utils"] = previous_eval_module

    records: list[dict[str, Any]] = []
    replicate_seeds = tuple(int(value) for value in (rollout_seeds or (seed,)))
    if not replicate_seeds or len(set(replicate_seeds)) != len(replicate_seeds):
        raise ValueError("rollout_seeds must be non-empty and unique")
    iterator = ((replicate_seed, rollout_id) for replicate_seed in replicate_seeds
                for rollout_id in range(n_rollouts))
    for replicate_seed, rollout_id in tqdm(
        iterator, total=len(replicate_seeds) * n_rollouts,
        desc=f"Couzin expert {n_prey}", leave=False,
    ):
        rollout_seed = replicate_seed + rollout_id
        np.random.seed(rollout_seed)
        original_enforce = original_update = None
        if wall_mode == "post_step_reflect":
            original_enforce = couzin_utils.enforce_walls
            original_update = couzin_utils.Agent.update_position
            couzin_utils.enforce_walls = lambda *_args, **_kwargs: None

            def update_then_reflect(agent: Any, delta_t: float) -> None:
                """Mirror ``Agent.update_position(delta_t)`` and reflect afterwards."""
                original_update(agent, delta_t)
                original_enforce(agent, area_width, area_height)

            couzin_utils.Agent.update_position = update_then_reflect
        try:
            _, _, metrics_list, _, _ = couzin_utils.run_couzin_simulation(
                visualization="off", n=n_prey, max_steps=rollout_steps,
                number_of_sharks=1, area_width=area_width, area_height=area_height,
                dt=dt, alpha=alpha, theta_dot_max=theta_dot_max,
                theta_dot_max_shark=theta_dot_max_shark,
                constant_speed=constant_speed, shark_speed=shark_speed,
            )
        finally:
            if original_enforce is not None and original_update is not None:
                couzin_utils.enforce_walls = original_enforce
                couzin_utils.Agent.update_position = original_update
        for step, metrics in enumerate(metrics_list):
            xs = np.asarray(metrics["xs"], dtype=float) * area_width
            ys = np.asarray(metrics["ys"], dtype=float) * area_height
            if wall_mode == "post_step_reflect" and (
                np.any((xs < -1e-6) | (xs > area_width + 1e-6))
                or np.any((ys < -1e-6) | (ys > area_height + 1e-6))
            ):
                raise AssertionError("Corrected Couzin rollout left the declared arena")
            vxs = np.asarray(metrics["vxs"], dtype=float)
            vys = np.asarray(metrics["vys"], dtype=float)
            headings = np.arctan2(vys, vxs)
            for agent in range(n_prey + 1):
                role = "predator" if agent == 0 else "prey"
                records.append({
                    "source": "couzin_expert", "condition": "approach",
                    "clip_id": f"seed_{replicate_seed}_rollout_{rollout_id:04d}",
                    "frame": step, "time_step": step,
                    "agent_id": f"{role}_{agent if agent == 0 else agent - 1}", "role": role,
                    "x": xs[agent], "y": ys[agent], "heading": headings[agent],
                    "vx": vxs[agent], "vy": vys[agent], "fps": np.nan,
                    "timestamp": step * dt, "step_duration": dt,
                    "sampling_stride": 1, "rollout_seed": replicate_seed,
                    "simulation_seed": rollout_seed,
                    "wall_mode": wall_mode,
                })
    return to_canonical_trajectory(records)


def trajectory_diagnostics(table: TrajectoryTable) -> dict[str, Any]:
    """Concise data diagnostics suitable for notebook display."""

    prey = table["role"].astype(str) == "prey"
    frame_keys = set(zip(table["clip_id"].astype(str), table["time_step"]))
    clip_ids = np.unique(table["clip_id"].astype(str))
    counts_per_frame = []
    for idx in frame_groups(table):
        counts_per_frame.append(int(np.sum(prey[idx])))
    finite_fps = table["fps"][np.isfinite(table["fps"])]
    finite_dt = table["step_duration"][np.isfinite(table["step_duration"])]
    finite_seeds = table["rollout_seed"][np.isfinite(table["rollout_seed"])]
    finite_simulation_seeds = table["simulation_seed"][np.isfinite(table["simulation_seed"])]
    return {
        "rows": len(table), "clips_or_rollouts": len(clip_ids), "frames": len(frame_keys),
        "prey_per_frame_min": min(counts_per_frame, default=0),
        "prey_per_frame_max": max(counts_per_frame, default=0),
        "x_range": (float(np.nanmin(table["x"])), float(np.nanmax(table["x"]))) if len(table) else (np.nan, np.nan),
        "y_range": (float(np.nanmin(table["y"])), float(np.nanmax(table["y"]))) if len(table) else (np.nan, np.nan),
        "heading_range": (float(np.nanmin(table["heading"])), float(np.nanmax(table["heading"]))) if len(table) else (np.nan, np.nan),
        "conditions": {value: int(np.sum(table["condition"].astype(str) == value)) for value in np.unique(table["condition"].astype(str))},
        "fps_values": sorted(np.unique(finite_fps).tolist()),
        "step_duration_values": sorted(np.unique(finite_dt).tolist()),
        "rollout_replicate_seeds": sorted(int(x) for x in np.unique(finite_seeds)),
        "independent_simulation_seeds": len(np.unique(finite_simulation_seeds)),
        "wall_modes": sorted(set(str(x) for x in table["wall_mode"] if str(x))),
        "timestep_status": "explicit step duration" if len(finite_dt) else "PHYSICAL TIME UNKNOWN",
    }


# ---------------------------------------------------------------------------
# Analysis 1: risk-conditioned collective metrics

# This is the original NumPy implementation retained for compatibility with
# earlier notebook cells.  The complete workflow below uses the corresponding
# vectorized tensor functions.


def compute_prey_nnd(prey_positions: np.ndarray) -> float:
    """Mean prey-only nearest-neighbor distance (predator excluded)."""

    positions = np.asarray(prey_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 2:
        return float("nan")
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(np.mean(np.min(distances, axis=1)))


def compute_polarization(prey_headings: np.ndarray) -> float:
    """Magnitude of the mean prey heading vector; one means perfect alignment."""

    heading = np.asarray(prey_headings, dtype=float)
    if len(heading) == 0:
        return float("nan")
    return float(np.hypot(np.mean(np.cos(heading)), np.mean(np.sin(heading))))


def compute_escape_alignment(
    predator_position: np.ndarray, prey_positions: np.ndarray, prey_headings: np.ndarray
) -> float:
    """Jannik's prey-only escape alignment, expressed via headings.

    This is the mean dot product between each prey's unit heading and the unit
    vector from predator to that prey.  Positive means motion away, negative
    means motion toward the predator, and zero means average orthogonality.
    """

    positions = np.asarray(prey_positions, dtype=float)
    headings = np.asarray(prey_headings, dtype=float)
    delta = positions - np.asarray(predator_position, dtype=float)
    norm = np.linalg.norm(delta, axis=1)
    valid = norm > 1e-12
    if not np.any(valid):
        return float("nan")
    away = delta[valid] / norm[valid, None]
    velocity_unit = np.stack((np.cos(headings[valid]), np.sin(headings[valid])), axis=1)
    return float(np.mean(np.sum(away * velocity_unit, axis=1)))


def compute_risk_conditioned_metrics(
    table: TrajectoryTable, *, risk_variable: str = "nearest_prey", require_32_prey: bool = True
) -> dict[str, np.ndarray]:
    """Compute one prey-group observation per eligible frame."""

    if risk_variable not in {"nearest_prey", "centroid"}:
        raise ValueError("risk_variable must be 'nearest_prey' or 'centroid'")
    output: dict[str, list[Any]] = {k: [] for k in (
        "source", "condition", "clip_id", "time_step", "predator_distance",
        "prey_nnd", "polarization", "escape_alignment",
    )}
    for indices in frame_groups(table):
        roles = table["role"][indices].astype(str)
        pred_idx, prey_idx = indices[roles == "predator"], indices[roles == "prey"]
        if len(pred_idx) != 1 or len(prey_idx) < 2 or (require_32_prey and len(prey_idx) != 32):
            continue
        pred = np.array([table["x"][pred_idx[0]], table["y"][pred_idx[0]]], dtype=float)
        positions = np.column_stack((table["x"][prey_idx], table["y"][prey_idx])).astype(float)
        headings = table["heading"][prey_idx].astype(float)
        risk = (predator_to_nearest_prey_distance(pred, positions) if risk_variable == "nearest_prey"
                else float(euclidean_distance(pred, prey_centroid(positions))))
        values = {
            "source": table["source"][indices[0]], "condition": table["condition"][indices[0]],
            "clip_id": table["clip_id"][indices[0]], "time_step": table["time_step"][indices[0]],
            "predator_distance": risk, "prey_nnd": compute_prey_nnd(positions),
            "polarization": compute_polarization(headings),
            "escape_alignment": compute_escape_alignment(pred, positions, headings),
        }
        for key, value in values.items():
            output[key].append(value)
    return {key: np.asarray(value) for key, value in output.items()}


def make_bin_edges(values: Any, n_bins: int, value_range: tuple[float, float] | None = None) -> np.ndarray:
    """Create explicit equal-width bin edges from a declared or observed range."""

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if n_bins < 1 or len(finite) == 0:
        raise ValueError("At least one bin and one finite value are required")
    low, high = value_range if value_range is not None else (float(finite.min()), float(finite.max()))
    if not high > low:
        high = np.nextafter(low, np.inf)
    return np.linspace(low, high, n_bins + 1)


def binned_cluster_summary(
    x: Any,
    y: Any,
    clusters: Any,
    bin_edges: np.ndarray,
    *,
    statistic: str = "mean",
    min_samples: int = 1,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Binned descriptive statistics and clip/rollout cluster-bootstrap CIs."""

    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters).astype(str)
    edges = np.asarray(bin_edges, float)
    if not (len(x) == len(y) == len(clusters)):
        raise ValueError("x, y, and clusters must have equal length")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    if not 0 < ci_level < 1 or min_samples < 1 or n_bootstrap < 0:
        raise ValueError("Invalid CI, minimum sample, or bootstrap configuration")
    reducer = np.mean if statistic == "mean" else np.median
    n_bins = len(edges) - 1
    center = (edges[:-1] + edges[1:]) / 2
    estimate = np.full(n_bins, np.nan)
    ci_low = np.full(n_bins, np.nan)
    ci_high = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    cluster_count = np.zeros(n_bins, dtype=int)
    rng = np.random.default_rng(seed)
    finite = np.isfinite(x) & np.isfinite(y)
    for b in range(n_bins):
        in_bin = finite & (x >= edges[b]) & ((x < edges[b + 1]) if b < n_bins - 1 else (x <= edges[b + 1]))
        count[b] = int(in_bin.sum())
        unique_clusters = np.unique(clusters[in_bin])
        cluster_count[b] = len(unique_clusters)
        if count[b] < min_samples:
            continue
        estimate[b] = float(reducer(y[in_bin]))
        if n_bootstrap and len(unique_clusters) >= 2:
            boot = np.empty(n_bootstrap)
            for j in range(n_bootstrap):
                sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
                sampled_values = np.concatenate([y[in_bin & (clusters == cluster)] for cluster in sampled])
                boot[j] = reducer(sampled_values)
            alpha = (1.0 - ci_level) / 2.0
            ci_low[b], ci_high[b] = np.quantile(boot, [alpha, 1.0 - alpha])
    return {"bin_left": edges[:-1], "bin_right": edges[1:], "bin_center": center,
            "estimate": estimate, "ci_low": ci_low, "ci_high": ci_high,
            "count": count, "cluster_count": cluster_count}


def summarize_risk_metrics(
    metrics: Mapping[str, np.ndarray], bin_edges: np.ndarray, **summary_kwargs: Any
) -> dict[str, dict[str, np.ndarray]]:
    """Create matched binned summaries for all three Analysis-1 metrics."""

    return {name: binned_cluster_summary(
        metrics["predator_distance"], metrics[name], metrics["clip_id"], bin_edges, **summary_kwargs
    ) for name in ("prey_nnd", "polarization", "escape_alignment")}


# ---------------------------------------------------------------------------
# Shared temporal pairing helpers for continuous predator response

# Temporal pairs never bridge missing source timesteps.  This matters because
# response lags are later scaled by each case's declared step duration.


def _ordered_next_rows(table: TrajectoryTable, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag < 1:
        raise ValueError("lag must be >= 1 source timestep")
    current: list[int] = []
    future: list[int] = []
    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for i, key in enumerate(zip(
        table["source"].astype(str), table["clip_id"].astype(str),
        table["role"].astype(str), table["agent_id"].astype(str),
    )):
        groups.setdefault(key, []).append(i)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda i: (table["time_step"][i], table["frame"][i]))
        if len(ordered) > lag:
            for i, j in zip(ordered[:-lag], ordered[lag:]):
                # Canonical continuous records and policy rollouts use
                # consecutive integer source timesteps. Do not bridge missing
                # detections and mislabel a multi-step gap as lag=1.
                if np.isclose(table["time_step"][j] - table["time_step"][i], lag):
                    current.append(i)
                    future.append(j)
    return np.asarray(current, dtype=int), np.asarray(future, dtype=int)


def _predator_by_frame(table: TrajectoryTable) -> dict[tuple[str, str, float], int]:
    result: dict[tuple[str, str, float], int] = {}
    for i in np.where(table["role"].astype(str) == "predator")[0]:
        key = (str(table["source"][i]), str(table["clip_id"][i]), float(table["time_step"][i]))
        if key in result:
            raise ValueError(f"Multiple predators in frame {key}")
        result[key] = int(i)
    return result


# ---------------------------------------------------------------------------
# Analysis 2: continuous predator response

# The threat direction is fixed at the current frame and reused for the future
# heading, matching the response definition used throughout the paper.


def compute_continuous_predator_response(table: TrajectoryTable, *, lag: int = 1) -> dict[str, np.ndarray]:
    """Compute fixed-threat-direction away-alignment change for every focal prey."""

    current, future = _ordered_next_rows(table, lag)
    predator = _predator_by_frame(table)
    rows: dict[str, list[Any]] = {k: [] for k in (
        "source", "condition", "clip_id", "time_step", "agent_id", "distance",
        "bearing", "response", "alignment_before", "alignment_after",
    )}
    for i, j in zip(current, future):
        if str(table["role"][i]) != "prey" or str(table["role"][j]) != "prey":
            continue
        key = (str(table["source"][i]), str(table["clip_id"][i]), float(table["time_step"][i]))
        if key not in predator:
            continue
        p = predator[key]
        prey_pos = np.array([table["x"][i], table["y"][i]])
        pred_pos = np.array([table["x"][p], table["y"][p]])
        theta_away = away_from_predator_direction(prey_pos, pred_pos)
        before = heading_alignment(table["heading"][i], theta_away)
        after = heading_alignment(table["heading"][j], theta_away)
        bearing = angular_difference(vector_angle(pred_pos - prey_pos), table["heading"][i])
        values = (table["source"][i], table["condition"][i], table["clip_id"][i],
                  table["time_step"][i], table["agent_id"][i], euclidean_distance(pred_pos, prey_pos),
                  bearing, after - before, before, after)
        for name, value in zip(rows, values):
            rows[name].append(value)
    return {name: np.asarray(value) for name, value in rows.items()}


def compute_response_curve(
    samples: Mapping[str, np.ndarray], distance_edges: np.ndarray, **summary_kwargs: Any
) -> dict[str, np.ndarray]:
    """Cluster-bootstrap continuous response curve versus predator distance."""

    return binned_cluster_summary(samples["distance"], samples["response"], samples["clip_id"],
                                  distance_edges, **summary_kwargs)


def compute_distance_bearing_response(
    samples: Mapping[str, np.ndarray], distance_edges: np.ndarray, bearing_edges: np.ndarray,
    *, min_observations: int = 20,
) -> dict[str, np.ndarray]:
    """Mean continuous response over a distance x relative-bearing grid."""

    distance = np.asarray(samples["distance"], float)
    bearing = np.asarray(samples["bearing"], float)
    response = np.asarray(samples["response"], float)
    nd, nb = len(distance_edges) - 1, len(bearing_edges) - 1
    count = np.zeros((nb, nd), dtype=int)
    mean = np.full((nb, nd), np.nan)
    di = np.searchsorted(distance_edges, distance, side="right") - 1
    bi = np.searchsorted(bearing_edges, bearing, side="right") - 1
    di[distance == distance_edges[-1]], bi[bearing == bearing_edges[-1]] = nd - 1, nb - 1
    valid = np.isfinite(distance) & np.isfinite(bearing) & np.isfinite(response)
    valid &= (di >= 0) & (di < nd) & (bi >= 0) & (bi < nb)
    for row in range(nb):
        for col in range(nd):
            values = response[valid & (di == col) & (bi == row)]
            count[row, col] = len(values)
            if len(values) >= min_observations:
                mean[row, col] = np.mean(values)
    return {"mean": mean, "count": count, "distance_edges": np.asarray(distance_edges),
            "bearing_edges": np.asarray(bearing_edges), "min_observations": np.asarray(min_observations)}


# ---------------------------------------------------------------------------
# Efficient tensor metrics used by the complete multi-case workflow

# The functions in this section preserve tensor devices, reuse derived
# geometry, and transfer only compact results back to CPU for presentation.


# Tensor and reusable-geometry preparation


def _as_float_tensor(value: Any, *, device: Any = None, dtype: Any = torch.float32) -> torch.Tensor:
    """Convert once while preserving an existing tensor's device by default."""

    if torch.is_tensor(value):
        target_device = value.device if device is None else device
        return value.to(device=target_device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _unit_vectors(vectors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(eps)


def compute_prey_nearest_neighbors(
    prey_positions: Any, *, time_chunk_size: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return prey-only nearest-neighbor distances and indices for ``[T,N,2]``.

    ``torch.cdist`` is applied to time chunks so long recordings do not create
    one large ``[T,N,N]`` allocation.  The predator is never included.
    """

    positions = _as_float_tensor(prey_positions)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("prey_positions must have shape [T, N, 2]")
    if positions.shape[1] < 2:
        shape = positions.shape[:2]
        return (torch.full(shape, torch.nan, device=positions.device, dtype=positions.dtype),
                torch.full(shape, -1, device=positions.device, dtype=torch.long))
    if time_chunk_size < 1:
        raise ValueError("time_chunk_size must be positive")
    distances, indices = [], []
    for start in range(0, positions.shape[0], time_chunk_size):
        chunk = positions[start:start + time_chunk_size]
        pairwise = torch.cdist(chunk, chunk)
        pairwise.diagonal(dim1=-2, dim2=-1).fill_(torch.inf)
        distance, index = pairwise.min(dim=-1)
        distances.append(distance)
        indices.append(index)
    return torch.cat(distances), torch.cat(indices)


def prepare_geometry_cache(
    data: Mapping[str, Any] | Any,
    prey_positions: Any | None = None,
    predator_headings: Any | None = None,
    prey_headings: Any | None = None,
    *,
    d_source: float,
    time_chunk_size: int = 4096,
    device: Any = None,
    dtype: Any = torch.float32,
    include_prey_neighbors: bool = True,
) -> dict[str, torch.Tensor]:
    """Calculate reusable geometry for one clip/rollout.

    ``data`` may be a small mapping with the four named raw tensors, or the
    predator positions themselves followed by the other arrays.  This is a
    derived cache, not a replacement trajectory representation.
    """

    if isinstance(data, Mapping):
        predator_positions = data["predator_positions"]
        prey_positions = data["prey_positions"]
        predator_headings = data.get("predator_headings", predator_headings)
        prey_headings = data.get("prey_headings", prey_headings)
    else:
        predator_positions = data
    if prey_positions is None:
        raise ValueError("prey_positions are required")
    if not np.isfinite(d_source) or d_source <= 0:
        raise ValueError("d_source must be a positive environment maximum distance")

    predator_positions = _as_float_tensor(predator_positions, device=device, dtype=dtype)
    prey_positions = _as_float_tensor(prey_positions, device=predator_positions.device, dtype=dtype)
    if predator_positions.ndim != 2 or predator_positions.shape[-1] != 2:
        raise ValueError("predator_positions must have shape [T, 2]")
    if prey_positions.ndim != 3 or prey_positions.shape[-1] != 2:
        raise ValueError("prey_positions must have shape [T, N, 2]")
    if predator_positions.shape[0] != prey_positions.shape[0]:
        raise ValueError("predator and prey positions must share T")

    displacement = prey_positions - predator_positions[:, None, :]
    distance = displacement.norm(dim=-1)
    nearest_distance, nearest_index = distance.min(dim=1)
    nearest_displacement = displacement.gather(
        1, nearest_index[:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    cache: dict[str, torch.Tensor] = {
        "predator_positions": predator_positions,
        "prey_positions": prey_positions,
        "predator_to_prey": displacement,
        "predator_to_prey_distance": distance,
        "nearest_prey_index": nearest_index,
        "nearest_prey_distance": nearest_distance,
        "nearest_prey_distance_norm": nearest_distance / float(d_source),
        "nearest_prey_direction": _unit_vectors(nearest_displacement),
    }
    if predator_headings is not None:
        heading = _as_float_tensor(predator_headings, device=predator_positions.device, dtype=dtype)
        if heading.shape != predator_positions.shape[:1]:
            raise ValueError("predator_headings must have shape [T]")
        cache["predator_headings"] = heading
        cache["predator_heading_vectors"] = torch.stack((heading.cos(), heading.sin()), dim=-1)
    if prey_headings is not None:
        heading = _as_float_tensor(prey_headings, device=predator_positions.device, dtype=dtype)
        if heading.shape != prey_positions.shape[:2]:
            raise ValueError("prey_headings must have shape [T, N]")
        cache["prey_headings"] = heading
        cache["prey_heading_vectors"] = torch.stack((heading.cos(), heading.sin()), dim=-1)
    if include_prey_neighbors:
        nnd, nn_idx = compute_prey_nearest_neighbors(prey_positions, time_chunk_size=time_chunk_size)
        cache["prey_nnd"] = nnd
        cache["prey_nn_index"] = nn_idx
    return cache


def compute_predator_nearest_distance(geometry: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and environment-normalized nearest-prey distance."""

    return geometry["nearest_prey_distance"], geometry["nearest_prey_distance_norm"]


# Predator- and prey-level directional metrics


def compute_closing_speed(
    nearest_distance: Any, *, lag: int = 3, step_duration: float = 1.0,
) -> torch.Tensor:
    """Mean nearest-distance decrease per declared time unit."""

    distance = _as_float_tensor(nearest_distance)
    if distance.ndim != 1:
        raise ValueError("nearest_distance must be one-dimensional")
    if lag < 1:
        raise ValueError("lag must be positive")
    if step_duration <= 0:
        raise ValueError("step_duration must be positive")
    if lag >= len(distance):
        return distance.new_empty(0)
    return (distance[:-lag] - distance[lag:]) / float(lag * step_duration)


def compute_pursuit_alignment(geometry: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Predator heading alignment with the independently nearest prey."""

    if "predator_heading_vectors" not in geometry:
        raise ValueError("geometry does not contain predator headings")
    return (geometry["predator_heading_vectors"] * geometry["nearest_prey_direction"]).sum(dim=-1)


def compute_escape_alignment_tensor(geometry: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Per-prey heading alignment with the predator-away direction, ``[T,N]``."""

    if "prey_heading_vectors" not in geometry:
        raise ValueError("geometry does not contain prey headings")
    away = _unit_vectors(geometry["predator_to_prey"])
    return (geometry["prey_heading_vectors"] * away).sum(dim=-1)


def compute_continuous_predator_response_tensor(
    geometry: Mapping[str, torch.Tensor], *, lag: int = 1
) -> torch.Tensor:
    """Fixed-threat-direction continuous response ``R`` for every prey."""

    if lag < 1:
        raise ValueError("lag must be positive")
    headings = geometry.get("prey_heading_vectors")
    if headings is None:
        raise ValueError("geometry does not contain prey headings")
    if lag >= headings.shape[0]:
        return headings.new_empty((0, headings.shape[1]))
    fixed_away = _unit_vectors(geometry["predator_to_prey"][:-lag])
    before = (headings[:-lag] * fixed_away).sum(dim=-1)
    after = (headings[lag:] * fixed_away).sum(dim=-1)
    return after - before


# Approach events and the shared first-response calculation


def detect_approach_events(
    normalized_distance: Any, *, threshold: float = 0.15, closing_lag: int = 3,
    exit_threshold: float | None = None, min_duration: int = 2,
    cooldown_steps: int = 20, target_index: Any | None = None,
    require_target_persistence: bool = True, return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Detect separated approaches with hysteresis, persistence, and cooldown.

    One event is emitted per threshold encounter.  The detector stays active
    until distance crosses ``exit_threshold``; short closing-sign flicker can
    therefore no longer create multiple overlapping response windows.
    """

    distance = _as_float_tensor(normalized_distance)
    if distance.ndim != 1:
        raise ValueError("normalized_distance must be one-dimensional")
    if closing_lag < 1:
        raise ValueError("closing_lag must be positive")
    if min_duration < 1 or cooldown_steps < 0:
        raise ValueError("min_duration must be >= 1 and cooldown_steps >= 0")
    exit_threshold = threshold * 1.10 if exit_threshold is None else float(exit_threshold)
    if exit_threshold < threshold:
        raise ValueError("exit_threshold must be at least threshold")
    condition = torch.zeros_like(distance, dtype=torch.bool)
    if closing_lag < len(distance):
        closing = (distance[:-closing_lag] - distance[closing_lag:]) / float(closing_lag)
        condition[closing_lag:] = (distance[closing_lag:] < threshold) & (closing > 0)
    if target_index is not None and require_target_persistence:
        target = torch.as_tensor(target_index, device=distance.device, dtype=torch.long).flatten()
        if target.shape != distance.shape:
            raise ValueError("target_index must match normalized_distance")
        persistent = torch.zeros_like(condition)
        for t in range(closing_lag, len(distance)):
            persistent[t] = bool(torch.all(target[t - closing_lag:t + 1] == target[t]))
        condition &= persistent

    active_mask = torch.zeros_like(condition)
    onsets: list[int] = []
    active = False
    next_allowed = 0
    t = closing_lag
    while t < len(distance):
        if active:
            active_mask[t] = True
            if not bool(torch.isfinite(distance[t])) or bool(distance[t] > exit_threshold):
                active = False
                next_allowed = t + cooldown_steps
            t += 1
            continue
        stop = min(t + min_duration, len(distance))
        sustained = stop - t == min_duration and bool(condition[t:stop].all())
        if t >= next_allowed and sustained:
            onsets.append(t)
            active = True
            active_mask[t] = True
        t += 1
    onset_tensor = torch.as_tensor(onsets, device=distance.device, dtype=torch.long)
    return (onset_tensor, active_mask) if return_mask else onset_tensor


def kaplan_meier_response_probability(
    durations: Any, events: Any, *, horizon: float,
) -> torch.Tensor:
    """Kaplan-Meier estimate of response probability by ``horizon``."""

    duration = _as_float_tensor(durations).flatten()
    observed_event = torch.as_tensor(events, device=duration.device, dtype=torch.bool).flatten()
    valid = torch.isfinite(duration) & (duration > 0)
    duration, observed_event = duration[valid], observed_event[valid]
    if not len(duration):
        return duration.new_tensor(torch.nan)
    survival = duration.new_tensor(1.0)
    event_times = torch.unique(duration[observed_event & (duration <= horizon)]).sort().values
    for event_time in event_times:
        at_risk = (duration >= event_time).sum()
        failures = ((duration == event_time) & observed_event).sum()
        if at_risk > 0:
            survival = survival * (1.0 - failures.to(duration.dtype) / at_risk.to(duration.dtype))
    return 1.0 - survival


def bootstrap_clip_mean(
    values: Any, *, n_bootstrap: int = 1000, ci_level: float = 0.95, seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Equal-clip mean and percentile interval for scalar or profile data."""

    tensor = _as_float_tensor(values)
    if tensor.ndim < 1:
        tensor = tensor.reshape(1)
    if not 0 < ci_level < 1 or n_bootstrap < 0:
        raise ValueError("ci_level must lie in (0,1) and n_bootstrap be non-negative")
    mean = torch.nanmean(tensor, dim=0)
    count = torch.isfinite(tensor).sum(dim=0)
    if n_bootstrap == 0 or tensor.shape[0] < 2:
        nan = torch.full_like(mean, torch.nan)
        return {"mean": mean, "ci_low": nan, "ci_high": nan, "clip_count": count}
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed))
    draws = []
    for _ in range(n_bootstrap):
        indices = torch.randint(tensor.shape[0], (tensor.shape[0],), generator=generator,
                                device=tensor.device)
        draws.append(torch.nanmean(tensor.index_select(0, indices), dim=0))
    samples = torch.stack(draws)
    alpha = (1.0 - ci_level) / 2.0
    return {
        "mean": mean,
        "ci_low": torch.nanquantile(samples, alpha, dim=0),
        "ci_high": torch.nanquantile(samples, 1.0 - alpha, dim=0),
        "clip_count": count,
    }


def compute_reaction_metrics(
    response: Any,
    approach_onsets: Any,
    predator_positions: Any,
    prey_positions: Any,
    *,
    response_threshold: float = 0.10,
    response_window_steps: int = 20,
    response_lag: int = 1,
    d_source: float | None = None,
    step_duration: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute first responses over exactly ``response_window_steps`` transitions.

    Latencies and reaction distances refer to the end frame at which a response
    first becomes observable.  Non-responders are confirmed only when their full
    response window is observed; otherwise they are explicitly right-censored.
    The primary response fraction is a Kaplan-Meier estimate and therefore
    incorporates right censoring.  Classified-only and complete-case fractions
    are retained explicitly as diagnostics.
    """

    response = _as_float_tensor(response)
    predator_positions = _as_float_tensor(predator_positions, device=response.device, dtype=response.dtype)
    prey_positions = _as_float_tensor(prey_positions, device=response.device, dtype=response.dtype)
    onsets = torch.as_tensor(approach_onsets, device=response.device, dtype=torch.long).flatten()
    if response.ndim != 2 or prey_positions.ndim != 3:
        raise ValueError("response and prey_positions must have shapes [T,N] and [T,N,2]")
    if response.shape[1] != prey_positions.shape[1]:
        raise ValueError("response and prey positions must have the same prey count")
    if response_window_steps < 1 or response_lag < 1:
        raise ValueError("response_window_steps and response_lag must be positive")
    if step_duration <= 0:
        raise ValueError("step_duration must be positive")

    n_events, n_prey = len(onsets), response.shape[1]
    # W response samples encode exactly W transitions: e->e+lag through
    # e+W-1->e+W-1+lag.  For lag=1, reported latencies are therefore 1..W.
    offsets = torch.arange(response_window_steps, device=response.device)
    window_index = onsets[:, None] + offsets[None, :]
    availability = (window_index >= 0) & (window_index < response.shape[0])
    safe_index = (window_index.clamp(0, response.shape[0] - 1)
                  if response.shape[0] else torch.zeros_like(window_index))
    if response.shape[0]:
        response_window = response[safe_index]
        observed = torch.isfinite(response_window) & availability[..., None]
        response_mask = (response_window > response_threshold) & observed
    else:
        observed = torch.zeros(
            (n_events, response_window_steps, n_prey),
            device=response.device, dtype=torch.bool,
        )
        response_mask = torch.zeros(
            (n_events, response_window_steps, n_prey),
            device=response.device, dtype=torch.bool,
        )
    # Only a contiguous observed prefix constitutes follow-up.  This prevents a
    # response after an internal tracking gap from being treated as observable.
    observed_prefix = observed.to(torch.int8).cumprod(dim=1).bool()
    response_mask &= observed_prefix
    responded = response_mask.any(dim=1)
    full_prey_window = observed_prefix.all(dim=1)
    confirmed_non_responder = (~responded) & full_prey_window
    censored = (~responded) & (~full_prey_window)
    first_offset = response_mask.to(torch.int8).argmax(dim=1).long()
    first_times = torch.where(
        responded, onsets[:, None] + first_offset + int(response_lag), -1)
    reaction_distance = torch.full((n_events, n_prey), torch.nan, device=response.device, dtype=response.dtype)
    event_index, prey_index = torch.nonzero(responded, as_tuple=True)
    if len(event_index):
        event_times = first_times[event_index, prey_index]
        valid_time = event_times < min(predator_positions.shape[0], prey_positions.shape[0])
        event_index, prey_index, event_times = (
            event_index[valid_time], prey_index[valid_time], event_times[valid_time]
        )
        reaction_distance[event_index, prey_index] = (
            prey_positions[event_times, prey_index] - predator_positions[event_times]
        ).norm(dim=-1)

    latency = torch.where(
        responded,
        (first_offset + int(response_lag)).to(response.dtype),
        torch.nan,
    )
    observed_responder_count = responded.sum(dim=1)
    confirmed_non_responder_count = confirmed_non_responder.sum(dim=1)
    censored_count = censored.sum(dim=1)
    classification_denominator = observed_responder_count + confirmed_non_responder_count
    classified_fraction = (observed_responder_count.to(response.dtype)
                           / classification_denominator.clamp_min(1).to(response.dtype))
    classified_fraction[classification_denominator == 0] = torch.nan
    classification_complete = censored_count == 0
    complete_case_fraction = observed_responder_count.to(response.dtype) / float(n_prey)
    complete_case_fraction[~classification_complete] = torch.nan
    prefix_steps = observed_prefix.sum(dim=1)
    duration_steps = torch.where(
        responded, first_offset + int(response_lag), prefix_steps,
    ).to(response.dtype)
    response_fraction = torch.stack([
        kaplan_meier_response_probability(
            duration_steps[event], responded[event], horizon=float(response_window_steps),
        )
        for event in range(n_events)
    ]) if n_events else response.new_empty(0)
    cascade_size = response_fraction * float(n_prey)
    propagation_time = torch.full((n_events,), torch.nan, device=response.device, dtype=response.dtype)
    propagation_delay = torch.full_like(propagation_time, torch.nan)
    if n_events:
        sentinel = torch.iinfo(first_times.dtype).max
        first_response = torch.where(responded, first_times, sentinel).amin(dim=1)
        last_response = torch.where(responded, first_times, -1).amax(dim=1)
        multi_response = (observed_responder_count >= 2) & classification_complete
        propagation_time[multi_response] = (
            last_response[multi_response] - first_response[multi_response]
        ).to(response.dtype)
        relative_times = torch.where(
            responded, first_times - first_response[:, None], 0
        ).to(response.dtype)
        propagation_delay[multi_response] = (
            relative_times[multi_response].sum(dim=1)
            / observed_responder_count[multi_response].to(response.dtype)
        )
    result = {
        "event_onset": onsets,
        "availability": availability,
        "available_steps": availability.sum(dim=1),
        # ``availability`` only checks tensor bounds; these fields additionally
        # respect per-prey NaNs caused by gaps or a missing terminal transition.
        "observed": observed,
        "observed_prefix": observed_prefix,
        "available_steps_per_prey": prefix_steps,
        "full_prey_window": full_prey_window,
        "full_window_available": full_prey_window.all(dim=1),
        "first_response_time": first_times,
        "responded": responded,
        "confirmed_non_responder": confirmed_non_responder,
        "censored": censored,
        "observed_responder_count": observed_responder_count,
        "confirmed_non_responder_count": confirmed_non_responder_count,
        "censored_count": censored_count,
        "classification_denominator": classification_denominator,
        "classification_complete": classification_complete,
        "reaction_latency": latency,
        "reaction_latency_time": latency * float(step_duration),
        "reaction_distance": reaction_distance,
        "response_fraction": response_fraction,
        "response_fraction_km": response_fraction.clone(),
        "response_fraction_classified": classified_fraction,
        "response_fraction_complete_case": complete_case_fraction,
        "followup_duration_steps": duration_steps,
        "followup_duration_time": duration_steps * float(step_duration),
        "cascade_size": cascade_size,
        "cascade_fraction": response_fraction.clone(),
        "observed_cascade_size": observed_responder_count.to(response.dtype),
        "propagation_time": propagation_time,
        "propagation_time_scaled": propagation_time * float(step_duration),
        "mean_propagation_delay": propagation_delay,
        "mean_propagation_delay_scaled": propagation_delay * float(step_duration),
    }
    if d_source is not None:
        if d_source <= 0:
            raise ValueError("d_source must be positive")
        result["reaction_distance_norm"] = reaction_distance / float(d_source)
    return result


# Collective time series and event-relative changes


def compute_polarization_tensor(prey_headings: Any) -> torch.Tensor:
    """Global polarization time evolution for headings ``[T,N]``."""

    heading = _as_float_tensor(prey_headings)
    if heading.ndim != 2:
        raise ValueError("prey_headings must have shape [T,N]")
    return torch.stack((heading.cos().mean(dim=1), heading.sin().mean(dim=1)), dim=-1).norm(dim=-1)


def compute_event_relative_change(
    values: Any, approach_onsets: Any, *, response_window_steps: int = 20,
    relative: bool = False, eps: float = 1e-12,
) -> torch.Tensor:
    """Return padded ``[events, window+1, ...]`` changes relative to onset."""

    values = _as_float_tensor(values)
    onsets = torch.as_tensor(approach_onsets, device=values.device, dtype=torch.long).flatten()
    if response_window_steps < 0:
        raise ValueError("response_window_steps must be non-negative")
    output_shape = (len(onsets), response_window_steps + 1, *values.shape[1:])
    if not len(onsets) or not len(values):
        return torch.full(output_shape, torch.nan, device=values.device, dtype=values.dtype)
    offsets = torch.arange(response_window_steps + 1, device=values.device)
    index = onsets[:, None] + offsets[None, :]
    availability = (onsets[:, None] >= 0) & (index < values.shape[0])
    safe_index = index.clamp(0, values.shape[0] - 1)
    baseline_index = onsets.clamp(0, values.shape[0] - 1)
    baseline = values[baseline_index]
    change = values[safe_index] - baseline[:, None]
    if relative:
        change = change / baseline[:, None].clamp_min(eps)
    mask_shape = (*availability.shape, *((1,) * (values.ndim - 1)))
    return torch.where(availability.reshape(mask_shape), change, torch.nan)


def compute_dos(geometry: Mapping[str, torch.Tensor], *, d_source: float) -> dict[str, torch.Tensor]:
    """Li et al. Degree of Swarm: mean prey NND divided by arena maximum distance."""

    if d_source <= 0:
        raise ValueError("d_source must be positive")
    timeline = geometry["prey_nnd"].mean(dim=1) / float(d_source)
    return {"timeline": timeline, "aggregate": timeline.mean()}


def compute_doa(geometry: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Li et al. local nearest-neighbor Degree of Alignment."""

    headings = geometry.get("prey_heading_vectors")
    indices = geometry.get("prey_nn_index")
    if headings is None or indices is None:
        raise ValueError("geometry requires prey headings and nearest-neighbor indices")
    if (indices < 0).any():
        timeline = torch.full(headings.shape[:1], torch.nan, device=headings.device, dtype=headings.dtype)
    else:
        neighbor = headings.gather(1, indices[..., None].expand(-1, -1, 2))
        timeline = (headings + neighbor).norm(dim=-1).mean(dim=1) / 2.0
    return {"timeline": timeline, "aggregate": timeline.mean()}


# Risk-conditioned aggregation and independent-trajectory uncertainty


def risk_bin_indices(x: Any, bin_edges: Any) -> torch.Tensor:
    """Compute common zero-based bin indices once; invalid/outside values are -1."""

    x = _as_float_tensor(x)
    edges = _as_float_tensor(bin_edges, device=x.device, dtype=x.dtype)
    if edges.ndim != 1 or len(edges) < 2 or not bool(torch.all(edges[1:] > edges[:-1])):
        raise ValueError("bin_edges must be a strictly increasing vector")
    index = torch.bucketize(x, edges, right=True) - 1
    index = torch.where(x == edges[-1], len(edges) - 2, index)
    valid = torch.isfinite(x) & (x >= edges[0]) & (x <= edges[-1])
    return torch.where(valid, index, -1)


def risk_conditioned_mean(
    x: Any,
    values: Any,
    bin_edges: Any,
    *,
    min_count: int = 1,
    bin_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Vectorized binned mean/count using ``bucketize`` and ``bincount``."""

    x = _as_float_tensor(x)
    values = _as_float_tensor(values, device=x.device, dtype=x.dtype)
    if x.shape != values.shape:
        raise ValueError("x and values must have equal shapes")
    edges = _as_float_tensor(bin_edges, device=x.device, dtype=x.dtype)
    index = risk_bin_indices(x, edges) if bin_index is None else bin_index.to(x.device)
    n_bins = len(edges) - 1
    valid = (index >= 0) & torch.isfinite(values)
    count = torch.bincount(index[valid], minlength=n_bins)
    sums = torch.bincount(index[valid], weights=values[valid], minlength=n_bins)
    mean = sums / count.clamp_min(1).to(values.dtype)
    mean[count < min_count] = torch.nan
    return {"bin_left": edges[:-1], "bin_right": edges[1:],
            "bin_center": (edges[:-1] + edges[1:]) / 2, "mean": mean, "count": count}


def cluster_bootstrap_risk_conditioned_mean(
    x_by_cluster: Sequence[Any],
    values_by_cluster: Sequence[Any],
    bin_edges: Any,
    *,
    min_count: int = 1,
    min_cluster_count: int = 1,
    bin_indices_by_cluster: Sequence[Any] | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Return sample-weighted and clip-balanced binned means with cluster CIs.

    The historical ``mean``/``ci_*`` fields remain aliases for the
    sample-weighted estimate.  The clip-balanced estimate first averages within
    every clip and bin and then gives each contributing clip equal weight.
    """

    if len(x_by_cluster) != len(values_by_cluster) or not x_by_cluster:
        raise ValueError("Equal non-empty cluster sequences are required")
    if n_bootstrap < 0 or min_count < 1 or min_cluster_count < 1 or not 0 < ci_level < 1:
        raise ValueError("Invalid bootstrap configuration")
    first = _as_float_tensor(x_by_cluster[0])
    edges = _as_float_tensor(bin_edges, device=first.device, dtype=first.dtype)
    n_bins = len(edges) - 1
    cluster_sums, cluster_counts, cluster_candidate_counts = [], [], []
    nonfinite_distance_count = below_range_count = above_range_count = 0
    if bin_indices_by_cluster is not None and len(bin_indices_by_cluster) != len(x_by_cluster):
        raise ValueError("bin_indices_by_cluster must match the cluster sequences")
    for cluster_index, (x, values) in enumerate(zip(x_by_cluster, values_by_cluster)):
        x = _as_float_tensor(x, device=first.device, dtype=first.dtype).flatten()
        values = _as_float_tensor(values, device=first.device, dtype=first.dtype).flatten()
        index = (risk_bin_indices(x, edges) if bin_indices_by_cluster is None else
                 torch.as_tensor(bin_indices_by_cluster[cluster_index], device=first.device, dtype=torch.long).flatten())
        if index.shape != x.shape:
            raise ValueError("Each cached bin-index tensor must match its x cluster")
        finite_distance = torch.isfinite(x)
        in_range = index >= 0
        valid = in_range & torch.isfinite(values)
        cluster_candidate_counts.append(torch.bincount(index[in_range], minlength=n_bins))
        cluster_counts.append(torch.bincount(index[valid], minlength=n_bins))
        cluster_sums.append(torch.bincount(index[valid], weights=values[valid], minlength=n_bins))
        nonfinite_distance_count += int((~finite_distance).sum())
        below_range_count += int((finite_distance & (x < edges[0])).sum())
        above_range_count += int((finite_distance & (x > edges[-1])).sum())
    sums = torch.stack(cluster_sums)
    counts = torch.stack(cluster_counts)
    candidate_counts = torch.stack(cluster_candidate_counts)
    total_count = counts.sum(dim=0)
    total_candidate_count = candidate_counts.sum(dim=0)
    cluster_count = (counts > 0).sum(dim=0)
    candidate_cluster_count = (candidate_counts > 0).sum(dim=0)
    missing_value_count = total_candidate_count - total_count
    mean = sums.sum(dim=0) / total_count.clamp_min(1).to(sums.dtype)
    per_cluster_mean = sums / counts.clamp_min(1).to(sums.dtype)
    per_cluster_mean[counts == 0] = torch.nan
    clip_balanced_mean = torch.nanmean(per_cluster_mean, dim=0)
    supported = (total_count >= min_count) & (cluster_count >= min_cluster_count)
    mean[~supported] = torch.nan
    clip_balanced_mean[~supported] = torch.nan
    ci_low = torch.full_like(mean, torch.nan)
    ci_high = torch.full_like(mean, torch.nan)
    clip_balanced_ci_low = torch.full_like(mean, torch.nan)
    clip_balanced_ci_high = torch.full_like(mean, torch.nan)
    if n_bootstrap and len(sums) >= 2:
        generator = torch.Generator(device=sums.device).manual_seed(seed)
        sample = torch.randint(len(sums), (n_bootstrap, len(sums)), device=sums.device, generator=generator)
        boot_count = counts[sample].sum(dim=1)
        boot_mean = sums[sample].sum(dim=1) / boot_count.clamp_min(1).to(sums.dtype)
        boot_mean[boot_count < min_count] = torch.nan
        sampled_cluster_means = per_cluster_mean[sample]
        boot_clip_balanced_mean = torch.nanmean(sampled_cluster_means, dim=1)
        boot_clip_balanced_mean[boot_count < min_count] = torch.nan
        alpha = (1.0 - ci_level) / 2.0
        ci_low = torch.nanquantile(boot_mean, alpha, dim=0)
        ci_high = torch.nanquantile(boot_mean, 1.0 - alpha, dim=0)
        clip_balanced_ci_low = torch.nanquantile(boot_clip_balanced_mean, alpha, dim=0)
        clip_balanced_ci_high = torch.nanquantile(boot_clip_balanced_mean, 1.0 - alpha, dim=0)
        ci_low[~supported] = torch.nan
        ci_high[~supported] = torch.nan
        clip_balanced_ci_low[~supported] = torch.nan
        clip_balanced_ci_high[~supported] = torch.nan
    return {"bin_left": edges[:-1], "bin_right": edges[1:],
            "bin_center": (edges[:-1] + edges[1:]) / 2, "mean": mean,
            "ci_low": ci_low, "ci_high": ci_high, "count": total_count,
            "sample_weighted_mean": mean, "sample_weighted_ci_low": ci_low,
            "sample_weighted_ci_high": ci_high,
            "clip_balanced_mean": clip_balanced_mean,
            "clip_balanced_ci_low": clip_balanced_ci_low,
            "clip_balanced_ci_high": clip_balanced_ci_high,
            "cluster_count": cluster_count, "supported": supported,
            # Diagnostics distinguish metric NaNs from distance-range clipping.
            "candidate_count": total_candidate_count,
            "candidate_cluster_count": candidate_cluster_count,
            "missing_value_count": missing_value_count,
            "nonfinite_distance_count": torch.tensor(nonfinite_distance_count, device=edges.device),
            "below_range_count": torch.tensor(below_range_count, device=edges.device),
            "above_range_count": torch.tensor(above_range_count, device=edges.device)}


def compute_group_size_difference(value_16: Any, value_32: Any) -> torch.Tensor:
    """Return ``M_32 - M_16`` while preserving tensors/device."""

    value_32 = _as_float_tensor(value_32)
    value_16 = _as_float_tensor(value_16, device=value_32.device, dtype=value_32.dtype)
    return value_32 - value_16


def compute_generalization_error(
    expert_16: Any, expert_32: Any, imitation_16: Any, imitation_32: Any
) -> dict[str, torch.Tensor]:
    """Quantify consistency across two group sizes seen during training.

    Historical field names are retained as aliases for compatibility.  They
    must not be interpreted as out-of-distribution generalization estimates.
    """

    delta_expert = compute_group_size_difference(expert_16, expert_32)
    delta_imitation = compute_group_size_difference(imitation_16, imitation_32)
    error = delta_imitation - delta_expert
    valid = torch.isfinite(delta_expert) & torch.isfinite(delta_imitation)
    mage = error[valid].abs().mean() if valid.any() else error.new_tensor(torch.nan)
    expert_16 = _as_float_tensor(expert_16, device=error.device, dtype=error.dtype)
    expert_32 = _as_float_tensor(expert_32, device=error.device, dtype=error.dtype)
    imitation_16 = _as_float_tensor(imitation_16, device=error.device, dtype=error.dtype)
    imitation_32 = _as_float_tensor(imitation_32, device=error.device, dtype=error.dtype)
    fidelity_16 = (imitation_16 - expert_16).abs()
    fidelity_32 = (imitation_32 - expert_32).abs()
    valid_fidelity_16 = torch.isfinite(fidelity_16)
    valid_fidelity_32 = torch.isfinite(fidelity_32)
    mafe_16 = (fidelity_16[valid_fidelity_16].mean() if valid_fidelity_16.any()
               else fidelity_16.new_tensor(torch.nan))
    mafe_32 = (fidelity_32[valid_fidelity_32].mean() if valid_fidelity_32.any()
               else fidelity_32.new_tensor(torch.nan))
    return {"delta_expert": delta_expert, "delta_imitation": delta_imitation,
            "consistency_error": error, "mean_absolute_consistency_error": mage,
            "generalization_error": error, "mage": mage, "valid": valid,
            "fidelity_error_16": fidelity_16, "fidelity_error_32": fidelity_32,
            "mean_absolute_fidelity_error_16": mafe_16,
            "mean_absolute_fidelity_error_32": mafe_32,
            "valid_fidelity_16": valid_fidelity_16,
            "valid_fidelity_32": valid_fidelity_32}


# Per-clip extraction and complete case orchestration


def trajectory_table_to_metric_segments(
    table: TrajectoryTable, *, expected_n_prey: int | None = None,
    device: Any = None, dtype: Any = torch.float32,
) -> dict[str, Any]:
    """Extract all valid frames plus gap- and identity-safe temporal segments.

    Frame clips retain every valid individual frame. Temporal segments split at
    source-time gaps. Response segments additionally split whenever the sorted
    prey identity tuple changes. Segment indices reference the frame clip so
    expensive geometry and nearest-neighbor tensors can be calculated once.
    """

    frame_clips: list[dict[str, Any]] = []
    temporal_segments: list[dict[str, Any]] = []
    response_segments: list[dict[str, Any]] = []
    source_values = table["source"].astype(str)
    clip_values = table["clip_id"].astype(str)
    # Infer the stored frame-index cadence globally for this table.  Some
    # sources retain raw frame numbers (normal delta=2), while reconstructed
    # windows and rollouts use consecutive sample indices (normal delta=1).
    cadence_candidates: list[float] = []
    for source, clip_id in sorted(set(zip(source_values, clip_values))):
        rows = np.where((source_values == source) & (clip_values == clip_id))[0]
        times = np.sort(np.unique(table["time_step"][rows].astype(float)))
        cadence_candidates.extend(np.diff(times)[np.diff(times) > 0].tolist())
    if cadence_candidates:
        # The smallest observed positive increment is the acquisition cadence;
        # larger increments are missing-sample gaps.  This supports both raw
        # frame ids (delta=2) and re-indexed samples (delta=1).
        expected_source_delta = float(np.min(np.round(cadence_candidates, 6)))
    else:
        expected_source_delta = 1.0
    invalid_frames = 0
    for source, clip_id in sorted(set(zip(source_values, clip_values))):
        clip_rows = np.where((source_values == source) & (clip_values == clip_id))[0]
        frames: list[dict[str, Any]] = []
        for time in sorted(np.unique(table["time_step"][clip_rows]).tolist()):
            rows = clip_rows[np.isclose(table["time_step"][clip_rows], time)]
            roles = table["role"][rows].astype(str)
            pred_rows, prey_rows = rows[roles == "predator"], rows[roles == "prey"]
            valid = len(pred_rows) == 1 and len(prey_rows) >= 2
            if expected_n_prey is not None:
                valid &= len(prey_rows) == expected_n_prey
            if not valid:
                invalid_frames += 1
                continue
            prey_rows = prey_rows[np.argsort(table["agent_id"][prey_rows].astype(str))]
            p = pred_rows[0]
            pred_position = np.asarray([table["x"][p], table["y"][p]], dtype=float)
            prey_position = np.column_stack((table["x"][prey_rows], table["y"][prey_rows])).astype(float)
            pred_heading = float(table["heading"][p])
            prey_heading = table["heading"][prey_rows].astype(float)
            if not (np.all(np.isfinite(pred_position)) and np.all(np.isfinite(prey_position))
                    and np.isfinite(pred_heading) and np.all(np.isfinite(prey_heading))):
                invalid_frames += 1
                continue
            frames.append({
                "time": float(time), "predator_position": pred_position,
                "predator_heading": pred_heading, "prey_position": prey_position,
                "prey_heading": prey_heading,
                "prey_ids": tuple(table["agent_id"][prey_rows].astype(str)),
            })
        if not frames:
            continue
        frame_index = len(frame_clips)
        frame_clips.append({
            "clip_id": clip_id, "source": source,
            "time_steps": torch.as_tensor([row["time"] for row in frames], device=device),
            "predator_positions": _as_float_tensor(
                np.stack([row["predator_position"] for row in frames]), device=device, dtype=dtype),
            "predator_headings": _as_float_tensor(
                [row["predator_heading"] for row in frames], device=device, dtype=dtype),
            "prey_positions": _as_float_tensor(
                np.stack([row["prey_position"] for row in frames]), device=device, dtype=dtype),
            "prey_headings": _as_float_tensor(
                np.stack([row["prey_heading"] for row in frames]), device=device, dtype=dtype),
        })
        times = np.asarray([row["time"] for row in frames])
        temporal_starts = [0] + (
            np.where(~np.isclose(np.diff(times), expected_source_delta))[0] + 1
        ).tolist()
        temporal_stops = temporal_starts[1:] + [len(frames)]
        for part, (start, stop) in enumerate(zip(temporal_starts, temporal_stops)):
            indices = torch.arange(start, stop, device=device)
            temporal_segments.append({
                "clip_index": frame_index, "clip_id": clip_id,
                "segment_id": f"{clip_id}_t{part}", "indices": indices,
            })
            block = frames[start:stop]
            identity_starts = [0] + [
                i for i in range(1, len(block))
                if block[i]["prey_ids"] != block[i - 1]["prey_ids"]
            ]
            identity_stops = identity_starts[1:] + [len(block)]
            for subpart, (sub_start, sub_stop) in enumerate(zip(identity_starts, identity_stops)):
                response_segments.append({
                    "clip_index": frame_index, "clip_id": clip_id,
                    "segment_id": f"{clip_id}_t{part}_r{subpart}",
                    "indices": torch.arange(start + sub_start, start + sub_stop, device=device),
                })
    return {
        "frame_clips": frame_clips, "temporal_segments": temporal_segments,
        "response_segments": response_segments,
        "diagnostics": {
            "valid_frames": sum(len(clip["time_steps"]) for clip in frame_clips),
            "invalid_frames": invalid_frames,
            "frame_clips": len(frame_clips),
            "frame_level_segments": len(temporal_segments),
            "temporal_segments": len(temporal_segments),
            "response_segments": len(response_segments),
            "response_capable_segments": sum(len(s["indices"]) >= 2 for s in response_segments),
            "expected_source_timestep_delta": expected_source_delta,
        },
    }


def trajectory_table_to_tensor_clips(
    table: TrajectoryTable, *, device: Any = None, dtype: Any = torch.float32
) -> list[dict[str, Any]]:
    """Backward-compatible materialization of gap-safe temporal segments."""

    bundle = trajectory_table_to_metric_segments(table, device=device, dtype=dtype)
    clips: list[dict[str, Any]] = []
    for segment in bundle["temporal_segments"]:
        if len(segment["indices"]) < 2:
            continue
        parent = bundle["frame_clips"][segment["clip_index"]]
        indices = segment["indices"]
        clips.append({
            "clip_id": segment["segment_id"], "source": parent["source"],
            **{key: parent[key][indices] for key in (
                "time_steps", "predator_positions", "predator_headings",
                "prey_positions", "prey_headings")},
        })
    return clips


@torch.inference_mode()
def analyze_tensor_clip(
    clip: Mapping[str, Any],
    *,
    d_source: float,
    risk_bin_edges: Any,
    approach_distance_threshold: float = 0.15,
    approach_closing_lag: int = 3,
    approach_exit_threshold: float | None = None,
    approach_min_duration: int = 2,
    approach_cooldown_steps: int = 20,
    closing_speed_lag: int = 3,
    response_threshold: float = 0.10,
    response_window_steps: int = 20,
    response_lag: int = 1,
    time_chunk_size: int = 4096,
    min_bin_count: int = 30,
    step_duration: float = 1.0,
) -> dict[str, Any]:
    """Calculate every requested metric for one already-loaded clip/rollout."""

    geometry = prepare_geometry_cache(
        clip, d_source=d_source, time_chunk_size=time_chunk_size,
        device=clip["prey_positions"].device if torch.is_tensor(clip["prey_positions"]) else None,
    )
    raw_distance, normalized_distance = compute_predator_nearest_distance(geometry)
    closing_speed_raw = compute_closing_speed(
        raw_distance, lag=closing_speed_lag, step_duration=step_duration)
    closing_speed = compute_closing_speed(
        normalized_distance, lag=closing_speed_lag, step_duration=step_duration)
    pursuit_alignment = compute_pursuit_alignment(geometry)
    escape_alignment = compute_escape_alignment_tensor(geometry)
    continuous_response = compute_continuous_predator_response_tensor(geometry, lag=response_lag)
    response_risk = geometry["predator_to_prey_distance"][:-response_lag] / float(d_source)
    onsets = detect_approach_events(
        normalized_distance, threshold=approach_distance_threshold,
        closing_lag=approach_closing_lag, exit_threshold=approach_exit_threshold,
        min_duration=approach_min_duration, cooldown_steps=approach_cooldown_steps,
        target_index=geometry["nearest_prey_index"], require_target_persistence=True)
    reactions = compute_reaction_metrics(
        continuous_response, onsets, geometry["predator_positions"], geometry["prey_positions"],
        response_threshold=response_threshold, response_window_steps=response_window_steps,
        response_lag=response_lag, d_source=d_source, step_duration=step_duration,
    )
    mean_nnd = geometry["prey_nnd"].mean(dim=1) / float(d_source)
    polarization = compute_polarization_tensor(geometry["prey_headings"])
    dos, doa = compute_dos(geometry, d_source=d_source), compute_doa(geometry)
    risk_index = risk_bin_indices(normalized_distance, risk_bin_edges)
    return {
        "clip_id": clip.get("clip_id", "clip"), "n_prey": geometry["prey_positions"].shape[1],
        "predator_distance": {"raw": raw_distance, "normalized": normalized_distance},
        "closing_speed": {"values": closing_speed, "curve": risk_conditioned_mean(
            normalized_distance[:-closing_speed_lag], closing_speed, risk_bin_edges,
            min_count=min_bin_count, bin_index=risk_index[:-closing_speed_lag])},
        "closing_speed_raw": closing_speed_raw,
        "pursuit_alignment": {"values": pursuit_alignment, "curve": risk_conditioned_mean(
            normalized_distance, pursuit_alignment, risk_bin_edges, min_count=min_bin_count,
            bin_index=risk_index)},
        "escape_alignment": {"values": escape_alignment, "curve": risk_conditioned_mean(
            normalized_distance, escape_alignment.mean(dim=1), risk_bin_edges,
            min_count=min_bin_count, bin_index=risk_index)},
        "risk_conditioned_nnd": {"values": mean_nnd, "curve": risk_conditioned_mean(
            normalized_distance, mean_nnd, risk_bin_edges,
            min_count=min_bin_count, bin_index=risk_index)},
        "risk_conditioned_polarization": {"values": polarization, "curve": risk_conditioned_mean(
            normalized_distance, polarization, risk_bin_edges,
            min_count=min_bin_count, bin_index=risk_index)},
        "continuous_predator_response": continuous_response,
        "continuous_predator_response_curve": {
            "values": continuous_response,
            "risk": response_risk,
            "curve": risk_conditioned_mean(
                response_risk.flatten(), continuous_response.flatten(), risk_bin_edges,
                min_count=min_bin_count),
        },
        "approach_onsets": onsets,
        "reaction": reactions,
        "nnd_during_approach": compute_event_relative_change(
            mean_nnd, onsets, response_window_steps=response_window_steps, relative=True),
        "polarization_during_approach": compute_event_relative_change(
            polarization, onsets, response_window_steps=response_window_steps),
        "dos": dos, "doa": doa,
    }


def _nanmean_or_nan(values: torch.Tensor) -> torch.Tensor:
    if not values.is_floating_point() and not values.is_complex():
        values = values.to(torch.float32)
    finite = values[torch.isfinite(values)]
    return finite.mean() if len(finite) else values.new_tensor(torch.nan)


@torch.inference_mode()
def analyze_case(
    data: TrajectoryTable | Sequence[Mapping[str, Any]],
    *,
    d_source: float,
    risk_bin_edges: Any,
    expected_n_prey: int | None = None,
    device: Any = None,
    dtype: Any = torch.float32,
    approach_distance_threshold: float = 0.15,
    approach_closing_lag: int = 3,
    approach_exit_threshold: float | None = None,
    approach_min_duration: int = 2,
    approach_cooldown_steps: int = 20,
    require_target_persistence: bool = True,
    closing_speed_lag: int = 3,
    response_threshold: float = 0.10,
    response_threshold_sensitivity: Sequence[float] = (0.075, 0.10, 0.125),
    response_window_steps: int = 20,
    response_lag: int = 1,
    time_chunk_size: int = 4096,
    min_bin_count: int = 30,
    min_cluster_count: int = 5,
    n_bootstrap: int = 0,
    ci_level: float = 0.95,
    bootstrap_seed: int = 0,
    step_duration: float = 1.0,
) -> dict[str, Any]:
    """Analyze static frames, gap-safe temporal segments, and identity-safe responses."""

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    if isinstance(data, TrajectoryTable):
        if expected_n_prey is None:
            first_group = next(iter(frame_groups(data)), np.asarray([], dtype=int))
            expected_n_prey = (int(np.sum(data["role"][first_group].astype(str) == "prey"))
                               if len(first_group) else None)
        bundle = trajectory_table_to_metric_segments(
            data, expected_n_prey=expected_n_prey, device="cpu", dtype=dtype)
    else:
        frame_clips = list(data)
        temporal_segments, response_segments = [], []
        for clip_index, clip in enumerate(frame_clips):
            indices = torch.arange(len(clip["time_steps"]))
            base = {"clip_index": clip_index, "clip_id": clip.get("clip_id", f"clip_{clip_index}"),
                    "segment_id": clip.get("clip_id", f"clip_{clip_index}"), "indices": indices}
            temporal_segments.append(base); response_segments.append(dict(base))
        bundle = {"frame_clips": frame_clips, "temporal_segments": temporal_segments,
                  "response_segments": response_segments,
                  "diagnostics": {"valid_frames": sum(len(c["time_steps"]) for c in frame_clips),
                                  "invalid_frames": 0, "frame_clips": len(frame_clips),
                                  "frame_level_segments": len(temporal_segments),
                                  "temporal_segments": len(temporal_segments),
                                  "response_segments": len(response_segments),
                                  "response_capable_segments": sum(len(c["time_steps"]) >= 2 for c in frame_clips)}}
    frame_clips = bundle["frame_clips"]
    if not frame_clips:
        raise ValueError("No valid predator/prey frames")
    edges = _as_float_tensor(risk_bin_edges, device=target_device, dtype=dtype)

    temporal_by_clip: dict[int, list[Mapping[str, Any]]] = {i: [] for i in range(len(frame_clips))}
    response_by_clip: dict[int, list[Mapping[str, Any]]] = {i: [] for i in range(len(frame_clips))}
    for segment in bundle["temporal_segments"]:
        temporal_by_clip[segment["clip_index"]].append(segment)
    for segment in bundle["response_segments"]:
        response_by_clip[segment["clip_index"]].append(segment)

    curve_names = ("closing_speed", "pursuit_alignment", "escape_alignment",
                   "risk_conditioned_nnd", "risk_conditioned_polarization",
                   "continuous_predator_response_curve")
    curve_x: dict[str, list[torch.Tensor]] = {name: [] for name in curve_names}
    curve_y: dict[str, list[torch.Tensor]] = {name: [] for name in curve_names}
    scalar_names = (
        "predator_distance", "closing_speed", "pursuit_alignment", "escape_alignment",
        "continuous_predator_response", "reaction_latency", "reaction_distance",
        "reaction_distance_norm", "response_fraction", "nnd_during_approach",
        "polarization_during_approach", "dos", "doa", "cascade_size",
        "cascade_fraction", "propagation_time", "mean_propagation_delay",
        "reaction_latency_time", "propagation_time_scaled",
        "mean_propagation_delay_scaled",
    )
    per_clip: dict[str, list[torch.Tensor]] = {name: [] for name in scalar_names}
    event_names = ("reaction_latency", "reaction_distance", "reaction_distance_norm",
                   "response_fraction", "cascade_size", "cascade_fraction",
                   "propagation_time", "mean_propagation_delay", "available_steps",
                   "observed_responder_count", "confirmed_non_responder_count",
                   "censored_count", "classification_denominator",
                   "response_fraction_classified", "response_fraction_complete_case",
                   "reaction_latency_time", "propagation_time_scaled",
                   "mean_propagation_delay_scaled", "observed_cascade_size")
    event_results: dict[str, list[torch.Tensor]] = {name: [] for name in event_names}
    sensitivity_values = {float(t): {"fraction": [], "two_plus": [], "clips": set()}
                          for t in response_threshold_sensitivity}
    response_availability: list[torch.Tensor] = []
    distance_raw, distance_norm, dos_timelines, doa_timelines = [], [], [], []
    nnd_profiles, polarization_profiles = [], []
    nnd_profiles_per_clip, polarization_profiles_per_clip = [], []
    temporal_approach_events = response_approach_events = responding_events = 0

    for clip_index, clip_cpu in enumerate(frame_clips):
        clip = {key: (value.to(target_device) if torch.is_tensor(value) else value)
                for key, value in clip_cpu.items()}
        geometry = prepare_geometry_cache(
            clip, d_source=d_source, time_chunk_size=time_chunk_size,
            device=target_device, dtype=dtype, include_prey_neighbors=True)
        raw, normalized = compute_predator_nearest_distance(geometry)
        pursuit = compute_pursuit_alignment(geometry)
        escape = compute_escape_alignment_tensor(geometry).mean(dim=1)
        mean_nnd = geometry["prey_nnd"].mean(dim=1) / float(d_source)
        polarization = compute_polarization_tensor(geometry["prey_headings"])
        dos, doa = compute_dos(geometry, d_source=d_source), compute_doa(geometry)
        risk_index = risk_bin_indices(normalized, edges)

        distance_raw.append(raw); distance_norm.append(normalized)
        dos_timelines.append(dos["timeline"]); doa_timelines.append(doa["timeline"])
        for name, values in (("pursuit_alignment", pursuit), ("escape_alignment", escape),
                             ("risk_conditioned_nnd", mean_nnd),
                             ("risk_conditioned_polarization", polarization)):
            curve_x[name].append(normalized); curve_y[name].append(values)

        clip_closing, clip_closing_risk = [], []
        clip_nnd_profiles, clip_pol_profiles = [], []
        for segment in temporal_by_clip[clip_index]:
            idx = segment["indices"].to(target_device)
            segment_distance = normalized[idx]
            closing = compute_closing_speed(
                segment_distance, lag=closing_speed_lag, step_duration=step_duration,
            )
            if len(closing):
                clip_closing.append(closing); clip_closing_risk.append(segment_distance[:-closing_speed_lag])
            onsets = detect_approach_events(
                segment_distance, threshold=approach_distance_threshold,
                closing_lag=approach_closing_lag,
                exit_threshold=approach_exit_threshold,
                min_duration=approach_min_duration,
                cooldown_steps=approach_cooldown_steps,
                target_index=geometry["nearest_prey_index"][idx],
                require_target_persistence=require_target_persistence)
            temporal_approach_events += len(onsets)
            if len(onsets):
                relative_nnd = compute_event_relative_change(
                    mean_nnd[idx], onsets, response_window_steps=response_window_steps, relative=True)
                delta_pol = compute_event_relative_change(
                    polarization[idx], onsets, response_window_steps=response_window_steps)
                clip_nnd_profiles.append(relative_nnd); clip_pol_profiles.append(delta_pol)
                nnd_profiles.append(relative_nnd); polarization_profiles.append(delta_pol)
        closing_values = torch.cat(clip_closing) if clip_closing else normalized.new_empty(0)
        closing_risk = torch.cat(clip_closing_risk) if clip_closing_risk else normalized.new_empty(0)
        if len(closing_values):
            curve_x["closing_speed"].append(closing_risk)
            curve_y["closing_speed"].append(closing_values)

        clip_response, clip_response_risk, clip_reactions = [], [], []
        for segment in response_by_clip[clip_index]:
            idx = segment["indices"].to(target_device)
            if len(idx) <= max(response_lag, approach_closing_lag):
                continue
            sub_geometry = {
                "prey_heading_vectors": geometry["prey_heading_vectors"][idx],
                "predator_to_prey": geometry["predator_to_prey"][idx],
            }
            response = compute_continuous_predator_response_tensor(sub_geometry, lag=response_lag)
            response_risk = geometry["predator_to_prey_distance"][idx][:-response_lag] / float(d_source)
            clip_response.append(response.flatten()); clip_response_risk.append(response_risk.flatten())
            segment_distance = normalized[idx]
            onsets = detect_approach_events(
                segment_distance, threshold=approach_distance_threshold,
                closing_lag=approach_closing_lag,
                exit_threshold=approach_exit_threshold,
                min_duration=approach_min_duration,
                cooldown_steps=approach_cooldown_steps,
                target_index=geometry["nearest_prey_index"][idx],
                require_target_persistence=require_target_persistence)
            response_approach_events += len(onsets)
            reactions = compute_reaction_metrics(
                response, onsets, geometry["predator_positions"][idx], geometry["prey_positions"][idx],
                response_threshold=response_threshold, response_window_steps=response_window_steps,
                response_lag=response_lag, d_source=d_source, step_duration=step_duration)
            clip_reactions.append(reactions)
            response_availability.append(reactions["availability"])
            responding_events += int((reactions["observed_responder_count"] > 0).sum())
            for threshold in sensitivity_values:
                sensitivity = (reactions if np.isclose(threshold, response_threshold) else
                    compute_reaction_metrics(
                        response, onsets, geometry["predator_positions"][idx], geometry["prey_positions"][idx],
                        response_threshold=threshold, response_window_steps=response_window_steps,
                        response_lag=response_lag, step_duration=step_duration))
                sensitivity_values[threshold]["fraction"].append(sensitivity["response_fraction"])
                complete_two_plus = torch.where(
                    sensitivity["classification_complete"],
                    (sensitivity["observed_responder_count"] >= 2).to(response.dtype),
                    torch.nan)
                sensitivity_values[threshold]["two_plus"].append(complete_two_plus)
                if len(onsets): sensitivity_values[threshold]["clips"].add(clip["clip_id"])
        response_values = torch.cat(clip_response) if clip_response else normalized.new_empty(0)
        response_risk_values = torch.cat(clip_response_risk) if clip_response_risk else normalized.new_empty(0)
        if len(response_values):
            curve_x["continuous_predator_response_curve"].append(response_risk_values)
            curve_y["continuous_predator_response_curve"].append(response_values)

        reaction_by_name: dict[str, torch.Tensor] = {}
        for name in event_names:
            parts = [reaction[name].flatten() for reaction in clip_reactions if reaction[name].numel()]
            values = torch.cat(parts) if parts else normalized.new_empty(0)
            if values.is_floating_point(): values = values[torch.isfinite(values)]
            reaction_by_name[name] = values; event_results[name].append(values)
        clip_nnd = torch.cat(clip_nnd_profiles) if clip_nnd_profiles else normalized.new_empty((0, response_window_steps + 1))
        clip_pol = torch.cat(clip_pol_profiles) if clip_pol_profiles else normalized.new_empty((0, response_window_steps + 1))
        nnd_profiles_per_clip.append(
            torch.nanmean(clip_nnd, dim=0) if len(clip_nnd)
            else normalized.new_full((response_window_steps + 1,), torch.nan)
        )
        polarization_profiles_per_clip.append(
            torch.nanmean(clip_pol, dim=0) if len(clip_pol)
            else normalized.new_full((response_window_steps + 1,), torch.nan)
        )
        clip_scalars = {
            "predator_distance": _nanmean_or_nan(normalized),
            "closing_speed": _nanmean_or_nan(closing_values),
            "pursuit_alignment": _nanmean_or_nan(pursuit),
            "escape_alignment": _nanmean_or_nan(escape),
            "continuous_predator_response": _nanmean_or_nan(response_values),
            "reaction_latency": _nanmean_or_nan(reaction_by_name["reaction_latency"]),
            "reaction_latency_time": _nanmean_or_nan(reaction_by_name["reaction_latency_time"]),
            "reaction_distance": _nanmean_or_nan(reaction_by_name["reaction_distance"]),
            "reaction_distance_norm": _nanmean_or_nan(reaction_by_name["reaction_distance_norm"]),
            "response_fraction": _nanmean_or_nan(reaction_by_name["response_fraction"]),
            "nnd_during_approach": _nanmean_or_nan(clip_nnd[:, -1]),
            "polarization_during_approach": _nanmean_or_nan(clip_pol[:, -1]),
            "dos": dos["aggregate"], "doa": doa["aggregate"],
            "cascade_size": _nanmean_or_nan(reaction_by_name["cascade_size"]),
            "cascade_fraction": _nanmean_or_nan(reaction_by_name["cascade_fraction"]),
            "propagation_time": _nanmean_or_nan(reaction_by_name["propagation_time"]),
            "propagation_time_scaled": _nanmean_or_nan(reaction_by_name["propagation_time_scaled"]),
            "mean_propagation_delay": _nanmean_or_nan(reaction_by_name["mean_propagation_delay"]),
            "mean_propagation_delay_scaled": _nanmean_or_nan(reaction_by_name["mean_propagation_delay_scaled"]),
        }
        for name, value in clip_scalars.items(): per_clip[name].append(value)

    curves: dict[str, dict[str, torch.Tensor]] = {}
    bin_index_cache: dict[tuple[tuple[int, int], ...], list[torch.Tensor]] = {}
    for name in curve_names:
        if not curve_x[name]:
            empty = edges.new_empty(len(edges) - 1); empty[:] = torch.nan
            curves[name] = {"bin_left": edges[:-1], "bin_right": edges[1:],
                            "bin_center": (edges[:-1] + edges[1:]) / 2,
                            "mean": empty, "ci_low": empty.clone(), "ci_high": empty.clone(),
                            "sample_weighted_mean": empty.clone(),
                            "sample_weighted_ci_low": empty.clone(),
                            "sample_weighted_ci_high": empty.clone(),
                            "clip_balanced_mean": empty.clone(),
                            "clip_balanced_ci_low": empty.clone(),
                            "clip_balanced_ci_high": empty.clone(),
                            "count": torch.zeros(len(empty), device=edges.device, dtype=torch.long),
                            "cluster_count": torch.zeros(len(empty), device=edges.device, dtype=torch.long),
                            "candidate_count": torch.zeros(len(empty), device=edges.device, dtype=torch.long),
                            "candidate_cluster_count": torch.zeros(len(empty), device=edges.device, dtype=torch.long),
                            "missing_value_count": torch.zeros(len(empty), device=edges.device, dtype=torch.long),
                            "nonfinite_distance_count": torch.tensor(0, device=edges.device),
                            "below_range_count": torch.tensor(0, device=edges.device),
                            "above_range_count": torch.tensor(0, device=edges.device),
                            "supported": torch.zeros(len(empty), device=edges.device, dtype=torch.bool)}
        else:
            cache_key = tuple((x.data_ptr(), x.numel()) for x in curve_x[name])
            if cache_key not in bin_index_cache:
                bin_index_cache[cache_key] = [risk_bin_indices(x, edges) for x in curve_x[name]]
            curves[name] = cluster_bootstrap_risk_conditioned_mean(
                curve_x[name], curve_y[name], edges, min_count=min_bin_count,
                min_cluster_count=min_cluster_count, n_bootstrap=n_bootstrap,
                ci_level=ci_level, seed=bootstrap_seed,
                bin_indices_by_cluster=bin_index_cache[cache_key])

    stacked_per_clip = {name: torch.stack(values) for name, values in per_clip.items()}
    aggregate = {name: _nanmean_or_nan(values) for name, values in stacked_per_clip.items()}
    empty = edges.new_empty(0)
    events = {name: torch.cat(values) if values and any(v.numel() for v in values) else empty
              for name, values in event_results.items()}
    sensitivity_summary = {}
    for threshold, values in sensitivity_values.items():
        fraction = torch.cat(values["fraction"]) if values["fraction"] else empty
        two_plus = torch.cat(values["two_plus"]) if values["two_plus"] else empty
        sensitivity_summary[threshold] = {
            "events": len(fraction), "contributing_clips": len(values["clips"]),
            "mean_response_fraction": _nanmean_or_nan(fraction),
            "fraction_events_two_plus": _nanmean_or_nan(two_plus),
            "mean_cascade_fraction": _nanmean_or_nan(fraction),
        }
    diagnostics = dict(bundle["diagnostics"])
    diagnostics.update({
        "approach_events": temporal_approach_events,
        "response_approach_events": response_approach_events,
        "responding_events": responding_events,
        "total_approach_events": response_approach_events,
        "valid_propagation_events": int(torch.isfinite(events["propagation_time"]).sum()),
    })
    diagnostics["fraction_valid_propagation_events"] = (
        diagnostics["valid_propagation_events"] / max(response_approach_events, 1))
    return {
        "n_prey": frame_clips[0]["prey_positions"].shape[1], "n_clips": len(frame_clips),
        "step_duration": float(step_duration),
        "clip_ids": [str(clip.get("clip_id", f"clip_{i}")) for i, clip in enumerate(frame_clips)],
        "curves": curves, "aggregate": aggregate, "per_clip": stacked_per_clip,
        "events": events, "diagnostics": diagnostics,
        "response_threshold_sensitivity": sensitivity_summary,
        "predator_distance_raw": distance_raw,
        "predator_distance_normalized": distance_norm,
        "dos_timeline": dos_timelines, "doa_timeline": doa_timelines,
        "nnd_during_approach": nnd_profiles,
        "polarization_during_approach": polarization_profiles,
        "nnd_during_approach_per_clip": torch.stack(nnd_profiles_per_clip),
        "polarization_during_approach_per_clip": torch.stack(polarization_profiles_per_clip),
        "response_availability": response_availability,
        # Retain the already-computed per-clip samples for diagnostic rebinning.
        "risk_curve_clusters": {
            name: {"x": curve_x[name], "values": curve_y[name]} for name in curve_names
        },
    }


def compute_all_group_size_comparisons(
    analysis_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Compute 16/32 consistency shifts for every complete source quartet."""

    output: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for source in ("couzin", "bio"):
        keys = {
            "expert_16": f"{source}_expert_16", "expert_32": f"{source}_expert_32",
            "imitation_16": f"{source}_imitation_16", "imitation_32": f"{source}_imitation_32",
        }
        if not all(key in analysis_results for key in keys.values()):
            continue
        source_output: dict[str, dict[str, torch.Tensor]] = {}
        scalar_metrics = set.intersection(*[
            set(analysis_results[key]["aggregate"]) for key in keys.values()
        ])
        for metric in scalar_metrics:
            source_output[metric] = compute_generalization_error(*[
                analysis_results[keys[name]]["aggregate"][metric]
                for name in ("expert_16", "expert_32", "imitation_16", "imitation_32")
            ])
        curve_metrics = set.intersection(*[
            set(analysis_results[key]["curves"]) for key in keys.values()
        ])
        for metric in curve_metrics:
            sample_weighted = compute_generalization_error(*[
                analysis_results[keys[name]]["curves"][metric]["sample_weighted_mean"]
                for name in ("expert_16", "expert_32", "imitation_16", "imitation_32")])
            clip_balanced = compute_generalization_error(*[
                analysis_results[keys[name]]["curves"][metric]["clip_balanced_mean"]
                for name in ("expert_16", "expert_32", "imitation_16", "imitation_32")])
            # The historical curve key remains a sample-weighted alias.
            source_output[f"{metric}_curve"] = sample_weighted
            source_output[f"{metric}_curve_sample_weighted"] = sample_weighted
            source_output[f"{metric}_curve_clip_balanced"] = clip_balanced
        output[source] = source_output
    return output


# Modular-network interpretation maps


@torch.inference_mode()
def compute_modular_network_maps(
    policy: Any,
    *,
    role: str,
    grid_size: int = 100,
    n_orientations: int = 100,
    batch_size: int = 65536,
    device: Any = None,
) -> dict[str, torch.Tensor]:
    """Vectorized PIN/attention maps adapted from Jannik's experiment notebook.

    Current policies append an active-mask feature, while older policies do
    not.  The input width is inferred from the trained PIN so both layouts are
    supported without changing either source repository.
    """

    if role not in {"predator", "prey", "prey_pred"}:
        raise ValueError("role must be 'predator', 'prey', or 'prey_pred'")
    if grid_size < 2 or n_orientations < 1 or batch_size < 1:
        raise ValueError("grid_size >= 2, n_orientations >= 1 and batch_size >= 1 are required")
    if not hasattr(policy, "pairwise") or not hasattr(policy, "attention"):
        raise ValueError("policy must expose pairwise and attention modules")

    original_parameter = next(policy.parameters())
    original_device = original_parameter.device
    target_device = torch.device(device) if device is not None else original_device
    was_training = policy.training
    policy.to(target_device).eval()

    xs = torch.linspace(-1.0, 1.0, grid_size, device=target_device)
    ys = torch.linspace(-1.0, 1.0, grid_size, device=target_device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    positions = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    theta = torch.linspace(-torch.pi, torch.pi, n_orientations + 1, device=target_device)[:-1]
    relative_velocity = torch.stack((theta.cos(), theta.sin()), dim=-1)

    base = torch.cat((
        positions[:, None, :].expand(-1, n_orientations, -1),
        relative_velocity[None, :, :].expand(grid_size * grid_size, -1, -1),
    ), dim=-1).reshape(-1, 4)
    in_features = int(policy.pairwise.fc1.in_features)
    if role == "predator":
        inputs = base if in_features == 4 else torch.cat((base, torch.ones_like(base[:, :1])), dim=1)
    else:
        flag = torch.full_like(base[:, :1], 1.0 if role == "prey_pred" else 0.0)
        inputs = torch.cat((flag, base), dim=1)
        if in_features == 6:
            inputs = torch.cat((inputs, torch.ones_like(flag)), dim=1)
    if inputs.shape[1] != in_features:
        policy.to(original_device).train(was_training)
        raise ValueError(
            f"Cannot build {role} map for PIN input width {in_features}; constructed {inputs.shape[1]} features"
        )

    action_chunks, attention_chunks = [], []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        mu, _ = policy.pairwise(batch)
        action_chunks.append(torch.tanh(mu.squeeze(-1)))
        attention_chunks.append(policy.attention(batch).squeeze(-1))
    action = torch.cat(action_chunks).reshape(grid_size * grid_size, n_orientations).mean(dim=1)
    attention = torch.cat(attention_chunks).reshape(grid_size * grid_size, n_orientations).mean(dim=1)
    attention_min, attention_max = attention.min(), attention.max()
    attention = ((attention - attention_min) / (attention_max - attention_min)
                 if bool(attention_max > attention_min) else torch.zeros_like(attention))
    result = {
        "x": xs.detach().cpu(), "y": ys.detach().cpu(),
        "action": action.reshape(grid_size, grid_size).detach().cpu(),
        "attention": attention.reshape(grid_size, grid_size).detach().cpu(),
    }
    policy.to(original_device).train(was_training)
    return result


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib is imported only when plotting is requested)

# Plotting remains separate from metric calculation so headless analysis and
# result serialization do not require importing matplotlib.


def _plt() -> Any:
    return importlib.import_module("matplotlib.pyplot")


def plot_risk_conditioned_metrics(
    summaries: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]], *, distance_label: str = "Predator to nearest prey distance [source units]"
) -> Any:
    """Three-panel Expert/Policy Analysis-1 figure."""

    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    labels = {"prey_nnd": "Prey NND [source units]", "polarization": "Polarization",
              "escape_alignment": "Escape alignment"}
    for ax, metric in zip(axes, labels):
        for series_label, result in summaries.items():
            summary = result[metric]
            line, = ax.plot(summary["bin_center"], summary["estimate"], marker="o", ms=3, label=series_label)
            valid_ci = np.isfinite(summary["ci_low"]) & np.isfinite(summary["ci_high"])
            ax.fill_between(summary["bin_center"], summary["ci_low"], summary["ci_high"],
                            where=valid_ci, color=line.get_color(), alpha=0.18)
        ax.set_xlabel(distance_label)
        ax.set_ylabel(labels[metric])
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Predator proximity versus collective prey state")
    fig.tight_layout()
    return fig


def plot_predator_response_curves(
    curves: Mapping[str, Mapping[str, np.ndarray]],
    bearing_maps: Mapping[str, Mapping[str, np.ndarray]],
) -> Any:
    """Distance response curve plus bearing-sensitive heatmaps."""

    plt = _plt()
    n_maps = len(bearing_maps)
    fig = plt.figure(figsize=(6 + 5 * n_maps, 4.5))
    grid = fig.add_gridspec(1, 1 + n_maps)
    ax = fig.add_subplot(grid[0, 0])
    for label, curve in curves.items():
        line, = ax.plot(curve["bin_center"], curve["estimate"], marker="o", ms=3, label=label)
        valid = np.isfinite(curve["ci_low"]) & np.isfinite(curve["ci_high"])
        ax.fill_between(curve["bin_center"], curve["ci_low"], curve["ci_high"], where=valid,
                        color=line.get_color(), alpha=0.18)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xlabel("Predator distance [source units]")
    ax.set_ylabel("Change in away-alignment R [per source step]")
    ax.set_title("Continuous response versus predator distance")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    finite_maps = [np.abs(m["mean"][np.isfinite(m["mean"])]) for m in bearing_maps.values() if np.any(np.isfinite(m["mean"]))]
    vmax = float(np.max(np.concatenate(finite_maps))) if finite_maps else 1.0
    for position, (label, result) in enumerate(bearing_maps.items(), start=1):
        map_ax = fig.add_subplot(grid[0, position])
        image = map_ax.pcolormesh(result["distance_edges"], result["bearing_edges"],
                                  np.ma.masked_invalid(result["mean"]), shading="auto", cmap="coolwarm",
                                  vmin=-vmax, vmax=vmax)
        map_ax.set_xlabel("Predator distance [source units]")
        map_ax.set_ylabel("Predator bearing [rad; +left]")
        map_ax.set_title(f"{label}: distance x bearing")
        fig.colorbar(image, ax=map_ax, label="Mean R")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Deterministic sanity checks

# These inexpensive checks encode the most important sign, orientation, and
# known-distance conventions.  They are intentionally independent of data.


def run_geometry_sanity_tests() -> dict[str, bool]:
    """Run deterministic geometry, metric, and response-sign checks."""

    tests: dict[str, bool] = {}
    tests["wrap_pi"] = bool(np.isclose(wrap_angle(3 * np.pi), -np.pi))
    tests["predator_ahead"] = bool(np.allclose(focal_frame_transform([2, 0], 0), [2, 0]))
    tests["predator_left"] = bool(np.allclose(focal_frame_transform([0, 2], 0), [0, 2]))
    tests["predator_right"] = bool(np.allclose(focal_frame_transform([0, -2], 0), [0, -2]))
    tests["left_turn_positive"] = bool(np.isclose(angular_difference(np.pi / 4, 0), np.pi / 4))
    tests["right_turn_negative"] = bool(np.isclose(angular_difference(-np.pi / 4, 0), -np.pi / 4))
    tests["polarization_aligned"] = bool(np.isclose(compute_polarization(np.zeros(4)), 1.0))
    tests["polarization_opposed"] = bool(compute_polarization(np.array([0, np.pi, 0, np.pi])) < 1e-8)
    square = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    tests["known_nnd"] = bool(np.isclose(compute_prey_nnd(square), 1.0))
    prey = np.array([[2.0, 0.0]])
    tests["closer_risk_distance"] = bool(
        predator_to_nearest_prey_distance([1, 0], prey) < predator_to_nearest_prey_distance([0, 0], prey)
    )
    away = away_from_predator_direction([1, 0], [0, 0])
    before = heading_alignment(np.pi / 2, away)
    after = heading_alignment(0, away)
    tests["away_response_positive"] = bool(after - before > 0)
    km = kaplan_meier_response_probability([1, 2, 3, 3], [1, 0, 1, 0], horizon=3)
    tests["km_handles_censoring"] = bool(torch.isclose(km, torch.tensor(0.625)))
    event_distance = torch.tensor([0.30, 0.20, 0.14, 0.13, 0.14, 0.13, 0.14, 0.20])
    robust_events = detect_approach_events(
        event_distance, threshold=0.15, closing_lag=1, exit_threshold=0.17,
        min_duration=1, cooldown_steps=3, require_target_persistence=False,
    )
    tests["approach_hysteresis_prevents_flicker"] = len(robust_events) == 1
    if not all(tests.values()):
        failed = [name for name, passed in tests.items() if not passed]
        raise AssertionError(f"Geometry/metric sanity tests failed: {failed}")
    return tests


__all__ = [
    "ExpertDataUnavailable", "TrajectoryTable", "to_canonical_trajectory",
    "concatenate_trajectory_tables", "with_constant_metadata",
    "initial_state_pool_from_trajectory", "wrap_angle",
    "angular_difference", "euclidean_distance", "prey_centroid",
    "predator_to_nearest_prey_distance", "focal_frame_transform", "vector_angle",
    "away_from_predator_direction", "heading_alignment", "frame_groups", "active_prey_filter",
    "load_expert_32prey", "load_expert_trajectories", "load_biological_window_trajectories",
    "load_policy_pair", "load_jannik_policy_pair",
    "generate_policy_rollouts", "generate_couzin_expert_rollouts", "trajectory_diagnostics",
    "compute_prey_nnd", "compute_polarization", "compute_escape_alignment",
    "compute_risk_conditioned_metrics", "make_bin_edges", "binned_cluster_summary",
    "summarize_risk_metrics", "compute_continuous_predator_response", "compute_response_curve",
    "compute_distance_bearing_response", "plot_risk_conditioned_metrics",
    "plot_predator_response_curves", "run_geometry_sanity_tests",
    "compute_prey_nearest_neighbors", "prepare_geometry_cache",
    "compute_predator_nearest_distance", "compute_closing_speed", "compute_pursuit_alignment",
    "compute_escape_alignment_tensor", "compute_continuous_predator_response_tensor",
    "detect_approach_events", "kaplan_meier_response_probability", "bootstrap_clip_mean",
    "compute_reaction_metrics", "compute_polarization_tensor",
    "compute_event_relative_change", "compute_dos", "compute_doa", "risk_bin_indices",
    "risk_conditioned_mean", "cluster_bootstrap_risk_conditioned_mean",
    "compute_group_size_difference", "compute_generalization_error",
    "trajectory_table_to_metric_segments", "trajectory_table_to_tensor_clips",
    "analyze_tensor_clip", "analyze_case",
    "compute_all_group_size_comparisons", "compute_modular_network_maps",
]
