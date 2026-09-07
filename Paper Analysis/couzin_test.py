"""Isolated Couzin-only sensitivity analyses for the ICRA 2027 paper.

Nothing in this module loads, transforms, or plots the biological data.  It is
an experimental harness for checking whether Couzin/GAIL discrepancies are
caused by evaluation mismatch, checkpoint pairing, rollout horizon, event
sampling, or explicitly declared process noise.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import importlib
import math
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
NICOLE_ROOT = PROJECT_ROOT / "Nicole" / "Predator-Prey-Thesis"
CHECKPOINT_DIR = (
    NICOLE_ROOT
    / "Data/2. Training/CouzinPredPrey - GAIL/stage1_seed42_32and16"
)

if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import final_analysis as fa
import paper_analysis2 as pa


PAPER_METRICS = OrderedDict([
    ("predator_distance", "Normalized predator distance"),
    ("pursuit_alignment", "Pursuit alignment"),
    ("escape_alignment", "Escape alignment"),
    ("continuous_predator_response", "Prey reorientation R"),
    ("dos", "Degree of Swarm (DoS)"),
    ("doa", "Degree of Alignment (DoA)"),
])

METRIC_SCALES = {
    "predator_distance": 0.10,
    "pursuit_alignment": 1.0,
    "escape_alignment": 1.0,
    "continuous_predator_response": 0.10,
    "dos": 0.10,
    "doa": 1.0,
}


@dataclass(frozen=True)
class VariantSpec:
    """One fully declared Couzin evaluation condition."""

    name: str
    label: str
    explanation: str
    dt: float = 0.25
    alpha: float = 0.01
    theta_dot_max: float = 0.5
    rollout_steps: int = 100
    checkpoint: str = "paper_best"
    deterministic_policy: bool = False
    wall_mode: str = "legacy"
    turn_noise_std: float = 0.0
    speed_noise_fraction: float = 0.0
    position_noise_std: float = 0.0
    feature_schema: str = "active"

    @property
    def max_turn(self) -> float:
        return self.dt * self.theta_dot_max


CURRENT_PAPER = VariantSpec(
    name="current_paper",
    label="Current paper baseline",
    explanation=(
        "Reproduces the former final-analysis setup: evaluation at dt=0.5, "
        "Couzin alpha=0.1, 1,000 steps, stochastic policies, and corrected "
        "post-step wall reflection."
    ),
    dt=0.5,
    alpha=0.1,
    rollout_steps=1000,
    wall_mode="post_step_reflect",
)

TRAINING_MATCHED = VariantSpec(
    name="training_matched",
    label="Training-matched evaluation",
    explanation=(
        "Uses the environment recorded in the Couzin training notebook: "
        "dt=0.25, alpha=0.01, a 100-step policy horizon, and legacy wall order."
    ),
)


def default_screening_variants() -> list[VariantSpec]:
    """Small, interpretable ablation set used by the notebook."""

    return [
        TRAINING_MATCHED,
        VariantSpec(
            name="training_matched_deterministic",
            label="Training-matched, deterministic",
            explanation=(
                "Removes policy sampling at evaluation to test whether action "
                "sampling is responsible for excess imitation variability."
            ),
            deterministic_policy=True,
        ),
        VariantSpec(
            name="joint_last",
            label="Joint final-generation pair",
            explanation=(
                "Uses predator and prey states from the same final training "
                "generation instead of independently selected best policies."
            ),
            checkpoint="joint_last",
        ),
        VariantSpec(
            name="joint_ema",
            label="Joint final-generation EMA pair",
            explanation=(
                "Uses the coexisting EMA predator/prey pair from the final "
                "generation, preserving co-adaptation between roles."
            ),
            checkpoint="joint_ema",
        ),
        VariantSpec(
            name="heading_schema_diagnostic",
            label="Heading-feature diagnostic",
            explanation=(
                "Feeds neighbor heading in the last feature slot. This is a "
                "diagnostic for the discovered expert/rollout feature mismatch, "
                "not a valid replacement for retraining."
            ),
            feature_schema="heading",
        ),
    ]


def noise_variants(base: VariantSpec = TRAINING_MATCHED) -> list[VariantSpec]:
    """Predeclared low-to-moderate process-noise sensitivity conditions."""

    variants = [base]
    for degrees in (1.0, 2.5, 5.0):
        variants.append(VariantSpec(
            **{
                **asdict(base),
                "name": f"turn_noise_{str(degrees).replace('.', 'p')}deg",
                "label": f"Turn noise {degrees:g}°/step",
                "explanation": (
                    f"Adds zero-mean Gaussian heading noise with SD {degrees:g}° "
                    "per transition to both Couzin and policy dynamics."
                ),
                "turn_noise_std": math.radians(degrees),
            }
        ))
    return variants


def _load_policy_class() -> Any:
    config = {"root": NICOLE_ROOT, "code_subdir": "."}
    _, policy_class, _ = pa._load_policy_source_modules(config, NICOLE_ROOT)
    return policy_class


def _extract_ema_state(state: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "ema_model."
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def load_policy_pair(checkpoint: str = "paper_best") -> tuple[Any, Any]:
    """Load a declared predator/prey checkpoint pairing on CPU."""

    policy_class = _load_policy_class()
    prey_policy = policy_class(features=6).cpu()
    pred_policy = policy_class(features=5).cpu()
    if checkpoint == "paper_best":
        prey_state = torch.load(
            CHECKPOINT_DIR / "prey_policy_stage1.pth", map_location="cpu", weights_only=True)
        pred_state = torch.load(
            CHECKPOINT_DIR / "pred_policy_stage1.pth", map_location="cpu", weights_only=True)
    else:
        archive = torch.load(
            CHECKPOINT_DIR / "ckpt_latest.pt", map_location="cpu", weights_only=False)
        if checkpoint == "joint_last":
            prey_state, pred_state = archive["prey_policy"], archive["pred_policy"]
        elif checkpoint == "joint_ema":
            prey_state = _extract_ema_state(archive["ema_prey"])
            pred_state = _extract_ema_state(archive["ema_pred"])
        else:
            raise ValueError(f"Unknown checkpoint pairing: {checkpoint}")
    prey_policy.load_state_dict(prey_state, strict=True)
    pred_policy.load_state_dict(pred_state, strict=True)
    prey_policy.eval()
    pred_policy.eval()
    return prey_policy, pred_policy


def _load_couzin_module() -> Any:
    """Load Nicole's Couzin simulator without retaining generic package names."""

    saved = {
        name: module for name, module in tuple(sys.modules.items())
        if name == "utils" or name.startswith("utils.")
    }
    for name in saved:
        sys.modules.pop(name, None)
    root_text = str(NICOLE_ROOT)
    sys.path.insert(0, root_text)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    stub = types.ModuleType("utils.eval_utils")
    stub.compute_polarization = lambda *args: float("nan")
    stub.compute_angular_momentum = lambda *args: float("nan")
    stub.degree_of_sparsity = lambda *args: float("nan")
    stub.distance_to_predator = lambda *args: float("nan")
    stub.escape_alignment = lambda *args: float("nan")
    stub.pred_distance_to_nearest_prey = lambda *args: float("nan")
    sys.modules["utils.eval_utils"] = stub
    try:
        module = importlib.import_module("utils.couzin_utils")
    finally:
        sys.dont_write_bytecode = previous_bytecode
        sys.path.remove(root_text)
        for name in tuple(sys.modules):
            if name == "utils" or name.startswith("utils."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
    return module


def _reflect_agents(position: np.ndarray, heading: np.ndarray, width: float, height: float) -> None:
    """Clamp at walls and reflect headings, matching the paper correction."""

    left, right = position[:, 0] < 0, position[:, 0] > width
    bottom, top = position[:, 1] < 0, position[:, 1] > height
    x_bounce, y_bounce = left | right, bottom | top
    position[:, 0] = np.clip(position[:, 0], 0.0, width)
    position[:, 1] = np.clip(position[:, 1], 0.0, height)
    heading[x_bounce] = np.pi - heading[x_bounce]
    heading[y_bounce] = -heading[y_bounce]
    heading[:] = (heading + np.pi) % (2.0 * np.pi) - np.pi


def generate_couzin_expert_rollouts(
    spec: VariantSpec,
    *,
    n_prey: int,
    n_rollouts: int,
    rollout_seeds: Sequence[int],
    area_width: float = 50.0,
    area_height: float = 50.0,
    constant_speed: float = 5.0,
    shark_speed: float = 5.0,
) -> pa.TrajectoryTable:
    """Generate Couzin experts with optional transition-level process noise."""

    couzin = _load_couzin_module()
    records: list[dict[str, Any]] = []
    for replicate_seed in rollout_seeds:
        for rollout_id in range(n_rollouts):
            simulation_seed = int(replicate_seed) + rollout_id
            rng = np.random.default_rng(simulation_seed + 700_001)
            np.random.seed(simulation_seed)
            original_update = couzin.Agent.update_position
            original_enforce = couzin.enforce_walls
            base_speeds: dict[int, float] = {}

            def noisy_update(agent: Any, delta_t: float) -> None:
                speed = float(np.linalg.norm(agent.vel))
                base_speed = base_speeds.setdefault(id(agent), speed)
                angle = math.atan2(agent.vel[1], agent.vel[0])
                angle += float(rng.normal(0.0, spec.turn_noise_std))
                speed = base_speed * max(
                    0.05, 1.0 + float(rng.normal(0.0, spec.speed_noise_fraction)))
                agent.vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                original_update(agent, delta_t)
                if spec.position_noise_std:
                    agent.pos += rng.normal(0.0, spec.position_noise_std, size=2)
                if spec.wall_mode == "post_step_reflect":
                    original_enforce(agent, area_width, area_height)

            if spec.wall_mode == "post_step_reflect":
                couzin.enforce_walls = lambda *_args, **_kwargs: None
            couzin.Agent.update_position = noisy_update
            try:
                _, _, metrics_list, _, _ = couzin.run_couzin_simulation(
                    visualization="off", n=n_prey, max_steps=spec.rollout_steps,
                    number_of_sharks=1, area_width=area_width, area_height=area_height,
                    dt=spec.dt, alpha=spec.alpha, theta_dot_max=spec.theta_dot_max,
                    theta_dot_max_shark=spec.theta_dot_max,
                    constant_speed=constant_speed, shark_speed=shark_speed,
                )
            finally:
                couzin.Agent.update_position = original_update
                couzin.enforce_walls = original_enforce
            clip_id = f"seed_{replicate_seed}_rollout_{rollout_id:04d}"
            for step, metrics in enumerate(metrics_list):
                xs = np.asarray(metrics["xs"], dtype=float) * area_width
                ys = np.asarray(metrics["ys"], dtype=float) * area_height
                vxs = np.asarray(metrics["vxs"], dtype=float)
                vys = np.asarray(metrics["vys"], dtype=float)
                headings = np.arctan2(vys, vxs)
                for agent in range(n_prey + 1):
                    role = "predator" if agent == 0 else "prey"
                    records.append({
                        "source": "couzin_expert", "condition": "approach",
                        "clip_id": clip_id, "frame": step, "time_step": step,
                        "agent_id": f"{role}_{agent if agent == 0 else agent - 1}",
                        "role": role, "x": xs[agent], "y": ys[agent],
                        "heading": headings[agent], "vx": vxs[agent], "vy": vys[agent],
                        "step_duration": spec.dt, "sampling_stride": 1,
                        "rollout_seed": replicate_seed, "simulation_seed": simulation_seed,
                        "wall_mode": spec.wall_mode,
                    })
    return pa.to_canonical_trajectory(records)


def _initial_states(table: pa.TrajectoryTable, n_prey: int) -> tuple[np.ndarray, list[str]]:
    states, clip_ids = [], []
    for indices in pa.frame_groups(table):
        clip_id = str(table["clip_id"][indices[0]])
        if clip_id in clip_ids:
            continue
        roles = table["role"][indices].astype(str)
        pred, prey = indices[roles == "predator"], indices[roles == "prey"]
        if len(pred) != 1 or len(prey) != n_prey:
            continue
        prey = prey[np.argsort(table["agent_id"][prey].astype(str))]
        ordered = np.concatenate((pred, prey))
        states.append(np.column_stack((
            table["x"][ordered], table["y"][ordered], table["heading"][ordered])))
        clip_ids.append(clip_id)
    if not states:
        raise ValueError("No complete Couzin initial states")
    return np.stack(states), clip_ids


def _policy_states(
    position: torch.Tensor,
    heading: torch.Tensor,
    speed: torch.Tensor,
    *,
    n_prey: int,
    width: float,
    height: float,
    feature_schema: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact neighbor tensor used by a selected schema."""

    batch, n_agents, _ = position.shape
    velocity = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1) * speed[..., None]
    x, y = position[..., 0], position[..., 1]
    dx = x[:, None, :] / width - x[:, :, None] / width
    dy = y[:, None, :] / height - y[:, :, None] / height
    cos_t, sin_t = torch.cos(heading), torch.sin(heading)
    vx, vy = velocity[..., 0], velocity[..., 1]
    rel_vx = (cos_t[:, :, None] * vx[:, None, :] + sin_t[:, :, None] * vy[:, None, :]) / 5.0
    rel_vy = (-sin_t[:, :, None] * vx[:, None, :] + cos_t[:, :, None] * vy[:, None, :]) / 5.0
    base = torch.stack((dx, dy, rel_vx.clamp(-1, 1), rel_vy.clamp(-1, 1)), dim=-1)
    mask = ~torch.eye(n_agents, dtype=torch.bool)
    neighbors = base[:, mask].reshape(batch, n_agents, n_agents - 1, 4)
    pred, prey = neighbors[:, :1], neighbors[:, 1:]
    if feature_schema == "active":
        last = torch.ones((batch, n_agents, n_agents - 1, 1), dtype=position.dtype)
    elif feature_schema == "heading":
        heading_scaled = ((heading + math.pi) / (2.0 * math.pi)).clamp(0, 1)
        expanded = heading_scaled[:, None, :].expand(-1, n_agents, -1)
        last = expanded[:, mask].reshape(batch, n_agents, n_agents - 1, 1)
    else:
        raise ValueError(f"Unknown feature schema: {feature_schema}")
    pred = torch.cat((pred, last[:, :1]), dim=-1)
    predator_flag = torch.zeros((batch, n_prey, n_agents - 1, 1), dtype=position.dtype)
    predator_flag[:, :, 0] = 1.0
    prey = torch.cat((predator_flag, prey, last[:, 1:]), dim=-1)
    return pred, prey


def generate_policy_rollouts(
    spec: VariantSpec,
    *,
    n_prey: int,
    initial_states: np.ndarray,
    clip_ids: Sequence[str],
    seed: int,
    policy_pair: tuple[Any, Any] | None = None,
    area_width: float = 50.0,
    area_height: float = 50.0,
    prey_speed: float = 5.0,
    pred_speed: float = 5.0,
) -> pa.TrajectoryTable:
    """Run a policy pair from exactly matched expert initial states."""

    prey_policy, pred_policy = policy_pair or load_policy_pair(spec.checkpoint)
    state = torch.as_tensor(initial_states, dtype=torch.float32)
    position, heading = state[..., :2].clone(), state[..., 2].clone()
    batch, n_agents, _ = position.shape
    if n_agents != n_prey + 1 or len(clip_ids) != batch:
        raise ValueError("Initial-state dimensions do not match the requested rollouts")
    speed = torch.full((batch, n_agents), prey_speed, dtype=torch.float32)
    speed[:, 0] = pred_speed
    rng = np.random.default_rng(seed + 900_001)
    torch.manual_seed(seed)
    records: list[dict[str, Any]] = []
    for step in range(spec.rollout_steps):
        position_np, heading_np = position.numpy(), heading.numpy()
        for rollout in range(batch):
            for agent in range(n_agents):
                role = "predator" if agent == 0 else "prey"
                records.append({
                    "source": "policy", "condition": "approach",
                    "clip_id": str(clip_ids[rollout]), "frame": step, "time_step": step,
                    "agent_id": f"{role}_{agent if agent == 0 else agent - 1}",
                    "role": role, "x": float(position_np[rollout, agent, 0]),
                    "y": float(position_np[rollout, agent, 1]),
                    "heading": float(heading_np[rollout, agent]),
                    "step_duration": spec.dt, "sampling_stride": 1,
                    "rollout_seed": seed, "simulation_seed": seed + rollout,
                    "wall_mode": spec.wall_mode, "policy_id": spec.checkpoint,
                })
        pred_state, prey_state = _policy_states(
            position, heading, speed, n_prey=n_prey, width=area_width,
            height=area_height, feature_schema=spec.feature_schema)
        with torch.inference_mode():
            pred_action, _ = pred_policy(
                pred_state.reshape(batch, n_prey, 5),
                deterministic=spec.deterministic_policy)
            prey_action, _ = prey_policy(
                prey_state.reshape(batch * n_prey, n_prey, 6),
                deterministic=spec.deterministic_policy)
        action = torch.cat((
            pred_action.reshape(batch, 1), prey_action.reshape(batch, n_prey)), dim=1)
        heading += (action - 0.5) * (2.0 * spec.max_turn)
        if spec.turn_noise_std:
            heading += torch.as_tensor(
                rng.normal(0.0, spec.turn_noise_std, size=heading.shape), dtype=heading.dtype)
        heading[:] = (heading + math.pi) % (2.0 * math.pi) - math.pi
        step_speed = speed.clone()
        if spec.speed_noise_fraction:
            multiplier = rng.normal(1.0, spec.speed_noise_fraction, size=step_speed.shape)
            step_speed *= torch.as_tensor(np.clip(multiplier, 0.05, None), dtype=step_speed.dtype)
        velocity = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1) * step_speed[..., None]
        if spec.wall_mode == "legacy":
            for rollout in range(batch):
                _reflect_agents(position_np[rollout], heading_np[rollout], area_width, area_height)
            velocity = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1) * step_speed[..., None]
            position += velocity * spec.dt
            if spec.position_noise_std:
                position += torch.as_tensor(
                    rng.normal(0.0, spec.position_noise_std, size=position.shape), dtype=position.dtype)
        else:
            position += velocity * spec.dt
            if spec.position_noise_std:
                position += torch.as_tensor(
                    rng.normal(0.0, spec.position_noise_std, size=position.shape), dtype=position.dtype)
            position_np, heading_np = position.numpy(), heading.numpy()
            for rollout in range(batch):
                _reflect_agents(position_np[rollout], heading_np[rollout], area_width, area_height)
    return pa.to_canonical_trajectory(records)


def build_supervised_transition_data(
    table: pa.TrajectoryTable,
    *,
    n_prey: int,
    spec: VariantSpec = TRAINING_MATCHED,
    distance_threshold: float = 0.15,
    closing_lag: int = 3,
) -> dict[str, torch.Tensor]:
    """Build schema-consistent state/action pairs and objective attack labels."""

    bundle = pa.trajectory_table_to_metric_segments(
        table, expected_n_prey=n_prey, device="cpu", dtype=torch.float32)
    pred_states, pred_targets, pred_attack = [], [], []
    prey_states, prey_targets, prey_attack = [], [], []
    for clip in bundle["frame_clips"]:
        position = torch.cat((
            clip["predator_positions"][:, None], clip["prey_positions"]), dim=1)
        heading = torch.cat((
            clip["predator_headings"][:, None], clip["prey_headings"]), dim=1)
        if len(position) < 2:
            continue
        speed = torch.full(heading[:-1].shape, 5.0, dtype=torch.float32)
        pred_state, prey_state = _policy_states(
            position[:-1], heading[:-1], speed, n_prey=n_prey,
            width=50.0, height=50.0, feature_schema="active")
        delta = (heading[1:] - heading[:-1] + math.pi) % (2.0 * math.pi) - math.pi
        target = (delta / (2.0 * spec.max_turn) + 0.5).clamp(0.0, 1.0)
        valid = delta.abs() <= spec.max_turn * 1.05
        geometry = pa.prepare_geometry_cache(
            clip, d_source=math.hypot(50.0, 50.0), device="cpu", dtype=torch.float32)
        _, distance = pa.compute_predator_nearest_distance(geometry)
        attack = torch.zeros(len(distance) - 1, dtype=torch.bool)
        if len(attack) > closing_lag:
            index = torch.arange(closing_lag, len(attack))
            attack[index] = (
                (distance[index] <= distance_threshold)
                & (distance[index] < distance[index - closing_lag]))

        pred_valid = valid[:, 0]
        pred_states.append(pred_state[:, 0][pred_valid])
        pred_targets.append(target[:, 0][pred_valid, None])
        pred_attack.append(attack[pred_valid])

        prey_valid = valid[:, 1:]
        prey_states.append(prey_state[prey_valid])
        prey_targets.append(target[:, 1:][prey_valid, None])
        prey_attack.append(attack[:, None].expand(-1, n_prey)[prey_valid])
    output = {
        "pred_state": torch.cat(pred_states), "pred_target": torch.cat(pred_targets),
        "pred_attack": torch.cat(pred_attack), "prey_state": torch.cat(prey_states),
        "prey_target": torch.cat(prey_targets), "prey_attack": torch.cat(prey_attack),
    }
    return output


def supervised_repair(
    training_data: Mapping[str, torch.Tensor],
    *,
    checkpoint: str = "paper_best",
    optimization_steps: int = 500,
    batch_size: int = 1024,
    learning_rate: float = 3e-4,
    attack_weight: float = 4.0,
    seed: int = 2027,
) -> tuple[tuple[Any, Any], pd.DataFrame]:
    """Diagnostic BC repair with attack-weighted batches.

    This is intentionally labelled a diagnostic: it tests whether a corrected,
    attack-balanced supervised warm start can recover the Couzin controller. It
    is not a substitute for the subsequent adversarial training required for a
    GAIL claim.
    """

    if optimization_steps < 1 or batch_size < 1 or attack_weight < 1:
        raise ValueError("Invalid supervised-repair settings")
    prey_policy, pred_policy = load_policy_pair(checkpoint)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    for role, policy in (("pred", pred_policy), ("prey", prey_policy)):
        state = training_data[f"{role}_state"]
        target = training_data[f"{role}_target"]
        attack = training_data[f"{role}_attack"]
        weights = torch.where(
            attack, torch.full_like(attack, attack_weight, dtype=torch.float32),
            torch.ones_like(attack, dtype=torch.float32))
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        policy.train()
        for step in range(optimization_steps):
            index = torch.multinomial(
                weights, min(batch_size, len(weights)), replacement=True, generator=generator)
            prediction, _ = policy(state[index], deterministic=True)
            loss = torch.nn.functional.mse_loss(prediction, target[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            if step in {0, optimization_steps // 4, optimization_steps // 2,
                        3 * optimization_steps // 4, optimization_steps - 1}:
                history.append({
                    "role": role, "step": step + 1, "mse": float(loss.detach()),
                    "samples": len(state), "attack_fraction": float(attack.float().mean()),
                    "effective_attack_fraction": float(
                        (weights * attack.float()).sum() / weights.sum()),
                })
        policy.eval()
    return (prey_policy, pred_policy), pd.DataFrame(history)


def combine_supervised_data(
    datasets: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad neighbor axes and combine 16-/32-prey transition datasets."""

    if not datasets:
        raise ValueError("At least one transition dataset is required")
    output: dict[str, torch.Tensor] = {}
    for role in ("pred", "prey"):
        state_key = f"{role}_state"
        width = max(data[state_key].shape[1] for data in datasets)
        padded = []
        for data in datasets:
            state = data[state_key]
            if state.shape[1] < width:
                pad = state.new_zeros((len(state), width - state.shape[1], state.shape[2]))
                state = torch.cat((state, pad), dim=1)
            padded.append(state)
        output[state_key] = torch.cat(padded)
        output[f"{role}_target"] = torch.cat([data[f"{role}_target"] for data in datasets])
        output[f"{role}_attack"] = torch.cat([data[f"{role}_attack"] for data in datasets])
    return output


def analyze_trajectory(
    table: pa.TrajectoryTable,
    *,
    n_prey: int,
    step_duration: float,
) -> dict[str, Any]:
    edges = torch.linspace(0.0144023038, 0.2365715653, 7)
    return pa.analyze_case(
        table, d_source=math.hypot(50.0, 50.0), risk_bin_edges=edges,
        expected_n_prey=n_prey, device="cpu", dtype=torch.float32,
        approach_distance_threshold=0.15, approach_exit_threshold=0.165,
        approach_closing_lag=3, approach_min_duration=2,
        approach_cooldown_steps=20, closing_speed_lag=3,
        response_threshold=0.10, response_window_steps=20, response_lag=1,
        min_bin_count=30, min_cluster_count=2, n_bootstrap=0,
        step_duration=step_duration,
    )


def extract_approach_windows(
    table: pa.TrajectoryTable,
    *,
    n_prey: int,
    window_steps: int = 40,
    pre_steps: int = 5,
    distance_threshold: float = 0.15,
    closing_lag: int = 3,
    exit_threshold: float = 0.165,
    min_duration: int = 2,
    cooldown_steps: int = 20,
    min_event_step: int = 0,
    one_window_per_rollout: bool = True,
) -> tuple[pa.TrajectoryTable, pd.DataFrame]:
    """Extract fixed Couzin windows around objectively detected approaches.

    Event onsets come from the same robust detector used by ``analyze_trajectory``:
    threshold plus closing trend, target persistence, minimum duration,
    hysteresis, and cooldown.  Selecting at most one window per source rollout
    keeps boxplot observations independent even though training may later
    oversample more windows.
    """

    if window_steps < 2 or pre_steps < 0 or pre_steps >= window_steps:
        raise ValueError("Require window_steps >= 2 and 0 <= pre_steps < window_steps")
    if min_event_step < 0:
        raise ValueError("min_event_step must be non-negative")
    bundle = pa.trajectory_table_to_metric_segments(
        table, expected_n_prey=n_prey, device="cpu", dtype=torch.float32)
    selections: list[dict[str, Any]] = []
    for clip in bundle["frame_clips"]:
        geometry = pa.prepare_geometry_cache(
            clip, d_source=math.hypot(50.0, 50.0), device="cpu", dtype=torch.float32)
        _, distance = pa.compute_predator_nearest_distance(geometry)
        target = geometry["nearest_prey_index"]
        detected = pa.detect_approach_events(
            distance, threshold=distance_threshold, closing_lag=closing_lag,
            exit_threshold=exit_threshold, min_duration=min_duration,
            cooldown_steps=cooldown_steps, target_index=target,
            require_target_persistence=True)
        candidates = []
        for index_tensor in detected:
            index = int(index_tensor)
            start = index - pre_steps
            stop = start + window_steps
            if index < max(pre_steps, min_event_step) or stop > len(distance):
                continue
            candidates.append(index)
        if not candidates:
            continue
        chosen = candidates[:1] if one_window_per_rollout else candidates[::window_steps]
        for event_number, index in enumerate(chosen):
            start, stop = index - pre_steps, index - pre_steps + window_steps
            selections.append({
                "source_clip_id": str(clip["clip_id"]),
                "window_id": f"{clip['clip_id']}_approach_{event_number:02d}",
                "start_time": float(clip["time_steps"][start]),
                "stop_time": float(clip["time_steps"][stop - 1]),
                "approach_time": float(clip["time_steps"][index]),
                "approach_index": int(index),
                "approach_distance": float(distance[index]),
            })
    if not selections:
        raise ValueError("No approach windows satisfy the declared criteria")

    parts: dict[str, list[np.ndarray]] = {name: [] for name in table.columns}
    for selection in selections:
        mask = (
            (table["clip_id"].astype(str) == selection["source_clip_id"])
            & (table["time_step"].astype(float) >= selection["start_time"])
            & (table["time_step"].astype(float) <= selection["stop_time"])
        )
        rows = np.where(mask)[0]
        for name, values in table.columns.items():
            value = values[rows].copy()
            if name == "clip_id":
                value = np.full(len(rows), selection["window_id"], dtype=object)
            elif name in {"frame", "time_step"}:
                value = value.astype(float) - selection["start_time"]
            elif name == "timestamp":
                finite = np.isfinite(value.astype(float))
                value = value.astype(float)
                if finite.any():
                    value[finite] -= np.nanmin(value[finite])
            parts[name].append(value)
    window_table = pa.TrajectoryTable({
        name: np.concatenate(values) for name, values in parts.items()
    })
    return window_table, pd.DataFrame(selections)


def _select_clips(
    table: pa.TrajectoryTable, clip_ids: Sequence[str],
) -> pa.TrajectoryTable:
    """Select complete clips while preserving table columns and row order."""

    wanted = {str(value) for value in clip_ids}
    mask = np.asarray([str(value) in wanted for value in table["clip_id"]], dtype=bool)
    return pa.TrajectoryTable({
        name: values[mask].copy() for name, values in table.columns.items()
    })


def truncate_trajectory_windows(
    table: pa.TrajectoryTable, window_steps: int,
) -> pa.TrajectoryTable:
    """Keep the first ``window_steps`` recorded states of every matched clip."""

    if window_steps < 2:
        raise ValueError("window_steps must be at least two")
    mask = np.zeros(len(table), dtype=bool)
    clip_values = table["clip_id"].astype(str)
    for clip_id in dict.fromkeys(clip_values):
        clip_rows = np.where(clip_values == clip_id)[0]
        times = np.unique(table["time_step"][clip_rows].astype(float))
        if len(times) < window_steps:
            raise ValueError(f"Clip {clip_id!r} has fewer than {window_steps} states")
        keep_times = set(times[:window_steps].tolist())
        mask[clip_rows] = np.asarray([
            float(value) in keep_times for value in table["time_step"][clip_rows]
        ])
    return pa.TrajectoryTable({
        name: values[mask].copy() for name, values in table.columns.items()
    })


def collect_independent_approach_windows(
    spec: VariantSpec = TRAINING_MATCHED,
    *,
    n_prey: int,
    target_windows: int = 40,
    minimum_windows: int = 30,
    max_source_rollouts: int = 120,
    batch_size: int = 10,
    source_steps: int = 300,
    window_steps: int = 80,
    burn_in_steps: int = 50,
    base_seed: int = 2027,
) -> tuple[pa.TrajectoryTable, pd.DataFrame, pd.DataFrame]:
    """Adaptively collect one valid approach window per independent rollout.

    Source rollouts are generated from distinct simulation seeds.  A rollout is
    accepted only when the robust detector finds an event after burn-in with a
    complete maximum-horizon future.  At most its first valid event is retained.
    """

    if not (1 <= minimum_windows <= target_windows <= max_source_rollouts):
        raise ValueError(
            "Require 1 <= minimum_windows <= target_windows <= max_source_rollouts")
    if batch_size < 1 or source_steps < window_steps + burn_in_steps:
        raise ValueError("Invalid batch size or insufficient source horizon")

    source_spec = replace(spec, rollout_steps=source_steps)
    accepted_windows: list[pa.TrajectoryTable] = []
    event_audits: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    attempted = accepted = 0

    while accepted < target_windows and attempted < max_source_rollouts:
        count = min(batch_size, max_source_rollouts - attempted)
        seeds = tuple(
            int(base_seed + n_prey * 100_000 + attempted + offset)
            for offset in range(count)
        )
        batch = generate_couzin_expert_rollouts(
            source_spec, n_prey=n_prey, n_rollouts=1, rollout_seeds=seeds)
        batch_clip_ids = list(dict.fromkeys(batch["clip_id"].astype(str)))
        try:
            windows, audit = extract_approach_windows(
                batch, n_prey=n_prey, window_steps=window_steps, pre_steps=0,
                min_event_step=burn_in_steps, one_window_per_rollout=True)
        except ValueError:
            windows, audit = None, pd.DataFrame()

        accepted_ids = set() if audit.empty else set(audit["source_clip_id"].astype(str))
        remaining = target_windows - accepted
        chosen_ids = list(dict.fromkeys(
            audit["source_clip_id"].astype(str).tolist()))[:remaining] if not audit.empty else []
        chosen = set(chosen_ids)
        if chosen_ids:
            chosen_audit = audit[audit["source_clip_id"].astype(str).isin(chosen)].copy()
            accepted_windows.append(_select_clips(windows, chosen_audit["window_id"].astype(str)))
            event_audits.append(chosen_audit)
            accepted += len(chosen_ids)

        audit_by_source = (
            audit.set_index("source_clip_id").to_dict("index") if not audit.empty else {})
        for clip_id in batch_clip_ids:
            event = audit_by_source.get(clip_id, {})
            source_rows.append({
                "n_prey": int(n_prey), "source_clip_id": clip_id,
                "source_seed": int(clip_id.split("_rollout_")[0].replace("seed_", "")),
                "valid_event_found": clip_id in accepted_ids,
                "selected": clip_id in chosen,
                "approach_index": event.get("approach_index", np.nan),
                "approach_distance": event.get("approach_distance", np.nan),
            })
        attempted += count

    if not accepted_windows:
        raise ValueError(
            f"No valid N={n_prey} approach events found in {attempted} independent rollouts")
    windows = pa.concatenate_trajectory_tables(accepted_windows)
    event_audit = pd.concat(event_audits, ignore_index=True)
    source_audit = pd.DataFrame(source_rows)
    source_audit.attrs.update({
        "target_windows": target_windows,
        "minimum_windows": minimum_windows,
        "minimum_reached": accepted >= minimum_windows,
        "selected_windows": accepted,
        "attempted_rollouts": attempted,
    })
    return windows, event_audit, source_audit


def evaluate_large_approach_cohort(
    spec: VariantSpec = TRAINING_MATCHED,
    *,
    group_sizes: Sequence[int] = (16, 32),
    horizons: Sequence[int] = (20, 40, 80),
    target_windows: int = 40,
    minimum_windows: int = 30,
    max_source_rollouts: int = 120,
    batch_size: int = 10,
    source_steps: int = 300,
    burn_in_steps: int = 50,
    base_seed: int = 2027,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    """Evaluate nested horizons on one large independent event cohort.

    Every group uses one event per source rollout.  Couzin and GAIL start at the
    identical event state, and the same 80-step Couzin/GAIL realization is
    truncated to each requested horizon so horizon differences are paired.
    """

    horizons = tuple(sorted({int(value) for value in horizons}))
    if not horizons or horizons[0] < 2:
        raise ValueError("At least one horizon of two or more steps is required")
    maximum = max(horizons)
    collected: dict[int, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for n_prey in group_sizes:
        expert_max, event_audit, source_audit = collect_independent_approach_windows(
            spec, n_prey=n_prey, target_windows=target_windows,
            minimum_windows=minimum_windows, max_source_rollouts=max_source_rollouts,
            batch_size=batch_size, source_steps=source_steps,
            window_steps=maximum, burn_in_steps=burn_in_steps,
            base_seed=base_seed)
        starts, clip_ids = _initial_states(expert_max, n_prey)
        max_spec = replace(spec, rollout_steps=maximum)
        imitation_max = generate_policy_rollouts(
            max_spec, n_prey=n_prey, initial_states=starts, clip_ids=clip_ids,
            seed=base_seed + 900_000 + n_prey)
        collected[n_prey] = {
            "expert": expert_max, "imitation": imitation_max,
            "window_audit": event_audit, "source_audit": source_audit,
        }
        selected = int(source_audit["selected"].sum())
        attempted = len(source_audit)
        summary_rows.append({
            "n_prey": int(n_prey), "target_windows": int(target_windows),
            "minimum_windows": int(minimum_windows), "selected_windows": selected,
            "source_rollouts_attempted": attempted,
            "source_rollouts_with_valid_event": int(source_audit["valid_event_found"].sum()),
            "acceptance_rate": selected / attempted,
            "minimum_reached": bool(selected >= minimum_windows),
            "unique_source_rollouts": int(event_audit["source_clip_id"].nunique()),
            "mean_onset_distance": float(event_audit["approach_distance"].mean()),
        })

    results: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        horizon_spec = replace(
            spec, name=f"large_independent_approach_{horizon}",
            label=f"Large independent approach cohort, {horizon} steps",
            explanation=(
                "One robustly detected post-burn-in event per independent Couzin "
                f"rollout; exact matched event states and a paired {horizon}-step horizon."),
            rollout_steps=horizon)
        result: dict[str, Any] = {"spec": horizon_spec, "groups": {}}
        for n_prey in group_sizes:
            expert = truncate_trajectory_windows(collected[n_prey]["expert"], horizon)
            imitation = truncate_trajectory_windows(collected[n_prey]["imitation"], horizon)
            result["groups"][n_prey] = {
                "expert": expert, "imitation": imitation,
                "window_audit": collected[n_prey]["window_audit"].copy(),
                "source_audit": collected[n_prey]["source_audit"].copy(),
                "expert_analysis": analyze_trajectory(
                    expert, n_prey=n_prey, step_duration=spec.dt),
                "imitation_analysis": analyze_trajectory(
                    imitation, n_prey=n_prey, step_duration=spec.dt),
            }
        results[horizon] = result
    return results, pd.DataFrame(summary_rows)


def evaluate_approach_variant(
    spec: VariantSpec = TRAINING_MATCHED,
    *,
    group_sizes: Sequence[int] = (16, 32),
    rollout_seeds: Sequence[int] = (2027,),
    n_source_rollouts_per_seed: int = 10,
    source_steps: int = 300,
    window_steps: int = 40,
    pre_steps: int = 5,
) -> dict[str, Any]:
    """Evaluate fixed attack/approach windows from matched expert states."""

    source_spec = replace(spec, rollout_steps=source_steps)
    window_spec = replace(
        spec, name=f"{spec.name}_approach_{window_steps}",
        label=f"{spec.label}, {window_steps}-step approaches",
        explanation=(
            f"{spec.explanation} Evaluation is restricted to objectively detected "
            f"approaches and uses matched {window_steps}-step windows."
        ),
        rollout_steps=window_steps)
    result: dict[str, Any] = {"spec": window_spec, "groups": {}}
    for n_prey in group_sizes:
        long_expert = generate_couzin_expert_rollouts(
            source_spec, n_prey=n_prey, n_rollouts=n_source_rollouts_per_seed,
            rollout_seeds=rollout_seeds)
        expert, audit = extract_approach_windows(
            long_expert, n_prey=n_prey, window_steps=window_steps, pre_steps=pre_steps)
        starts, clip_ids = _initial_states(expert, n_prey)
        imitation = generate_policy_rollouts(
            window_spec, n_prey=n_prey, initial_states=starts, clip_ids=clip_ids,
            seed=int(rollout_seeds[0]) + 70_000 + n_prey)
        result["groups"][n_prey] = {
            "expert": expert, "imitation": imitation, "window_audit": audit,
            "expert_analysis": analyze_trajectory(expert, n_prey=n_prey, step_duration=spec.dt),
            "imitation_analysis": analyze_trajectory(imitation, n_prey=n_prey, step_duration=spec.dt),
        }
    return result


def evaluate_variant(
    spec: VariantSpec,
    *,
    group_sizes: Sequence[int] = (16, 32),
    rollout_seeds: Sequence[int] = (2027,),
    n_rollouts_per_seed: int = 4,
    policy_pair: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Generate, analyze, and summarize one Couzin-only condition."""

    result: dict[str, Any] = {"spec": spec, "groups": {}}
    for n_prey in group_sizes:
        expert = generate_couzin_expert_rollouts(
            spec, n_prey=n_prey, n_rollouts=n_rollouts_per_seed,
            rollout_seeds=rollout_seeds)
        starts, clip_ids = _initial_states(expert, n_prey)
        imitation = generate_policy_rollouts(
            spec, n_prey=n_prey, initial_states=starts, clip_ids=clip_ids,
            seed=int(rollout_seeds[0]) + 50_000 + n_prey,
            policy_pair=policy_pair)
        expert_analysis = analyze_trajectory(expert, n_prey=n_prey, step_duration=spec.dt)
        imitation_analysis = analyze_trajectory(imitation, n_prey=n_prey, step_duration=spec.dt)
        result["groups"][n_prey] = {
            "expert": expert, "imitation": imitation,
            "expert_analysis": expert_analysis, "imitation_analysis": imitation_analysis,
        }
    return result


def result_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return tidy per-metric comparison rows for one variant."""

    spec = result["spec"]
    rows = []
    for n_prey, group in result["groups"].items():
        for metric, label in PAPER_METRICS.items():
            expert = group["expert_analysis"]["per_clip"][metric]
            imitation = group["imitation_analysis"]["per_clip"][metric]
            expert_np = expert[torch.isfinite(expert)].cpu().numpy()
            imitation_np = imitation[torch.isfinite(imitation)].cpu().numpy()
            expert_mean = float(np.mean(expert_np)) if len(expert_np) else np.nan
            imitation_mean = float(np.mean(imitation_np)) if len(imitation_np) else np.nan
            pooled_sd = float(np.sqrt((np.var(expert_np) + np.var(imitation_np)) / 2.0))
            rows.append({
                "variant": spec.name, "variant_label": spec.label,
                "n_prey": int(n_prey), "metric": metric, "metric_label": label,
                "expert_n": len(expert_np), "expert_mean": expert_mean,
                "expert_sd": float(np.std(expert_np)) if len(expert_np) else np.nan,
                "expert_q25": float(np.quantile(expert_np, 0.25)) if len(expert_np) else np.nan,
                "expert_q75": float(np.quantile(expert_np, 0.75)) if len(expert_np) else np.nan,
                "imitation_n": len(imitation_np), "imitation_mean": imitation_mean,
                "imitation_sd": float(np.std(imitation_np)) if len(imitation_np) else np.nan,
                "imitation_q25": float(np.quantile(imitation_np, 0.25)) if len(imitation_np) else np.nan,
                "imitation_q75": float(np.quantile(imitation_np, 0.75)) if len(imitation_np) else np.nan,
                "mean_gap": imitation_mean - expert_mean,
                "absolute_gap": abs(imitation_mean - expert_mean),
                "scaled_absolute_gap": abs(imitation_mean - expert_mean) / METRIC_SCALES[metric],
                "pooled_sd": pooled_sd,
                "iqr_overlap": max(
                    0.0,
                    min(np.quantile(expert_np, 0.75), np.quantile(imitation_np, 0.75))
                    - max(np.quantile(expert_np, 0.25), np.quantile(imitation_np, 0.25)),
                ) if len(expert_np) and len(imitation_np) else np.nan,
            })
    return rows


def evaluate_variants(
    specs: Sequence[VariantSpec],
    *,
    group_sizes: Sequence[int] = (16, 32),
    rollout_seeds: Sequence[int] = (2027,),
    n_rollouts_per_seed: int = 4,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate variants and rank them without using biological results."""

    results = {}
    rows = []
    for spec in specs:
        result = evaluate_variant(
            spec, group_sizes=group_sizes, rollout_seeds=rollout_seeds,
            n_rollouts_per_seed=n_rollouts_per_seed)
        results[spec.name] = result
        rows.extend(result_rows(result))
    details = pd.DataFrame(rows)
    ranking = (
        details.groupby(["variant", "variant_label"], as_index=False)
        .agg(mean_scaled_gap=("scaled_absolute_gap", "mean"),
             median_scaled_gap=("scaled_absolute_gap", "median"),
             comparisons=("metric", "size"),
             iqr_overlaps=("iqr_overlap", lambda value: int((value > 0).sum())))
        .sort_values(["mean_scaled_gap", "median_scaled_gap"], ignore_index=True)
    )
    return results, details, ranking


def provenance_audit() -> pd.DataFrame:
    """Document the Couzin train/evaluation mismatches motivating the tests."""

    return pd.DataFrame([
        {"component": "Integration step", "training": "0.25", "former_evaluation": "0.5",
         "consequence": "Different displacement and maximum turn per step"},
        {"component": "Couzin alpha", "training": "0.01", "former_evaluation": "0.1",
         "consequence": "Different balance of threat avoidance and social motion"},
        {"component": "Policy horizon", "training": "100 steps", "former_evaluation": "1,000 steps",
         "consequence": "Tenfold long-horizon distribution shift"},
        {"component": "Last predator feature", "training_expert": "neighbor heading",
         "training": "policy rollout: active=1", "former_evaluation": "active=1",
         "consequence": "Expert and generated discriminator inputs use different semantics"},
        {"component": "Last prey feature", "training_expert": "neighbor heading",
         "training": "policy rollout: active=1", "former_evaluation": "active=1",
         "consequence": "Same semantic mismatch for the prey discriminator"},
        {"component": "Checkpoint pairing", "training": "role-wise best on N=32",
         "former_evaluation": "independently selected files",
         "consequence": "Predator and prey may come from different generations"},
    ])


def quality_gate(details: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent tolerances; boxplot overlap alone is not a pass rule."""

    thresholds = {
        "predator_distance": 0.015,
        "pursuit_alignment": 0.15,
        "escape_alignment": 0.15,
        "continuous_predator_response": 0.015,
        "dos": 0.015,
        "doa": 0.10,
    }
    checked = details.copy()
    checked["tolerance"] = checked["metric"].map(thresholds)
    checked["mean_gap_pass"] = checked["absolute_gap"] <= checked["tolerance"]
    checked["iqr_overlap_pass"] = checked["iqr_overlap"] > 0
    return checked


def plot_behavior_boxplots(result: Mapping[str, Any], category: str) -> Any:
    """Use the same visual grammar as the final paper boxplots."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fa.set_paper_style()
    metrics = {
        "Predator": ("predator_distance", "pursuit_alignment"),
        "Prey": ("escape_alignment", "continuous_predator_response"),
    }
    if category not in metrics:
        raise ValueError("category must be 'Predator' or 'Prey'")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    positions = {(16, "expert"): 0.82, (16, "imitation"): 1.18,
                 (32, "expert"): 1.82, (32, "imitation"): 2.18}
    colors = {"expert": "#2563A6", "imitation": "#E2762D"}
    labels = {"expert": "Data", "imitation": "GAIL imitation"}
    for metric_index, (ax, metric) in enumerate(zip(axes, metrics[category])):
        for group_index, (n_prey, kind) in enumerate(
            (pair for n in (16, 32) for pair in ((n, "expert"), (n, "imitation")))
        ):
            values = result["groups"][n_prey][f"{kind}_analysis"]["per_clip"][metric]
            values = values[torch.isfinite(values)].cpu().numpy()
            position = positions[(n_prey, kind)]
            box = ax.boxplot(
                [values], positions=[position], widths=0.28, patch_artist=True,
                showfliers=False, whis=1.5, manage_ticks=False,
                medianprops={"color": "white", "linewidth": 1.5},
                boxprops={"edgecolor": colors[kind], "linewidth": 1.1},
                whiskerprops={"color": colors[kind], "linewidth": 1.0},
                capprops={"color": colors[kind], "linewidth": 1.0})
            box["boxes"][0].set_facecolor(colors[kind])
            box["boxes"][0].set_alpha(0.78)
            rng = np.random.default_rng(2027 + 100 * metric_index + group_index)
            ax.scatter(
                np.full(len(values), position) + rng.uniform(-0.055, 0.055, len(values)),
                values, s=10, color=colors[kind], alpha=0.38, edgecolors="none", zorder=3)
        spec = fa.METRIC_SPECS[metric]
        ax.set_title(spec["short_label"])
        ax.set_xlabel("Group size")
        ax.set_ylabel(spec["unit"])
        ax.set_xticks((1, 2), labels=("16", "32"))
        ax.set_xlim(0.55, 2.45)
        if metric != "predator_distance":
            ax.axhline(0, color="0.45", linewidth=0.7, zorder=0)
    handles = [Patch(facecolor=colors[k], edgecolor=colors[k], alpha=0.78, label=labels[k])
               for k in ("expert", "imitation")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.18, 1, 1), w_pad=1.4)
    return fig


def plot_collective_timelines(result: Mapping[str, Any]) -> Any:
    """Plot Couzin DoS/DoA using the unchanged final-paper visual grammar."""

    import matplotlib.pyplot as plt

    fa.set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    colors = {"expert": "#2563A6", "imitation": "#E2762D"}
    labels = {"expert": "Data", "imitation": "GAIL imitation"}
    for ax, metric in zip(axes, ("dos", "doa")):
        for n_prey in (16, 32):
            for kind in ("expert", "imitation"):
                timelines = result["groups"][n_prey][f"{kind}_analysis"][f"{metric}_timeline"]
                width = min(len(value) for value in timelines)
                matrix = torch.stack([value[:width].float().cpu() for value in timelines])
                mean = matrix.mean(dim=0)
                if len(matrix) > 1:
                    generator = torch.Generator(device="cpu").manual_seed(2027 + n_prey)
                    index = torch.randint(len(matrix), (1000, len(matrix)), generator=generator)
                    draws = matrix[index].mean(dim=1)
                    low, high = torch.quantile(draws, torch.tensor([0.025, 0.975]), dim=0)
                else:
                    low = high = mean
                x = np.arange(width) * result["spec"].dt
                linestyle = "-" if n_prey == 16 else "--"
                ax.plot(x, mean, color=colors[kind], linestyle=linestyle,
                        linewidth=1.5, label=f"{labels[kind]}, N={n_prey}")
                ax.fill_between(x, low, high, color=colors[kind], alpha=0.10, linewidth=0)
        ax.set_title("Degree of Swarm (DoS)" if metric == "dos" else "Degree of Alignment (DoA)")
        ax.set_xlabel("Couzin simulation time")
        ax.set_ylabel(fa.METRIC_SPECS[metric]["unit"])
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.20, 1, 1), w_pad=1.4)
    return fig


def plot_expert_collective_timelines(result: Mapping[str, Any]) -> Any:
    """Plot expert-only DoS/DoA for the Couzin data behind an analysis."""

    import matplotlib.pyplot as plt

    fa.set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    colors = {16: "#2563A6", 32: "#6F4E9C"}
    is_event_window = "approach" in result["spec"].name
    for metric_index, (ax, metric) in enumerate(zip(axes, ("dos", "doa"))):
        for n_prey in (16, 32):
            timelines = result["groups"][n_prey]["expert_analysis"][f"{metric}_timeline"]
            width = min(len(value) for value in timelines)
            matrix = torch.stack([value[:width].float().cpu() for value in timelines])
            mean = matrix.mean(dim=0)
            if len(matrix) > 1:
                generator = torch.Generator(device="cpu").manual_seed(
                    31_337 + 100 * metric_index + n_prey)
                index = torch.randint(len(matrix), (1000, len(matrix)), generator=generator)
                draws = matrix[index].mean(dim=1)
                low, high = torch.quantile(
                    draws, torch.tensor([0.025, 0.975]), dim=0)
            else:
                low = high = mean
            x = np.arange(width) * result["spec"].dt
            ax.plot(x, mean, color=colors[n_prey], linewidth=1.6,
                    label=f"Couzin, N={n_prey}")
            ax.fill_between(x, low, high, color=colors[n_prey], alpha=0.14, linewidth=0)
        ax.set_title("Couzin Degree of Swarm (DoS)" if metric == "dos"
                     else "Couzin Degree of Alignment (DoA)")
        ax.set_xlabel("Time from approach onset" if is_event_window
                      else "Couzin simulation time")
        ax.set_ylabel(fa.METRIC_SPECS[metric]["unit"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.20, 1, 1), w_pad=1.4)
    return fig


def save_variant_artifacts(
    result: Mapping[str, Any], output_dir: str | Path,
) -> list[Path]:
    """Save only Couzin result tables and figures for one variant."""

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = result["spec"].name
    paths: list[Path] = []
    table_path = output_dir / f"{name}_metrics.csv"
    pd.DataFrame(result_rows(result)).to_csv(table_path, index=False)
    paths.append(table_path)
    for category in ("Predator", "Prey"):
        fig = plot_behavior_boxplots(result, category)
        pdf, png = fa.save_figure(fig, output_dir / f"{name}_{category.lower()}_boxplots")
        paths.extend((pdf, png))
        plt.close(fig)
    fig = plot_collective_timelines(result)
    pdf, png = fa.save_figure(fig, output_dir / f"{name}_collective_timeline")
    paths.extend((pdf, png))
    plt.close(fig)
    fig = plot_expert_collective_timelines(result)
    pdf, png = fa.save_figure(fig, output_dir / f"{name}_couzin_dos_doa")
    paths.extend((pdf, png))
    plt.close(fig)
    return paths


__all__ = [
    "VariantSpec", "CURRENT_PAPER", "TRAINING_MATCHED", "default_screening_variants",
    "noise_variants", "load_policy_pair", "generate_couzin_expert_rollouts",
    "generate_policy_rollouts", "analyze_trajectory", "extract_approach_windows",
    "truncate_trajectory_windows", "collect_independent_approach_windows",
    "build_supervised_transition_data", "supervised_repair",
    "combine_supervised_data",
    "evaluate_approach_variant", "evaluate_large_approach_cohort",
    "evaluate_variant", "evaluate_variants",
    "result_rows", "provenance_audit", "quality_gate", "plot_behavior_boxplots",
    "plot_collective_timelines", "plot_expert_collective_timelines",
    "save_variant_artifacts",
]
