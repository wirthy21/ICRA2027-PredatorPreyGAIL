"""Biological-policy PIN/AN heat maps for 16- and 32-prey models.

This module is intentionally isolated from the Couzin experiments and from the
biological trajectory metric pipeline.  It reads only the separately trained
biological GAIL checkpoints and visualizes their Pairwise-Interaction Network
(PIN) and Attention Network (AN) responses.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import torch


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
NICOLE_ROOT = PROJECT_ROOT / "Nicole" / "Predator-Prey-Thesis"

if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import final_analysis as fa
import paper_analysis2 as pa


GROUP_SIZES = (16, 32)
INTERACTIONS = OrderedDict([
    ("predator_prey", {
        "label": "Predator → Prey",
        "role": "predator",
        "policy": "predator",
        "focal_icon": "predator.png",
        "description": (
            "Response of the focal predator to a prey neighbor. The PIN map "
            "encodes the signed turning contribution; the AN map encodes the "
            "relative attention assigned to that neighbor position."
        ),
    }),
    ("prey_predator", {
        "label": "Prey → Predator",
        "role": "prey_pred",
        "policy": "prey",
        "focal_icon": "prey.png",
        "description": (
            "Response of the focal prey to the predator. The predator flag is "
            "active, so this isolates the learned threat-specific interaction."
        ),
    }),
    ("prey_prey", {
        "label": "Prey → Prey",
        "role": "prey",
        "policy": "prey",
        "focal_icon": "prey.png",
        "description": (
            "Response of the focal prey to another prey. The predator flag is "
            "inactive, isolating the learned social interaction."
        ),
    }),
])


def biological_policy_config(n_prey: int) -> dict[str, Any]:
    """Return the explicit manifest for one group-specific biological policy."""

    if n_prey not in GROUP_SIZES:
        raise ValueError(f"n_prey must be one of {GROUP_SIZES}")
    run = f"stage1_seed42_{n_prey}only"
    checkpoint_root = Path("Data/2. Training/VideoPredPrey - GAIL") / run
    return {
        "root": NICOLE_ROOT,
        "code_subdir": ".",
        "prey_checkpoint": str(checkpoint_root / "prey_policy_stage1.pth"),
        "pred_checkpoint": str(checkpoint_root / "pred_policy_stage1.pth"),
        "prey_features": 6,
        "pred_features": 5,
        "group_size": int(n_prey),
        "training_run": run,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(group_sizes: Sequence[int] = GROUP_SIZES) -> pd.DataFrame:
    """Audit the exact biological checkpoints used for every map row."""

    rows: list[dict[str, Any]] = []
    for n_prey in group_sizes:
        config = biological_policy_config(int(n_prey))
        for policy_role, key in (
            ("predator", "pred_checkpoint"), ("prey", "prey_checkpoint")
        ):
            path = (NICOLE_ROOT / config[key]).resolve()
            available = path.is_file()
            rows.append({
                "n_prey": int(n_prey),
                "training_run": config["training_run"],
                "policy_role": policy_role,
                "checkpoint": str(path),
                "available": available,
                "bytes": path.stat().st_size if available else np.nan,
                "sha256": _sha256(path) if available else "",
            })
    return pd.DataFrame(rows)


def load_biological_policy_pairs(
    group_sizes: Sequence[int] = GROUP_SIZES,
) -> dict[int, tuple[Any, Any]]:
    """Load the separately trained N=16 and N=32 biological policy pairs."""

    pairs: dict[int, tuple[Any, Any]] = {}
    for n_prey in group_sizes:
        config = biological_policy_config(int(n_prey))
        pairs[int(n_prey)] = pa.load_policy_pair(config, NICOLE_ROOT)
    return pairs


@torch.inference_mode()
def _compute_policy_heat_map(
    policy: Any,
    *,
    role: str,
    grid_size: int,
    n_orientations: int,
    batch_size: int,
    device: Any,
) -> dict[str, torch.Tensor]:
    """Evaluate one policy with both legacy and policy-consistent PIN links."""

    if role not in {"predator", "prey", "prey_pred"}:
        raise ValueError("role must be 'predator', 'prey', or 'prey_pred'")
    if grid_size < 2 or n_orientations < 1 or batch_size < 1:
        raise ValueError(
            "grid_size >= 2, n_orientations >= 1 and batch_size >= 1 are required"
        )

    original_parameter = next(policy.parameters())
    original_device = original_parameter.device
    target_device = torch.device(device) if device is not None else original_device
    was_training = policy.training
    policy.to(target_device).eval()
    try:
        xs = torch.linspace(-1.0, 1.0, grid_size, device=target_device)
        ys = torch.linspace(-1.0, 1.0, grid_size, device=target_device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        positions = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        theta = torch.linspace(
            -torch.pi, torch.pi, n_orientations + 1, device=target_device
        )[:-1]
        relative_velocity = torch.stack((theta.cos(), theta.sin()), dim=-1)
        base = torch.cat((
            positions[:, None, :].expand(-1, n_orientations, -1),
            relative_velocity[None, :, :].expand(grid_size * grid_size, -1, -1),
        ), dim=-1).reshape(-1, 4)

        in_features = int(policy.pairwise.fc1.in_features)
        if role == "predator":
            inputs = (
                base if in_features == 4
                else torch.cat((base, torch.ones_like(base[:, :1])), dim=1)
            )
        else:
            predator_flag = torch.full_like(
                base[:, :1], 1.0 if role == "prey_pred" else 0.0
            )
            inputs = torch.cat((predator_flag, base), dim=1)
            if in_features == 6:
                inputs = torch.cat(
                    (inputs, torch.ones_like(predator_flag)), dim=1
                )
        if inputs.shape[1] != in_features:
            raise ValueError(
                f"Cannot build {role} map for PIN input width {in_features}; "
                f"constructed {inputs.shape[1]} features"
            )

        policy_action_chunks: list[torch.Tensor] = []
        expected_action_chunks: list[torch.Tensor] = []
        legacy_action_chunks: list[torch.Tensor] = []
        attention_chunks: list[torch.Tensor] = []
        quadrature_nodes_np, quadrature_weights_np = np.polynomial.hermite.hermgauss(12)
        quadrature_nodes = torch.as_tensor(
            quadrature_nodes_np * np.sqrt(2.0),
            dtype=inputs.dtype,
            device=target_device,
        )
        quadrature_weights = torch.as_tensor(
            quadrature_weights_np / np.sqrt(np.pi),
            dtype=inputs.dtype,
            device=target_device,
        )
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            mu, sigma = policy.pairwise(batch)
            mu = mu.squeeze(-1)
            sigma = sigma.squeeze(-1)
            # The deployed policy uses sigmoid(mu), followed by the simulator's
            # [0, 1] -> [-max_turn, +max_turn] conversion. In normalized units
            # this is 2*sigmoid(mu)-1 == tanh(mu/2).
            policy_action_chunks.append(2.0 * torch.sigmoid(mu) - 1.0)
            expected_action_chunks.append(
                (
                    (2.0 * torch.sigmoid(
                        mu[:, None] + sigma[:, None] * quadrature_nodes[None, :]
                    ) - 1.0)
                    * quadrature_weights[None, :]
                ).sum(dim=1)
            )
            legacy_action_chunks.append(torch.tanh(mu))
            attention_chunks.append(policy.attention(batch).squeeze(-1))

        output_shape = (grid_size * grid_size, n_orientations)
        policy_action = torch.cat(policy_action_chunks).reshape(output_shape).mean(1)
        expected_action = torch.cat(expected_action_chunks).reshape(output_shape).mean(1)
        legacy_action = torch.cat(legacy_action_chunks).reshape(output_shape).mean(1)
        attention = torch.cat(attention_chunks).reshape(output_shape).mean(1)
        attention_min, attention_max = attention.min(), attention.max()
        attention = (
            (attention - attention_min) / (attention_max - attention_min)
            if bool(attention_max > attention_min)
            else torch.zeros_like(attention)
        )
        return {
            "x": xs.detach().cpu(),
            "y": ys.detach().cpu(),
            "action": policy_action.reshape(grid_size, grid_size).detach().cpu(),
            "action_stochastic_mean": expected_action.reshape(
                grid_size, grid_size
            ).detach().cpu(),
            "action_legacy": legacy_action.reshape(grid_size, grid_size).detach().cpu(),
            "attention": attention.reshape(grid_size, grid_size).detach().cpu(),
        }
    finally:
        policy.to(original_device).train(was_training)


def _smooth_map(values: torch.Tensor, sigma_cells: float) -> torch.Tensor:
    if sigma_cells <= 0:
        return values.clone()
    smoothed = gaussian_filter(
        values.detach().cpu().numpy(), sigma=float(sigma_cells), mode="nearest"
    )
    return torch.as_tensor(smoothed, dtype=values.dtype)


def _roughness(values: np.ndarray) -> float:
    """Mean absolute first difference across both spatial dimensions."""

    return float(
        (np.mean(np.abs(np.diff(values, axis=0)))
         + np.mean(np.abs(np.diff(values, axis=1)))) / 2.0
    )


def compute_biological_heat_maps(
    *,
    group_sizes: Sequence[int] = GROUP_SIZES,
    interactions: Sequence[str] = tuple(INTERACTIONS),
    grid_size: int = 100,
    n_orientations: int = 100,
    batch_size: int = 65536,
    device: Any = "cpu",
    pin_smoothing_sigma: float = 2.5,
) -> dict[str, Any]:
    """Compute PIN/AN maps for both biological group-specific policies.

    Each map averages over uniformly sampled relative-velocity orientations.
    PIN integrates the stochastic action used by the deployed policy and uses
    a small, optional spatial Gaussian kernel to suppress grid-scale ReLU artifacts.
    Attention remains unsmoothed.
    """

    unknown = set(interactions) - set(INTERACTIONS)
    if unknown:
        raise ValueError(f"Unknown interactions: {sorted(unknown)}")
    if pin_smoothing_sigma < 0:
        raise ValueError("pin_smoothing_sigma must be non-negative")
    group_sizes = tuple(int(value) for value in group_sizes)
    policy_pairs = load_biological_policy_pairs(group_sizes)
    maps: dict[str, Any] = {
        "settings": {
            "group_sizes": group_sizes,
            "grid_size": int(grid_size),
            "n_orientations": int(n_orientations),
            "batch_size": int(batch_size),
            "device": str(device),
            "pin_action_statistic": "E[2*sigmoid(mu+sigma*epsilon)-1]",
            "pin_quadrature_nodes": 12,
            "pin_smoothing_sigma_cells": float(pin_smoothing_sigma),
            "pin_smoothing_sigma_position": (
                float(pin_smoothing_sigma) * 2.0 / (int(grid_size) - 1)
            ),
            "attention_smoothing_sigma_cells": 0.0,
        },
        "groups": {},
    }
    for n_prey in group_sizes:
        prey_policy, predator_policy = policy_pairs[n_prey]
        maps["groups"][n_prey] = {}
        for interaction in interactions:
            spec = INTERACTIONS[interaction]
            policy = predator_policy if spec["policy"] == "predator" else prey_policy
            result = _compute_policy_heat_map(
                policy,
                role=spec["role"],
                grid_size=grid_size,
                n_orientations=n_orientations,
                batch_size=batch_size,
                device=device,
            )
            result["action_policy_median"] = result["action"].clone()
            result["action_unsmoothed"] = result["action_stochastic_mean"].clone()
            result["attention_unsmoothed"] = result["attention"].clone()
            result["action"] = _smooth_map(
                result["action_unsmoothed"], pin_smoothing_sigma
            )
            maps["groups"][n_prey][interaction] = result
    return maps


def smoothing_variant_diagnostics(
    maps: Mapping[str, Any],
    sigmas: Sequence[float] = (0.0, 0.75, 1.5, 2.5),
) -> pd.DataFrame:
    """Compare smoothing strengths against policy-consistent unsmoothed maps."""

    rows: list[dict[str, Any]] = []
    grid_size = int(maps["settings"]["grid_size"])
    for sigma in sigmas:
        for n_prey, group in maps["groups"].items():
            for interaction, result in group.items():
                raw = result["action_unsmoothed"].numpy().astype(float)
                candidate = _smooth_map(
                    result["action_unsmoothed"], float(sigma)
                ).numpy().astype(float)
                value_range = float(np.ptp(raw)) or 1.0
                raw_roughness = _roughness(raw) or 1.0
                rows.append({
                    "sigma_cells": float(sigma),
                    "sigma_position": float(sigma) * 2.0 / (grid_size - 1),
                    "n_prey": int(n_prey),
                    "interaction": interaction,
                    "roughness_ratio": _roughness(candidate) / raw_roughness,
                    "normalized_RMSE": float(
                        np.sqrt(np.mean((candidate - raw) ** 2)) / value_range
                    ),
                    "correlation": float(
                        np.corrcoef(raw.ravel(), candidate.ravel())[0, 1]
                    ),
                })
    frame = pd.DataFrame(rows)
    return frame.groupby(["sigma_cells", "sigma_position"], as_index=False).agg(
        mean_roughness_ratio=("roughness_ratio", "mean"),
        max_normalized_RMSE=("normalized_RMSE", "max"),
        min_correlation=("correlation", "min"),
    )


def action_link_diagnostics(maps: Mapping[str, Any]) -> pd.DataFrame:
    """Audit legacy, policy-median, and stochastic-mean PIN transformations."""

    rows: list[dict[str, Any]] = []
    for n_prey, group in maps["groups"].items():
        for interaction, result in group.items():
            legacy = result["action_legacy"].numpy().astype(float)
            policy = result["action_policy_median"].numpy().astype(float)
            stochastic = result["action_stochastic_mean"].numpy().astype(float)
            rows.append({
                "n_prey": int(n_prey),
                "interaction": interaction,
                "legacy_vs_policy_median_correlation": float(
                    np.corrcoef(legacy.ravel(), policy.ravel())[0, 1]
                ),
                "legacy_normalized_roughness": (
                    _roughness(legacy) / (float(np.ptp(legacy)) or 1.0)
                ),
                "policy_normalized_roughness": (
                    _roughness(policy) / (float(np.ptp(policy)) or 1.0)
                ),
                "stochastic_mean_correlation": float(
                    np.corrcoef(policy.ravel(), stochastic.ravel())[0, 1]
                ),
                "stochastic_mean_normalized_roughness": (
                    _roughness(stochastic) / (float(np.ptp(stochastic)) or 1.0)
                ),
            })
    return pd.DataFrame(rows)


def heat_map_summary(maps: Mapping[str, Any]) -> pd.DataFrame:
    """Return compact numerical diagnostics without changing map normalization."""

    rows: list[dict[str, Any]] = []
    for n_prey, group in maps["groups"].items():
        for interaction, result in group.items():
            for module, key, scale in (
                ("PIN", "action", 180.0), ("AN", "attention", 1.0)
            ):
                values = result[key].float().flatten() * scale
                rows.append({
                    "n_prey": int(n_prey),
                    "interaction": interaction,
                    "module": module,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "mean": float(values.mean()),
                    "sd": float(values.std()),
                })
    return pd.DataFrame(rows)


def compare_group_maps(maps: Mapping[str, Any]) -> pd.DataFrame:
    """Quantify N=16 versus N=32 map similarity for each interaction/module."""

    if not all(n_prey in maps["groups"] for n_prey in GROUP_SIZES):
        raise ValueError("Both N=16 and N=32 maps are required")
    rows: list[dict[str, Any]] = []
    for interaction in INTERACTIONS:
        for module, key, scale in (
            ("PIN", "action", 180.0), ("AN", "attention", 1.0)
        ):
            values_16 = maps["groups"][16][interaction][key].float().flatten() * scale
            values_32 = maps["groups"][32][interaction][key].float().flatten() * scale
            difference = values_32 - values_16
            correlation = (
                float(torch.corrcoef(torch.stack((values_16, values_32)))[0, 1])
                if float(values_16.std()) > 0 and float(values_32.std()) > 0
                else np.nan
            )
            rows.append({
                "interaction": interaction,
                "module": module,
                "mean_N16": float(values_16.mean()),
                "mean_N32": float(values_32.mean()),
                "MAE": float(difference.abs().mean()),
                "RMSE": float(difference.square().mean().sqrt()),
                "correlation": correlation,
            })
    return pd.DataFrame(rows)


def plot_interaction_grid(
    maps: Mapping[str, Any],
    interaction: str,
    *,
    group_sizes: Sequence[int] = GROUP_SIZES,
) -> Any:
    """Plot one interaction as rows N=16/N=32 and columns PIN/AN."""

    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.ticker import FuncFormatter

    if interaction not in INTERACTIONS:
        raise ValueError(f"Unknown interaction: {interaction}")
    group_sizes = tuple(int(value) for value in group_sizes)
    if group_sizes != GROUP_SIZES:
        raise ValueError("The paper grid requires group_sizes=(16, 32)")
    for n_prey in group_sizes:
        if n_prey not in maps["groups"] or interaction not in maps["groups"][n_prey]:
            raise ValueError(f"Missing {interaction} map for N={n_prey}")

    fa.set_paper_style()
    pin_maps = [maps["groups"][n][interaction]["action"].numpy() * 180.0
                for n in group_sizes]
    attention_maps = [maps["groups"][n][interaction]["attention"].numpy()
                      for n in group_sizes]
    pin_limit = max(float(np.nanmax(np.abs(value))) for value in pin_maps) or 1.0
    pin_norm = colors.TwoSlopeNorm(vmin=-pin_limit, vcenter=0.0, vmax=pin_limit)
    attention_norm = colors.Normalize(vmin=0.0, vmax=1.0)
    pin_levels = np.linspace(-pin_limit, pin_limit, 31)
    attention_levels = np.linspace(0.0, 1.0, 31)
    colorbar_formatter = FuncFormatter(
        lambda value, _: f"{value:.2f}".rstrip("0").rstrip(".")
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.6), sharex=True, sharey=True)
    icon_path = ANALYSIS_DIR / "images" / INTERACTIONS[interaction]["focal_icon"]
    icon = plt.imread(icon_path)
    for row, n_prey in enumerate(group_sizes):
        result = maps["groups"][n_prey][interaction]
        x, y = np.meshgrid(result["x"].numpy(), result["y"].numpy())
        pin_image = axes[row, 0].contourf(
            x, y, pin_maps[row], levels=pin_levels, cmap="inferno",
            norm=pin_norm)
        attention_image = axes[row, 1].contourf(
            x, y, attention_maps[row], levels=attention_levels, cmap="RdBu",
            norm=attention_norm)
        for ax in axes[row]:
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.set_aspect("equal")
            ax.set_xticks(np.linspace(-1.0, 1.0, 5))
            ax.set_yticks(np.linspace(-1.0, 1.0, 5))
            ax.add_artist(AnnotationBbox(
                OffsetImage(icon, zoom=0.42), (0.0, 0.0),
                frameon=False, xycoords="data", zorder=5))
            ax.set_xlabel("Relative x-position")
            ax.set_ylabel("Relative y-position")
        axes[row, 0].set_anchor("E")
        axes[row, 1].set_anchor("W")
        axes[row, 0].set_title(
            f"N={n_prey} | PIN", pad=5)
        axes[row, 1].set_title(
            f"N={n_prey} | AN", pad=5)
        fig.colorbar(
            pin_image,
            ax=axes[row, 0],
            label="action [degrees]",
            format=colorbar_formatter,
        )
        fig.colorbar(
            attention_image,
            ax=axes[row, 1],
            label="attention",
            format=colorbar_formatter,
        )

    fig.suptitle(
        f"Biological GAIL: {INTERACTIONS[interaction]['label']}",
        fontsize=16,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.91,
        wspace=0.18,
        hspace=0.21,
    )
    return fig


def save_heat_map_artifacts(
    maps: Mapping[str, Any], output_dir: str | Path,
) -> list[Path]:
    """Save the three biological heat-map grids and numerical audits."""

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for interaction in INTERACTIONS:
        fig = plot_interaction_grid(maps, interaction)
        pdf, png = fa.save_figure(fig, output_dir / f"biological_{interaction}_pin_an")
        paths.extend((pdf, png))
        plt.close(fig)
    summary_path = output_dir / "biological_heat_map_summary.csv"
    comparison_path = output_dir / "biological_heat_map_N16_vs_N32.csv"
    heat_map_summary(maps).to_csv(summary_path, index=False)
    compare_group_maps(maps).to_csv(comparison_path, index=False)
    paths.extend((summary_path, comparison_path))
    return paths


__all__ = [
    "GROUP_SIZES", "INTERACTIONS", "biological_policy_config",
    "checkpoint_manifest", "load_biological_policy_pairs",
    "compute_biological_heat_maps", "smoothing_variant_diagnostics",
    "action_link_diagnostics", "heat_map_summary", "compare_group_maps",
    "plot_interaction_grid", "save_heat_map_artifacts",
]
