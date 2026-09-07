"""Paper-ready summaries for the final predator-prey analysis.

This module deliberately exposes only the six pre-selected paper metrics:

Predator
    * normalized predator-to-nearest-prey distance
    * pursuit alignment
Prey
    * escape alignment
    * continuous prey reorientation response R
Collective
    * Degree of Swarm/Sparsity (DoS) over a common supported horizon
    * Degree of Alignment (DoA) over the same horizon

The heavy trajectory loading and metric implementation remain in
``paper_analysis2.py``.  Here we freeze the paper estimands, use one observation
per independent clip, construct a fixed complete-prefix cohort for T_common,
bootstrap at clip level, and create the final comparison tables and figures.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import numpy as np
import torch


METRIC_SPECS: "OrderedDict[str, dict[str, str]]" = OrderedDict([
    ("predator_distance", {
        "category": "Predator",
        "label": "Normalized predator distance",
        "short_label": "Predator distance",
        "unit": "arena diagonals",
    }),
    ("pursuit_alignment", {
        "category": "Predator",
        "label": "Pursuit alignment",
        "short_label": "Pursuit alignment",
        "unit": "alignment [-1, 1]",
    }),
    ("escape_alignment", {
        "category": "Prey",
        "label": "Escape alignment",
        "short_label": "Escape alignment",
        "unit": "alignment [-1, 1]",
    }),
    ("continuous_predator_response", {
        "category": "Prey",
        "label": "Continuous prey reorientation R",
        "short_label": "Prey reorientation R",
        "unit": "alignment change",
    }),
    ("dos", {
        "category": "Collective",
        "label": "Degree of Swarm (DoS), T-common",
        "short_label": "DoS (T-common)",
        "unit": "normalized NND",
    }),
    ("doa", {
        "category": "Collective",
        "label": "Degree of Alignment (DoA), T-common",
        "short_label": "DoA (T-common)",
        "unit": "local alignment",
    }),
])

SOURCE_CASES = {
    "couzin": tuple(
        f"couzin_{kind}_{n}" for kind in ("expert", "imitation") for n in (16, 32)
    ),
    "biological": tuple(
        f"bio_{kind}_{n}" for kind in ("expert", "imitation") for n in (16, 32)
    ),
}

SOURCE_PREFIX = {"couzin": "couzin", "biological": "bio"}
KIND_LABEL = {"expert": "Data", "imitation": "GAIL imitation"}
KIND_COLOR = {"expert": "#2563A6", "imitation": "#E2762D"}
SIZE_MARKER = {16: "o", 32: "s"}
BEHAVIOR_METRICS = tuple(METRIC_SPECS)[:4]
CATEGORY_METRICS = {
    "Predator": BEHAVIOR_METRICS[:2],
    "Prey": BEHAVIOR_METRICS[2:],
}


def _finite_1d(values: Any) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cpu").flatten()
    return tensor[torch.isfinite(tensor)]


def _timeline_matrix(series: Sequence[Any]) -> torch.Tensor:
    """NaN-pad clip timelines without inventing observations."""

    if not series:
        return torch.empty((0, 0), dtype=torch.float64)
    tensors = [torch.as_tensor(value, dtype=torch.float64, device="cpu").flatten() for value in series]
    width = max((len(value) for value in tensors), default=0)
    matrix = torch.full((len(tensors), width), torch.nan, dtype=torch.float64)
    for row, value in enumerate(tensors):
        matrix[row, :len(value)] = value
    return matrix


def _bootstrap_means(
    values: Any, *, n_bootstrap: int, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = _finite_1d(values)
    if not len(values):
        return values.new_tensor(torch.nan), values.new_empty(0)
    observed = values.mean()
    if n_bootstrap <= 0:
        return observed, values.new_empty(0)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(len(values), (int(n_bootstrap), len(values)), generator=generator)
    return observed, values[indices].mean(dim=1)


def _bootstrap_curve_mean(
    cohort: torch.Tensor, *, n_bootstrap: int, seed: int, chunk_size: int = 256,
) -> torch.Tensor:
    """Clip-bootstrap a whole curve without a large B x clips x time tensor."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    chunks = []
    for start in range(0, int(n_bootstrap), int(chunk_size)):
        size = min(int(chunk_size), int(n_bootstrap) - start)
        indices = torch.randint(len(cohort), (size, len(cohort)), generator=generator)
        chunks.append(cohort[indices].mean(dim=1))
    return torch.cat(chunks, dim=0)


def summarize_values(
    values: Any, *, n_bootstrap: int = 2000, ci_level: float = 0.95, seed: int = 0,
) -> dict[str, float | int]:
    """Summarize independent clip values with a percentile clip bootstrap CI."""

    values = _finite_1d(values)
    mean, draws = _bootstrap_means(values, n_bootstrap=n_bootstrap, seed=seed)
    alpha = (1.0 - float(ci_level)) / 2.0
    if len(draws):
        quantiles = torch.tensor([alpha, 1.0 - alpha], dtype=draws.dtype)
        low, high = torch.quantile(draws, quantiles).tolist()
    else:
        low = high = float("nan")
    return {
        "n_clips": int(len(values)),
        "mean": float(mean),
        "std": float(values.std(unbiased=False)) if len(values) else float("nan"),
        "median": float(values.median()) if len(values) else float("nan"),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _last_supported_step(matrix: torch.Tensor, min_support_fraction: float) -> tuple[int, int]:
    if not len(matrix) or not matrix.shape[1]:
        return -1, 0
    required = max(1, math.ceil(float(min_support_fraction) * len(matrix)))
    supported = torch.isfinite(matrix).sum(dim=0) >= required
    first_failure = torch.nonzero(~supported).flatten()
    last = int(first_failure[0]) - 1 if len(first_failure) else matrix.shape[1] - 1
    return last, required


def build_final_results(
    case_results: Mapping[str, Mapping[str, Any]],
    *,
    step_duration: Mapping[str, float],
    min_support_fraction: float = 0.5,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 2027,
) -> dict[str, Any]:
    """Freeze the six paper estimands and their T-common collective cohorts.

    The four predator/prey metrics reuse the robust per-clip estimates from
    ``paper_analysis2.analyze_case``.  DoS and DoA are recomputed as clip means
    over ``0..T_common`` using only clips observed throughout that entire
    interval.  Consequently the cohort is fixed at every plotted time point.
    """

    missing = [case for cases in SOURCE_CASES.values() for case in cases if case not in case_results]
    if missing:
        raise ValueError(f"The final paper analysis requires all eight cases; missing: {missing}")
    if not 0 < min_support_fraction <= 1:
        raise ValueError("min_support_fraction must lie in (0, 1]")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least one for final paper intervals")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must lie in (0, 1)")
    missing_durations = set(SOURCE_CASES) - set(step_duration)
    if missing_durations:
        raise ValueError(f"Missing source step durations: {sorted(missing_durations)}")

    output: dict[str, Any] = {
        "metric_specs": METRIC_SPECS,
        "per_clip": {case: {} for cases in SOURCE_CASES.values() for case in cases},
        "summary": {case: {} for cases in SOURCE_CASES.values() for case in cases},
        "collective": {"dos": {}, "doa": {}},
        "support": [],
        "settings": {
            "min_support_fraction": float(min_support_fraction),
            "n_bootstrap": int(n_bootstrap),
            "ci_level": float(ci_level),
            "seed": int(seed),
            "step_duration": dict(step_duration),
        },
    }

    for case in output["per_clip"]:
        for metric in tuple(METRIC_SPECS)[:4]:
            values = _finite_1d(case_results[case]["per_clip"][metric])
            output["per_clip"][case][metric] = values

    timeline_key = {"dos": "dos_timeline", "doa": "doa_timeline"}
    timeline_matrices = {
        metric: {
            source: {
                case: _timeline_matrix(case_results[case][key]) for case in cases
            }
            for source, cases in SOURCE_CASES.items()
        }
        for metric, key in timeline_key.items()
    }
    last_supported: dict[str, dict[str, dict[str, int]]] = {
        metric: {source: {} for source in SOURCE_CASES} for metric in timeline_key
    }
    required_support: dict[str, dict[str, dict[str, int]]] = {
        metric: {source: {} for source in SOURCE_CASES} for metric in timeline_key
    }
    for metric in timeline_key:
        for source, cases in SOURCE_CASES.items():
            for case in cases:
                last, required = _last_supported_step(
                    timeline_matrices[metric][source][case], min_support_fraction,
                )
                last_supported[metric][source][case] = last
                required_support[metric][source][case] = required

    # One source-specific T_common is shared by DoS and DoA. This prevents the
    # two collective metrics from silently summarizing different time windows.
    common_horizon = {}
    for source, cases in SOURCE_CASES.items():
        candidates = [
            last_supported[metric][source][case]
            for metric in timeline_key for case in cases
        ]
        if any(value < 0 for value in candidates):
            raise ValueError(
                f"No shared DoS/DoA horizon for {source}: "
                f"{ {metric: last_supported[metric][source] for metric in timeline_key} }"
            )
        common_horizon[source] = min(candidates)
    output["settings"]["T_common"] = dict(common_horizon)

    # Freeze one identical cohort per case for both collective metrics.
    common_complete_mask: dict[str, dict[str, torch.Tensor]] = {
        source: {} for source in SOURCE_CASES
    }
    for source, cases in SOURCE_CASES.items():
        t_common = common_horizon[source]
        for case in cases:
            row_counts = {
                len(timeline_matrices[metric][source][case]) for metric in timeline_key
            }
            if len(row_counts) != 1:
                raise ValueError(f"{case}: DoS and DoA clip counts differ: {row_counts}")
            common_complete_mask[source][case] = torch.stack([
                torch.isfinite(
                    timeline_matrices[metric][source][case][:, :t_common + 1]
                ).all(dim=1)
                for metric in timeline_key
            ]).all(dim=0)

    for metric in timeline_key:
        for source, cases in SOURCE_CASES.items():
            matrices = timeline_matrices[metric][source]
            t_common = common_horizon[source]
            case_curves: dict[str, Any] = {}
            for case, matrix in matrices.items():
                prefix = matrix[:, :t_common + 1]
                complete = common_complete_mask[source][case]
                cohort = prefix[complete]
                if not len(cohort):
                    raise ValueError(f"{case}: no complete-prefix {metric} clips at T_common={t_common}")
                per_clip = cohort.mean(dim=1)
                output["per_clip"][case][metric] = per_clip

                curve_mean = cohort.mean(dim=0)
                if n_bootstrap > 0:
                    curves = _bootstrap_curve_mean(
                        cohort,
                        n_bootstrap=n_bootstrap,
                        seed=(
                            int(seed) + 1009 * (list(METRIC_SPECS).index(metric) + 1)
                            + 37 * list(cases).index(case)
                        ),
                    )
                    alpha = (1.0 - ci_level) / 2.0
                    curve_low = torch.quantile(curves, alpha, dim=0)
                    curve_high = torch.quantile(curves, 1.0 - alpha, dim=0)
                else:
                    curve_low = torch.full_like(curve_mean, torch.nan)
                    curve_high = torch.full_like(curve_mean, torch.nan)
                case_curves[case] = {
                    "mean": curve_mean,
                    "ci_low": curve_low,
                    "ci_high": curve_high,
                    "complete_clip_mask": complete,
                    "complete_clips": int(complete.sum()),
                    "original_clips": int(len(matrix)),
                    "per_clip": per_clip,
                }
                output["support"].append({
                    "metric": metric,
                    "source": source,
                    "case": case,
                    "T_common_step": int(t_common),
                    "T_common_time": float(t_common * step_duration[source]),
                    "case_last_supported_step": int(last_supported[metric][source][case]),
                    "required_clips": int(required_support[metric][source][case]),
                    "complete_prefix_clips": int(complete.sum()),
                    "original_clips": int(len(matrix)),
                    "complete_prefix_fraction": float(complete.float().mean()),
                })
            output["collective"][metric][source] = {
                "t_common": int(t_common),
                "time": torch.arange(t_common + 1, dtype=torch.float64) * step_duration[source],
                "cases": case_curves,
            }

    for case, metrics in output["per_clip"].items():
        for metric, values in metrics.items():
            metric_seed = int(seed) + 101 * (list(METRIC_SPECS).index(metric) + 1)
            output["summary"][case][metric] = summarize_values(
                values, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=metric_seed,
            )
    return output


def _difference_summary(
    before: Any,
    after: Any,
    *,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> dict[str, float]:
    before = _finite_1d(before)
    after = _finite_1d(after)
    if not len(before) or not len(after):
        return {"difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    observed = after.mean() - before.mean()
    _, before_draws = _bootstrap_means(before, n_bootstrap=n_bootstrap, seed=seed)
    _, after_draws = _bootstrap_means(after, n_bootstrap=n_bootstrap, seed=seed + 1)
    if len(before_draws) and len(after_draws):
        draws = after_draws - before_draws
        alpha = (1.0 - ci_level) / 2.0
        quantiles = torch.tensor([alpha, 1.0 - alpha], dtype=draws.dtype)
        low, high = torch.quantile(draws, quantiles).tolist()
    else:
        low = high = float("nan")
    return {"difference": float(observed), "ci_low": float(low), "ci_high": float(high)}


def stage_imitation_table(
    final_results: Mapping[str, Any],
    source: str,
) -> list[dict[str, Any]]:
    """Rows for Couzin/Bio data versus GAIL, separately at N=16 and N=32."""

    prefix = SOURCE_PREFIX[source]
    settings = final_results["settings"]
    rows: list[dict[str, Any]] = []
    for category in ("Predator", "Prey", "Collective"):
        for metric, spec in METRIC_SPECS.items():
            if spec["category"] != category:
                continue
            for n_prey in (16, 32):
                expert = f"{prefix}_expert_{n_prey}"
                imitation = f"{prefix}_imitation_{n_prey}"
                expert_summary = final_results["summary"][expert][metric]
                imitation_summary = final_results["summary"][imitation][metric]
                gap = _difference_summary(
                    final_results["per_clip"][expert][metric],
                    final_results["per_clip"][imitation][metric],
                    n_bootstrap=settings["n_bootstrap"], ci_level=settings["ci_level"],
                    seed=settings["seed"] + 10000 + len(rows) * 2,
                )
                rows.append({
                    "category": category,
                    "metric": spec["label"],
                    "n_prey": n_prey,
                    "data_n": expert_summary["n_clips"],
                    "data_mean": expert_summary["mean"],
                    "data_ci_low": expert_summary["ci_low"],
                    "data_ci_high": expert_summary["ci_high"],
                    "gail_n": imitation_summary["n_clips"],
                    "gail_mean": imitation_summary["mean"],
                    "gail_ci_low": imitation_summary["ci_low"],
                    "gail_ci_high": imitation_summary["ci_high"],
                    "gail_minus_data": gap["difference"],
                    "difference_ci_low": gap["ci_low"],
                    "difference_ci_high": gap["ci_high"],
                    "absolute_imitation_error": abs(gap["difference"]),
                })
    return rows


def paper_imitation_table(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Compact formatted DataFrame intended for direct paper transcription."""

    import pandas as pd

    def estimate(mean: float, low: float, high: float) -> str:
        return f"{mean:.3f} [{low:.3f}, {high:.3f}]"

    return pd.DataFrame([{
        "Category": row["category"],
        "Metric": row["metric"],
        "N": row["n_prey"],
        "Data mean [95% CI]": estimate(row["data_mean"], row["data_ci_low"], row["data_ci_high"]),
        "GAIL mean [95% CI]": estimate(row["gail_mean"], row["gail_ci_low"], row["gail_ci_high"]),
        "GAIL - Data [95% CI]": estimate(
            row["gail_minus_data"], row["difference_ci_low"], row["difference_ci_high"],
        ),
        "Absolute error": f"{row['absolute_imitation_error']:.3f}",
        "Clips Data/GAIL": f"{row['data_n']}/{row['gail_n']}",
    } for row in rows])


def set_paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
    })


def plot_source_overview(
    final_results: Mapping[str, Any], source: str, category: str,
) -> Any:
    """Compact two-panel clip distributions for one response category."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    set_paper_style()
    if category not in CATEGORY_METRICS:
        raise ValueError(f"category must be one of {tuple(CATEGORY_METRICS)}")
    prefix = SOURCE_PREFIX[source]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    positions = {
        (16, "expert"): 0.82, (16, "imitation"): 1.18,
        (32, "expert"): 1.82, (32, "imitation"): 2.18,
    }
    for metric_index, (ax, metric) in enumerate(zip(axes.flat, CATEGORY_METRICS[category])):
        spec = METRIC_SPECS[metric]
        for group_index, (n_prey, kind) in enumerate(
            (pair for n in (16, 32) for pair in ((n, "expert"), (n, "imitation")))
        ):
            case = f"{prefix}_{kind}_{n_prey}"
            values = _finite_1d(final_results["per_clip"][case][metric]).numpy()
            position = positions[(n_prey, kind)]
            if not len(values):
                continue
            box = ax.boxplot(
                [values], positions=[position], widths=0.28, patch_artist=True,
                showfliers=False, whis=1.5, manage_ticks=False,
                medianprops={"color": "white", "linewidth": 1.5},
                boxprops={"edgecolor": KIND_COLOR[kind], "linewidth": 1.1},
                whiskerprops={"color": KIND_COLOR[kind], "linewidth": 1.0},
                capprops={"color": KIND_COLOR[kind], "linewidth": 1.0},
            )
            box["boxes"][0].set_facecolor(KIND_COLOR[kind])
            box["boxes"][0].set_alpha(0.78)

            # Every independent clip remains visible; jitter is deterministic.
            rng = np.random.default_rng(2027 + 100 * metric_index + group_index)
            jitter = rng.uniform(-0.055, 0.055, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter, values, s=10,
                color=KIND_COLOR[kind], alpha=0.38, edgecolors="none", zorder=3,
            )
        ax.set_title(spec["short_label"])
        ax.set_xlabel("Group size")
        ax.set_ylabel(spec["unit"])
        ax.set_xticks((1, 2), labels=("16", "32"))
        ax.set_xlim(0.55, 2.45)
        ax.axhline(0, color="0.45", linewidth=0.7, zorder=0) if metric != "predator_distance" else None
    handles = [
        Patch(facecolor=KIND_COLOR[kind], edgecolor=KIND_COLOR[kind], alpha=0.78,
              label=KIND_LABEL[kind])
        for kind in ("expert", "imitation")
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=2, frameon=False,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1), w_pad=1.4)
    return fig


def plot_collective_timeline(final_results: Mapping[str, Any], source: str) -> Any:
    """Paper-ready fixed-cohort DoS/DoA curves up to T_common."""

    import matplotlib.pyplot as plt

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    for ax, metric in zip(axes, ("dos", "doa")):
        block = final_results["collective"][metric][source]
        x = block["time"].numpy()
        for case, values in block["cases"].items():
            kind = "imitation" if "imitation" in case else "expert"
            n_prey = 32 if case.endswith("32") else 16
            label = f"{KIND_LABEL[kind]}, N={n_prey}"
            linestyle = "-" if n_prey == 16 else "--"
            ax.plot(x, values["mean"], color=KIND_COLOR[kind], linestyle=linestyle,
                    linewidth=1.5, label=label)
            ax.fill_between(x, values["ci_low"], values["ci_high"],
                            color=KIND_COLOR[kind], alpha=0.10, linewidth=0)
        ax.set_title("Degree of Swarm (DoS)" if metric == "dos" else "Degree of Alignment (DoA)")
        ax.set_xlabel("Time [s]" if source == "biological" else "Couzin simulation time")
        ax.set_ylabel(METRIC_SPECS[metric]["unit"])
        ax.set_xlim(float(x[0]), float(x[-1]))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=4, frameon=False,
    )
    fig.tight_layout(rect=(0, 0.20, 1, 1), w_pad=1.4)
    return fig


def save_figure(fig: Any, output_stem: str | Path) -> tuple[Path, Path]:
    """Save a figure as vector PDF and high-resolution PNG."""

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path, png_path = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return pdf_path, png_path


def save_table_artifacts(table: Any, output_stem: str | Path) -> tuple[Path, Path]:
    """Save exact table values as CSV and a ready-to-edit LaTeX table."""

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = stem.with_suffix(".csv"), stem.with_suffix(".tex")
    table.to_csv(csv_path, index=False)
    tex_path.write_text(table.to_latex(index=False, escape=True), encoding="utf-8")
    return csv_path, tex_path


__all__ = [
    "METRIC_SPECS", "SOURCE_CASES", "build_final_results", "summarize_values",
    "stage_imitation_table", "paper_imitation_table", "plot_source_overview",
    "plot_collective_timeline", "save_figure", "save_table_artifacts",
]
