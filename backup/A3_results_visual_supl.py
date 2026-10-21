# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 08:27:15 2026

@author: juliu
"""


# Load original event datasets only once per scale
# dataset_by_scale = {}
# for scale in scales:
#     print(f"Loading dataset for scale: {scale}")
#     dataset_tmp = load_dataset(scale=scale)
#     dataset_tmp = dataset_tmp.dropna(subset=["flood_occurrence"]).reset_index(drop=True)
#     dataset_by_scale[scale] = dataset_tmp


#%% Supplementary figure: compact 2-subplot version

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

scales = ["catch", "valley", "downs"]

scale_label = {
    "catch": "Catchment",
    "valley": "Valley",
    "downs": "Downstream"
}

scale_colors = {
    "catch": "#4C78A8",
    "valley": "#F58518",
    "downs": "#54A24B"
}

state_order = ["below_Q50", "Q50_Q75", "Q75_Q90", "above_Q90"]

state_label = {
    "below_Q50": "Low",
    "Q50_Q75": "Moderate",
    "Q75_Q90": "High",
    "above_Q90": "Very high"
}

effect_col = "q75_to_q90_tau_report"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.unicode_minus": False,
    "savefig.dpi": 1200,
})

# --------------------------------------------------
# 2. Prepare data
# --------------------------------------------------
# effects_all should already contain:
# target_type, scale, dtp_state, flood_occurrence, flood_peak, fold, q75_to_q90_tau_report

required_cols = [
    "target_type",
    "scale",
    "dtp_state",
    "flood_occurrence",
    "flood_peak",
    "fold",
    effect_col
]

missing_cols = [c for c in required_cols if c not in effects_all.columns]
if missing_cols:
    raise ValueError(f"Missing columns in effects_all: {missing_cols}")

peak_flood = (
    effects_all[
        (effects_all["target_type"] == "peak") &
        (effects_all["flood_occurrence"] == 1)
    ]
    .dropna(subset=["scale", "dtp_state", "flood_peak", effect_col, "fold"])
    .copy()
)

peak_flood["dtp_state"] = pd.Categorical(
    peak_flood["dtp_state"],
    categories=state_order,
    ordered=True
)

# Fold-level causal effect, not event-level, for clearer plotting
causal_fold_state = (
    peak_flood
    .groupby(["scale", "dtp_state", "fold"], observed=True, as_index=False)
    [effect_col]
    .mean()
    .rename(columns={effect_col: "fold_mean_effect"})
)

# --------------------------------------------------
# 3. Helper: grouped boxplot by groundwater state and scale
# --------------------------------------------------

def grouped_boxplot_state_scale(
    ax,
    data,
    value_col,
    ylabel,
    ylim=None,
    show_points=True,
    point_level="event"
):
    """
    One compact grouped boxplot.

    x-axis = groundwater state
    grouped boxes = catch / valley / downstream
    """

    base_positions = np.arange(len(state_order))
    width = 0.22

    offsets = {
        "catch": -width,
        "valley": 0.0,
        "downs": width
    }

    rng = np.random.default_rng(42)

    for scale in scales:

        values_by_state = []
        positions = []

        for i, state in enumerate(state_order):
            vals = data.loc[
                (data["scale"] == scale) &
                (data["dtp_state"] == state),
                value_col
            ].dropna().values

            values_by_state.append(vals)
            positions.append(base_positions[i] + offsets[scale])

        bp = ax.boxplot(
            values_by_state,
            positions=positions,
            widths=0.16,
            patch_artist=True,
            showfliers=False,
            medianprops={"linewidth": 0.8, "color": "black"},
            whiskerprops={"linewidth": 0.6},
            capprops={"linewidth": 0.6},
            boxprops={"linewidth": 0.6}
        )

        for box in bp["boxes"]:
            box.set_facecolor(scale_colors[scale])
            box.set_alpha(0.60)
            box.set_edgecolor("black")

        if show_points:
            for pos, vals in zip(positions, values_by_state):
                if len(vals) == 0:
                    continue

                # avoid overplotting many observed flood peaks
                if point_level == "event" and len(vals) > 500:
                    vals_plot = rng.choice(vals, size=500, replace=False)
                else:
                    vals_plot = vals

                jitter = rng.normal(0, 0.018, size=len(vals_plot))

                ax.scatter(
                    np.full(len(vals_plot), pos) + jitter,
                    vals_plot,
                    s=3 if point_level == "event" else 7,
                    alpha=0.10 if point_level == "event" else 0.45,
                    color=scale_colors[scale],
                    edgecolor="none",
                    zorder=3
                )

    ax.set_xticks(base_positions)
    ax.set_xticklabels(
        [state_label[s] for s in state_order],
        rotation=0
    )

    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------
# 4. Create compact 2-row figure
# --------------------------------------------------

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(7.08, 3.1),   # GRL one-column style
    sharex=True,
    gridspec_kw={
        "hspace": 0.18
    }
)

# Row 1: observed flood peak
grouped_boxplot_state_scale(
    ax=axes[0],
    data=peak_flood,
    value_col="flood_peak",
    ylabel="Observed flood peak (mm d$^{-1}$)",
    ylim=(0, peak_flood["flood_peak"].quantile(0.99)),
    show_points=True,
    point_level="event"
)


# Row 2: causal effect
grouped_boxplot_state_scale(
    ax=axes[1],
    data=causal_fold_state,
    value_col="fold_mean_effect",
    ylabel="Q75 → Q90 effect on flood peak discharge (%)",
    ylim=None,
    show_points=True,
    point_level="fold"
)

axes[1].axhline(
    0,
    linestyle="--",
    linewidth=0.7,
    color="black",
    alpha=0.65
)

# Shared legend
legend_handles = [
    Patch(
        facecolor=scale_colors[scale],
        edgecolor="black",
        alpha=0.60,
        label=scale_label[scale]
    )
    for scale in scales
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.3, 0.8),
    columnspacing=0.9,
    handlelength=1.1
)

plt.tight_layout(rect=[0.0, 0.07, 1.0, 1.0])

out_fig = os.path.join(
    pkl_path,
    "supplement_compact_flood_peak_and_causal_peak_effect.png"
)

plt.savefig(out_fig, dpi=1200, bbox_inches="tight")
plt.show()

print("Saved figure to:", out_fig)
