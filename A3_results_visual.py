# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:23:04 2026

@author: juliu
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from A0_detect_events import load_camels_info, load_camels_dynamic, load_camels_static
from A2_causal_model_train import load_dataset


from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd

dataset = load_dataset(scale='catch') # catch, valley, downs
dataset = dataset.dropna(subset=["flood_occurrence"]).reset_index(drop=True)

pkl_path = r"\\geodata.geus.dk\HOME\causal_ml_gw_riverflood\results"
#%% hist
# -----------------------------
# Choose columns
# -----------------------------

x_col    = "event_precip_mm"
peak_col = "flood_peak"
occ_col  = "flood_occurrence"
gw_col   = "dtp_state"   # below_Q50, Q50_Q75, Q75_Q90, above_Q90

# -----------------------------
# Prepare data
# -----------------------------

needed_cols = [x_col, peak_col, occ_col, gw_col]

plot_df = (
    dataset[needed_cols]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

plot_df = plot_df[
    (plot_df[x_col] > 0) &
    (plot_df[peak_col] > 0)
].copy()

plot_df[occ_col] = plot_df[occ_col].astype(int)

# Restrict extreme precipitation outliers for clearer plotting
x_min, x_max = plot_df[x_col].quantile([0.01, 0.99])

plot_df = plot_df[
    (plot_df[x_col] >= x_min) &
    (plot_df[x_col] <= x_max)
].copy()

# -----------------------------
# Precipitation bins
# -----------------------------

n_bins = 30
bins = np.linspace(x_min, x_max, n_bins + 1)

plot_df["precip_bin"] = pd.cut(
    plot_df[x_col],
    bins=bins,
    include_lowest=True
)

# Overall bin summary
bin_summary = (
    plot_df
    .groupby("precip_bin", observed=False)
    .agg(
        n_events=(x_col, "count"),
        precip_mid=(x_col, "median"),
        peak_median=(peak_col, "median"),
        peak_q25=(peak_col, lambda x: np.percentile(x, 25)),
        peak_q75=(peak_col, lambda x: np.percentile(x, 75)),
        flood_occurrence=(occ_col, "mean"),
    )
    .reset_index()
)

bin_summary = bin_summary[bin_summary["n_events"] > 0].copy()

# Groundwater-state bin summary
bin_summary_gw = (
    plot_df
    .groupby(["precip_bin", gw_col], observed=False)
    .agg(
        n_events=(x_col, "count"),
        precip_mid=(x_col, "median"),
        flood_occurrence=(occ_col, "mean"),
    )
    .reset_index()
)

# Avoid noisy estimates from small bins
min_bin_count = 20
bin_summary_gw = bin_summary_gw[
    bin_summary_gw["n_events"] >= min_bin_count
].copy()

# -----------------------------
# Groundwater state colors
# -----------------------------
# -----------------------------
# High-contrast colors
# -----------------------------

hist_color = "#d9d9d9"          # light gray
peak_color = "#111111"          # nearly black
peak_band_color = "#bdbdbd"     # gray uncertainty band
overall_occ_color = "#d95f02"   # orange/red

state_order = ["below_Q50", "Q50_Q75", "Q75_Q90", "above_Q90"]

state_labels = {
    "below_Q50": "< P50 GW",
    "Q50_Q75": "P50 - P75  GW",
    "Q75_Q90": "P75 - P90 GW ",
    "above_Q90": "> P90 GW",
}

# Colorblind-safe, stronger separation
state_colors = {
    "below_Q50": "#7f3b08",   # dark brown
    "Q50_Q75":   "#e6ab02",     # mustard
    "Q75_Q90":   "#1b9e77",     # green/teal
    "above_Q90": "#1f78b4",   # strong blue
}

state_markers = {
    "below_Q50": "o",
    "Q50_Q75": "s",
    "Q75_Q90": "^",
    "above_Q90": "D",
}

state_linestyles = {
    "below_Q50": "-",
    "Q50_Q75": "--",
    "Q75_Q90": "-.",
    "above_Q90": ":",
}


# -----------------------------
# Single-panel plot:
# precipitation histogram + peak discharge + flood occurrence
# -----------------------------
fig, ax_count = plt.subplots(figsize=(6.5, 4.0))

# Right axis 1: peak discharge
ax_peak = ax_count.twinx()

# Right axis 2: flood occurrence, offset outward
ax_occ = ax_count.twinx()
ax_occ.spines["right"].set_position(("axes", 1.15))
ax_occ.spines["right"].set_visible(True)

# -----------------------------
# Histogram: event precipitation count
# -----------------------------
bin_width = bins[1] - bins[0]
ax_count.hist(
    plot_df[x_col],
    bins=bins,
    color=hist_color,
    edgecolor="white",
    linewidth=0.6,
    alpha=0.85,
    label="Count",
    zorder=1
)
ax_count.set_ylabel("Count")
ax_count.set_xlabel("Precipitation amount (mm)")

# -----------------------------
# Peak discharge response
# -----------------------------
peak_line, = ax_peak.plot(
    bin_summary["precip_mid"],
    bin_summary["peak_median"],
    color=peak_color,
    marker="o",
    markersize=4,
    linewidth=2.4,
    label="Median Peak discharge",
    zorder=5
)


ax_peak.set_ylabel("Peak discharge (mm/d)")
ax_peak.tick_params(axis="y", colors=peak_color)
ax_peak.yaxis.label.set_color(peak_color)
ax_peak.spines["right"].set_color(peak_color)

# -----------------------------
# Flood occurrence curves
# -----------------------------
overall_occ_line, = ax_occ.plot(
    bin_summary["precip_mid"],
    100 * bin_summary["flood_occurrence"],
    color=overall_occ_color,
    linewidth=2.4,
    linestyle="--",
    label="Flood occurrence",
    zorder=6
)

# Groundwater-state flood occurrence
occ_lines = [overall_occ_line]

for state in state_order:
    d = bin_summary_gw[bin_summary_gw[gw_col] == state].copy()

    if len(d) == 0:
        continue

    line, = ax_occ.plot(
        d["precip_mid"],
        100 * d["flood_occurrence"],
        marker=state_markers[state],
        markersize=4,
        linewidth=1.8,
        linestyle=state_linestyles[state],
        color=state_colors[state],
        label=state_labels[state],
        zorder=7
    )

    occ_lines.append(line)

ax_occ.set_ylabel("Flood occurrence (%)")
ax_occ.set_ylim(0, 101)


# -----------------------------
# Styling
# -----------------------------
ax_count.grid(True, axis="y", alpha=0.25)
ax_count.spines["top"].set_visible(False)
ax_peak.spines["top"].set_visible(False)
ax_occ.spines["top"].set_visible(False)

# Make axis labels visually distinct but calm
ax_count.tick_params(axis="y", colors="0.35")
ax_count.yaxis.label.set_color("0.35")

ax_peak.tick_params(axis="y", colors="0.10")
ax_peak.yaxis.label.set_color("0.10")

ax_occ.set_ylabel("Flood occurrence (%)")
ax_occ.tick_params(axis="y", colors=overall_occ_color)
ax_occ.yaxis.label.set_color(overall_occ_color)
ax_occ.spines["right"].set_color(overall_occ_color)

# Combined legend
handles = []
labels = []

for ax in [ax_count, ax_peak, ax_occ]:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

ax_count.legend(handles,labels,frameon=False,fontsize=8,loc="upper left",ncol=1)

# Note
ax_count.text(
    0.99,
    0.05,
    f"Groundwater-state occurrence curves \nshown only for bins with n ≥ {min_bin_count}",
    transform=ax_count.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    color="0.35"
)

plt.tight_layout()
plt.show()
#%% data statistics

# -----------------------------
# Settings
# -----------------------------

x_col = "event_precip_mm"
gw_col = "dtp_quantile"  # 0-1, higher = shallower groundwater
soil_col = "wcr_quantile"   # 0-1, higher = wetter soil

plot_vars = [
    "flood_peak",
    "flood_volume_above_q95",
    "duration_above_q95"
]


y_labels = {
    "flood_peak": "Peak discharge (mm/d)",
    "flood_volume_above_q95": "Flood volume above Q95 (mm)",
    "duration_above_q95": "Flood duration above Q95 (d)"
}

# Use log10 for all three because volume/duration are usually highly skewed.
# If you want no log for peak, set "flood_peak": False.
log10_transform = {
    "flood_peak": False,
    "flood_volume_above_q95": False,
    "duration_above_q95": False
}

# Duration does not need contours
show_contours = {
    "flood_peak": True,
    "flood_volume_above_q95": True,
    "duration_above_q95": False
}

n_bins_x = 50
n_bins_y = 50
min_count = 5


# -----------------------------
# Groundwater colormap
# -----------------------------

gw_cmap = LinearSegmentedColormap.from_list(
    "groundwater_percentile",
    [
        "#8c510a",  # brown: low percentile / deep groundwater
        "#d8b365",  # tan
        "#f6e8c3",  # pale yellow
        "#c7eae5",  # pale blue-green
        "#5ab4ac",  # teal
        "#01665e",  # dark teal: high percentile / shallow groundwater
    ],
    N=256
)

gw_cmap.set_bad(color="white")


# -----------------------------
# Helper function
# -----------------------------
def prepare_binned_groundwater(
    dataset,
    x_col,
    y_col,
    gw_col,
    soil_col=None,
    log10_y=True,
    n_bins_x=50,
    n_bins_y=50,
    min_count=10
):
    """
    Prepare binned mean groundwater percentile and optionally soil moisture
    over precipitation-response space.
    """

    cols = [x_col, y_col, gw_col]
    if soil_col is not None:
        cols.append(soil_col)

    plot_df = (
        dataset[cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    plot_df = plot_df[
        (plot_df[x_col] > 0) &
        (plot_df[y_col] > 0)
    ].copy()

    if log10_y:
        plot_df["y_plot"] = np.log10(plot_df[y_col])
    else:
        plot_df["y_plot"] = plot_df[y_col]

    x_min, x_max = plot_df[x_col].quantile([0.01, 0.99])
    y_min, y_max = plot_df["y_plot"].quantile([0.01, 0.99])

    plot_df = plot_df[
        (plot_df[x_col] >= x_min) & (plot_df[x_col] <= x_max) &
        (plot_df["y_plot"] >= y_min) & (plot_df["y_plot"] <= y_max)
    ].copy()

    x_edges = np.linspace(x_min, x_max, n_bins_x + 1)
    y_edges = np.linspace(y_min, y_max, n_bins_y + 1)

    gw_mean, x_edges, y_edges, _ = binned_statistic_2d(
        plot_df[x_col],
        plot_df["y_plot"],
        plot_df[gw_col],
        statistic="mean",
        bins=[x_edges, y_edges]
    )

    event_count, _, _, _ = binned_statistic_2d(
        plot_df[x_col],
        plot_df["y_plot"],
        plot_df[gw_col],
        statistic="count",
        bins=[x_edges, y_edges]
    )

    gw_mean_plot = gw_mean.T
    event_count_plot = event_count.T

    gw_mean_plot[event_count_plot < min_count] = np.nan

    soil_mean_plot = None

    if soil_col is not None:
        soil_mean, _, _, _ = binned_statistic_2d(
            plot_df[x_col],
            plot_df["y_plot"],
            plot_df[soil_col],
            statistic="mean",
            bins=[x_edges, y_edges]
        )

        soil_mean_plot = soil_mean.T
        soil_mean_plot[event_count_plot < min_count] = np.nan

        # Light smoothing makes the soil-moisture contours cleaner
        soil_mean_plot = gaussian_filter(soil_mean_plot, sigma=1.0)

    event_density = gaussian_filter(event_count_plot, sigma=1.2)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers)

    return {
        "plot_df": plot_df,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "gw_mean_plot": gw_mean_plot,
        "soil_mean_plot": soil_mean_plot,
        "event_count_plot": event_count_plot,
        "event_density": event_density,
        "Xc": Xc,
        "Yc": Yc
    }



def plot_groundwater_panel(
    ax,
    dataset,
    x_col,
    y_col,
    gw_col,
    ylabel,
    soil_col=None,
    show_soil_contours=False,
    log10_y=True,
    contour=False,
    n_bins_x=50,
    n_bins_y=50,
    min_count=10
):
    """
    Plot one panel of mean groundwater percentile across precipitation-response space.
    Optionally overlay soil-moisture contours.
    """

    data = prepare_binned_groundwater(
        dataset=dataset,
        x_col=x_col,
        y_col=y_col,
        gw_col=gw_col,
        soil_col=soil_col,
        log10_y=log10_y,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        min_count=min_count
    )

    mesh = ax.pcolormesh(
        data["x_edges"],
        data["y_edges"],
        data["gw_mean_plot"],
        shading="auto",
        cmap=gw_cmap,
        vmin=0,
        vmax=1
    )

    # Event-density contours
    if contour:
        positive_density = data["event_density"][data["event_density"] > 0]

        if len(positive_density) > 0:
            levels = np.nanpercentile(positive_density, [60, 75, 90])
            levels = np.unique(levels)

            if len(levels) > 1:
                cs = ax.contour(
                    data["Xc"],
                    data["Yc"],
                    data["event_density"],
                    levels=levels,
                    colors="0.15",
                    linewidths=1.0,
                    alpha=0.85
                )

                labels = ax.clabel(
                    cs,
                    inline=True,
                    fmt="%.0f"
                )

                for txt in labels:
                    txt.set_path_effects([
                        pe.withStroke(linewidth=2.3, foreground="white")
                    ])

    # Soil moisture contours, only where requested
    # Highlight cells where soil is very wet but groundwater is not yet extremely shallow
    if show_soil_contours and soil_col is not None and data["soil_mean_plot"] is not None:
        print("soil_mean_plot range:",
              np.nanmin(data["soil_mean_plot"]),
              np.nanmax(data["soil_mean_plot"]))
        
        soil_high = data["soil_mean_plot"] >= 0.75
        groundwater_not_max = data["gw_mean_plot"] < 0.90
        wet_soil_but_gw_can_rise = soil_high & groundwater_not_max
    
        ax.contourf(
            data["Xc"],
            data["Yc"],
            wet_soil_but_gw_can_rise.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["///"],
            alpha=0
        )

    ax.set_xlabel("Precipitation amount (mm)")
    ax.set_ylabel(ylabel)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.text(
    #     0.02,
    #     0.98,
    #     f"n ≥ {min_count} per cell",
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=8,
    #     bbox=dict(
    #         facecolor="white",
    #         edgecolor="none",
    #         alpha=0.75,
    #         boxstyle="round,pad=0.25"
    #     )
    # )

    if show_soil_contours:
        ax.text(
            0.02,
            0.98,
            "Hatching: wet soil, GW still below p90",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
                boxstyle="round,pad=0.25"
            )
        )

    return mesh


# -----------------------------
# Make figure
# -----------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(13, 3),
    dpi=1200,
    constrained_layout=True
)
import matplotlib as mpl


last_mesh = None

for i, (ax, plot_var) in enumerate(zip(axes, plot_vars)):
    last_mesh = plot_groundwater_panel(
        ax=ax,
        dataset=dataset,
        x_col=x_col,
        y_col=plot_var,
        gw_col=gw_col,
        soil_col=soil_col,
        show_soil_contours=(i in [0, 1]),   # soil moisture only in subplot (a) and (b)
        ylabel=y_labels[plot_var],
        log10_y=log10_transform[plot_var],
        contour=False, #show_contours[plot_var]
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        min_count=min_count
    )

    # ax.text(
    #     0.02,
    #     1.04,
    #     f"({chr(97 + i)})",
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="bottom",
    #     fontsize=10,
    #     fontweight="bold"
    # )

cbar = fig.colorbar(
    last_mesh,
    ax=axes,
    orientation="vertical",
    fraction=0.025,
    pad=0.02
)

cbar.set_label("Mean groundwater percentile\nhigher = shallower groundwater", )

cbar.set_ticks([0, 0.25, 0.50, 0.75, 0.90, 1.00])
cbar.set_ticklabels(["0", "0.25", "0.50", "0.75", "0.90", "1.00"])
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 12,
    "axes.unicode_minus": False,
})

plt.show()


#%% nuisance
# --------------------------------------------------
# 1. Load all nuisance results
# --------------------------------------------------

scales = ["catch", "valley", "downs"] #
target_types = ["peak", "occurrence", "volume", "duration"]  #"peak", "occurrence", "volume", "duration"

scale_label = {
    "catch": "Catchment",
    "valley": "Valley",
    "downs": "Downstream"
}

target_label = {
    "peak": "Peak",
    "occurrence": "Occurrence",
    "volume": "Volume",
    "duration": "Duration"
}

all_results = []

for scale in scales:
    for target_type in target_types:
        pkl_file = os.path.join(
            pkl_path,
            f"causal_ml_results_{target_type}_{scale}_all_contrasts.pkl"
        )

        with open(pkl_file, "rb") as f:
            loaded = pickle.load(f)

        results_df = loaded["nuisance_results"].copy()
        results_df["scale"] = scale
        results_df["scale_label"] = scale_label[scale]
        results_df["target_type"] = target_type
        results_df["target_label"] = target_label[target_type]

        all_results.append(results_df)

nuisance_all = pd.concat(all_results, ignore_index=True)

print(nuisance_all.head())
print(nuisance_all.columns)


# --------------------------------------------------
# 2. Prepare plotting table
# --------------------------------------------------
# --------------------------------------------------
# Plot nuisance model performance: 2 rows × 5 columns
# Top row: R2 or PR-AUC
# Bottom row: RMSE or Brier score
# --------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

scale_colors = {
    "catch": "#4C78A8",
    "valley": "#F58518",
    "downs": "#54A24B"
}

scale_label = {
    "catch": "Catchment",
    "valley": "Valley",
    "downs": "Downstream"
}

plot_columns = ["treatment", "peak", "occurrence", "volume", "duration"]

column_label = {
    "treatment": "T model(dtp)",
    "peak": "Y model (Peak)",
    "occurrence": "Y model (Occurrence)",
    "volume": "Y model (Volume)",
    "duration": "Y model (Duration)"
}

metric_layout = {
    "treatment": {
        "top": ("T_R2", "R²"),
        "bottom": ("T_RMSE", "RMSE")
    },
    "peak": {
        "top": ("Y_R2", "R²"),
        "bottom": ("Y_RMSE", "RMSE")
    },
    "occurrence": {
        "top": ("Y_PR_AUC", "PR-AUC"),
        "bottom": ("Y_Brier", "Brier")
    },
    "volume": {
        "top": ("Y_R2", "R²"),
        "bottom": ("Y_RMSE", "RMSE")
    },
    "duration": {
        "top": ("Y_R2", "R²"),
        "bottom": ("Y_RMSE", "RMSE")
    }
}


def get_values_for_panel(nuisance_all, column_name, metric_name, scale):
    """
    Extract fold-level values for each panel.

    For the treatment model, use one target file as representative because
    the treatment model predicts the same dtp variable using the same X/W
    for a given scale.
    """

    if column_name == "treatment":
        # Use peak as representative to avoid repeating treatment results
        sub = nuisance_all[
            (nuisance_all["target_type"] == "peak") &
            (nuisance_all["scale"] == scale)
        ]
    else:
        sub = nuisance_all[
            (nuisance_all["target_type"] == column_name) &
            (nuisance_all["scale"] == scale)
        ]

    if metric_name not in sub.columns:
        return np.array([])

    return sub[metric_name].dropna().values


def colored_scale_boxplot_metric(
    ax,
    nuisance_all,
    column_name,
    metric_name,
    metric_label,
    scales,
    show_title=False,
    show_ylabel=False
):
    positions = np.arange(len(scales)) + 1

    values = [
        get_values_for_panel(
            nuisance_all=nuisance_all,
            column_name=column_name,
            metric_name=metric_name,
            scale=scale
        )
        for scale in scales
    ]

    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.32,          # slim boxes
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 1.2, "color": "black"},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9}
    )

    for box, scale in zip(bp["boxes"], scales):
        box.set_facecolor(scale_colors[scale])
        box.set_alpha(0.65)
        box.set_edgecolor("black")
        box.set_linewidth(0.8)

    # Fold points
    rng = np.random.default_rng(42)

    for i, scale in enumerate(scales):
        y = values[i]

        if len(y) == 0:
            continue

        x = rng.normal(positions[i], 0.025, size=len(y))
        ax.scatter(x,y,s=9,alpha=0.40,color=scale_colors[scale],edgecolor="none")

    ax.set_xticks([])
    ax.set_xlim(0.45, len(scales) + 0.55)
    ax.grid(axis="y", alpha=0.25)

    if show_title:
        ax.set_title(column_label[column_name], fontsize=11)

    # if show_ylabel:
    #     ax.set_ylabel(metric_label, fontsize=10)

    # Add metric label inside each panel
    if metric_label in ['RMSE', 'Brier']: 
        ax.text(0.40,0.90,metric_label,transform=ax.transAxes,ha="left", va="top",fontsize=10)
    else: 
        ax.text(0.40,0.10,metric_label,transform=ax.transAxes,ha="left", va="top",fontsize=10)

    # Sensible y-limits
    nonempty = [v for v in values if len(v) > 0]

    if len(nonempty) > 0:
        all_y = np.concatenate(nonempty)
        ymin = np.nanmin(all_y)
        ymax = np.nanmax(all_y)
        pad = 0.08 * (ymax - ymin + 1e-9)

        bounded_metrics = [
            "T_R2",
            "Y_R2",
            "Y_PR_AUC",
            "Y_AUC",
            "Y_positive_PR_AUC"
        ]

        if metric_name in bounded_metrics:
            ax.set_ylim(max(0, ymin - pad), min(1.05, ymax + pad))
        else:
            ax.set_ylim(max(0, ymin - pad), ymax + pad)


fig, axes = plt.subplots(
    nrows=2,
    ncols=5,
    figsize=(10.5, 4.4),
    sharey=False
)

for col, column_name in enumerate(plot_columns):

    # Top row
    metric_name, metric_label = metric_layout[column_name]["top"]

    colored_scale_boxplot_metric(
        ax=axes[0, col],
        nuisance_all=nuisance_all,
        column_name=column_name,
        metric_name=metric_name,
        metric_label=metric_label,
        scales=scales,
        show_title=True,
        show_ylabel=(col == 0)
    )

    # Bottom row
    metric_name, metric_label = metric_layout[column_name]["bottom"]

    colored_scale_boxplot_metric(
        ax=axes[1, col],
        nuisance_all=nuisance_all,
        column_name=column_name,
        metric_name=metric_name,
        metric_label=metric_label,
        scales=scales,
        show_title=False,
        show_ylabel=(col == 0)
    )


# Shared legend
legend_handles = [
    Patch(
        facecolor=scale_colors[scale],
        edgecolor="black",
        alpha=0.65,
        label=scale_label[scale]
    )
    for scale in scales
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02)
)

plt.tight_layout(rect=[0, 0.06, 1, 0.98])

out_fig = os.path.join(
    pkl_path,
    "figure_nuisance_model_performance_2x5.png"
)

plt.savefig(out_fig, dpi=1200, bbox_inches="tight")
plt.show()

print("Saved figure to:", out_fig)


# --------------------------------------------------
# 4. Summary table
# --------------------------------------------------
summary_rows = []
for target_type in target_types:
    for scale in scales:
        sub = nuisance_all[
            (nuisance_all["target_type"] == target_type) &
            (nuisance_all["scale"] == scale)
        ]

        row = {
            "target": target_type,
            "scale": scale,
            "n_folds": len(sub),
            "T_R2_mean": sub["T_R2"].mean(),
            "T_R2_sd": sub["T_R2"].std(),
            "T_RMSE_mean": sub["T_RMSE"].mean(),
            "T_RMSE_sd": sub["T_RMSE"].std(),
        }

        if target_type == "occurrence":
            if "Y_PR_AUC" in sub.columns:
                row["Y_metric"] = "PR-AUC"
                row["Y_metric_mean"] = sub["Y_PR_AUC"].mean()
                row["Y_metric_sd"] = sub["Y_PR_AUC"].std()
            if "Y_AUC" in sub.columns:
                row["Y_ROC_AUC_mean"] = sub["Y_AUC"].mean()
                row["Y_ROC_AUC_sd"] = sub["Y_AUC"].std()
            if "Y_Brier" in sub.columns:
                row["Y_Brier_mean"] = sub["Y_Brier"].mean()
                row["Y_Brier_sd"] = sub["Y_Brier"].std()
        else:
            row["Y_metric"] = "R2"
            row["Y_metric_mean"] = sub["Y_R2"].mean()
            row["Y_metric_sd"] = sub["Y_R2"].std()
            row["Y_RMSE_mean"] = sub["Y_RMSE"].mean()
            row["Y_RMSE_sd"] = sub["Y_RMSE"].std()
            row["Y_MAE_mean"] = sub["Y_MAE"].mean()
            row["Y_MAE_sd"] = sub["Y_MAE"].std()

            if "Y_positive_PR_AUC" in sub.columns:
                row["Y_positive_PR_AUC_mean"] = sub["Y_positive_PR_AUC"].mean()
                row["Y_positive_PR_AUC_sd"] = sub["Y_positive_PR_AUC"].std()

        summary_rows.append(row)

nuisance_summary = pd.DataFrame(summary_rows)


# #%% causal - box
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch

# # --------------------------------------------------
# # 1. Load causal fold results
# # --------------------------------------------------
# scales = ["catch", "valley", "downs"]
# target_types = ["peak", "occurrence", "volume", "duration"]

# scale_label = {
#     "catch": "Catchment",
#     "valley": "Valley",
#     "downs": "Downstream"
# }

# target_label = {
#     "peak": "Peak",
#     "occurrence": "Occurrence",
#     "volume": "Volume",
#     "duration": "Duration"
# }

# effect_unit_label = {
#     "peak": "% change in peak",
#     "occurrence": "Percentage-point change",
#     "volume": "% change in volume + 1",
#     "duration": "% change in duration + 1"
# }

# all_effect_results = []

# for scale in scales:
#     for target_type in target_types:
#         pkl_file = os.path.join(
#             pkl_path,
#             f"causal_ml_results_{target_type}_{scale}_all_contrasts.pkl")

#         with open(pkl_file, "rb") as f:
#             loaded = pickle.load(f)

#         effect_results_df = loaded["fold_contrast_results"].copy()

#         effect_results_df["scale"] = scale
#         effect_results_df["scale_label"] = scale_label[scale]
#         effect_results_df["target_type"] = target_type
#         effect_results_df["target_label"] = target_label[target_type]

#         all_effect_results.append(effect_results_df)

# causal_fold_all = pd.concat(all_effect_results, ignore_index=True)

# print(causal_fold_all.head())
# print(causal_fold_all.columns)

# # --------------------------------------------------
# # 2. Standardize effect column names
# # --------------------------------------------------
# def get_first_existing_column(df, candidates):
#     for col in candidates:
#         if col in df.columns:
#             return col
#     raise ValueError(f"None of these columns found: {candidates}")

# mean_col   = get_first_existing_column(causal_fold_all, ["mean_effect",   "mean_effect_percent"])
# median_col = get_first_existing_column(causal_fold_all, ["median_effect", "median_effect_percent"])
# p05_col    = get_first_existing_column(causal_fold_all, ["p05_effect",    "p05_effect_percent"])
# p95_col    = get_first_existing_column(causal_fold_all, ["p95_effect",    "p95_effect_percent"])

# causal_fold_all["mean_effect_plot"]   = causal_fold_all[mean_col]
# causal_fold_all["median_effect_plot"] = causal_fold_all[median_col]
# causal_fold_all["p05_effect_plot"]    = causal_fold_all[p05_col]
# causal_fold_all["p95_effect_plot"]    = causal_fold_all[p95_col]

# # Width of event-level heterogeneity within each fold
# causal_fold_all["heterogeneity_width"] = (
#     causal_fold_all["p95_effect_plot"] - causal_fold_all["p05_effect_plot"])


# out_csv = os.path.join(pkl_path, "Causal_model_performance_summary.csv")
# causal_fold_all.to_csv(out_csv, index=False)
# print("Saved summary table to:", out_csv)

# # --------------------------------------------------
# # 3. Plot causal / residualized effect stability
# # --------------------------------------------------
# scale_colors = {"catch": "#4C78A8",
#                 "valley": "#F58518",
#                 "downs": "#54A24B"}

# def colored_scale_boxplot(ax, data, value_col, scales, title=None, ylabel=None, zero_line=False):
#     positions = np.arange(len(scales)) + 1

#     values = [
#         data.loc[data["scale"] == scale, value_col].dropna().values
#         for scale in scales
#     ]

#     bp = ax.boxplot(
#         values,
#         positions=positions,
#         widths=0.55,
#         patch_artist=True,
#         showfliers=False,
#         medianprops={"linewidth": 1.5, "color": "black"},
#         whiskerprops={"linewidth": 1.0},
#         capprops={"linewidth": 1.0}
#     )

#     for box, scale in zip(bp["boxes"], scales):
#         box.set_facecolor(scale_colors[scale])
#         box.set_alpha(0.65)
#         box.set_edgecolor("black")

#     rng = np.random.default_rng(42)

#     for i, scale in enumerate(scales):
#         y = data.loc[data["scale"] == scale, value_col].dropna().values
#         x = rng.normal(positions[i], 0.045, size=len(y))

#         ax.scatter(
#             x,
#             y,
#             s=12,
#             alpha=0.45,
#             color=scale_colors[scale],
#             edgecolor="none"
#         )

#     if zero_line:
#         ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.6)

#     ax.set_xticks([])
#     ax.set_xlim(0.4, len(scales) + 0.6)

#     if title is not None:
#         ax.set_title(title, fontsize=11)

#     if ylabel is not None:
#         ax.set_ylabel(ylabel)

#     ax.grid(axis="y", alpha=0.25)


# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=4,
#     figsize=(10.5, 5.0),
#     sharey=False
# )

# for col, target_type in enumerate(target_types):
#     sub = causal_fold_all[causal_fold_all["target_type"] == target_type].copy()

#     # -----------------------------
#     # Top row: fold-mean causal effect
#     # -----------------------------
#     ax = axes[0, col]

#     colored_scale_boxplot(
#         ax=ax,
#         data=sub,
#         value_col="mean_effect_plot",
#         scales=scales,
#         title=f"{target_label[target_type]}",
#         ylabel="Mean effect (%)", # effect_unit_label[target_type] if col == 0 else None,
#         zero_line=True
#     )

#     # -----------------------------
#     # Bottom row: event-level heterogeneity width
#     # -----------------------------
#     ax = axes[1, col]

#     colored_scale_boxplot(
#         ax=ax,
#         data=sub,
#         value_col="heterogeneity_width",
#         scales=scales,
#         ylabel="Heterogeneity range (p95 - p05)" if col == 0 else None,
#         zero_line=False
#     )

# # Shared legend
# legend_handles = [
#     Patch(
#         facecolor=scale_colors[scale],
#         edgecolor="black",
#         alpha=0.65,
#         label=scale_label[scale]
#     )
#     for scale in scales
# ]

# fig.legend(
#     handles=legend_handles,
#     loc="lower center",
#     ncol=3,
#     frameon=False,
#     bbox_to_anchor=(0.5, -0.02)
# )
# plt.tight_layout(rect=[0, 0.05, 1, 0.98])
# out_fig = os.path.join(pkl_path,"figure_causal_effect_stability_2x4_colored_scales.png")
# plt.savefig(out_fig, dpi=1200, bbox_inches="tight")
# plt.show()
# print("Saved figure to:", out_fig)
 
#%% causal - forest plot: Q75 -> Q90 only

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

scales = ["catch", "valley", "downs"]
target_types = ["peak", "occurrence", "volume", "duration"]

scale_label = {
    "catch": "Catch",
    "valley": "Valley",
    "downs": "Downs"
}

target_label = {
    "peak": "Peak discharge",
    "occurrence": "Flood occurrence",
    "volume": "Flood volume Q95",
    "duration": "Duration Q95"
}

effect_unit_label = {
    "peak": "Percentage change",
    "occurrence": "Percentage-point change",
    "volume": "Percentage change",
    "duration": "Percentage change"
}

scale_colors = {
    "catch": "#4C78A8",
    "valley": "#F58518",
    "downs": "#54A24B"
}

scale_markers = {
    "catch": "o",
    "valley": "s",
    "downs": "^"
}

state_order = ["below_Q50", "Q50_Q75", "Q75_Q90", "above_Q90"]

state_label = {
    "below_Q50": "Low",
    "Q50_Q75": "Moderate",
    "Q75_Q90": "High",
    "above_Q90": "Very high"
}

forest_xlim = {
    "peak": (10, 20),
    "occurrence": (2, 4),
    "volume": (2, 5),
    "duration": (5, 8),
}

# Only keep this contrast
main_contrast       = "q75_to_q90"
main_contrast_label = "High → Very high state"

# --------------------------------------------------
# GRL-style figure settings
# --------------------------------------------------
# GRL two-column width is approximately 7 inches.
GRL_WIDTH = 7.08
GRL_HEIGHT= 3.50

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "savefig.dpi": 1200,
})

# --------------------------------------------------
# 2. Load all event-level causal results
# --------------------------------------------------

all_event_rows = []

for scale in scales:
    for target_type in target_types:

        pkl_file = os.path.join(
            pkl_path,
            f"causal_ml_results_{target_type}_{scale}_all_contrasts.pkl"
        )

        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"Cannot find file: {pkl_file}")

        with open(pkl_file, "rb") as f:
            loaded = pickle.load(f)

        effects_event_df = loaded["causal_event_results"].copy()

        effects_event_df["scale"] = scale
        effects_event_df["scale_label"] = scale_label[scale]
        effects_event_df["target_type"] = target_type
        effects_event_df["target_label"] = target_label[target_type]
        effects_event_df["effect_unit"] = effect_unit_label[target_type]

        all_event_rows.append(effects_event_df)

effects_all = pd.concat(all_event_rows, ignore_index=True)

print(effects_all.head())
print(effects_all.columns)

# --------------------------------------------------
# 3. Fold-level summary for Q75 -> Q90 forest row
# --------------------------------------------------

def make_fold_summary(effects_all, contrast):
    """
    Summarize one contrast by fold, scale, and target.
    """
    col = f"{contrast}_tau_report"

    if col not in effects_all.columns:
        raise ValueError(f"Column not found: {col}")

    fold_df = (
        effects_all
        .groupby(
            ["target_type", "target_label", "effect_unit",
             "scale", "scale_label", "fold"],
            as_index=False
        )[col]
        .mean()
        .rename(columns={col: "fold_mean_effect"})
    )

    summary_df = (
        fold_df
        .groupby(
            ["target_type", "target_label", "effect_unit",
             "scale", "scale_label"],
            as_index=False
        )
        .agg(
            mean_effect=("fold_mean_effect", "mean"),
            sd_effect=("fold_mean_effect", "std"),
            n_folds=("fold", "nunique"),
            positive_folds=("fold_mean_effect", lambda x: int((x > 0).sum())),
            negative_folds=("fold_mean_effect", lambda x: int((x < 0).sum())),
            min_fold=("fold_mean_effect", "min"),
            max_fold=("fold_mean_effect", "max"),
        )
    )

    summary_df["se_effect"] = summary_df["sd_effect"] / np.sqrt(summary_df["n_folds"])
    summary_df["ci95_effect"] = 1.96 * summary_df["se_effect"]
    summary_df["contrast"] = contrast

    return fold_df, summary_df


main_fold_df, main_summary = make_fold_summary(effects_all, main_contrast)

# --------------------------------------------------
# 4. State-dependent summaries for Q75 -> Q90
# --------------------------------------------------
# Row 2 also uses Q75 -> Q90.
# Each box contains fold-level mean effects within each observed groundwater state.

state_effect_col = f"{main_contrast}_tau_report"

if state_effect_col not in effects_all.columns:
    raise ValueError(f"Column not found: {state_effect_col}")

if "dtp_state" not in effects_all.columns:
    raise ValueError("Column 'dtp_state' not found in effects_all.")

state_fold_df = (
    effects_all
    .dropna(subset=["dtp_state", state_effect_col])
    .copy()
)

state_fold_df["dtp_state"] = pd.Categorical(
    state_fold_df["dtp_state"],
    categories=state_order,
    ordered=True
)

state_fold_summary = (
    state_fold_df
    .groupby(
        ["target_type", "target_label", "effect_unit",
         "scale", "scale_label", "fold", "dtp_state"],
        observed=False,
        as_index=False
    )[state_effect_col]
    .mean()
    .rename(columns={state_effect_col: "fold_state_mean_effect"})
)

state_fold_summary["contrast"] = main_contrast

print(state_fold_summary.head())

# --------------------------------------------------
# 5. Plot helper functions
# --------------------------------------------------

def plot_forest_row(
    ax,
    summary_df,
    fold_df,
    target_type,
    show_ylabel=False,
    error_type="ci95",
    show_fold_points=True,
    xlim=None,
):
    """
    Forest-style panel.

    y-axis = groundwater spatial scale
    x-axis = Q75 -> Q90 causal effect

    Small points = fold-level mean effects
    Large point = mean across folds
    Error bar = 95% CI of fold means or SD across folds
    """

    sub_summary = summary_df[summary_df["target_type"] == target_type].copy()
    sub_fold = fold_df[fold_df["target_type"] == target_type].copy()

    y_positions = {
        "catch": 2,
        "valley": 1,
        "downs": 0
    }

    rng = np.random.default_rng(42)

    for scale in scales:
        row = sub_summary[sub_summary["scale"] == scale]

        if row.empty:
            continue

        y = y_positions[scale]

        # Fold-level points
        if show_fold_points:
            fold_vals = sub_fold.loc[
                sub_fold["scale"] == scale,
                "fold_mean_effect"
            ].dropna().values

            y_jitter = rng.normal(0, 0.045, size=len(fold_vals))

            ax.scatter(
                fold_vals,
                np.full(len(fold_vals), y) + y_jitter,
                s=8,
                color=scale_colors[scale],
                alpha=0.35,
                edgecolor="none",
                zorder=2
            )

        # Mean effect and uncertainty
        x = row["mean_effect"].values[0]

        if error_type == "sd":
            xerr = row["sd_effect"].values[0]
        elif error_type == "ci95":
            xerr = row["ci95_effect"].values[0]
        else:
            raise ValueError("error_type must be 'ci95' or 'sd'")

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt=scale_markers[scale],
            color=scale_colors[scale],
            markersize=4.5,
            linewidth=1.1,
            capsize=2.5,
            alpha=1.0,
            markeredgecolor="black",
            markeredgewidth=0.35,
            zorder=4
        )

    ax.axvline(0, linestyle="--", linewidth=0.7, color="black", alpha=0.65)

    ax.set_yticks([2, 1, 0])

    if show_ylabel:
        ax.set_yticklabels(["Catch", "Valley", "Downs"], rotation=90, va="center", ha="center",)
    else:
        ax.set_yticklabels([])
        
    if xlim is not None:
        ax.set_xlim(xlim) 
    
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_state_boxplot(
    ax,
    state_fold_summary,
    target_type,
    show_ylabel=False
):
    """
    Boxplot panel.

    x-axis = observed groundwater state
    y-axis = fold-level mean Q75 -> Q90 effect
    boxes = groundwater spatial scales
    """

    sub = state_fold_summary[
        state_fold_summary["target_type"] == target_type
    ].copy()

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
            vals = sub.loc[
                (sub["scale"] == scale) &
                (sub["dtp_state"] == state),
                "fold_state_mean_effect"
            ].dropna().values

            values_by_state.append(vals)
            positions.append(base_positions[i] + offsets[scale])

        bp = ax.boxplot(
            values_by_state,
            positions=positions,
            widths=0.15,
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

        # Fold-level points
        for pos, vals in zip(positions, values_by_state):
            if len(vals) == 0:
                continue

            jitter = rng.normal(0, 0.020, size=len(vals))

            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                s=5,
                alpha=0.45,
                color=scale_colors[scale],
                edgecolor="none",
                zorder=3
            )

    ax.axhline(0, linestyle="--", linewidth=0.7, color="black", alpha=0.65)

    ax.set_xticks(base_positions)
    ax.set_xticklabels(
        [state_label[s] for s in state_order],
        rotation=35,
        ha="right"
    )

    # if show_ylabel:
    #     ax.set_ylabel("Fold mean effect")

    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------
# 6. Create GRL two-row figure
# --------------------------------------------------
fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(GRL_WIDTH, GRL_HEIGHT),
    sharey=False,
    gridspec_kw={
        "height_ratios": [0.85, 1.05],
        "hspace": 0.45,
        "wspace": 0.32
    }
)

# panel_letters = [
#     ["a", "b", "c", "d"],
#     ["e", "f", "g", "h"]
# ]

for col, target_type in enumerate(target_types):

    # Column title
    axes[0, col].set_title(target_label[target_type], pad=3)

    # -----------------------------
    # Row 1: Q75 -> Q90 forest plot
    # -----------------------------
    plot_forest_row(
        ax=axes[0, col],
        summary_df=main_summary,
        fold_df=main_fold_df,
        target_type=target_type,
        show_ylabel=(col == 0),
        error_type="ci95",
        show_fold_points=True,
        xlim=forest_xlim[target_type],
    )

    axes[0, col].set_xlabel(effect_unit_label[target_type], labelpad=1.5)

    # -----------------------------
    # Row 2: state dependence of Q75 -> Q90 effect
    # -----------------------------
    plot_state_boxplot(
        ax=axes[1, col],
        state_fold_summary=state_fold_summary,
        target_type=target_type,
        show_ylabel=(col == 0)
    )

    # axes[1, col].set_xlabel("Groundwater state", labelpad=1.5)

    # Panel letters
    # for row in range(2):
    #     axes[row, col].text(
    #         0.02,
    #         0.96,
    #         panel_letters[row][col],
    #         transform=axes[row, col].transAxes,
    #         ha="left",
    #         va="top",
    #         fontweight="bold"
    #     )

# Row labels on the far left
fig.text(
    0.09, 0.70,
    "High → Very hight state",
    rotation=90,
    va="center",
    ha="center",
    fontweight="bold"
)

fig.text(
    0.09, 0.28,
    "Fold mean effect",
    rotation=90,
    va="center",
    ha="center",
    fontweight="bold"
)

# Shared legend for groundwater scale
# legend_handles = [
#     Line2D(
#         [0], [0],
#         marker=scale_markers[scale],
#         color=scale_colors[scale],
#         label=scale_label[scale],
#         linestyle="none",
#         markersize=4.5,
#         markeredgecolor="black",
#         markeredgewidth=0.3
#     )
#     for scale in scales
# ]

# fig.legend(
#     handles=legend_handles,
#     loc="lower center",
#     ncol=3,
#     frameon=False,
#     bbox_to_anchor=(0.53, 0.005),
#     handletextpad=0.4,
#     columnspacing=1.0
# )

plt.tight_layout(rect=[0.035, 0.075, 1.0, 0.98])

out_fig = os.path.join(
    pkl_path,
    "figure3_GRL_q75_to_q90_two_rows.png")

plt.savefig(out_fig, dpi=1200, bbox_inches="tight")
plt.show()

print("Saved figure to:", out_fig)

# --------------------------------------------------
# 7. Save summary tables
# --------------------------------------------------

main_summary.to_csv(
    os.path.join(pkl_path, "figure3_q75_to_q90_summary.csv"),
    index=False
)

main_fold_df.to_csv(
    os.path.join(pkl_path, "figure3_q75_to_q90_fold_means.csv"),
    index=False
)

state_fold_summary.to_csv(
    os.path.join(pkl_path, "figure3_q75_to_q90_state_fold_summary.csv"),
    index=False
)

print("Saved summary tables.")


#%% heterogeneity analysis - box
def assign_season_from_month(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "autumn"
    else:
        return np.nan
    
def add_season_column(effects_df):
    """
    Adds a season column if it does not already exist.

    This function assumes one of the following columns exists:
    - season
    - month
    - event_date
    - date
    - flood_peak_date
    """

    if "season" in effects_df.columns:
        return effects_df

    if "month" in effects_df.columns:
        effects_df["season"] = effects_df["month"].apply(assign_season_from_month)
        return effects_df

    date_candidates = ["event_date", "date", "flood_peak_date", "rain_start", "response_start"]

    for col in date_candidates:
        if col in effects_df.columns:
            effects_df[col] = pd.to_datetime(effects_df[col])
            effects_df["month"] = effects_df[col].dt.month
            effects_df["season"] = effects_df["month"].apply(assign_season_from_month)
            return effects_df

    raise ValueError(
        "No season, month, or date column found. "
        "Please save event_date/month/season into effects_event_df when creating causal results."
    )


def get_effect_column(effects_df, target_type):
    """
    Finds the correct effect column.
    For your newer script, tau_report is generic.
    For older peak scripts, tau_percent may already exist.
    """

    if target_type == "occurrence":
        if "tau_percentage_points" in effects_df.columns:
            return "tau_percentage_points"
        elif "tau_report" in effects_df.columns:
            return "tau_report"
        else:
            raise ValueError("No occurrence effect column found.")

    else:
        if "tau_percent" in effects_df.columns:
            return "tau_percent"
        elif "tau_report" in effects_df.columns:
            return "tau_report"
        else:
            raise ValueError(f"No effect column found for {target_type}.")

def add_quantile_bins(df, col, q=4):
    """
    Add quantile bins for one catchment attribute.
    """
    bin_col = f"{col}_bin"

    df = df.copy()
    df[bin_col] = pd.qcut(
        df[col],
        q=q,
        duplicates="drop"
    )

    return df, bin_col


def add_qcut_bin(df, var, q=4):
    """
    Add quantile bins safely, even when duplicate values reduce the number of bins.
    """

    x = df[var]

    # First get bins without labels
    binned, bins = pd.qcut(
        x,
        q=q,
        retbins=True,
        duplicates="drop"
    )

    n_bins = len(bins) - 1

    if n_bins == 4:
        labels = ["Q1 low", "Q2", "Q3", "Q4 high"]
    else:
        labels = [f"Q{i+1}" for i in range(n_bins)]
        labels[0] = f"{labels[0]} low"
        labels[-1] = f"{labels[-1]} high"

    df[f"{var}_bin"] = pd.cut(
        x,
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


def summarize_effect_by_attribute_bin(catch_effects, attr_col, effect_col="mean_effect", q=4):
    d = catch_effects.dropna(subset=[attr_col, effect_col]).copy()
    d, bin_col = add_quantile_bins(d, attr_col, q=q)

    summary = (
        d
        .groupby(["target_type", bin_col], observed=True)[effect_col]
        .agg(
            n_catchments="count",
            mean="mean",
            median="median",
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
            p05=lambda x: np.percentile(x, 5),
            p95=lambda x: np.percentile(x, 95),
        )
        .reset_index()
        .rename(columns={bin_col: "attribute_bin"})
    )

    summary["attribute"] = attr_col

    return summary

def add_quantile_bins_with_ranges(df, col, q=4, decimals=2):
    df = df.copy()
    bin_col = f"{col}_bin"

    # Create quantile bins
    df[bin_col] = pd.qcut(
        df[col],
        q=q,
        duplicates="drop"
    )

    # Create readable range labels from the interval categories
    categories = df[bin_col].cat.categories

    labels = []
    for interval in categories:
        left = interval.left
        right = interval.right
        labels.append(f"{left:.{decimals}f}–{right:.{decimals}f}")

    # Rename interval categories to range labels
    df[bin_col] = df[bin_col].cat.rename_categories(labels)

    return df, bin_col, labels


# -----------------------------
# Load and combine results
# -----------------------------
# 0. load dataset
scales      = 'catch'
target_types = ["peak", "occurrence", "volume", "duration"]

target_labels = {
    "peak": "Peak discharge",
    "occurrence": "Flood occurrence",
    "volume": "Flood volume (Q > Q95)",
    "duration": "Flood Duration",
}

effect_labels = {
    "peak": "Effect on peak discharge (%)",
    "occurrence": "Effect on flood probability (percentage points)",
    "volume": "Effect on flood-volume burden (%)",
    "duration": "Effect on flood-duration burden (%)",
}


# load camels-dk dataset
_, _, attributes = load_camels_info()
feature_cols     = [
    "catch_id",
    "elev_median",
    "slope_median",
    "pct_wetlands_corine_2018",
    'BFI',"catch_area"]
keep_cols = ["target_type","scale","season","effect_value","fold"] + feature_cols

all_dfs = []
for target_type in target_types:
    pkl_file = f"results_pickle/causal_ml_results_{target_type}_{scale}.pkl"
    
    dataset = load_dataset(scale=scale) # catch, valley, downs
    dataset = dataset.dropna(subset=["flood_occurrence"]).reset_index(drop=True)
    with open(pkl_file, "rb") as f:
        obj = pickle.load(f)
    effects_df = obj["causal_event_results"].copy()
    effects_df["target_type"] = target_type
    effects_df["scale"] = scale
    effects_df["event_date"] = pd.to_datetime(dataset.loc[effects_df["row_index"], 'rain_start_date'])
    effects_df = add_season_column(effects_df)
    effect_col = get_effect_column(effects_df, target_type)
    effects_df["effect_value"] = effects_df[effect_col]

    # add additional figures
    effects_with_features = effects_df.merge(attributes[feature_cols],on="catch_id",how="left")
    
    all_dfs.append(effects_with_features[keep_cols])


# Combine all target results
effects_all = pd.concat(all_dfs, ignore_index=True)

catch_effects = (
    effects_all
    .groupby(["target_type", "scale", "catch_id"], observed=True)
    .agg(
        mean_effect=("effect_value", "mean"),
        median_effect=("effect_value", "median"),
        n_events=("effect_value", "count"),
        elev_median=("elev_median", "first"),
        slope_median=("slope_median", "first"),
        pct_wetlands_corine_2018=("pct_wetlands_corine_2018", "first"),
        BFI=("BFI", "first"),
        catch_area=("catch_area", "first"),
    )
    .reset_index()
)

# print(catch_effects.head())


# catch_effects["catch_area_km2"] = catch_effects["catch_area"] / 1e6
# attrs_to_plot = ["elev_median", "BFI", "catch_area_km2"]
# target_order = ["peak", "occurrence", "volume", "duration"]

# attr_labels = {
#     "elev_median": "Median elevation",
#     "BFI": "Baseflow index",
#     "catch_area": "Catchment area (km²)"
# }


# fig, axes = plt.subplots(
#     nrows=len(attrs_to_plot),
#     ncols=len(target_order),
#     figsize=(4.4 * len(target_order), 3.8 * len(attrs_to_plot)),
#     sharey=False, dpi=1200
# )

# for r, attr_col in enumerate(attrs_to_plot):

#     # Choose decimal places
#     if attr_col in ["BFI"]:
#         decimals = 2
#     else:
#         decimals = 1

#     d_attr = catch_effects.dropna(subset=[attr_col, "mean_effect"]).copy()
#     d_attr, bin_col, bin_labels = add_quantile_bins_with_ranges(
#         d_attr,
#         attr_col,
#         q=4,
#         decimals=decimals
#     )

#     for c, target in enumerate(target_order):
#         ax = axes[r, c]

#         d = d_attr[d_attr["target_type"] == target].copy()

#         data_to_plot = [
#             d.loc[d[bin_col] == b, "mean_effect"].dropna().values
#             for b in bin_labels
#         ]

#         ax.boxplot(
#             data_to_plot,
#             labels=bin_labels,
#             showfliers=False
#         )

#         # Add jittered catchment-level points
#         rng = np.random.default_rng(100 + r * 10 + c)

#         for i, b in enumerate(bin_labels, start=1):
#             vals = d.loc[d[bin_col] == b, "mean_effect"].dropna().values
#             x = rng.normal(i, 0.05, size=len(vals))
#             ax.scatter(x, vals, s=12, alpha=0.35)

#         ax.axhline(0, linestyle="--", linewidth=1)

#         if r == 0:
#             ax.set_title(target)

#         if c == 0:
#             ax.set_ylabel(f"{attr_labels.get(attr_col, attr_col)}\nMean effect")

#         ax.set_xlabel(attr_labels.get(attr_col, attr_col))
#         ax.tick_params(axis="x", rotation=35)

# plt.tight_layout()
# plt.show()
    

# all_attr_summaries = []

# for attr in attrs_to_plot:
#     s = summarize_effect_by_attribute_bin(
#         catch_effects,
#         attr_col=attr,
#         effect_col="mean_effect",
#         q=4
#     )
#     all_attr_summaries.append(s)

# attr_summary = pd.concat(all_attr_summaries, ignore_index=True)

# print(attr_summary)


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Attribute setup
# --------------------------------------------------

catch_effects["catch_area_km2"] = catch_effects["catch_area"] / 1e6

attrs_to_plot = ["elev_median", "BFI"] # , "catch_area_km2"
target_order = ["peak", "volume"]      #"occurrence",  , "duration"

target_labels = {
    "peak": "Peak discharge",
    "occurrence": "Flood occurrence",
    "volume": "Flood volume",
    "duration": "Flood duration",
}

effect_labels = {
    "peak": "Effect on peak discharge (%)",
    "occurrence": "Effect on flood probability (percentage points)",
    "volume": "Effect on flood-volume burden (%)",
    "duration": "Effect on flood-duration burden (%)",
}

attr_labels = {
    "elev_median": "Median elevation (m)",
    "BFI": "Baseflow index",
    "catch_area_km2": "Catchment area (km²)",
}

# --------------------------------------------------
# Helper: create common bins for each attribute
# --------------------------------------------------

def make_attribute_bins(df, attr_col, n_bins=8, binning="quantile"):
    """
    Make common bins for one attribute, shared across all targets.
    This ensures that peak and volume use exactly the same DEM/BFI bins.
    """

    d = df.dropna(subset=[attr_col]).copy()

    # Use unique catchments if catch_id exists, otherwise use all rows
    if "catch_id" in d.columns:
        d_unique = d.drop_duplicates(subset=["catch_id"])
    else:
        d_unique = d.copy()

    if binning == "quantile":
        _, bins = pd.qcut(
            d_unique[attr_col],
            q=n_bins,
            retbins=True,
            duplicates="drop"
        )

    elif binning == "equal_width":
        _, bins = pd.cut(
            d_unique[attr_col],
            bins=n_bins,
            retbins=True,
            include_lowest=True
        )

    else:
        raise ValueError("binning must be 'quantile' or 'equal_width'")

    # slightly extend edges to avoid boundary problems
    bins[0] = bins[0] - 1e-9
    bins[-1] = bins[-1] + 1e-9

    return bins


# --------------------------------------------------
# Helper: summarize target effects using fixed bins
# --------------------------------------------------
def binned_effect_summary_fixed_bins(
    df,
    attr_col,
    bins,
    effect_col="mean_effect"
):
    """
    Summarize catchment-level causal effects using pre-defined bins.
    """

    d = df.dropna(subset=[attr_col, effect_col]).copy()

    d["attr_bin"] = pd.cut(
        d[attr_col],
        bins=bins,
        include_lowest=True
    )

    # bin center
    bin_centers = []
    for interval in d["attr_bin"].cat.categories:
        bin_centers.append((interval.left + interval.right) / 2)

    center_map = {
        interval: center
        for interval, center in zip(d["attr_bin"].cat.categories, bin_centers)
    }

    d["bin_center"] = d["attr_bin"].map(center_map).astype(float)

    summary = (
        d
        .groupby("attr_bin", observed=True)
        .agg(
            bin_center=("bin_center", "first"),
            n_catchments=(effect_col, "count"),
            mean_effect=(effect_col, "mean"),
            median_effect=(effect_col, "median"),
            p25_effect=(effect_col, lambda x: np.percentile(x, 25)),
            p75_effect=(effect_col, lambda x: np.percentile(x, 75)),
            p05_effect=(effect_col, lambda x: np.percentile(x, 5)),
            p95_effect=(effect_col, lambda x: np.percentile(x, 95)),
            attr_min=(attr_col, "min"),
            attr_max=(attr_col, "max"),
        )
        .reset_index()
    )

    return summary, d

# --------------------------------------------------
# Helper: bin attribute and summarize effect
# --------------------------------------------------
def plot_attribute_effect_dual_left_axes(
    catch_effects,
    attrs_to_plot,
    target_peak="peak",
    target_volume="volume",
    n_bins=8,
    binning="quantile",
    use_iqr_band=True,
    show_points=True
):
    """
    One subplot per spatial attribute.

    x-axis: attribute bins, e.g. elevation or BFI
    left y-axis 1: peak-discharge effect
    left y-axis 2: flood-volume effect, offset to the left
    right y-axis: number of catchments per bin

    This is useful when peak and volume effects have different magnitudes.
    """

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(attrs_to_plot),
        figsize=(5.8 * len(attrs_to_plot), 4.2),
        dpi=600,
        sharey=False
    )

    if len(attrs_to_plot) == 1:
        axes = [axes]

    for i, attr_col in enumerate(attrs_to_plot):

        ax_peak = axes[i]

        # --------------------------------------------------
        # Make common bins for this attribute
        # --------------------------------------------------
        bins = make_attribute_bins(
            catch_effects,
            attr_col=attr_col,
            n_bins=n_bins,
            binning=binning
        )

        # --------------------------------------------------
        # Catchment count bars, calculated once
        # --------------------------------------------------
        d_count = catch_effects.dropna(subset=[attr_col]).copy()

        if "catch_id" in d_count.columns:
            d_count = d_count.drop_duplicates(subset=["catch_id"])

        d_count["attr_bin"] = pd.cut(
            d_count[attr_col],
            bins=bins,
            include_lowest=True
        )

        count_summary = (
            d_count
            .groupby("attr_bin", observed=True)
            .agg(
                n_catchments=(attr_col, "count")
            )
            .reset_index()
        )

        count_summary["bin_center"] = count_summary["attr_bin"].apply(
            lambda interval: (interval.left + interval.right) / 2
        ).astype(float)

        x_count = count_summary["bin_center"].values
        counts = count_summary["n_catchments"].values

        if len(x_count) > 1:
            bar_width = np.nanmedian(np.diff(np.sort(x_count))) * 0.75
        else:
            bar_width = 0.1

        # Right axis for catchment count
        ax_count = ax_peak.twinx()

        ax_count.bar(
            x_count,
            counts,
            width=bar_width,
            alpha=0.14,
            color="grey",
            edgecolor="none",
            label="Catchment count"
        )

        ax_count.set_ylabel("No. catchments", fontsize=9, color="grey")
        ax_count.tick_params(axis="y", labelsize=8, colors="grey")
        ax_count.grid(False)

        # --------------------------------------------------
        # Second left axis for volume effect
        # --------------------------------------------------
        ax_volume = ax_peak.twinx()

        # Move volume axis from right to left, slightly outward
        ax_volume.spines["right"].set_visible(False)
        ax_volume.spines["left"].set_position(("outward", 52))
        ax_volume.spines["left"].set_visible(True)
        ax_volume.yaxis.set_label_position("left")
        ax_volume.yaxis.set_ticks_position("left")

        # --------------------------------------------------
        # Peak effect summary
        # --------------------------------------------------
        d_peak = catch_effects[
            catch_effects["target_type"] == target_peak
        ].dropna(subset=[attr_col, "mean_effect"]).copy()

        summary_peak, d_peak_binned = binned_effect_summary_fixed_bins(
            d_peak,
            attr_col=attr_col,
            bins=bins,
            effect_col="mean_effect"
        )

        x_peak = summary_peak["bin_center"].values
        y_peak = summary_peak["mean_effect"].values
        y_peak25 = summary_peak["p25_effect"].values
        y_peak75 = summary_peak["p75_effect"].values

        if use_iqr_band:
            ax_peak.fill_between(
                x_peak,
                y_peak25,
                y_peak75,
                alpha=0.16,
                color="tab:blue",
                linewidth=0
            )

        line_peak = ax_peak.plot(
            x_peak,
            y_peak,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color="tab:blue",
            label=target_labels.get(target_peak, target_peak)
        )

        # --------------------------------------------------
        # Volume effect summary
        # --------------------------------------------------
        d_volume = catch_effects[
            catch_effects["target_type"] == target_volume
        ].dropna(subset=[attr_col, "mean_effect"]).copy()

        summary_volume, d_volume_binned = binned_effect_summary_fixed_bins(
            d_volume,
            attr_col=attr_col,
            bins=bins,
            effect_col="mean_effect"
        )

        x_volume = summary_volume["bin_center"].values
        y_volume = summary_volume["mean_effect"].values
        y_volume25 = summary_volume["p25_effect"].values
        y_volume75 = summary_volume["p75_effect"].values

        if use_iqr_band:
            ax_volume.fill_between(
                x_volume,
                y_volume25,
                y_volume75,
                alpha=0.14,
                color="tab:orange",
                linewidth=0
            )

        line_volume = ax_volume.plot(
            x_volume,
            y_volume,
            marker="s",
            linewidth=2.2,
            markersize=5,
            color="tab:orange",
            label=target_labels.get(target_volume, target_volume)
        )

        # --------------------------------------------------
        # Optional individual catchment points
        # --------------------------------------------------
        if show_points:
            rng = np.random.default_rng(1000 + i)

            jitter_scale = 0.01 * (
                catch_effects[attr_col].max() - catch_effects[attr_col].min()
            )

            x_peak_jitter = d_peak[attr_col].values + rng.normal(
                0,
                jitter_scale,
                size=len(d_peak)
            )

            ax_peak.scatter(
                x_peak_jitter,
                d_peak["mean_effect"].values,
                s=8,
                alpha=0.13,
                color="tab:blue",
                linewidths=0
            )

            x_volume_jitter = d_volume[attr_col].values + rng.normal(
                0,
                jitter_scale,
                size=len(d_volume)
            )

            ax_volume.scatter(
                x_volume_jitter,
                d_volume["mean_effect"].values,
                s=8,
                alpha=0.13,
                color="tab:orange",
                linewidths=0
            )

        # --------------------------------------------------
        # Zero lines
        # --------------------------------------------------
        ax_peak.axhline(
            0,
            linestyle="--",
            linewidth=1,
            color="tab:blue",
            alpha=0.6
        )

        ax_volume.axhline(
            0,
            linestyle=":",
            linewidth=1,
            color="tab:orange",
            alpha=0.6
        )

        # --------------------------------------------------
        # Axis labels and formatting
        # --------------------------------------------------
        ax_peak.set_xlabel(
            attr_labels.get(attr_col, attr_col),
            fontsize=10
        )

        ax_peak.set_ylabel(
            "Average effect on peak discharge (%)",
            fontsize=10,
            color="tab:blue"
        )

        ax_volume.set_ylabel(
            "Average effect on flood volume (%)",
            fontsize=10,
            color="tab:orange"
        )

        ax_peak.tick_params(axis="y", labelsize=8, colors="tab:blue")
        ax_volume.tick_params(axis="y", labelsize=8, colors="tab:orange")
        ax_peak.tick_params(axis="x", labelsize=8)

        ax_peak.spines["left"].set_color("tab:blue")
        ax_volume.spines["left"].set_color("tab:orange")
        ax_count.spines["right"].set_color("grey")

        # ax_peak.set_title(
        #     attr_labels.get(attr_col, attr_col),
        #     fontsize=11
        # )

        if attr_col == "catch_area_km2":
            ax_peak.set_xscale("log")
            ax_volume.set_xscale("log")
            ax_count.set_xscale("log")

        # --------------------------------------------------
        # Combined legend
        # --------------------------------------------------
        handles = line_peak + line_volume
        labels = [
            target_labels.get(target_peak, target_peak),
            target_labels.get(target_volume, target_volume)
        ]

        # Add count bar handle
        bar_handle = ax_count.patches[0] if len(ax_count.patches) > 0 else None
        if bar_handle is not None:
            handles.append(bar_handle)
            labels.append("Catchment count")

        ax_peak.legend(
            handles,
            labels,
            fontsize=8,
            frameon=False,
            loc="best"
        )

    plt.tight_layout()
    return fig, axes

fig, axes = plot_attribute_effect_dual_left_axes(
    catch_effects=catch_effects,
    attrs_to_plot=["elev_median", "BFI"],
    target_peak="peak",
    target_volume="volume",
    n_bins=20,
    binning="equal_width",
    use_iqr_band=True,
    show_points=True
)

plt.show()

