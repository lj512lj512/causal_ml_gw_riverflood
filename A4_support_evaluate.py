# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:44:51 2026

@author: juliu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_well_metrics(group):
    data = (
        group[["obs_value", "sim_cell"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    obs = data["obs_value"].to_numpy(dtype=float)
    sim = data["sim_cell"].to_numpy(dtype=float)
    n = len(data)

    if n < 3:
        return pd.Series({
            "n": n,
            "R2": np.nan,
            "Spearman_rho": np.nan,
            "RMSE": np.nan,
            "relative_RMSE": np.nan,
            "bias": np.nan,
        })

    # Absolute groundwater levels
    residual = sim - obs
    rmse = np.sqrt(np.mean(residual**2))
    bias = np.mean(residual)

    # Remove each series' own mean
    obs_relative = obs - np.mean(obs)
    sim_relative = sim - np.mean(sim)

    relative_rmse = np.sqrt(
        np.mean((sim_relative - obs_relative) ** 2)
    )

    if np.std(obs) > 0 and np.std(sim) > 0:
        spearman_rho = spearmanr(obs, sim).statistic
    else:
        spearman_rho = np.nan

    if np.var(obs) > 0:
        r2 = r2_score(obs, sim)
    else:
        r2 = np.nan

    return pd.Series({
        "n": n,
        "R2": r2,
        "Spearman_rho": spearman_rho,
        "RMSE": rmse,
        "relative_RMSE": relative_rmse,
        "bias": bias,
    })





#%% read observations

# -------------------------------------------------------
# Time-series plotting function
# -------------------------------------------------------
def plot_well_timeseries(ax, well_id):
    well_data = (
        plot_data.loc[
            plot_data["OBS_ID"] == well_id,
            ["dato", "obs_value", "dtp_dkm"],
        ]
        .sort_values("dato")
        .copy()
    )

    # Mean-center observations and simulations separately
    well_data["obs_relative"] = (
        well_data["obs_value"]
        - well_data["obs_value"].mean()
    )

    well_data["sim_relative"] = (
        well_data["dtp_dkm"]
        - well_data["dtp_dkm"].mean()
    )

    ax.plot(
        well_data["dato"],
        well_data["obs_relative"],
        linewidth=0.8,
        label="Observed anomaly",
    )

    ax.plot(
        well_data["dato"],
        well_data["sim_relative"],
        linewidth=0.8,
        alpha=0.85,
        label="Simulated anomaly",
    )

    # Zero represents the well-specific mean
    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.6,
        alpha=0.7,
    )

    # Get metrics for this well
    metric_row = well_metrics.loc[
        well_metrics["OBS_ID"] == well_id
    ]

    if not metric_row.empty:
        rho = metric_row["Spearman_rho"].iloc[0]
        relative_rmse = metric_row["relative_RMSE"].iloc[0]
        n = int(metric_row["n"].iloc[0])

        title = (
            f"Well {well_id} "
            f"(n = {n}, "
            rf"$\rho$ = {rho:.2f}, "
            f"RMSE = {relative_rmse:.2f} m)"
        )
    else:
        title = f"Well {well_id}"

    ax.set_title(
        title,
        loc="left",
    )

    ax.set_ylabel(
        "Groundwater depth anomaly (m)",
    )

    ax.grid(
        alpha=0.25,
        linewidth=0.4,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
    )

    ax.tick_params(
        axis="both",
        width=0.5,
        length=2,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        
        
# Load dataset
datapath= r'\\geodata.geus.dk\DKmodel_users\FloodWarning\GWH_emulator\Hydro03_E_temporary\juliu\GWH_hybrid'
# dataset = pd.read_csv(os.path.join(datapath, 
#                                    'lightGBM', 
#                                    'Dataset_for_ML_obs_length406150_DK123456_500m.csv'), index_col='Unnamed: 0')

paired_data = (
    dataset
    .dropna(subset=["OBS_ID", "obs_value", "dtp_dkm"])
    .copy()
)

well_counts = paired_data.groupby("OBS_ID").size()

selected_wells = well_counts[well_counts > 100].index

dataset_100 = (
    paired_data[
        paired_data["OBS_ID"].isin(selected_wells)
    ]
    .copy()
)

well_metrics = (
    dataset_100
    .groupby("OBS_ID")
    .apply(calculate_well_metrics, include_groups=False)
    .reset_index()
)

# Ensure dates are datetime
dataset_100["dato"] = pd.to_datetime(
    dataset_100["dato"],
    errors="coerce",
)

# Keep valid paired observations and simulations
plot_data = (
    dataset_100
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["OBS_ID", "dato", "obs_value", "dtp_dkm"])
    .copy()
)

# Identify the two wells with the most valid paired records
two_longest_wells = (
    plot_data
    .groupby("OBS_ID")
    .size()
    .nlargest(2)
    .index
    .tolist()
)
print("Two longest wells:", two_longest_wells)

# -------------------------------------------------------
# Figure layout
# -------------------------------------------------------
fig = plt.figure(
    figsize=(7.5, 6.0),
    constrained_layout=True,
)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 10,
})

gs = fig.add_gridspec(
    nrows=3,
    ncols=3,
    height_ratios=[1.0, 1.4, 1.4],
)

# Top-row boxplots
ax_rho = fig.add_subplot(gs[0, 0])
ax_rmse = fig.add_subplot(gs[0, 1])
ax_relative = fig.add_subplot(gs[0, 2])

# Time series spanning all columns
ax_well1 = fig.add_subplot(gs[1, :])
ax_well2 = fig.add_subplot(gs[2, :])

# -------------------------------------------------------
# Boxplot 1: Spearman correlation
# -------------------------------------------------------
ax_rho.boxplot(
    well_metrics["Spearman_rho"].dropna(),
    widths=0.45,
    showfliers=True,
    showmeans=True,
)

ax_rho.set_xticks([1])
ax_rho.set_xticklabels([])
ax_rho.set_ylabel(r"Spearman $\rho$", fontsize=12)
ax_rho.axhline(
    0,
    linestyle="--",
    linewidth=0.7,
    alpha=0.7,
)
ax_rho.grid(
    axis="y",
    alpha=0.25,
    linewidth=0.4,
)
ax_rho.tick_params(axis="both")

# -------------------------------------------------------
# Boxplot 2: Absolute RMSE
# -------------------------------------------------------
ax_rmse.boxplot(
    well_metrics["RMSE"].dropna(),
    widths=0.45,
    showfliers=True,
    showmeans=True,
)

ax_rmse.set_xticks([1])
ax_rmse.set_xticklabels([])
ax_rmse.set_ylabel("GWD RMSE (m)")
ax_rmse.grid(
    axis="y",
    alpha=0.25,
    linewidth=0.4,
)
ax_rmse.tick_params(axis="both")
# -------------------------------------------------------
# Boxplot 3: Mean-centered RMSE
# -------------------------------------------------------
ax_relative.boxplot(
    well_metrics["relative_RMSE"].dropna(),
    widths=0.45,
    showfliers=True,
    showmeans=True,
)

ax_relative.set_xticks([1])
ax_relative.set_xticklabels([])
ax_relative.set_ylabel("Relative GWD RMSE (m)")
ax_relative.grid(
    axis="y",
    alpha=0.25,
    linewidth=0.4,
)
ax_relative.tick_params(axis="both")

# Plot the two longest wells
plot_well_timeseries(ax_well1, two_longest_wells[0])
plot_well_timeseries(ax_well2, two_longest_wells[1])

# ------------------------------------------------------
# General formatting
# -------------------------------------------------------
for ax in [ax_rho, ax_rmse, ax_relative]:
    ax.tick_params(
        axis="both",
        width=0.5,
        length=2,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

# Optional panel labels
panel_axes = [
    ax_rho,
    ax_rmse,
    ax_relative,
    ax_well1,
    ax_well2,
]

plt.show()


#%% statistics R2 and RMSE


#%% plot two time series



