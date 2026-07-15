# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:12:50 2026

@author: juliu
"""

# ==================================================
# National groundwater-sensitivity analysis
# ==================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely import force_2d
from matplotlib.patches import Patch
from econml.dml import NonParamDML
from lightgbm import LGBMRegressor
from datetime import datetime
import ee
import geemap

from A2_causal_model_train import load_dataset, make_model_y, make_model_t


# ==================================================
# User settings
# ==================================================

RUN_EE_EXTRACTION = False

SCALE = "catch"  # "catch", "valley", "downs"
TARGET_TYPES = ["peak", "volume"]

BASE_DIR = Path(r"\\geodata.geus.dk\DKmodel_users\FloodWarning")

DYNAMIC_DIR = BASE_DIR / (
    r"LSTM_postprocessing\CAMELS-DK\Level_3\Dynamics\Ungauged_catchments"
)

CATCHMENT_SHP = BASE_DIR / (
    r"LSTM_postprocessing\CAMELS-DK\Level_3\Shapefile"
    r"\CAMELS_DK_3330_catchment_boundaries.shp"
)

DK_SHP = Path(
    r"\\geodata.geus.dk\HOME\Discharge_correction_with_LSTM"
    r"\Shapfile\DKDomains2013\kort10_land_DK.shp"
)

OUT_DIR = Path(r"results_pickle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FEATURE_CSV = OUT_DIR / "all_catchments_groundwater_spatial_features.csv"
OUT_MAP_FIG = OUT_DIR / "national_groundwater_sensitivity_maps.png"


# ==================================================
# Feature names
# ==================================================

ealpha_e_features = [
    "A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07",
    "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15",
    "A16", "A17", "A18", "A19", "A20", "A21", "A22", "A23",
    "A24", "A25", "A26", "A27", "A28", "A29", "A30", "A31",
    "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39",
    "A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47",
    "A48", "A49", "A50", "A51", "A52", "A53", "A54", "A55",
    "A56", "A57", "A58", "A59", "A60", "A61", "A62", "A63",
]

CONFOUNDERS = [
    "rain_duration_days",
    "event_precip_mm",
    "max_1d_precip_mm",
    "split_from_long_event",
    "wcr",
    "sdy",
    "cdy",
    "tem",
    "pet",
]

HETEROGENEITY_FEATURES = ealpha_e_features


# ==================================================
# Effect-bin settings
# ==================================================

PEAK_LABELS = ["<0%", "0–2%", "2–4%", "4–6%", "6–8%", ">8%"]
PEAK_BINS = [-np.inf, 0, 2, 4, 6, 8, np.inf]

VOLUME_LABELS = ["<0%", "0–1%", "1–2%", "2–3%", "3–4%", ">4%"]
VOLUME_BINS = [-np.inf, 0, 1, 2, 3, 4, np.inf]

COMBINED_LABELS = ["0–25%", "25–50%", "50–75%", "75–90%", "90–100%"]
COMBINED_BINS = [0, 0.25, 0.50, 0.75, 0.90, 1.00]

# Warm sequential palette: light = weak sensitivity, dark = strong sensitivity
EFFECT_COLORS_6 = {
    "<0%":  "#bdbdbd",
    "0–2%": "#f0f9e8",
    "2–4%": "#bae4bc",
    "4–6%": "#7bccc4",
    "6–8%": "#43a2ca",
    ">8%": "#0868ac",

    "0–1%": "#f0f9e8",
    "1–2%": "#bae4bc",
    "2–3%": "#7bccc4",
    "3–4%": "#43a2ca",
    ">4%": "#0868ac",
}

COMBINED_COLORS = {
    "0–25%": "#fff7ec",
    "25–50%": "#fee8c8",
    "50–75%": "#fdbb84",
    "75–90%": "#e34a33",
    "90–100%": "#7f0000",
}

TYPE_ORDER = ["Both high", "Peak", "Volume", "Mixed"]

TYPE_COLORS = {
    "Both high": "#6a51a3",
    "Peak": "#2b8cbe",
    "Volume": "#f03b20",
    "Mixed": "#bdbdbd",
}


# ==================================================
# Helper functions: Earth Engine
# ==================================================

def initialize_earth_engine():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def get_ee_features_for_catchment(catch, year=2020):

    # Reproject to WGS84
    catchments = catch.to_crs(epsg=4326)
    
    # Drop Z dimension
    catch["geometry"] = catch.geometry.apply(force_2d)
    
    # to ee 
    catchments_ee = geemap.geopandas_to_ee(catch)
    
    # Load AlphaEarth / satellite embedding annual image
    alphaearth = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"2020-01-01", f"2020-12-31")
        .filterBounds(catchments_ee.geometry())
        .mosaic() # combine all matching images into one image
    )
    
    # Extract catchment mean embeddings
    stats = alphaearth.reduceRegions(
        collection=catchments_ee,
        reducer=ee.Reducer.mean(),
        scale=10
    )
    
    # data frame 
    df_alpha = geemap.ee_to_df(stats, remove_geom=False)
    
    return df_alpha


# ==================================================
# Step 1: Prepare national catchment predictors
# ==================================================
import xarray as xr 
def load_jumpfilled_dtp():
    resolution = 500
    dataPath = r'\\geodata.geus.dk\DKmodel_users\FloodWarning\GWH_emulator\Hydro03_E_temporary\juliu\GWH_emulator\Dataset\DKM_sim'
    
    data_all = []
    for DKM_id in range(1, 7):
        data = xr.load_dataset(os.path.join(dataPath, 
                                            f'DK{DKM_id}_HIP_{resolution}m_2DSZ_dtp_jumps_filled_1989_2024.nc'))
        data_all.append(data)
    
    return xr.merge(data_all).sel(time = slice(datetime(1989, 1, 1), datetime(2019, 12, 31)))
        

def build_national_feature_table(
    catchments_gdf,
    dynamic_dir,
    dtp, 
    out_csv,
    run_ee_extraction=False,
):
    if out_csv.exists() and not run_ee_extraction:
        print(f"Loading existing feature table: {out_csv}")
        return pd.read_csv(out_csv)

    if run_ee_extraction:
        initialize_earth_engine()

    all_rows = []
    catch_ids = catchments_gdf["catch_id"].astype(str).tolist()

    for i, catch_id in enumerate(catch_ids):
        print(f"{i + 1}/{len(catchments_gdf)} catch_id={catch_id}")

        # ts_path = dynamic_dir / f"CAMELS_DK_sim_based_{catch_id}.csv"
        # if not ts_path.exists():
        #     print(f"  Missing dynamic file, skipping: {ts_path}")
        #     continue
        
        # get dtp 
        catch = catchments_gdf[catchments_gdf["catch_id"] == catch_id].geometry
        try:
            dyn = dtp.rio.clip(catch).mean(dim=('x', 'y')).to_dataframe().rename(columns = {'dtp': 'DKM_dtp'})
        except:
            print('No data found in bounds. Data variable: dtp')
            continue
        # dyn = pd.read_csv(ts_path, index_col="time", parse_dates=True)
        dtp_raw = dyn['DKM_dtp'].dropna()
        dtp_anom = dtp_raw - dtp_raw.mean()

        row = {
            "catch_id": str(catch_id),
            "dtp_q75_value": dtp_anom.quantile(0.75),
            "dtp_q90_value": dtp_anom.quantile(0.90),
            "dtp_q75_q90_delta": dtp_anom.quantile(0.90) - dtp_anom.quantile(0.75),
            "dtp_mean": dtp_raw.mean(),
            "dtp_std": dtp_anom.std(),
            "n_dtp_days": len(dtp_anom),
        }

        if run_ee_extraction:
            catch_geom = catchments_gdf.loc[catchments_gdf["catch_id"].astype(str) == str(catch_id)].copy()

            alpha_dict = get_ee_features_for_catchment(catch_geom)[ealpha_e_features].iloc[0].to_dict()
            row.update(alpha_dict)

        all_rows.append(row)

    all_catchments_df = pd.DataFrame(all_rows)
    all_catchments_df.to_csv(out_csv, index=False)

    print(f"Saved national feature table: {out_csv}")
    return all_catchments_df


# ==================================================
# Step 2: Outcome preparation
# ==================================================
def prepare_outcome(dataset, target_type):
    if target_type == "peak":
        dataset["peak_discharge_log"] = np.log(dataset["flood_peak"] + 1e-6)

        return {
            "dataset": dataset,
            "outcome": "peak_discharge_log",
            "raw_outcome": "flood_peak",
            "is_binary_outcome": False,
            "is_log_outcome": True,
        }

    elif target_type == "volume":
        dataset["flood_volume_above_q95_log"] = np.log1p(
            dataset["flood_volume_above_q95"]
        )

        print(
            "Percentage of events with flood_volume_above_q95 > 0:",
            100 * (dataset["flood_volume_above_q95"] > 0).mean()
        )

        return {
            "dataset": dataset,
            "outcome": "flood_volume_above_q95_log",
            "raw_outcome": "flood_volume_above_q95",
            "is_binary_outcome": False,
            "is_log_outcome": True,
        }

    else:
        raise ValueError(f"Unsupported target_type: {target_type}")


# ==================================================
# Step 3: Train final causal model
# ==================================================
def train_final_causal_model(
    target_type,
    scale,
    confounders,
    heterogeneity_features,
    random_state=752
):
    print(f"\nTraining final causal model: target={target_type}, scale={scale}")

    dataset = load_dataset(scale=scale)
    dataset = dataset.dropna(subset=["flood_occurrence"]).reset_index(drop=True)

    outcome_info = prepare_outcome(dataset, target_type)

    dataset = outcome_info["dataset"]
    outcome = outcome_info["outcome"]
    is_binary_outcome = outcome_info["is_binary_outcome"]

    required_cols = [outcome, "dtp"] + confounders + heterogeneity_features
    dataset = dataset.dropna(subset=required_cols).reset_index(drop=True)

    model_y = make_model_y(target_type=target_type, random_state=random_state)
    model_t = make_model_t(random_state=random_state + 100)

    model_final = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state + 200,
        n_jobs=48,
        verbose=-1
    )

    cf = NonParamDML(
        model_y=model_y,
        model_t=model_t,
        model_final=model_final,
        discrete_treatment=False,
        discrete_outcome=is_binary_outcome,
        cv=3,
        random_state=random_state + 300
    )

    cf.fit(
        Y=dataset[outcome].values,
        T=dataset["dtp"].values,
        W=dataset[confounders].values,
        X=dataset[heterogeneity_features].values
    )

    return cf, outcome_info


# ==================================================
# Step 4: Predict national Q90 sensitivity
# ==================================================
def predict_q90_sensitivity(
    model,
    all_catchments_df,
    heterogeneity_features,
    target_type
):
    required_cols = heterogeneity_features + ["dtp_q90_value"]

    missing = [c for c in required_cols if c not in all_catchments_df.columns]
    if missing:
        raise ValueError(f"Missing columns in national feature table: {missing}")

    d = all_catchments_df.dropna(subset=required_cols).copy()

    X_all = d[heterogeneity_features].values
    T0 = d["dtp_q90_value"].values
    T1 = d["dtp_q90_value"].values + 0.1

    tau = model.effect(X_all, T0=T0, T1=T1)
    tau = np.asarray(tau).ravel()

    sensitivity = (np.exp(tau) - 1) * 100

    out = d[["catch_id"]].copy()
    out[f"{target_type}_q90_plus10"] = sensitivity

    return out


# ==================================================
# Step 5: Effect bins and vulnerability type
# ==================================================
def add_vulnerability_metrics_effect_bins(map_gdf):
    required_cols = ["peak_q90_plus10", "volume_q90_plus10"]
    missing = [c for c in required_cols if c not in map_gdf.columns]

    if missing:
        raise ValueError(f"Missing sensitivity columns: {missing}")

    # Effect bins for maps a and b
    map_gdf["peak_effect_bin"] = pd.cut(
        map_gdf["peak_q90_plus10"],
        bins=PEAK_BINS,
        labels=PEAK_LABELS,
        include_lowest=True
    )

    map_gdf["volume_effect_bin"] = pd.cut(
        map_gdf["volume_q90_plus10"],
        bins=VOLUME_BINS,
        labels=VOLUME_LABELS,
        include_lowest=True
    )

    # Ranks are only used for combined vulnerability and vulnerability type
    map_gdf["peak_rank"] = map_gdf["peak_q90_plus10"].rank(pct=True)
    map_gdf["volume_rank"] = map_gdf["volume_q90_plus10"].rank(pct=True)

    map_gdf["combined_vulnerability_score"] = (
        map_gdf["peak_rank"] + map_gdf["volume_rank"]
    ) / 2

    map_gdf["combined_effect_bin"] = pd.cut(
        map_gdf["combined_vulnerability_score"],
        bins=COMBINED_BINS,
        labels=COMBINED_LABELS,
        include_lowest=True
    )

    rank_diff = map_gdf["peak_rank"] - map_gdf["volume_rank"]

    conditions = [
        (map_gdf["peak_rank"] >= 0.75) & (map_gdf["volume_rank"] >= 0.75),
        rank_diff >= 0.25,
        rank_diff <= -0.25,
    ]

    choices = ["Both high", "Peak", "Volume"]

    map_gdf["vulnerability_type"] = np.select(
        conditions,
        choices,
        default="Mixed"
    )

    return map_gdf


# ==================================================
# Step 6: Mapping helpers
# ==================================================
def subplot_setup(ax, extent, edgecolor="black"):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])

    for spine in ax.spines.values():
        spine.set_edgecolor(edgecolor)


def plot_categorical_map(
    gdf,
    ax,
    column,
    order,
    colors,
    alpha=0.94,
    zorder=10
):
    for label in order:
        subset = gdf[gdf[column] == label]

        if len(subset) == 0:
            continue

        subset.plot(
            ax=ax,
            facecolor=colors[label],
            edgecolor="none",
            linewidth=0,
            alpha=alpha,
            zorder=zorder
        )


def add_circular_percent_legend(
    ax,
    counts,
    colors,
    order,
    title,
    loc=(0.60, 0.56, 0.45, 0.45),
    fontsize=5.5
):
    """
    Circular bar legend.
    Bar height = percentage of catchments in each bin.
    Text = percentage only, no class words.
    """

    counts = counts.reindex(order).fillna(0)
    total = counts.sum()

    inset = ax.inset_axes(loc, projection="polar")

    values = counts.values.astype(float)

    if total == 0:
        percentages = np.zeros_like(values)
    else:
        percentages = values / total * 100

    if percentages.max() == 0:
        heights = percentages
    else:
        heights = 0.55 * percentages / percentages.max()

    n = len(order)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bar_width = 2 * np.pi / n * 0.72
    bottom = 0.35

    for t, h, label, pct in zip(theta, heights, order, percentages):
        inset.bar(
            t,
            h,
            width=bar_width,
            bottom=bottom,
            color=colors[label],
            edgecolor="white",
            linewidth=0.35
        )

        inset.text(
            t,
            bottom + h + 0.13,
            f"{pct:.0f}%",
            ha="center",
            va="center",
            fontsize=fontsize
        )


    inset.set_ylim(0, 1.18)
    inset.set_axis_off()


def add_subplot_legend_below(
    ax,
    labels,
    colors,
    title=None,
    ncol=None,
    fontsize=5.8,
    title_fontsize=6.2,
    y_offset=-0.06):
    """
    Add a discrete color legend below one subplot.

    y_offset controls how far below the subplot the legend appears.
    """

    if ncol is None:
        ncol = min(len(labels), 3)

    handles = [
        Patch(
            facecolor=colors[label],
            edgecolor="none",
            label=label
        )
        for label in labels
    ]

    ax.legend(
        handles=handles,
        title=title,
        loc="upper center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        handlelength=1.0,
        handleheight=0.7,
        columnspacing=0.7,
        labelspacing=0.35,
        borderaxespad=0.0
    )


def make_four_panel_sensitivity_map(
    map_gdf,
    catchment_polygons,
    dk_gdf,
    out_file=None):
    extent = {
        "DK": [[439000, 739000], [6049000, 6404000]],
    }

    fig, axes = plt.subplots(
        1,
        4,
        sharex=True,
        sharey=True,
        figsize=(9.5, 8.5),
        dpi=600,
        constrained_layout=False
    )

    panel_titles = [
        "(a) Peak sensitivity",
        "(b) Volume sensitivity",
        "(c) Combined vulnerability",
        "(d) Vulnerability type",
    ]

    for ax, title in zip(axes, panel_titles):
        subplot_setup(ax, extent["DK"], edgecolor="black")

        dk_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=0.35,
            zorder=5
        )

        catchment_polygons.plot(
            ax=ax,
            facecolor="none",
            edgecolor="gray",
            linewidth=0.05,
            zorder=6
        )

        # ax.set_title(title, fontsize=8, pad=2)

    # --------------------------------------------------
    # Panel a: peak effect bins
    # --------------------------------------------------
    plot_categorical_map(
        gdf=map_gdf,
        ax=axes[0],
        column="peak_effect_bin",
        order=PEAK_LABELS,
        colors=EFFECT_COLORS_6
    )

    peak_counts = (
        map_gdf["peak_effect_bin"]
        .value_counts()
        .reindex(PEAK_LABELS)
        .fillna(0)
    )

    add_circular_percent_legend(
        ax=axes[0],
        counts=peak_counts,
        colors=EFFECT_COLORS_6,
        order=PEAK_LABELS,
        title="Peak",
        loc=(0.56, 0.54, 0.45, 0.45)
    )

    # --------------------------------------------------
    # Panel b: volume effect bins
    # --------------------------------------------------
    plot_categorical_map(
        gdf=map_gdf,
        ax=axes[1],
        column="volume_effect_bin",
        order=VOLUME_LABELS,
        colors=EFFECT_COLORS_6
    )

    volume_counts = (
        map_gdf["volume_effect_bin"]
        .value_counts()
        .reindex(VOLUME_LABELS)
        .fillna(0)
    )

    add_circular_percent_legend(
        ax=axes[1],
        counts=volume_counts,
        colors=EFFECT_COLORS_6,
        order=VOLUME_LABELS,
        title="Volume",
        loc=(0.56, 0.54, 0.45, 0.45)
    )

    # --------------------------------------------------
    # Panel c: combined vulnerability percentile
    # --------------------------------------------------
    plot_categorical_map(
        gdf=map_gdf,
        ax=axes[2],
        column="combined_effect_bin",
        order=COMBINED_LABELS,
        colors=COMBINED_COLORS
    )

    combined_counts = (
        map_gdf["combined_effect_bin"]
        .value_counts()
        .reindex(COMBINED_LABELS)
        .fillna(0)
    )

    add_circular_percent_legend(
        ax=axes[2],
        counts=combined_counts,
        colors=COMBINED_COLORS,
        order=COMBINED_LABELS,
        title="Both",
        loc=(0.56, 0.54, 0.45, 0.45)
    )

    # --------------------------------------------------
    # Panel d: vulnerability type
    # --------------------------------------------------
    plot_categorical_map(
        gdf=map_gdf,
        ax=axes[3],
        column="vulnerability_type",
        order=TYPE_ORDER,
        colors=TYPE_COLORS
    )

    type_counts = (
        map_gdf["vulnerability_type"]
        .value_counts()
        .reindex(TYPE_ORDER)
        .fillna(0)
    )

    add_circular_percent_legend(
        ax=axes[3],
        counts=type_counts,
        colors=TYPE_COLORS,
        order=TYPE_ORDER,
        title="Type",
        loc=(0.56, 0.54, 0.45, 0.45)
    )

    # --------------------------------------------------
    # Redraw boundaries
    # --------------------------------------------------
    for ax in axes:
        dk_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=0.35,
            zorder=20
        )

        catchment_polygons.boundary.plot(
            ax=ax,
            edgecolor="black",
            linewidth=0.015,
            alpha=0.20,
            zorder=21
        )

    # --------------------------------------------------
    # Figure-level legends below maps
    # --------------------------------------------------
    add_subplot_legend_below(
        ax=axes[0],
        labels=PEAK_LABELS,
        colors=EFFECT_COLORS_6,
        title="Peak effect",
        ncol=3,
        y_offset=-0.035
    )
    
    add_subplot_legend_below(
        ax=axes[1],
        labels=VOLUME_LABELS,
        colors=EFFECT_COLORS_6,
        title="Volume effect",
        ncol=3,
        y_offset=-0.035
    )
    
    add_subplot_legend_below(
        ax=axes[2],
        labels=COMBINED_LABELS,
        colors=COMBINED_COLORS,
        title="Combined percentile",
        ncol=3,
        y_offset=-0.035
    )
    
    type_labels = TYPE_ORDER
    
    add_subplot_legend_below(
        ax=axes[3],
        labels=type_labels,
        colors=TYPE_COLORS,
        title="Type",
        ncol=2,
        y_offset=-0.035
    )

    # type_handles = [
    #     Patch(facecolor=TYPE_COLORS[label], edgecolor="none", label=label)
    #     for label in TYPE_ORDER
    # ]

    # fig.legend(
    #     handles=type_handles,
    #     title="Dominant vulnerability type",
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.015),
    #     ncol=len(TYPE_ORDER),
    #     frameon=False,
    #     fontsize=6.2,
    #     title_fontsize=6.8
    # )

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.96,
        bottom=0.14,
        wspace=0.02
    )

    if out_file is not None:
        fig.savefig(out_file, dpi=600, bbox_inches="tight")
        print(f"Saved map figure: {out_file}")

    return fig, axes


# ==================================================
#%% Main workflow
# ==================================================

if __name__ == "__main__":

    # -----------------------------
    # Load shapefiles
    # -----------------------------
    catchments_gdf = gpd.read_file(CATCHMENT_SHP)
    dk_gdf = gpd.read_file(DK_SHP)
    catchments_gdf["catch_id"] = catchments_gdf["catch_id"].astype(str)

    # -----------------------------
    # Build / load national feature table
    # -----------------------------
    if RUN_EE_EXTRACTION:
        dtp = load_jumpfilled_dtp().rio.write_crs(catchments_gdf.crs)
        all_catchments_df = build_national_feature_table(
            catchments_gdf=catchments_gdf,
            dynamic_dir=DYNAMIC_DIR,
            dtp = dtp, 
            out_csv=OUT_FEATURE_CSV,
            run_ee_extraction=RUN_EE_EXTRACTION,
        )
    else:
        all_catchments_df = pd.read_csv(OUT_FEATURE_CSV)

    all_catchments_df["catch_id"] = all_catchments_df["catch_id"].astype(str)

    print(
        all_catchments_df[
            ["dtp_q75_value", "dtp_q90_value", "dtp_q75_q90_delta"]
        ].describe()
    )

    # -----------------------------
    # Train target models and predict national sensitivity
    # -----------------------------
    sensitivity_tables = []

    for target_type in TARGET_TYPES:

        model, outcome_info = train_final_causal_model(
            target_type=target_type,
            scale=SCALE,
            confounders=CONFOUNDERS,
            heterogeneity_features=HETEROGENEITY_FEATURES,
            random_state=752
        )

        sens = predict_q90_sensitivity(
            model=model,
            all_catchments_df=all_catchments_df,
            heterogeneity_features=HETEROGENEITY_FEATURES,
            target_type=target_type
        )

        sensitivity_tables.append(sens)

    sensitivity_df = sensitivity_tables[0]

    for table in sensitivity_tables[1:]:
        sensitivity_df = sensitivity_df.merge(
            table,
            on="catch_id",
            how="outer"
        )

    # -----------------------------
    # Join to catchment polygons
    # -----------------------------
    map_gdf = catchment_polygons.merge(
        sensitivity_df,
        on="catch_id",
        how="left"
    )

    # -----------------------------
    # Add effect bins and vulnerability type
    # -----------------------------
    map_gdf = add_vulnerability_metrics_effect_bins(map_gdf)

    # -----------------------------
    # Make national sensitivity map
    # -----------------------------
    fig, axes = make_four_panel_sensitivity_map(
        map_gdf=map_gdf,
        catchment_polygons=catchment_polygons,
        dk_gdf=dk_gdf,
        out_file=OUT_MAP_FIG
    )

    plt.show()
    
    

