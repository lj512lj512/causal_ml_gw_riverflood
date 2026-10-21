# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:10:48 2026

@author: juliu
"""

# -*- coding: utf-8 -*-
"""
CATE interpretation of the groundwater Q75 -> Q90 contrast.

The causal estimand is the change in flood response when relative groundwater
depth moves from the catchment-specific Q75 state to the catchment-specific
Q90 state.

Higher relative DTP means shallower groundwater in this dataset.

Outputs
-------
For every target and spatial scale:

1. Final CausalForestDML model fitted to the complete analysis dataset.
2. EconML SingleTreeCateInterpreter figure.
3. Event-level Q75 -> Q90 CATE estimates.
4. Terminal-leaf summaries.
5. Interpreter fidelity diagnostics.
6. CSV and pickle output files.

Important
---------
The interpreter is a simplified representation of the fitted causal forest.
It does not re-estimate the causal effect.
"""

# ============================================================
# Imports
# ============================================================

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

from econml.dml import CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter


# ------------------------------------------------------------
# Change this import to the name of your existing script
# ------------------------------------------------------------
from A2_causal_model_train import (
    load_dataset,
    make_model_y,
    make_model_t,
    ealpha_e_features,
)


# ============================================================
# Settings
# ============================================================

SCALES = [
    "catch",
    "valley",
    "downs",
]

TARGET_TYPES = [
    "peak",
    "occurrence",
    "volume",
    "duration",
]

GROUP_COL = "catch_id"
TREATMENT = "dtp"

RANDOM_STATE = 42

# Grouped nuisance cross-fitting inside the final DML model
N_NUISANCE_SPLITS = 3

# Causal forest settings
N_ESTIMATORS = 1000
CF_MIN_SAMPLES_LEAF = 20

# Interpretation-tree settings
TREE_MAX_DEPTH = 3

# This is an absolute event count.
# Increase it when the tree creates very small subgroups.
TREE_MIN_SAMPLES_LEAF = 1000

INCLUDE_MODEL_UNCERTAINTY = True
UNCERTAINTY_LEVEL = 0.05

OUT_DIR = Path(
    r"\\geodata.geus.dk\HOME\causal_ml_gw_riverflood"
    r"\results\cate_interpretation_q75_q90"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Feature definitions
# ============================================================

CONFOUNDERS = [
    "rain_duration_days",
    "event_precip_mm",
    "max_1d_precip_mm",
    "split_from_long_event",
    "pre_event_precip_3d",
    "wcr",
    "sdy",
    "cdy",
    "tem",
    "pet",
]

# These must match X in the original causal model.
HETEROGENEITY_FEATURES = (
    ealpha_e_features
    + [
        "sdy",
        "cdy",
    ]
)

QUANTILE_COLUMNS = [
    "dtp_q75_value",
    "dtp_q90_value",
]


# ============================================================
# General helpers
# ============================================================

def unique_preserve_order(items):
    """Remove duplicate names while preserving their order."""

    seen = set()
    output = []

    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)

    return output


def prepare_target(dataset, target_type):
    """
    Prepare one outcome and return its metadata.
    """

    dataset = dataset.copy()

    if target_type == "peak":

        dataset["peak_discharge_log"] = np.log(
            dataset["flood_peak"] + 1e-6
        )

        return {
            "dataset": dataset,
            "target_type": target_type,
            "outcome": "peak_discharge_log",
            "raw_outcome": "flood_peak",
            "is_binary_outcome": False,
            "is_log_outcome": True,
            "title": "Peak discharge",
            "raw_effect_unit": "log outcome units",
            "report_effect_unit": "percent",
        }

    if target_type == "occurrence":

        dataset["flood_occurrence_bool"] = (
            dataset["flood_occurrence"]
            .astype(int)
        )

        return {
            "dataset": dataset,
            "target_type": target_type,
            "outcome": "flood_occurrence_bool",
            "raw_outcome": "flood_occurrence",
            "is_binary_outcome": True,
            "is_log_outcome": False,
            "title": "Flood occurrence",
            "raw_effect_unit": "probability",
            "report_effect_unit": "percentage points",
        }

    if target_type == "volume":

        dataset["flood_volume_above_q95_log"] = np.log1p(
            dataset["flood_volume_above_q95"]
        )

        return {
            "dataset": dataset,
            "target_type": target_type,
            "outcome": "flood_volume_above_q95_log",
            "raw_outcome": "flood_volume_above_q95",
            "is_binary_outcome": False,
            "is_log_outcome": True,
            "title": "Flood volume above Q95",
            "raw_effect_unit": "log1p outcome units",
            "report_effect_unit": "percent change in outcome + 1",
        }

    if target_type == "duration":

        dataset["duration_above_q95_log"] = np.log1p(
            dataset["duration_above_q95"]
        )

        return {
            "dataset": dataset,
            "target_type": target_type,
            "outcome": "duration_above_q95_log",
            "raw_outcome": "duration_above_q95",
            "is_binary_outcome": False,
            "is_log_outcome": True,
            "title": "Flood duration above Q95",
            "raw_effect_unit": "log1p outcome units",
            "report_effect_unit": "percent change in outcome + 1",
        }

    raise ValueError(
        f"Unsupported target type: {target_type}"
    )


def convert_effect_to_report_units(
    tau,
    target_type,
):
    """
    Convert raw CATE values to reporting units.

    occurrence
        Probability difference -> percentage points.

    peak
        Difference in log peak -> percent change in peak.

    volume and duration
        Difference in log1p outcome -> percent change in outcome + 1.
        This is not exactly the percentage change in the raw outcome,
        especially when the raw value is close to zero.
    """

    tau = np.asarray(tau, dtype=float).ravel()

    if target_type == "occurrence":
        return 100.0 * tau

    if target_type in {
        "peak",
        "volume",
        "duration",
    }:
        return 100.0 * np.expm1(tau)

    return tau


def get_interpreter_tree_model(interpreter):
    """
    Retrieve the fitted sklearn tree from different EconML versions.
    """

    possible_attributes = [
        "tree_model_",
        "_tree_model",
        "tree_model",
        "_tree_model_",
    ]

    for attribute in possible_attributes:

        if hasattr(interpreter, attribute):

            tree_model = getattr(
                interpreter,
                attribute,
            )

            if tree_model is not None:
                return tree_model

    available = [
        name
        for name in dir(interpreter)
        if "tree" in name.lower()
    ]

    raise AttributeError(
        "Could not find the fitted interpretation tree. "
        f"Tree-related attributes available: {available}"
    )


def make_tree_feature_labels(feature_names):
    """
    Create shorter labels for the tree figure.

    AlphaEarth features remain A00, A01, ..., A63.
    """

    label_map = {
        "sdy": "sin(day)",
        "cdy": "cos(day)",
    }

    return [
        label_map.get(feature, feature)
        for feature in feature_names
    ]


# ============================================================
# Data preparation
# ============================================================

def build_analysis_data(
    dataset,
    target_info,
):
    """
    Construct the complete-case dataset and model matrices.
    """

    outcome = target_info["outcome"]
    raw_outcome = target_info["raw_outcome"]

    required_columns = (
        [
            outcome,
            raw_outcome,
            TREATMENT,
            GROUP_COL,
        ]
        + CONFOUNDERS
        + HETEROGENEITY_FEATURES
        + QUANTILE_COLUMNS
    )

    # Keep date when available for monthly summaries.
    if "rain_start_date" in dataset.columns:
        required_columns.append(
            "rain_start_date"
        )

    required_columns = unique_preserve_order(
        required_columns
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            f"{missing_columns}"
        )

    df = (
        dataset[required_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .copy()
    )

    # Ensure Q90 represents a shallower state than Q75.
    df["q75_q90_delta_dtp"] = (
        df["dtp_q90_value"]
        - df["dtp_q75_value"]
    )

    invalid_quantile_order = (
        df["q75_q90_delta_dtp"] <= 0
    )

    if invalid_quantile_order.any():

        n_invalid = int(
            invalid_quantile_order.sum()
        )

        warnings.warn(
            f"Removing {n_invalid} events for which "
            "dtp_q90_value <= dtp_q75_value."
        )

        df = df.loc[
            ~invalid_quantile_order
        ].copy()

    if len(df) == 0:
        raise ValueError(
            "No valid observations remain after data cleaning."
        )

    Y = df[outcome].to_numpy()
    T = df[TREATMENT].to_numpy()

    X_df = df[
        HETEROGENEITY_FEATURES
    ].copy()

    W_df = df[
        CONFOUNDERS
    ].copy()

    groups = df[
        GROUP_COL
    ].to_numpy()

    T0 = df[
        "dtp_q75_value"
    ].to_numpy()

    T1 = df[
        "dtp_q90_value"
    ].to_numpy()

    return {
        "df": df,
        "Y": Y,
        "T": T,
        "X_df": X_df,
        "W_df": W_df,
        "groups": groups,
        "T0": T0,
        "T1": T1,
    }


# ============================================================
# Final causal forest
# ============================================================

def fit_final_causal_forest(
    analysis,
    target_info,
):
    """
    Fit one final causal forest using all available events.

    Nuisance cross-fitting remains grouped by catchment.
    """

    Y = analysis["Y"]
    T = analysis["T"]
    X = analysis["X_df"].to_numpy()
    W = analysis["W_df"].to_numpy()
    groups = analysis["groups"]

    nuisance_cv = GroupKFold(
        n_splits=N_NUISANCE_SPLITS
    )

    model_y = make_model_y(
        target_type=target_info["target_type"],
        random_state=RANDOM_STATE + 100,
    )

    model_t = make_model_t(
        random_state=RANDOM_STATE + 200,
    )

    causal_forest = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=CF_MIN_SAMPLES_LEAF,
        discrete_treatment=False,
        discrete_outcome=target_info[
            "is_binary_outcome"
        ],
        cv=nuisance_cv,
        random_state=RANDOM_STATE + 300,
    )

    print(
        "Fitting final causal forest using "
        f"{len(Y):,} events and "
        f"{len(np.unique(groups)):,} catchments."
    )

    causal_forest.fit(
        Y=Y,
        T=T,
        X=X,
        W=W,
        groups=groups,
        inference="auto",
    )

    return causal_forest


# ============================================================
# Q75 -> Q90 CATE estimation
# ============================================================

def estimate_q75_q90_effects(
    causal_forest,
    analysis,
    target_info,
):
    """
    Estimate event-specific effects for the Q75 -> Q90 contrast.
    """

    X = analysis["X_df"].to_numpy()
    T0 = analysis["T0"]
    T1 = analysis["T1"]

    print(
        "Estimating Q75 -> Q90 event-level CATEs."
    )

    tau = causal_forest.effect(
        X,
        T0=T0,
        T1=T1,
    )

    tau = np.asarray(
        tau
    ).ravel()

    try:

        tau_lower, tau_upper = (
            causal_forest.effect_interval(
                X,
                T0=T0,
                T1=T1,
                alpha=0.05,
            )
        )

        tau_lower = np.asarray(
            tau_lower
        ).ravel()

        tau_upper = np.asarray(
            tau_upper
        ).ravel()

    except Exception as error:

        warnings.warn(
            "Effect intervals could not be calculated: "
            f"{error}"
        )

        tau_lower = np.full(
            len(tau),
            np.nan,
        )

        tau_upper = np.full(
            len(tau),
            np.nan,
        )

    tau_report = convert_effect_to_report_units(
        tau,
        target_type=target_info["target_type"],
    )

    tau_lower_report = convert_effect_to_report_units(
        tau_lower,
        target_type=target_info["target_type"],
    )

    tau_upper_report = convert_effect_to_report_units(
        tau_upper,
        target_type=target_info["target_type"],
    )

    effect_df = analysis["df"].copy()

    effect_df["q75_q90_T0"] = T0
    effect_df["q75_q90_T1"] = T1
    effect_df["q75_q90_delta_dtp"] = T1 - T0

    effect_df["q75_q90_tau_raw"] = tau
    effect_df["q75_q90_tau_report"] = tau_report

    effect_df[
        "q75_q90_tau_lower_raw"
    ] = tau_lower

    effect_df[
        "q75_q90_tau_upper_raw"
    ] = tau_upper

    effect_df[
        "q75_q90_tau_lower_report"
    ] = tau_lower_report

    effect_df[
        "q75_q90_tau_upper_report"
    ] = tau_upper_report

    return effect_df


# ============================================================
# EconML CATE interpreter
# ============================================================

def fit_q75_q90_interpreter(
    causal_forest,
    analysis,):
    """
    Fit an EconML SingleTreeCateInterpreter for the Q75 -> Q90 effect.

    T0 and T1 are event-specific because every catchment has its own
    groundwater Q75 and Q90 thresholds.
    """

    X = analysis["X_df"].to_numpy()
    T0 = analysis["T0"]
    T1 = analysis["T1"]

    interpreter = SingleTreeCateInterpreter(
        include_model_uncertainty=(
            INCLUDE_MODEL_UNCERTAINTY
        ),
        uncertainty_level=(
            UNCERTAINTY_LEVEL
        ),
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=(
            TREE_MIN_SAMPLES_LEAF
        ),
        random_state=RANDOM_STATE,
    )

    print(
        "Fitting SingleTreeCateInterpreter "
        "for the Q75 -> Q90 contrast."
    )

    interpreter.interpret(
        causal_forest,
        X,
    )

    return interpreter


def plot_interpreter(
    interpreter,
    target_info,
    scale,
    output_file,
):
    """
    Plot and save the EconML CATE interpretation tree.
    """

    feature_labels = make_tree_feature_labels(
        HETEROGENEITY_FEATURES
    )

    plt.figure(
        figsize=(22, 11)
    )

    interpreter.plot(
        feature_names=feature_labels,
        fontsize=8,
    )

    plt.suptitle(
        (
            "CATE interpretation of the groundwater "
            "Q75 → Q90 contrast\n"
            f"{target_info['title']} | {scale} scale"
        ),
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved interpretation tree: {output_file}"
    )


# ============================================================
# Leaf summaries and fidelity
# ============================================================

def add_interpreter_leaf_ids(
    interpreter,
    analysis,
    effect_df,
):
    """
    Assign every event to a terminal interpretation-tree leaf.
    """

    tree_model = get_interpreter_tree_model(
        interpreter
    )

    X = analysis["X_df"].to_numpy()

    leaf_ids = tree_model.apply(X)

    output = effect_df.copy()
    output["cate_leaf"] = leaf_ids

    return output, tree_model


def summarize_interpreter_leaves(
    effect_df,
    target_info,
):
    """
    Summarize Q75 -> Q90 effects in each terminal leaf.
    """

    effect_column = "q75_q90_tau_report"

    leaf_summary = (
        effect_df
        .groupby(
            "cate_leaf",
            observed=True,
        )
        .agg(
            n_events=(
                effect_column,
                "count",
            ),
            n_catchments=(
                GROUP_COL,
                "nunique",
            ),
            mean_delta_dtp=(
                "q75_q90_delta_dtp",
                "mean",
            ),
            median_delta_dtp=(
                "q75_q90_delta_dtp",
                "median",
            ),
            mean_effect=(
                effect_column,
                "mean",
            ),
            median_effect=(
                effect_column,
                "median",
            ),
            std_effect=(
                effect_column,
                "std",
            ),
            mean_lower_bound=(
                "q75_q90_tau_lower_report",
                "mean",
            ),
            mean_upper_bound=(
                "q75_q90_tau_upper_report",
                "mean",
            ),
        )
        .reset_index()
    )

    quantile_summary = (
        effect_df
        .groupby(
            "cate_leaf",
            observed=True,
        )[effect_column]
        .quantile(
            [
                0.05,
                0.25,
                0.75,
                0.95,
            ]
        )
        .unstack()
        .reset_index()
        .rename(
            columns={
                0.05: "p05_effect",
                0.25: "p25_effect",
                0.75: "p75_effect",
                0.95: "p95_effect",
            }
        )
    )

    leaf_summary = leaf_summary.merge(
        quantile_summary,
        on="cate_leaf",
        how="left",
    )

    leaf_summary[
        "effect_unit"
    ] = target_info["report_effect_unit"]

    leaf_summary = leaf_summary.sort_values(
        "mean_effect",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return leaf_summary


def calculate_tree_fidelity(
    tree_model,
    analysis,
    effect_df,
):
    """
    Calculate how well the shallow tree reproduces the causal forest CATEs.

    The EconML interpretation tree is trained on raw effect units, so
    fidelity is calculated against q75_q90_tau_raw.
    """

    X = analysis["X_df"].to_numpy()

    tree_prediction = (
        tree_model.predict(X)
    )

    tree_prediction = np.asarray(
        tree_prediction
    ).ravel()

    forest_effect = effect_df[
        "q75_q90_tau_raw"
    ].to_numpy()

    fidelity_r2 = r2_score(
        forest_effect,
        tree_prediction,
    )

    fidelity_mae = np.mean(
        np.abs(
            forest_effect
            - tree_prediction
        )
    )

    return {
        "tree_fidelity_R2": fidelity_r2,
        "tree_fidelity_MAE_raw_units": fidelity_mae,
    }


def summarize_overall_effect(
    effect_df,
    target_info,
):
    """
    Overall event-level summary of the Q75 -> Q90 contrast.
    """

    values = (
        effect_df[
            "q75_q90_tau_report"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .to_numpy()
    )

    summary = pd.DataFrame(
        [
            {
                "contrast": "q75_to_q90",
                "n_events": len(values),
                "n_catchments": effect_df[
                    GROUP_COL
                ].nunique(),
                "mean_effect": np.mean(values),
                "median_effect": np.median(values),
                "std_effect": np.std(
                    values,
                    ddof=1,
                ),
                "p05_effect": np.percentile(
                    values,
                    5,
                ),
                "p25_effect": np.percentile(
                    values,
                    25,
                ),
                "p75_effect": np.percentile(
                    values,
                    75,
                ),
                "p95_effect": np.percentile(
                    values,
                    95,
                ),
                "effect_unit": target_info[
                    "report_effect_unit"
                ],
                "mean_q75_q90_delta_dtp": (
                    effect_df[
                        "q75_q90_delta_dtp"
                    ].mean()
                ),
                "median_q75_q90_delta_dtp": (
                    effect_df[
                        "q75_q90_delta_dtp"
                    ].median()
                ),
            }
        ]
    )

    return summary


# ============================================================
# Optional monthly summary
# ============================================================

def summarize_effect_by_month(
    effect_df,
    target_info,
):
    """
    Summarize effects by calendar month when dates are available.
    """

    if "rain_start_date" not in effect_df.columns:
        return None

    monthly_df = effect_df.copy()

    monthly_df["month"] = (
        pd.to_datetime(
            monthly_df["rain_start_date"]
        ).dt.month
    )

    monthly_summary = (
        monthly_df
        .groupby(
            "month",
            observed=True,
        )
        .agg(
            n_events=(
                "q75_q90_tau_report",
                "count",
            ),
            n_catchments=(
                GROUP_COL,
                "nunique",
            ),
            mean_effect=(
                "q75_q90_tau_report",
                "mean",
            ),
            median_effect=(
                "q75_q90_tau_report",
                "median",
            ),
            std_effect=(
                "q75_q90_tau_report",
                "std",
            ),
        )
        .reset_index()
    )

    monthly_summary[
        "effect_unit"
    ] = target_info["report_effect_unit"]

    return monthly_summary


# ============================================================
# Plot leaf-level effects
# ============================================================

def plot_leaf_effect_summary(
    leaf_summary,
    target_info,
    scale,
    output_file,
):
    """
    Plot mean Q75 -> Q90 effects for terminal interpretation-tree leaves.
    """

    plot_df = leaf_summary.copy()

    plot_df = plot_df.sort_values(
        "mean_effect",
        ascending=True,
    )

    y_position = np.arange(
        len(plot_df)
    )

    lower_error = (
        plot_df["mean_effect"]
        - plot_df["p25_effect"]
    )

    upper_error = (
        plot_df["p75_effect"]
        - plot_df["mean_effect"]
    )

    lower_error = np.maximum(
        lower_error,
        0,
    )

    upper_error = np.maximum(
        upper_error,
        0,
    )

    fig, ax = plt.subplots(
        figsize=(7.2, max(4.0, 0.55 * len(plot_df)))
    )

    ax.errorbar(
        plot_df["mean_effect"],
        y_position,
        xerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        capsize=3,
        linewidth=1.2,
    )

    labels = [
        (
            f"Leaf {leaf} "
            f"(n={n_events:,}; "
            f"{n_catchments:,} catchments)"
        )
        for leaf, n_events, n_catchments
        in zip(
            plot_df["cate_leaf"],
            plot_df["n_events"],
            plot_df["n_catchments"],
        )
    ]

    ax.set_yticks(
        y_position
    )

    ax.set_yticklabels(
        labels
    )

    ax.axvline(
        0,
        color="0.35",
        linewidth=0.9,
        linestyle="--",
    )

    ax.set_xlabel(
        (
            "Estimated Q75 → Q90 effect "
            f"({target_info['report_effect_unit']})"
        )
    )

    ax.set_title(
        (
            "Groundwater-effect heterogeneity by "
            "interpretation-tree subgroup\n"
            f"{target_info['title']} | {scale} scale"
        ),
        fontweight="bold",
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved leaf-effect figure: {output_file}"
    )


# ============================================================
# One target-scale analysis
# ============================================================
def run_one_analysis(
    dataset0,
    scale,
    target_type,):
    """
    Run the complete Q75 -> Q90 interpretation workflow.
    """

    print("\n" + "=" * 90)
    print(
        f"Scale={scale} | Target={target_type}"
    )
    print("=" * 90)

    target_info = prepare_target(
        dataset0,
        target_type,
    )

    dataset = target_info["dataset"]

    analysis = build_analysis_data(
        dataset=dataset,
        target_info=target_info,
    )

    causal_forest = fit_final_causal_forest(
        analysis=analysis,
        target_info=target_info,
    )

    effect_df = estimate_q75_q90_effects(
        causal_forest=causal_forest,
        analysis=analysis,
        target_info=target_info,
    )

    interpreter = fit_q75_q90_interpreter(
        causal_forest=causal_forest,
        analysis=analysis,
    )

    target_scale_stem = (
        f"{target_type}_{scale}_q75_to_q90"
    )

    tree_figure_file = (
        OUT_DIR
        / f"cate_tree_{target_scale_stem}.png"
    )

    plot_interpreter(
        interpreter=interpreter,
        target_info=target_info,
        scale=scale,
        output_file=tree_figure_file,
    )

    effect_df, tree_model = (
        add_interpreter_leaf_ids(
            interpreter=interpreter,
            analysis=analysis,
            effect_df=effect_df,
        )
    )

    overall_summary = summarize_overall_effect(
        effect_df=effect_df,
        target_info=target_info,
    )

    leaf_summary = summarize_interpreter_leaves(
        effect_df=effect_df,
        target_info=target_info,
    )

    monthly_summary = summarize_effect_by_month(
        effect_df=effect_df,
        target_info=target_info,
    )

    fidelity = calculate_tree_fidelity(
        tree_model=tree_model,
        analysis=analysis,
        effect_df=effect_df,
    )

    fidelity_df = pd.DataFrame(
        [
            {
                "target_type": target_type,
                "scale": scale,
                **fidelity,
            }
        ]
    )

    leaf_figure_file = (
        OUT_DIR
        / f"cate_leaf_effects_{target_scale_stem}.png"
    )

    plot_leaf_effect_summary(
        leaf_summary=leaf_summary,
        target_info=target_info,
        scale=scale,
        output_file=leaf_figure_file,
    )

    # --------------------------------------------------------
    # Save tabular outputs
    # --------------------------------------------------------

    effect_csv = (
        OUT_DIR
        / f"cate_event_effects_{target_scale_stem}.csv"
    )

    overall_csv = (
        OUT_DIR
        / f"cate_overall_summary_{target_scale_stem}.csv"
    )

    leaf_csv = (
        OUT_DIR
        / f"cate_leaf_summary_{target_scale_stem}.csv"
    )

    fidelity_csv = (
        OUT_DIR
        / f"cate_tree_fidelity_{target_scale_stem}.csv"
    )

    effect_df.to_csv(
        effect_csv,
        index=False,
    )

    overall_summary.to_csv(
        overall_csv,
        index=False,
    )

    leaf_summary.to_csv(
        leaf_csv,
        index=False,
    )

    fidelity_df.to_csv(
        fidelity_csv,
        index=False,
    )

    if monthly_summary is not None:

        monthly_csv = (
            OUT_DIR
            / f"cate_monthly_summary_{target_scale_stem}.csv"
        )

        monthly_summary.to_csv(
            monthly_csv,
            index=False,
        )

    # --------------------------------------------------------
    # Save Python results
    # --------------------------------------------------------

    save_object = {
        "target_type": target_type,
        "scale": scale,
        "contrast": "q75_to_q90",
        "contrast_description": (
            "Catchment-specific high groundwater state "
            "(Q75) to very high groundwater state (Q90)"
        ),
        "target_info": target_info,
        "confounders": CONFOUNDERS,
        "heterogeneity_features": HETEROGENEITY_FEATURES,
        "event_effects": effect_df,
        "overall_summary": overall_summary,
        "leaf_summary": leaf_summary,
        "monthly_summary": monthly_summary,
        "fidelity": fidelity_df,
        "interpreter": interpreter,
        "interpreter_tree_model": tree_model,
        # The causal forest may be large.
        # Keep this entry only when sufficient disk space is available.
        "causal_forest": causal_forest,
    }

    pickle_file = (
        OUT_DIR
        / f"cate_interpretation_{target_scale_stem}.pkl"
    )

    try:

        with open(
            pickle_file,
            "wb",
        ) as file:
            pickle.dump(
                save_object,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(
            f"Saved complete result: {pickle_file}"
        )

    except Exception as error:

        warnings.warn(
            "The complete model object could not be pickled. "
            f"Reason: {error}"
        )

        # Save a smaller object without the fitted models.
        reduced_object = {
            key: value
            for key, value in save_object.items()
            if key not in {
                "causal_forest",
                "interpreter",
                "interpreter_tree_model",
            }
        }

        reduced_pickle_file = (
            OUT_DIR
            / (
                "cate_interpretation_reduced_"
                f"{target_scale_stem}.pkl"
            )
        )

        with open(
            reduced_pickle_file,
            "wb",
        ) as file:
            pickle.dump(
                reduced_object,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(
            "Saved reduced result without fitted models: "
            f"{reduced_pickle_file}"
        )

    print("\nOverall Q75 -> Q90 effect:")
    print(
        overall_summary.to_string(
            index=False
        )
    )

    print("\nInterpretation-tree leaves:")
    print(
        leaf_summary.to_string(
            index=False
        )
    )

    print("\nTree fidelity:")
    print(
        fidelity_df.to_string(
            index=False
        )
    )

    return {
        "causal_forest": causal_forest,
        "interpreter": interpreter,
        "event_effects": effect_df,
        "overall_summary": overall_summary,
        "leaf_summary": leaf_summary,
        "monthly_summary": monthly_summary,
        "fidelity": fidelity_df,
    }

#%%
# ============================================================
# Main workflow
# ============================================================
if __name__ == "__main__":
    all_results = {}
    for scale in SCALES:
        print("\n" + "#" * 100)
        print(f"Loading dataset for scale: {scale}")
        print("#" * 100)

        dataset0 = load_dataset(scale=scale)
        dataset0 = (dataset0.dropna(subset=["flood_occurrence"]).reset_index(drop=True))

        for target_type in TARGET_TYPES:
            key = (scale,target_type) 
            all_results[key] = run_one_analysis(dataset0=dataset0,
                                                scale=scale,
                                                target_type=target_type,)
    print("\nFinished all requested analyses.")
