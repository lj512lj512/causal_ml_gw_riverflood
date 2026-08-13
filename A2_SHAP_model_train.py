# -*- coding: utf-8 -*-
"""
TabPFN predictive benchmark:
Predictive evidence for groundwater compounding

Figure layout:
    Left column  = event peak discharge
    Right column = flood occurrence

Rows:
    1. Stepwise gain / forest-bar plot
    2. SHAP feature interconnection network
    3. Feature-target correlation matrix
"""

# ==================================================
# Imports
# ==================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier
import shap
from tabpfn_extensions.interpretability import shapiq_to_shap_explanation
from tabpfn_extensions.interpretability.shapiq import get_tabpfn_imputation_explainer
from A2_causal_model_train import load_dataset
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_737DETUKbWBFIApc_LTT-AcyKG1phd5_ULEtRLaxyWo"

# ==================================================
# Settings
# ==================================================

SCALE = "catch"
TARGET_TYPES = ["peak", "occurrence"]

N_SPLITS = 20
RANDOM_STATE = 42
MAX_TRAIN_SIZE_PER_FOLD = 20000

RUN_SHAP = False
RUN_PDP  = False
SHAP_BACKGROUND_SIZE = 150
SHAP_EXPLAIN_SIZE = 400

# ==================================================
# Feature sets
# ==================================================

FEATURE_SETS = {
    "Rainfall": [
        "rain_duration_days",
        "event_precip_mm",
        "max_1d_precip_mm",
    ],

    "Rainfall + meteo": [
        "rain_duration_days",
        "event_precip_mm",
        "max_1d_precip_mm",
        "sdy",
        "cdy",
        "tem",
        "pet",
    ],

    "+ antecedent P": [
        "rain_duration_days",
        "event_precip_mm",
        "max_1d_precip_mm",
        "pre_event_precip_3d",
        "sdy",
        "cdy",
        "tem",
        "pet",
    ],

    "+ soil water": [
        "rain_duration_days",
        "event_precip_mm",
        "max_1d_precip_mm",
        "pre_event_precip_3d",
        "wcr",
        "sdy",
        "cdy",
        "tem",
        "pet",
    ],
    
    "+ groundwater": [
        "rain_duration_days",
        "event_precip_mm",
        "max_1d_precip_mm",
        "pre_event_precip_3d",
        "dtp",
        "sdy",
        "cdy",
        "tem",
        "pet",
    ],

    "+ soil water + groundwater": [
        "rain_duration_days",
        "event_precip_mm",
        "max_1d_precip_mm",
        "pre_event_precip_3d",
        "wcr",
        "dtp",
        "sdy",
        "cdy",
        "tem",
        "pet",
    ],
}

PREDICTOR_ORDER = list(FEATURE_SETS.keys())
FINAL_SET_NAME = "+ soil water + groundwater"
FINAL_FEATURES = FEATURE_SETS[FINAL_SET_NAME]

SHORT_LABELS = {
    "rain_duration_days": "Rain duration",
    "event_precip_mm": "Event P",
    "max_1d_precip_mm": "Max 1-d P",
    "pre_event_precip_3d": "Pre-event P",
    "wcr": "Soil water",
    "dtp": "Groundwater",
    "sdy": "sin(day)",
    "cdy": "cos(day)",
    "tem": "Temp.",
    "pet": "PET",
}


# ==================================================
# Target preparation
# ==================================================

def prepare_target(dataset, target_type):
    dataset = dataset.copy()

    if target_type == "peak":
        dataset["peak_discharge_log"] = np.log(dataset["flood_peak"] + 1e-6)
        return {
            "target_type": target_type,
            "dataset": dataset,
            "outcome": "peak_discharge_log",
            "raw_outcome": "flood_peak",
            "is_classification": False,
            "title": "Event peak discharge",
            "metric_col": "R2",
            "metric_mean_col": "R2_mean",
            "metric_std_col": "R2_std",
            "metric_label": "Cross-validated R²",
            "target_label": "log peak discharge",
        }

    elif target_type == "occurrence":
        dataset["flood_occurrence"] = dataset["flood_occurrence"].astype(int)
        return {
            "target_type": target_type,
            "dataset": dataset,
            "outcome": "flood_occurrence",
            "raw_outcome": "flood_occurrence",
            "is_classification": True,
            "title": "Flood occurrence",
            "metric_col": "ROC_AUC",
            "metric_mean_col": "ROC_AUC_mean",
            "metric_std_col": "ROC_AUC_std",
            "metric_label": "Cross-validated ROC-AUC",
            "target_label": "flood occurrence",
        }

    else:
        raise ValueError(f"Unsupported target_type: {target_type}")


# ==================================================
# Model helpers
# ==================================================
def make_tabpfn_model(is_classification, random_state=42, fit_with_cache=False):
    fit_mode = "fit_with_cache" if fit_with_cache else "standard"

    if is_classification:
        try:
            return TabPFNClassifier(
                device="cuda",
                random_state=random_state,
                fit_mode=fit_mode
            )
        except TypeError:
            return TabPFNClassifier(device="cuda", fit_mode=fit_mode)
    else:
        try:
            return TabPFNRegressor(
                device="cuda",
                random_state=random_state,
                fit_mode=fit_mode
            )
        except TypeError:
            return TabPFNRegressor(device="cuda", fit_mode=fit_mode)


def subsample_balanced_by_catchment(
    X_train,
    y_train,
    groups_train,
    max_train_size=20000,
    random_state=42,
):
    """
    Subsample approximately equally across catchments.
    Good default for regression targets.
    """

    n = len(y_train)

    if max_train_size is None or n <= max_train_size:
        return X_train, y_train

    rng = np.random.default_rng(random_state)

    groups_train = pd.Series(groups_train).reset_index(drop=True)
    y_ser = pd.Series(y_train).reset_index(drop=True)

    if isinstance(X_train, pd.DataFrame):
        X_df = X_train.reset_index(drop=True)
    else:
        X_df = pd.DataFrame(X_train)

    index_df = pd.DataFrame({
        "idx": np.arange(n),
        "catch_id": groups_train.values,
    })

    unique_groups = index_df["catch_id"].unique()
    n_groups = len(unique_groups)

    base_per_group = max(1, max_train_size // n_groups)

    selected = []

    for _, sub in index_df.groupby("catch_id"):
        k = min(len(sub), base_per_group)
        selected.extend(
            rng.choice(sub["idx"].values, size=k, replace=False)
        )

    selected = np.array(selected, dtype=int)

    remaining = max_train_size - len(selected)

    if remaining > 0:
        unused = np.setdiff1d(np.arange(n), selected)
        extra = rng.choice(
            unused,
            size=min(remaining, len(unused)),
            replace=False,
        )
        selected = np.concatenate([selected, extra])

    rng.shuffle(selected)

    if isinstance(X_train, pd.DataFrame):
        X_sub = X_df.iloc[selected].copy()
    else:
        X_sub = X_df.iloc[selected].values

    y_sub = y_ser.iloc[selected].values

    return X_sub, y_sub


def subsample_balanced_by_catchment_and_target_bins(
    X_train,
    y_train,
    groups_train,
    max_train_size=20000,
    n_target_bins=5,
    random_state=42,
):
    """
    Subsample approximately across catchments and target magnitude bins.

    Good for peak-discharge regression because it keeps both ordinary and large
    flood-response events represented.

    y_train can be log peak discharge.
    """

    n = len(y_train)

    if max_train_size is None or n <= max_train_size:
        return X_train, y_train

    rng = np.random.default_rng(random_state)

    groups_train = pd.Series(groups_train).reset_index(drop=True)
    y_ser = pd.Series(y_train).reset_index(drop=True)

    if isinstance(X_train, pd.DataFrame):
        X_df = X_train.reset_index(drop=True)
    else:
        X_df = pd.DataFrame(X_train)

    # Quantile bins of the target.
    # duplicates='drop' prevents failure if repeated values exist.
    target_bin = pd.qcut(
        y_ser,
        q=n_target_bins,
        labels=False,
        duplicates="drop",
    )

    index_df = pd.DataFrame({
        "idx": np.arange(n),
        "catch_id": groups_train.values,
        "target_bin": target_bin.values,
    })

    # Remove rows that could not be binned
    index_df = index_df.dropna(subset=["target_bin"]).copy()
    index_df["target_bin"] = index_df["target_bin"].astype(int)

    strata = list(index_df.groupby(["catch_id", "target_bin"]))
    n_strata = len(strata)

    base_per_stratum = max(1, max_train_size // n_strata)

    selected = []

    for _, sub in strata:
        k = min(len(sub), base_per_stratum)
        selected.extend(
            rng.choice(sub["idx"].values, size=k, replace=False)
        )

    selected = np.array(selected, dtype=int)

    remaining = max_train_size - len(selected)

    if remaining > 0:
        unused = np.setdiff1d(np.arange(n), selected)
        extra = rng.choice(
            unused,
            size=min(remaining, len(unused)),
            replace=False,
        )
        selected = np.concatenate([selected, extra])

    rng.shuffle(selected)

    if isinstance(X_train, pd.DataFrame):
        X_sub = X_df.iloc[selected].copy()
    else:
        X_sub = X_df.iloc[selected].values

    y_sub = y_ser.iloc[selected].values

    return X_sub, y_sub

def subsample_balanced_by_catchment_and_class(
    X_train,
    y_train,
    groups_train,
    max_train_size=20000,
    class_balance="natural_with_boost",
    positive_fraction_target=0.35,
    random_state=42,
):
    """
    Subsample for binary classification while preserving catchment representation
    and ensuring enough positive flood events.

    Parameters
    ----------
    class_balance:
        "natural"             -> catchment/class balanced but keeps natural class prevalence as much as possible
        "balanced"            -> approximately 50/50 positive/negative
        "natural_with_boost"  -> boosts positives to at least positive_fraction_target
    positive_fraction_target:
        Used only for "natural_with_boost".
        Example: 0.35 means try to make at least 35% of sample positive.
    """

    n = len(y_train)

    if max_train_size is None or n <= max_train_size:
        return X_train, y_train

    rng = np.random.default_rng(random_state)

    groups_train = pd.Series(groups_train).reset_index(drop=True)
    y_ser = pd.Series(y_train).reset_index(drop=True).astype(int)

    if isinstance(X_train, pd.DataFrame):
        X_df = X_train.reset_index(drop=True)
    else:
        X_df = pd.DataFrame(X_train)

    index_df = pd.DataFrame({
        "idx": np.arange(n),
        "catch_id": groups_train.values,
        "class": y_ser.values,
    })

    pos_df = index_df[index_df["class"] == 1]
    neg_df = index_df[index_df["class"] == 0]

    n_pos_available = len(pos_df)
    n_neg_available = len(neg_df)

    if n_pos_available == 0 or n_neg_available == 0:
        print("Warning: only one class available in this fold. Falling back to catchment-balanced sampling.")
        return subsample_balanced_by_catchment(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            max_train_size=max_train_size,
            random_state=random_state,
        )

    if class_balance == "balanced":
        n_pos_target = min(n_pos_available, max_train_size // 2)
        n_neg_target = min(n_neg_available, max_train_size - n_pos_target)

    elif class_balance == "natural_with_boost":
        natural_pos_fraction = n_pos_available / n

        target_pos_fraction = max(natural_pos_fraction, positive_fraction_target)

        n_pos_target = int(max_train_size * target_pos_fraction)
        n_pos_target = min(n_pos_target, n_pos_available)

        n_neg_target = max_train_size - n_pos_target
        n_neg_target = min(n_neg_target, n_neg_available)

    elif class_balance == "natural":
        n_pos_target = int(max_train_size * (n_pos_available / n))
        n_pos_target = max(1, min(n_pos_target, n_pos_available))

        n_neg_target = max_train_size - n_pos_target
        n_neg_target = min(n_neg_target, n_neg_available)

    else:
        raise ValueError(
            "class_balance must be 'natural', 'balanced', or 'natural_with_boost'"
        )

    def sample_balanced_within_class(class_df, n_target):
        """
        Sample a target number from one class while balancing catchments.
        """
        if n_target <= 0:
            return np.array([], dtype=int)

        strata = list(class_df.groupby("catch_id"))
        n_strata = len(strata)

        base_per_group = max(1, n_target // n_strata)

        selected_class = []

        for _, sub in strata:
            k = min(len(sub), base_per_group)
            selected_class.extend(
                rng.choice(sub["idx"].values, size=k, replace=False)
            )

        selected_class = np.array(selected_class, dtype=int)

        remaining = n_target - len(selected_class)

        if remaining > 0:
            unused = np.setdiff1d(class_df["idx"].values, selected_class)
            if len(unused) > 0:
                extra = rng.choice(
                    unused,
                    size=min(remaining, len(unused)),
                    replace=False,
                )
                selected_class = np.concatenate([selected_class, extra])

        return selected_class

    selected_pos = sample_balanced_within_class(pos_df, n_pos_target)
    selected_neg = sample_balanced_within_class(neg_df, n_neg_target)

    selected = np.concatenate([selected_pos, selected_neg])

    # If still below max size because one class was exhausted, fill from remaining unused events
    remaining = max_train_size - len(selected)

    if remaining > 0:
        unused = np.setdiff1d(np.arange(n), selected)
        if len(unused) > 0:
            extra = rng.choice(
                unused,
                size=min(remaining, len(unused)),
                replace=False,
            )
            selected = np.concatenate([selected, extra])

    rng.shuffle(selected)

    if isinstance(X_train, pd.DataFrame):
        X_sub = X_df.iloc[selected].copy()
    else:
        X_sub = X_df.iloc[selected].values

    y_sub = y_ser.iloc[selected].values

    return X_sub, y_sub


def subsample_for_tabpfn(
    X_train,
    y_train,
    groups_train,
    target_type,
    max_train_size=20000,
    random_state=42,
):
    """
    Target-aware TabPFN subsampling.

    peak:
        catchment + target magnitude bins

    occurrence:
        catchment + class balance

    volume/duration:
        catchment + target bins, or catchment-only if you prefer
    """

    if max_train_size is None or len(y_train) <= max_train_size:
        return X_train, y_train

    if target_type == "occurrence":
        return subsample_balanced_by_catchment_and_class(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            max_train_size=max_train_size,
            class_balance="natural_with_boost",
            positive_fraction_target=0.35,
            random_state=random_state,
        )

    elif target_type == "peak":
        return subsample_balanced_by_catchment_and_target_bins(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            max_train_size=max_train_size,
            n_target_bins=5,
            random_state=random_state,
        )

    elif target_type in ["volume", "duration"]:
        return subsample_balanced_by_catchment_and_target_bins(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            max_train_size=max_train_size,
            n_target_bins=5,
            random_state=random_state,
        )

    else:
        print('wrong target subsampling !')
    
    
    
def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def classification_metrics(y_true, y_prob):
    eps = 1e-6
    y_prob_safe = np.clip(y_prob, eps, 1 - eps)

    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "LogLoss": log_loss(y_true, y_prob_safe),
    }


# ==================================================
# Compare predictor sets
# ==================================================

def compare_predictor_sets_tabpfn(
    dataset,
    target_info,
    feature_sets,
    group_col="catch_id",
    n_splits=20,
    max_train_size=20000,
    random_state=42,
):
    outcome = target_info["outcome"]
    raw_outcome = target_info["raw_outcome"]
    is_classification = target_info["is_classification"]

    all_results = []
    gkf = GroupKFold(n_splits=n_splits)

    for set_name, cols in feature_sets.items():
        print("=" * 80)
        print(f"Target={target_info['title']} | predictor set={set_name}")
        print("=" * 80)

        required_cols = [outcome, raw_outcome, group_col] + cols
        required_cols = unique_preserve_order(required_cols)
        d = (
            dataset[required_cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .copy()
        )

        X = d[cols]
        Y = d[outcome].values
        groups = d[group_col].values

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups),start=1):
            # -----------------------------
            # Split full fold data
            # -----------------------------
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = Y[train_idx]
            y_test = Y[test_idx]
            groups_train = groups[train_idx]
        
            # -----------------------------
            # Subsample training data only
            # -----------------------------
            X_tr_sub, y_tr_sub = subsample_for_tabpfn(
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                target_type=target_info["target_type"],
                max_train_size=MAX_TRAIN_SIZE_PER_FOLD,
                random_state=RANDOM_STATE + fold,)
        
            # -----------------------------
            # Create model after subsampling
            # -----------------------------
            model = make_tabpfn_model(
                is_classification=is_classification,
                random_state=RANDOM_STATE + fold,
                fit_with_cache=True)
        
            # -----------------------------
            # Fit on subsampled training data
            # -----------------------------
            model.fit(
                X_tr_sub.values if isinstance(X_tr_sub, pd.DataFrame) else X_tr_sub,
                y_tr_sub)

            row = {
                "predictor_set": set_name,
                "fold": fold,
                "n_features": len(cols),
                "n_train_full": len(train_idx),
                "n_train_used": len(y_tr_sub),
                "n_test": len(test_idx),
            }

            if is_classification:
                y_prob = model.predict_proba(X.iloc[test_idx].values)[:, 1]
                row.update(classification_metrics(Y[test_idx], y_prob))
            else:
                y_hat = model.predict(X.iloc[test_idx].values)
                row.update(regression_metrics(Y[test_idx], y_hat))

            all_results.append(row)

    results_df = pd.DataFrame(all_results)

    if is_classification:
        summary_df = (
            results_df
            .groupby("predictor_set")
            .agg(
                n_folds=("fold", "count"),
                n_features=("n_features", "first"),
                ROC_AUC_mean=("ROC_AUC", "mean"),
                ROC_AUC_std=("ROC_AUC", "std"),
                PR_AUC_mean=("PR_AUC", "mean"),
                Brier_mean=("Brier", "mean"),
                LogLoss_mean=("LogLoss", "mean"),
            )
            .reset_index()
        )
    else:
        summary_df = (
            results_df
            .groupby("predictor_set")
            .agg(
                n_folds=("fold", "count"),
                n_features=("n_features", "first"),
                R2_mean=("R2", "mean"),
                R2_std=("R2", "std"),
                RMSE_mean=("RMSE", "mean"),
                MAE_mean=("MAE", "mean"),
            )
            .reset_index()
        )

    return results_df, summary_df


# ==================================================
# Final model + SHAP for selected feature set
# ==================================================
def unique_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def train_final_model_and_shap(
    dataset,
    target_info,
    feature_cols,
    group_col="catch_id",
    out_file = None,
    max_train_size=20000,
    shap_background_size=150,
    shap_explain_size=400,
    random_state=42,
):
    outcome = target_info["outcome"]
    raw_outcome = target_info["raw_outcome"]
    is_classification = target_info["is_classification"]

    required_cols = [outcome, raw_outcome, group_col] + feature_cols
    required_cols = unique_preserve_order(required_cols)
    d = (
        dataset[required_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = d[feature_cols]
    Y = d[outcome].values

    rng = np.random.default_rng(random_state)

    if max_train_size is not None and len(X) > max_train_size:
        final_train_idx = rng.choice(
            np.arange(len(X)),
            size=max_train_size,
            replace=False,
        )
    else:
        final_train_idx = np.arange(len(X))

    model = make_tabpfn_model(
        is_classification=is_classification,
        random_state=random_state,
        fit_with_cache=True
    )

    print(f"Training final model for {target_info['title']}")
    model.fit(
        X.iloc[final_train_idx].values,
        Y[final_train_idx],
    )

    background_size = min(shap_background_size, len(X))
    explain_size = min(shap_explain_size, len(X))

    background_idx = rng.choice(
        np.arange(len(X)),
        size=background_size,
        replace=False,
    )

    explain_idx = rng.choice(
        np.arange(len(X)),
        size=explain_size,
        replace=False,
    )

    X_background = X.iloc[background_idx]
    X_explain = X.iloc[explain_idx]

    if is_classification:
        def predict_fn(x):
            return model.predict_proba(x)[:, 1]
    else:
        def predict_fn(x):
            return model.predict(x)
    
    print(f"Calculating TabPFN shapiq values for {target_info['title']}")
    
    explainer = get_tabpfn_imputation_explainer(
        model=model,
        data=X.iloc[final_train_idx].values
    )
    
    shap_values = shapiq_to_shap_explanation(
        explainer=explainer,
        X=X_explain.values,
        budget=128,
        feature_names=feature_cols
    )

    shap_importance = (
        pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    y_explain = d.loc[X_explain.index, outcome].values
    raw_y_explain = d.loc[X_explain.index, raw_outcome].values
    
    shap_results = {
        "d": d,
        "X": X,
        "Y": Y,
        "final_model": model,
        "X_explain": X_explain,
        "y_explain": y_explain,
        "raw_y_explain": raw_y_explain,
        "shap_values": shap_values,
        "shap_importance": shap_importance,
    }
    
    if out_file is not None:
        with open(out_file, "wb") as f:
            pickle.dump(shap_results, f)
        print(f"Saved PDP surfaces to: {out_file}")
    return shap_results


# ==================================================
# Plot: stepwise gain with forest bars
# ==================================================
def plot_stepwise_gain_forest_ax(
    summary_df,
    target_info,
    ax,
    predictor_order,
    metric_mean_col,
    metric_std_col,
    highlight_name="+ soil water + groundwater",
):
    s = (
        summary_df
        .set_index("predictor_set")
        .loc[predictor_order]
        .reset_index()
        .copy()
    )

    s["delta"] = s[metric_mean_col].diff()
    s.loc[0, "delta"] = np.nan

    y = np.arange(len(s))[::-1]

    for i, row in s.iterrows():
        color = "#d73027" if row["predictor_set"] == highlight_name else "#4c78a8"

        ax.errorbar(
            row[metric_mean_col],
            y[i],
            xerr=row[metric_std_col],
            fmt="o",
            markersize=6,
            linewidth=1.4,
            capsize=3,
            color=color,
            ecolor=color,
        )

        ax.text(
            row[metric_mean_col] + 0.01,
            y[i],
            f"{row[metric_mean_col]:.2f}",
            va="center",
            fontsize=8,
        )

        if not np.isnan(row["delta"]):
            ax.text(
                row[metric_mean_col] + 0.065,
                y[i],
                f"Δ={row['delta']:+.2f}",
                va="center",
                fontsize=8,
                color=color,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(s["predictor_set"])
    ax.set_xlabel(target_info["metric_label"])
    ax.set_title(target_info["title"], fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmin = max(0, s[metric_mean_col].min() - 2 * s[metric_std_col].max() - 0.05)
    xmax = min(1.05, s[metric_mean_col].max() + 0.18)
    ax.set_xlim(xmin, xmax)

    return s


# --------------------------------------------------
# Manual 2D PDP calculator
# --------------------------------------------------
def compute_manual_2d_pdp(
    model,
    X_ref,
    feature_x,
    feature_y,
    is_classification=False,
    x_quantiles=(0.05, 0.95),
    y_quantiles=(0.05, 0.95),
    grid_resolution=25,
    sample_size=1200,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    X_ref = X_ref.copy()

    if sample_size is not None and len(X_ref) > sample_size:
        idx = rng.choice(np.arange(len(X_ref)), size=sample_size, replace=False)
        X_ref = X_ref.iloc[idx].copy()

    x_low, x_high = X_ref[feature_x].quantile(x_quantiles)
    y_low, y_high = X_ref[feature_y].quantile(y_quantiles)

    x_grid = np.linspace(x_low, x_high, grid_resolution)
    y_grid = np.linspace(y_low, y_high, grid_resolution)

    Z = np.zeros((len(y_grid), len(x_grid)))

    for iy, yv in enumerate(y_grid):
        for ix, xv in enumerate(x_grid):
            X_tmp = X_ref.copy()
            X_tmp[feature_x] = xv
            X_tmp[feature_y] = yv

            if is_classification:
                pred = model.predict_proba(X_tmp.values)[:, 1]
            else:
                pred = model.predict(X_tmp.values)

            Z[iy, ix] = np.mean(pred)

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "Z": Z,
        "feature_x": feature_x,
        "feature_y": feature_y,
        "is_classification": is_classification,
    }

def compute_peak_occurrence_pdp_surfaces(
    peak_result,
    occurrence_result,
    grid_resolution=25,
    sample_size=1200,
    random_state=42,
    out_file=None,
):
    peak_model = peak_result["shap"]["final_model"]
    peak_X = peak_result["shap"]["X"]

    occ_model = occurrence_result["shap"]["final_model"]
    occ_X = occurrence_result["shap"]["X"]

    pdp_results = {}

    print("Computing peak: wcr × event_precip_mm")
    pdp_results["peak_wcr_precip"] = compute_manual_2d_pdp(
        model=peak_model,
        X_ref=peak_X,
        feature_x="wcr",
        feature_y="event_precip_mm",
        is_classification=False,
        grid_resolution=grid_resolution,
        sample_size=sample_size,
        random_state=random_state,
    )

    print("Computing occurrence: wcr × event_precip_mm")
    pdp_results["occurrence_wcr_precip"] = compute_manual_2d_pdp(
        model=occ_model,
        X_ref=occ_X,
        feature_x="wcr",
        feature_y="event_precip_mm",
        is_classification=True,
        grid_resolution=grid_resolution,
        sample_size=sample_size,
        random_state=random_state,
    )
    
    print("Computing peak: dtp × event_precip_mm")
    pdp_results["peak_dtp_precip"] = compute_manual_2d_pdp(
        model=peak_model,
        X_ref=peak_X,
        feature_x="dtp",
        feature_y="event_precip_mm",
        is_classification=False,
        grid_resolution=grid_resolution,
        sample_size=sample_size,
        random_state=random_state,
    )

    print("Computing occurrence: dtp × event_precip_mm")
    pdp_results["occurrence_dtp_precip"] = compute_manual_2d_pdp(
        model=occ_model,
        X_ref=occ_X,
        feature_x="dtp",
        feature_y="event_precip_mm",
        is_classification=True,
        grid_resolution=grid_resolution,
        sample_size=sample_size,
        random_state=random_state,
    )

    print("Computing peak: dtp × wcr")
    pdp_results["peak_dtp_wcr"] = compute_manual_2d_pdp(
        model=peak_model,
        X_ref=peak_X,
        feature_x="dtp",
        feature_y="wcr",
        is_classification=False,
        grid_resolution=grid_resolution,
        sample_size=sample_size,
        random_state=random_state,
    )

    print("Computing occurrence: dtp × wcr")
    pdp_results["occurrence_dtp_wcr"] = compute_manual_2d_pdp(
        model=occ_model,
        X_ref=occ_X,
        feature_x="dtp",
        feature_y="wcr",
        is_classification=True,
        grid_resolution=grid_resolution,
        sample_size=sample_size,
        random_state=random_state,
    )

    if out_file is not None:
        with open(out_file, "wb") as f:
            pickle.dump(pdp_results, f)
        print(f"Saved PDP surfaces to: {out_file}")

    return pdp_results


# --------------------------------------------------
# Stronger PDP figure
# --------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

def align_ylabels_by_column(fig, left_axes, right_axes, x_left=-0.12, x_right=-0.12):
    """
    Align y-axis labels within each column.

    x_left and x_right are in axes coordinates:
    more negative = label moves further left.
    """
    fig.canvas.draw()

    for ax in left_axes:
        ax.yaxis.set_label_coords(x_left, 0.5)

    for ax in right_axes:
        ax.yaxis.set_label_coords(x_right, 0.5)
        

def one_decimal_if_less_than_one(x, pos=None):
    if abs(x) < 1:
        return f"{x:.2f}"
    else:
        return f"{x:g}"


def apply_clean_tick_format(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(one_decimal_if_less_than_one))
    ax.yaxis.set_major_formatter(FuncFormatter(one_decimal_if_less_than_one))
    

def plot_2d_pdp_ax(
    ax,
    pdp,
    xlabel,
    ylabel,
    cmap="viridis",
    n_levels=14,
    contour_labels=True,
):
    x_grid = pdp["x_grid"]
    y_grid = pdp["y_grid"]
    Z = pdp["Z"]

    xx, yy = np.meshgrid(x_grid, y_grid)

    cf = ax.contourf(xx, yy, Z, levels=n_levels, cmap=cmap)

    cs = ax.contour(
        xx,
        yy,
        Z,
        levels=n_levels,
        colors="white",
        linewidths=0.45,
        alpha=0.75,
    )

    if contour_labels:
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title, fontsize=10)
    # ax.set_title(title,loc="left",fontweight="bold",)

    return cf


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_stepwise_gain_half_violin_ax(
    results_df,
    ax,
    predictor_order,
    metric_col,
    ylabel,
    final_set_name=None,
    jitter=0.045,
    violin_width=0.28,
    point_alpha=0.45,
    point_size=14,
    line_width=1.8,
    random_state=42,
):
    """
    Half-violin stepwise predictive-gain plot.

    Left side  = fold-wise metric values as jittered points
    Right side = half violin distribution
    Curve      = median metric across predictor sets
    """
    
    PREDICTOR_LABELS = {
    "Rainfall": "P",
    "Rainfall + meteo": "PM",
    "+ antecedent P": "PMaP",
    "+ soil water": "PMaPS",
    "+ groundwater": "PMaPG",
    "+ soil water + groundwater": "PMaPSG",}
    
    rng = np.random.default_rng(random_state)

    df = results_df.copy()

    # Try to detect the predictor-set column name
    possible_set_cols = ["predictor_set", "feature_set", "set_name", "model_name"]
    set_col = None
    for c in possible_set_cols:
        if c in df.columns:
            set_col = c
            break

    if set_col is None:
        raise ValueError(
            "Could not find predictor-set column. Expected one of: "
            f"{possible_set_cols}. Available columns: {list(df.columns)}"
        )

    if metric_col not in df.columns:
        raise ValueError(
            f"Metric column '{metric_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[df[set_col].isin(predictor_order)].copy()
    df[set_col] = pd.Categorical(df[set_col], categories=predictor_order, ordered=True)
    df = df.sort_values(set_col)

    x_positions = np.arange(len(predictor_order))

    medians = []
    means = []

    for i, set_name in enumerate(predictor_order):
        vals = (
            df.loc[df[set_col] == set_name, metric_col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
        )

        if len(vals) == 0:
            medians.append(np.nan)
            means.append(np.nan)
            continue

        medians.append(np.nanmedian(vals))
        means.append(np.nanmean(vals))

        color = "tab:red" if set_name == final_set_name else "0.45"

        # Left-side fold values
        x_points = i - 0.13 + rng.normal(0, jitter, size=len(vals))
        ax.scatter(
            x_points,
            vals,
            s=point_size,
            alpha=point_alpha,
            color=color,
            edgecolor="none",
            zorder=3,
        )

        # Right-side half violin
        # parts = ax.violinplot(
        #     vals,
        #     positions=[i + 0.10],
        #     widths=violin_width,
        #     showmeans=False,
        #     showmedians=False,
        #     showextrema=False,
        # )

        # for body in parts["bodies"]:
        #     body.set_facecolor(color)
        #     body.set_edgecolor("none")
        #     body.set_alpha(0.28)

        #     # Clip violin to right half
        #     path = body.get_paths()[0]
        #     verts = path.vertices
        #     verts[:, 0] = np.maximum(verts[:, 0], i + 0.10)

        # Median marker
        ax.scatter(
            i,
            np.nanmedian(vals),
            s=32,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )

        # Mean ± SD as thin vertical interval, optional but useful
        # mean_val = np.nanmean(vals)
        # sd_val = np.nanstd(vals)
        # ax.plot(
        #     [i, i],
        #     [mean_val - sd_val, mean_val + sd_val],
        #     color=color,
        #     lw=1.0,
        #     alpha=0.65,
        #     zorder=4,
        # )

    # Median-connecting curve
    ax.plot(
        x_positions,
        medians,
        color="black",
        lw=line_width,
        marker="o",
        markersize=3.5,
        zorder=6,
    )

    # Add Δ labels relative to previous median
    for i in range(1, len(predictor_order)):
        if np.isfinite(medians[i]) and np.isfinite(medians[i - 1]):
            delta = medians[i] - medians[i - 1]
            ax.text(
                i,
                medians[i],
                f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="black",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [PREDICTOR_LABELS.get(x, x) for x in predictor_order],
        rotation=0,ha="center",fontsize=8)

    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Slight x padding so left points and right violins fit
    ax.set_xlim(-0.55, len(predictor_order) - 0.45)

    return ax


def plot_peak_occurrence_pdp_figure_from_saved(
    peak_result,
    occurrence_result,
    pdp_results,
    predictor_order,
    out_file=None):
    
    fig = plt.figure(figsize=(7.8, 7), dpi=1200)

    gs = GridSpec(
        nrows=3,
        ncols=2,
        height_ratios=[0.9, 1.1, 1.1],
        hspace=0.35,
        wspace=0.35,)

    # First row: wider panels, slightly smaller gap
    gs_top = gs[0, :].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1, 1],
        wspace=0.35,
    )
    
    ax_gain_peak = fig.add_subplot(gs_top[0, 0])
    ax_gain_occ = fig.add_subplot(gs_top[0, 1])

    ax_pdp_rain_peak = fig.add_subplot(gs[1, 0])
    ax_pdp_rain_occ = fig.add_subplot(gs[1, 1])

    ax_pdp_wcr_peak = fig.add_subplot(gs[2, 0])
    ax_pdp_wcr_occ = fig.add_subplot(gs[2, 1])

    # Row 1: predictive gain
    plot_stepwise_gain_half_violin_ax(
        results_df=peak_result["results_df"],
        ax=ax_gain_peak,
        predictor_order=predictor_order,
        metric_col="R2",
        ylabel="R²",
        final_set_name=predictor_order[-1],
    )
    ax_gain_peak.set_title(
        "(a)                    Peak discharge",
        loc="left",
        fontweight="bold",
    )

    plot_stepwise_gain_half_violin_ax(
        results_df=occurrence_result["results_df"],
        ax=ax_gain_occ,
        predictor_order=predictor_order,
        metric_col="ROC_AUC",
        ylabel="ROC-AUC",
        final_set_name=predictor_order[-1],
    )
    ax_gain_occ.set_title(
        "(b)                    Flood occurance",
        loc="left",
        fontweight="bold",
    )

    # Row 2: dtp × precipitation
    cf1 = plot_2d_pdp_ax(
        ax=ax_pdp_rain_peak,
        pdp=pdp_results["peak_wcr_precip"], #"peak_wcr_precip"
        xlabel= "Soil moisture", #, "Relative groundwater depth (m)", #"Soil moisture",
        ylabel= "Event precipitation (mm)",
        title="(c)", # PDP: groundwater × event precipitation
        cmap="YlOrRd",
    )
    cbar1 = fig.colorbar(cf1, ax=ax_pdp_rain_peak, fraction=0.046, pad=0.02)
    cbar1.set_label("Predicted log peak discharge")

    cf2 = plot_2d_pdp_ax(
        ax=ax_pdp_rain_occ,
        pdp=pdp_results["occurrence_wcr_precip"], #"occurrence_dtp_precip"
        xlabel= "Soil moisture", # "Relative groundwater depth (m)", #"Soil moisture",
        ylabel="Event precipitation (mm)",
        title="(d)", # PDP: groundwater × event precipitation
        cmap="YlGnBu",
    )
    cbar2 = fig.colorbar(cf2, ax=ax_pdp_rain_occ, fraction=0.046, pad=0.02)
    cbar2.set_label("Predicted flood occurrence")

    # Row 3: dtp × soil water
    cf3 = plot_2d_pdp_ax(
        ax=ax_pdp_wcr_peak,
        pdp=pdp_results["peak_dtp_wcr"],
        xlabel="Relative groundwater depth (m)",
        ylabel="Soil moisture",
        title="(e)", # PDP: groundwater × soil water
        cmap="YlOrRd",
    )
    cbar3 = fig.colorbar(cf3, ax=ax_pdp_wcr_peak, fraction=0.046, pad=0.02)
    cbar3.set_label("Predicted log peak discharge")

    cf4 = plot_2d_pdp_ax(
        ax=ax_pdp_wcr_occ,
        pdp=pdp_results["occurrence_dtp_wcr"],
        xlabel="Relative groundwater depth (m)",
        ylabel="Soil moisture",
        title="(f)", # PDP: groundwater × soil water
        cmap="YlGnBu",
    )
    cbar4 = fig.colorbar(cf4, ax=ax_pdp_wcr_occ, fraction=0.046, pad=0.02)
    cbar4.set_label("Predicted flood occurrence")
    for ax in [
    # ax_gain_peak,
    # ax_gain_occ,
    ax_pdp_rain_peak,
    ax_pdp_rain_occ,
    ax_pdp_wcr_peak,
    ax_pdp_wcr_occ]:
        apply_clean_tick_format(ax)
    # Align y labels within each column
    align_ylabels_by_column(
        fig,
        left_axes=[
            ax_gain_peak,
            ax_pdp_rain_peak,
            ax_pdp_wcr_peak,
        ],
        right_axes=[
            ax_gain_occ,
            ax_pdp_rain_occ,
            ax_pdp_wcr_occ,
        ],
        x_left=-0.13,
        x_right=-0.13,
    )
    
    if out_file is not None:
        fig.savefig(out_file, dpi=1200, bbox_inches="tight")
        print(f"Saved figure: {out_file}")

    return fig

def plot_custom_shap_beeswarm_ax(
    explanation,
    ax,
    short_labels=None,
    max_display=10,
    point_size=11,
    alpha=0.55,
    random_state=42,
):
    """
    Custom SHAP beeswarm-like panel that works inside an existing axis.
    """
    rng = np.random.default_rng(random_state)

    shap_values = np.asarray(explanation.values)
    feature_values = np.asarray(explanation.data)
    feature_names = np.asarray(explanation.feature_names)

    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected 2-D SHAP values; obtained {shap_values.shape}"
        )

    if feature_values.shape != shap_values.shape:
        raise ValueError(
            "SHAP data and values have inconsistent shapes: "
            f"{feature_values.shape} versus {shap_values.shape}"
        )

    importance = np.nanmean(np.abs(shap_values), axis=0)
    order = np.argsort(importance)[::-1][:max_display]

    # Reverse so the most important feature appears at the top
    order = order[::-1]

    scatter_for_colorbar = None

    for row, feature_idx in enumerate(order):
        sv = shap_values[:, feature_idx]
        fv = feature_values[:, feature_idx]

        valid = np.isfinite(sv) & np.isfinite(fv)
        sv = sv[valid]
        fv = fv[valid]

        if len(sv) == 0:
            continue

        # Robust normalization of feature values for colour
        q_low, q_high = np.nanquantile(fv, [0.05, 0.95])

        if q_high > q_low:
            color_value = np.clip(
                (fv - q_low) / (q_high - q_low),
                0,
                1,
            )
        else:
            color_value = np.full(len(fv), 0.5)

        jitter = rng.normal(
            loc=0,
            scale=0.10,
            size=len(sv),
        )

        scatter_for_colorbar = ax.scatter(
            sv,
            row + jitter,
            c=color_value,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    labels = []
    for feature_idx in order:
        feature = feature_names[feature_idx]
        labels.append(
            short_labels.get(feature, feature)
            if short_labels is not None
            else feature
        )

    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="0.35", linewidth=0.8)
    ax.grid(axis="x", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("SHAP value")

    return scatter_for_colorbar

def plot_peak_occurrence_custom_shap(
    peak_result,
    occurrence_result,
    short_labels,
    out_file=None,
    max_display=100,
):
    peak_exp = peak_result["shap"]["shap_values"]
    occ_exp = occurrence_result["shap"]["shap_values"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.08, 3.7),
        constrained_layout=True,dpi=1200
    )

    sc1 = plot_custom_shap_beeswarm_ax(
        explanation=peak_exp,
        ax=axes[0],
        short_labels=short_labels,
        max_display=max_display,
        random_state=42,
    )

    axes[0].set_title(
        "(a) Peak discharge",
        loc="left",
        fontweight="bold",
    )

    sc2 = plot_custom_shap_beeswarm_ax(
        explanation=occ_exp,
        ax=axes[1],
        short_labels=short_labels,
        max_display=max_display,
        random_state=43,
    )

    axes[1].set_title(
        "(b) Flood occurrence",
        loc="left",
        fontweight="bold",
    )

    # One common feature-value colour bar
    cbar = fig.colorbar(
        sc2,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value")

    if out_file is not None:
        fig.savefig(
            out_file,
            dpi=600,
            bbox_inches="tight",
        )
        print(f"Saved figure: {out_file}")

    return fig


def plot_single_target_pdp_figure_from_saved(
    result,
    pdp_results,
    predictor_order,
    target_type="peak",     # "peak" or "occurrence"
    out_file=None,
):
    """
    Plot predictive performance and three 2-D PDP surfaces
    for ONE target only.

    Layout
    ------
    (a) Stepwise predictive performance
    (b) Soil moisture × event precipitation
    (c) Groundwater × event precipitation
    (d) Groundwater × soil moisture

    Parameters
    ----------
    result : dict
        Either all_results["peak"] or all_results["occurrence"].

    pdp_results : dict
        Saved output from compute_peak_occurrence_pdp_surfaces().

    predictor_order : list
        Predictor sets shown in the stepwise performance plot.

    target_type : str
        "peak" or "occurrence".

    out_file : str or Path, optional
        Output figure path.
    """

    # ==================================================
    # Target-dependent settings
    # ==================================================

    if target_type == "peak":

        metric_col = "R2"
        metric_ylabel = "R²"

        pdp_wcr_precip = pdp_results["peak_wcr_precip"]
        pdp_dtp_precip = pdp_results["peak_dtp_precip"]
        pdp_dtp_wcr = pdp_results["peak_dtp_wcr"]

        cmap = "YlOrRd"

        cbar_label = "Predicted log peak discharge"

        figure_title = "Peak discharge"

    elif target_type == "occurrence":

        metric_col = "ROC_AUC"
        metric_ylabel = "ROC-AUC"

        pdp_wcr_precip = pdp_results["occurrence_wcr_precip"]
        pdp_dtp_precip = pdp_results["occurrence_dtp_precip"]
        pdp_dtp_wcr = pdp_results["occurrence_dtp_wcr"]

        cmap = "YlGnBu"

        cbar_label = "Predicted flood probability"

        figure_title = "Flood occurrence"

    else:

        raise ValueError(
            "target_type must be either 'peak' or 'occurrence'"
        )

    # ==================================================
    # Figure
    # ==================================================

    fig = plt.figure(
        figsize=(7.08, 5.0),
        dpi=1200,
    )

    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        height_ratios=[1, 1],
        width_ratios=[1, 1],
        hspace=0.20,
        wspace=0.30,
    )

    # Each subplot spans both columns
    ax_gain = fig.add_subplot(gs[0, 0])

    ax_wcr_precip = fig.add_subplot(gs[0, 1])

    ax_dtp_precip = fig.add_subplot(gs[1, 0])

    ax_dtp_wcr = fig.add_subplot(gs[1, 1])

    # ==================================================
    # (a) Stepwise predictive performance
    # ==================================================
    plot_stepwise_gain_half_violin_ax(
        results_df=result["results_df"],
        ax=ax_gain,
        predictor_order=predictor_order,
        metric_col=metric_col,
        ylabel=metric_ylabel,
        final_set_name=predictor_order[-1],
    )

    # ax_gain.set_title(f"(a)",loc="left",fontweight="bold")
    plt.text(0.1, 0.95, '(a)', 
             horizontalalignment='center',verticalalignment='center',
             transform=ax_gain.transAxes, 
             fontweight="bold")

    # ==================================================
    # (b) Soil moisture × precipitation
    # ==================================================
    cf1 = plot_2d_pdp_ax(
        ax=ax_wcr_precip,
        pdp=pdp_wcr_precip,
        xlabel="Soil moisture",
        ylabel="Event precipitation (mm)",
        cmap=cmap,
    )
    plt.text(0.1, 0.95, '(b)', 
             horizontalalignment='center',verticalalignment='center',
             transform=ax_wcr_precip.transAxes, 
             fontweight="bold")

    # cbar1 = fig.colorbar(
    #     cf1,
    #     ax=ax_wcr_precip,
    #     fraction=0.025,
    #     pad=0.02,
    # )

    # cbar1.set_label(
    #     cbar_label
    # )

    # ==================================================
    # (c) Groundwater × precipitation
    # ==================================================
    cf2 = plot_2d_pdp_ax(
        ax=ax_dtp_precip,
        pdp=pdp_dtp_precip,
        xlabel="Relative groundwater depth (m)",
        ylabel="Event precipitation (mm)",
        cmap=cmap,
    )
    plt.text(0.1, 0.95, '(c)', 
             horizontalalignment='center',verticalalignment='center',
             transform=ax_dtp_precip.transAxes, 
             fontweight="bold")

    # cbar2 = fig.colorbar(
    #     cf2,
    #     ax=ax_dtp_precip,
    #     fraction=0.025,
    #     pad=0.02,
    # )

    # cbar2.set_label(
    #     cbar_label
    # )

    # ==================================================
    # (d) Groundwater × soil moisture
    # ==================================================
    cf3 = plot_2d_pdp_ax(
        ax=ax_dtp_wcr,
        pdp=pdp_dtp_wcr,
        xlabel="Relative groundwater depth (m)",
        ylabel="Soil moisture",
        cmap=cmap,
    )
    plt.text(0.1, 0.95, '(d)', 
             horizontalalignment='center',verticalalignment='center',
             transform=ax_dtp_wcr.transAxes, 
             fontweight="bold")
    # cbar3 = fig.colorbar(
    #     cf3,
    #     ax=ax_dtp_wcr,
    #     fraction=0.025,
    #     pad=0.02,
    # )

    # cbar3.set_label(
    #     cbar_label
    # )

    # ==================================================
    # Tick formatting
    # ==================================================

    for ax in [
        ax_wcr_precip,
        ax_dtp_precip,
        ax_dtp_wcr,
    ]:
        apply_clean_tick_format(ax)

    # ==================================================
    # Align y labels
    # ==================================================

    fig.canvas.draw()

    for ax in [
        ax_gain,
        ax_wcr_precip,
        ax_dtp_precip,
        ax_dtp_wcr,
    ]:
        ax.yaxis.set_label_coords(-0.18,0.5,)

    # ==================================================
    # Save
    # ==================================================
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    
        # Main text
        "font.size": 8,
    
        # Axes
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.7,
    
        # Ticks
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
    
        # Legend
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        "legend.frameon": False,
    
        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
    
        # Saving
        "savefig.dpi": 600,
    })
    
    if out_file is not None:

        fig.savefig(
            out_file,
            dpi=1200,
            bbox_inches="tight",
        )

        print(
            f"Saved figure: {out_file}"
        )

    return fig


# ==================================================
#%% Main workflow
# ==================================================
if __name__ == "__main__":

    dataset0 = load_dataset(scale=SCALE)
    dataset0 = dataset0[~dataset0["flood_occurrence"].isna()].reset_index(drop=True)

    all_results = {}

    # --------------------------------------------------
    # Step 1: run peak and occurrence models
    # --------------------------------------------------
    for target_type in TARGET_TYPES:

        print("\n" + "#" * 90)
        print(f"Running target: {target_type}")
        print("#" * 90)

        target_info = prepare_target(dataset0, target_type)
        dataset = target_info["dataset"]

        results_df, summary_df = compare_predictor_sets_tabpfn(
            dataset=dataset,
            target_info=target_info,
            feature_sets=FEATURE_SETS,
            group_col="catch_id",
            n_splits=N_SPLITS,
            max_train_size=MAX_TRAIN_SIZE_PER_FOLD,
            random_state=RANDOM_STATE,
        )

        print("\nSummary:")
        print(summary_df)

        if RUN_SHAP:
            shap_result = train_final_model_and_shap(
                dataset=dataset,
                target_info=target_info,
                feature_cols=FINAL_FEATURES,
                group_col="catch_id",
                out_file=f"../results/shap_result_{SCALE}_{target_type}.pkl",
                max_train_size=MAX_TRAIN_SIZE_PER_FOLD,
                shap_background_size=SHAP_BACKGROUND_SIZE,
                shap_explain_size=SHAP_EXPLAIN_SIZE,
                random_state=RANDOM_STATE,
            )
        else:
            with open(f"../results/shap_result_{SCALE}_{target_type}.pkl", "rb") as f:
                shap_result = pickle.load(f)

        all_results[target_type] = {
            "target_info": target_info,
            "results_df": results_df,
            "summary_df": summary_df,
            "shap": shap_result,
        }

    # --------------------------------------------------
    # Step 2: compute or load PDP after both targets exist
    # --------------------------------------------------
    if RUN_PDP:
        pdp_results = compute_peak_occurrence_pdp_surfaces(
            peak_result=all_results["peak"],
            occurrence_result=all_results["occurrence"],
            grid_resolution=25,
            sample_size=1200,
            random_state=RANDOM_STATE,
            out_file=f"../results/pdp_surfaces_{SCALE}.pkl",
        )
    else:
        with open(f"../results/pdp_surfaces_{SCALE}.pkl", "rb") as f:
            pdp_results = pickle.load(f)

    all_results["peak"]["pdp"] = pdp_results
    all_results["occurrence"]["pdp"] = pdp_results

    # --------------------------------------------------
    # Step 3: plot PDP
    # --------------------------------------------------
    fig_peak = plot_single_target_pdp_figure_from_saved(
    result=all_results["peak"],
    pdp_results=pdp_results,
    predictor_order=PREDICTOR_ORDER[1:],
    target_type="peak",
    out_file=f"../results/predictive_compounding_peak_{SCALE}.png",)
    plt.show()

    fig = plot_peak_occurrence_pdp_figure_from_saved(
        peak_result=all_results["peak"],
        occurrence_result=all_results["occurrence"],
        pdp_results=pdp_results,
        predictor_order=PREDICTOR_ORDER[1:],
        out_file=f"../results/predictive_compounding_pdp_{SCALE}.png",)
    plt.show()
    
    # --------------------------------------------------
    # Step 3: plot SHAP
    # --------------------------------------------------
    fig_shap = plot_peak_occurrence_custom_shap(
    peak_result=all_results["peak"],
    occurrence_result=all_results["occurrence"],
    short_labels=SHORT_LABELS,
    out_file=f"../results/shap_peak_occurrence_{SCALE}.png",
    max_display=50) 
    plt.show()