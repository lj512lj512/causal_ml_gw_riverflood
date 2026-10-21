# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:47:17 2026

@author: juliu
"""

# -*- coding: utf-8 -*-
"""
A2_causal_model_train.py - revised for the resubmission run.

Changes relative to the previous version (all marked with >>> tags):
 1. ANTECEDENT_LAG_DAYS switch: extract dtp/wcr on rain onset day (0, old
    behaviour) or the day before onset (1, recommended so "antecedent" is
    strictly pre-event).
 2. mean_dtp captured per catchment BEFORE the anomaly transform and carried
    into the outputs (needed for Figure 5).
 3. Per-event raw outcome (y0_raw) stored with the causal effects.
 4. Absolute-scale effects (delta_abs) computed per contrast by
    back-transforming the log-scale tau:
       duration -> days, volume -> raw volume units, peak -> raw discharge.
 5. headline_summary table: for each contrast, one row per outcome with
    - percent / percentage-point effect (unconditional, matches Figure 4)
    - conditional percent effect among flood events (intensive margin)
    - conditional absolute effect among flood events (days, volume units)
    - baseline statistics for anchoring sentences
 6. State-stability generalized to all contrasts (was hardcoded to
    event_plus10).
"""
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from sklearn.model_selection import GroupKFold
from econml.dml import CausalForestDML
from lightgbm import LGBMRegressor, LGBMClassifier

from A0_detect_events import load_camels_info, load_camels_dynamic, load_camels_static

# >>> NEW: 0 = extract antecedent states on rain onset day (old behaviour);
#          1 = extract on the day before onset (strictly pre-event).
ANTECEDENT_LAG_DAYS = 1

ealpha_e_features = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07',
       'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
       'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
       'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
       'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47',
       'A48', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57',
       'A58', 'A59', 'A60', 'A61', 'A62', 'A63']


def load_dataset(scale="catch"):

    # Load dynamics
    dynamics = {}
    for DKM_id in range(1, 7):
        with open(rf"../data/alphaearth/shp_df_dtp_wcr_DKM{DKM_id}.pkl", "rb") as f:
            dynamics.update(pickle.load(f))

    # Load statics
    with open(r"../data/alphaearth/shp_df_ee_features.pkl", "rb") as f:
        statics = pickle.load(f)

    # Load events
    all_events = pd.read_csv(r"../data/rainfall_runoff_events_all_304_catchments.csv", index_col="Unnamed: 0")
    all_events = all_events[~all_events['catch_id'].isin([82100426.0, 37710002.0, 91500006.0, 91200485.0, 91300007.0])]  # remove Bornholm
    all_events["rain_start_date"] = pd.to_datetime(all_events["rain_start_date"])
    all_events["catch_id_float"] = all_events["catch_id"].astype(float)

    # >>> NEW: antecedent extraction date (day before onset if lag = 1)
    all_events["antecedent_date"] = (
        all_events["rain_start_date"] - pd.Timedelta(days=ANTECEDENT_LAG_DAYS)
    )

    # -------------------------
    # Static EE features
    # -------------------------
    static_rows = []
    for catch_id in all_events["catch_id_float"].unique():
        vals = {"catch_id_float": catch_id}
        df_static = statics[catch_id][f"df_ee_{scale}"]
        for eef in ealpha_e_features:
            vals[eef] = df_static[eef].values[0]
        static_rows.append(vals)
    static_df = pd.DataFrame(static_rows)
    all_events = all_events.merge(static_df, on="catch_id_float", how="left")

    # -------------------------
    # Dynamic DTP / WCR features
    # -------------------------
    dynamic_rows = []

    for catch_id in all_events["catch_id_float"].unique():
        mask = all_events["catch_id_float"] == catch_id
        needed_onset = all_events.loc[mask, "rain_start_date"].unique()
        needed_dates = all_events.loc[mask, "antecedent_date"].unique()   # >>> CHANGED

        dtp_df = dynamics[catch_id][f"dtp_{scale}"].copy()

        # >>> NEW: keep the catchment temporal mean BEFORE building the anomaly
        mean_dtp_value = dtp_df["dtp"].mean()

        # relative groundwater anomaly (higher = shallower in this data)
        dtp_df["dtp"] = dtp_df["dtp"] - mean_dtp_value

        dtp_df["dtp_quantile"] = dtp_df["dtp"].rank(pct=True)
        dtp_df["dtp_state"] = pd.cut(dtp_df["dtp_quantile"],
                                     bins=[0, 0.50, 0.75, 0.90, 1.00],
                                     labels=["below_Q50", "Q50_Q75", "Q75_Q90", "above_Q90"],
                                     include_lowest=True)

        wcr_df = dynamics[catch_id][f"wcr_{scale}"].copy()
        wcr_df["wcr_quantile"] = wcr_df["wcr"].rank(pct=True)
        wcr_df["wcr_state"] = pd.cut(wcr_df["wcr_quantile"],
                                     bins=[0, 0.50, 0.75, 0.90, 1.00],
                                     labels=["below_Q50", "Q50_Q75", "Q75_Q90", "above_Q90"],
                                     include_lowest=True)

        tmp = pd.DataFrame({"catch_id_float": catch_id,
                            "rain_start_date": needed_onset})
        # >>> CHANGED: index dynamics on the antecedent date, keyed back to onset
        tmp["dtp"]           = dtp_df.reindex(needed_dates)["dtp"].values
        tmp["dtp_quantile"]  = dtp_df.reindex(needed_dates)["dtp_quantile"].values
        tmp["dtp_state"]     = dtp_df.reindex(needed_dates)["dtp_state"].values
        tmp["dtp_q50_value"] = dtp_df["dtp"].quantile(0.50)
        tmp["dtp_q75_value"] = dtp_df["dtp"].quantile(0.75)
        tmp["dtp_q90_value"] = dtp_df["dtp"].quantile(0.90)
        tmp["mean_dtp"]      = mean_dtp_value                       # >>> NEW

        tmp["wcr"]           = wcr_df.reindex(needed_dates)["wcr"].values
        tmp["wcr_quantile"]  = wcr_df.reindex(needed_dates)["wcr_quantile"].values
        tmp["wcr_state"]     = wcr_df.reindex(needed_dates)["wcr_state"].values
        tmp["wcr_q50_value"] = wcr_df["wcr"].quantile(0.50)
        tmp["wcr_q75_value"] = wcr_df["wcr"].quantile(0.75)
        tmp["wcr_q90_value"] = wcr_df["wcr"].quantile(0.90)

        dynamic_rows.append(tmp)

    dynamic_df = pd.concat(dynamic_rows, ignore_index=True)
    all_events = all_events.merge(dynamic_df, on=["catch_id_float", "rain_start_date"], how="left")

    # -------------------------
    # Seasonal day-of-year features
    # -------------------------
    doy = all_events["rain_start_date"].dt.dayofyear
    all_events["sdy"] = np.sin(2 * np.pi * doy / 365.25)
    all_events["cdy"] = np.cos(2 * np.pi * doy / 365.25)

    # -------------------------
    # Climate features
    # -------------------------
    datapath = (
        r"\\geodata.geus.dk\DKmodel_users\FloodWarning\LSTM_postprocessing"
        r"\CAMELS-DK\Level_3\Dynamics\Gauged_catchments"
    )

    climate_rows = []
    for catch_id in all_events["catch_id"].unique():
        mask = all_events["catch_id"] == catch_id
        needed_onset = all_events.loc[mask, "rain_start_date"].unique()
        needed_dates = all_events.loc[mask, "antecedent_date"].unique()   # >>> CHANGED

        camels_ts = pd.read_csv(os.path.join(datapath, f"CAMELS_DK_obs_based_{catch_id}.csv"),
                                index_col="time", parse_dates=True)
        # >>> CHANGED: tem/pet sampled on the antecedent date, keyed to onset
        tmp = camels_ts.reindex(needed_dates)[["temperature", "pet"]].reset_index(drop=True)
        tmp = tmp.rename(columns={"temperature": "tem"})
        tmp["rain_start_date"] = needed_onset
        tmp["catch_id"] = catch_id

        # 3-day antecedent precipitation, excluding the onset day
        precip = camels_ts["precipitation"].fillna(0)
        pre_event_precip_3d = []
        for date in needed_onset:
            date = pd.to_datetime(date).normalize()
            start_date = date - pd.Timedelta(days=3)
            end_date = date - pd.Timedelta(days=1)
            pre_event_precip_3d.append(precip.loc[start_date:end_date].sum())
        tmp["pre_event_precip_3d"] = pre_event_precip_3d

        climate_rows.append(tmp)
    climate_df = pd.concat(climate_rows, ignore_index=True)
    all_events = all_events.merge(climate_df, on=["catch_id", "rain_start_date"], how="left")
    all_events = all_events.drop(columns=["catch_id_float"])
    return all_events


def make_model_y(target_type, random_state):
    if target_type == "occurrence":
        return LGBMClassifier(
            objective="binary", metric="auc", n_estimators=1000,
            learning_rate=0.03, num_leaves=31, max_depth=-1,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=random_state,
        )
    return LGBMRegressor(
        n_estimators=1000, learning_rate=0.03, num_leaves=31, max_depth=-1,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
        random_state=random_state,
    )


def make_model_t(random_state):
    return LGBMRegressor(
        n_estimators=1000, learning_rate=0.03, num_leaves=31, max_depth=-1,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
        random_state=random_state,
    )


def convert_tau_to_report_units(tau, target_type, is_log_outcome):
    """occurrence: probability -> percentage points; log outcomes: percent."""
    tau = np.asarray(tau).ravel()
    if target_type == "occurrence":
        return 100 * tau, "percentage_points"
    elif is_log_outcome:
        return (np.exp(tau) - 1) * 100, "percent"
    else:
        return tau, "raw_units"


# >>> NEW ------------------------------------------------------------------
def compute_absolute_effect(tau, y0_raw, target_type):
    """
    Back-transform log-scale tau to an absolute per-event effect.

    duration : days       (log1p outcome)
    volume   : raw volume units (log1p outcome)
    peak     : raw discharge units (log(y + 1e-6) outcome)
    occurrence: not applicable (tau is already a probability change)

    Note: for zero-valued events (y0 = 0) the log1p back-transform describes
    flood creation, not intensification; summaries therefore condition on
    y0 > 0 (the intensive margin). Occurrence carries the extensive margin.
    """
    tau = np.asarray(tau).ravel()
    y0 = np.asarray(y0_raw, dtype=float).ravel()
    if target_type in ("volume", "duration"):
        y1 = np.expm1(np.log1p(y0) + tau)
    elif target_type == "peak":
        y1 = (y0 + 1e-6) * np.exp(tau) - 1e-6
    else:
        return np.full_like(tau, np.nan, dtype=float)
    return y1 - y0
# ---------------------------------------------------------------------------


out_dir = Path(r'\\geodata.geus.dk\HOME\causal_ml_gw_riverflood\results')
out_dir.mkdir(parents=True, exist_ok=True)
n_splits = 20

if __name__ == "__main__":
    for scale in ['catch', 'valley', 'downs']:
        dataset = load_dataset(scale=scale)
        dataset = dataset.dropna(subset=["flood_occurrence"]).reset_index(drop=True)

        for target_type in ["peak", "occurrence", "volume", "duration"]:

            if target_type == "peak":
                dataset["peak_discharge_log"] = np.log(dataset["flood_peak"] + 1e-6)
                outcome, raw_outcome = "peak_discharge_log", "flood_peak"
                is_binary_outcome, is_log_outcome, zero_heavy = False, True, False
            elif target_type == "occurrence":
                dataset["flood_occurrence_bool"] = dataset["flood_occurrence"].copy()
                outcome, raw_outcome = "flood_occurrence_bool", "flood_occurrence"
                is_binary_outcome, is_log_outcome, zero_heavy = True, False, False
            elif target_type == "volume":
                dataset["flood_volume_above_q95_log"] = np.log1p(dataset["flood_volume_above_q95"])
                outcome, raw_outcome = "flood_volume_above_q95_log", "flood_volume_above_q95"
                is_binary_outcome, is_log_outcome, zero_heavy = False, True, True
            elif target_type == "duration":
                dataset["duration_above_q95_log"] = np.log1p(dataset["duration_above_q95"])
                outcome, raw_outcome = "duration_above_q95_log", "duration_above_q95"
                is_binary_outcome, is_log_outcome, zero_heavy = False, True, True

            treatment = 'dtp'
            confounders = ["rain_duration_days", "event_precip_mm", "max_1d_precip_mm",
                           "split_from_long_event", 'pre_event_precip_3d',
                           "wcr", "sdy", "cdy", "tem", "pet"]
            heterogeneity_features = ealpha_e_features + ["sdy", "cdy"]
            extra_state_cols = ["dtp_q50_value", "dtp_q75_value", "dtp_q90_value",
                                "mean_dtp"]                              # >>> CHANGED

            group_col = 'catch_id'
            cols = [outcome, raw_outcome, treatment, group_col] + confounders \
                   + heterogeneity_features + extra_state_cols
            df = dataset[cols].dropna().copy()
            Y = df[outcome].values
            T = df[treatment].values
            W = df[confounders].values
            X = df[heterogeneity_features].values
            groups = df[group_col].values

            gkf = GroupKFold(n_splits=n_splits)

            # -----------------------------
            # Nuisance evaluation (unchanged)
            # -----------------------------
            results = []
            for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), start=1):
                print(f"Evaluating nuisance model fold {fold}")
                model_y = make_model_y(target_type=target_type, random_state=fold)
                model_t = make_model_t(random_state=fold + 100)
                XW_train = np.column_stack([X[train_idx], W[train_idx]])
                XW_test = np.column_stack([X[test_idx], W[test_idx]])
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                T_train, T_test = T[train_idx], T[test_idx]
                model_y.fit(XW_train, Y_train)
                model_t.fit(XW_train, T_train)

                if is_binary_outcome:
                    Y_pred = model_y.predict_proba(XW_test)[:, 1]
                    Y_pred_clip = np.clip(Y_pred, 1e-6, 1 - 1e-6)
                    if len(np.unique(Y_test)) == 2:
                        Y_auc = roc_auc_score(Y_test, Y_pred)
                        Y_pr_auc = average_precision_score(Y_test, Y_pred)
                        Y_log_loss = log_loss(Y_test, Y_pred_clip)
                    else:
                        Y_auc = Y_pr_auc = Y_log_loss = np.nan
                    Y_brier = brier_score_loss(Y_test, Y_pred)
                    Y_R2 = np.nan
                    Y_RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))
                    Y_MAE = mean_absolute_error(Y_test, Y_pred)
                    Y_positive_auc = Y_positive_pr_auc = np.nan
                    positive_rate = Y_test.mean()
                else:
                    Y_pred = model_y.predict(XW_test)
                    Y_R2 = r2_score(Y_test, Y_pred)
                    Y_RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))
                    Y_MAE = mean_absolute_error(Y_test, Y_pred)
                    Y_auc = Y_pr_auc = Y_brier = Y_log_loss = np.nan
                    if zero_heavy:
                        raw_test = df.iloc[test_idx][raw_outcome].values
                        has_positive = (raw_test > 0).astype(int)
                        if len(np.unique(has_positive)) == 2:
                            Y_positive_auc = roc_auc_score(has_positive, Y_pred)
                            Y_positive_pr_auc = average_precision_score(has_positive, Y_pred)
                        else:
                            Y_positive_auc = Y_positive_pr_auc = np.nan
                        positive_rate = has_positive.mean()
                    else:
                        Y_positive_auc = Y_positive_pr_auc = positive_rate = np.nan

                T_pred = model_t.predict(XW_test)
                results.append({
                    "fold": fold, "n_test": len(test_idx),
                    "Y_R2": Y_R2, "Y_RMSE": Y_RMSE, "Y_MAE": Y_MAE,
                    "Y_AUC": Y_auc, "Y_PR_AUC": Y_pr_auc,
                    "Y_Brier": Y_brier, "Y_LogLoss": Y_log_loss,
                    "positive_rate": positive_rate,
                    "Y_positive_AUC": Y_positive_auc,
                    "Y_positive_PR_AUC": Y_positive_pr_auc,
                    "T_R2": r2_score(T_test, T_pred),
                    "T_RMSE": np.sqrt(mean_squared_error(T_test, T_pred)),
                    "T_MAE": mean_absolute_error(T_test, T_pred),
                })
            results_df = pd.DataFrame(results)
            print(results_df.mean(numeric_only=True))

            # -----------------------------
            # Causal estimation
            # -----------------------------
            all_effects = []
            df["dtp_percentile"] = df.groupby(group_col)[treatment].rank(pct=True)
            df["dtp_state"] = pd.cut(df["dtp_percentile"],
                                     bins=[0, 0.50, 0.75, 0.90, 1.00],
                                     labels=["below_Q50", "Q50_Q75", "Q75_Q90", "above_Q90"],
                                     include_lowest=True)

            for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), start=1):
                print(f"Evaluating causal model fold {fold}")
                model_y_fold = make_model_y(target_type=target_type, random_state=fold)
                model_t_fold = make_model_t(random_state=fold + 100)

                cf = CausalForestDML(
                    model_y=model_y_fold, model_t=model_t_fold,
                    n_estimators=1000, min_samples_leaf=20,
                    discrete_treatment=False, discrete_outcome=is_binary_outcome,
                    cv=3, random_state=fold + 200,
                )
                cf.fit(Y=Y[train_idx], T=T[train_idx], X=X[train_idx], W=W[train_idx])

                X_test = X[test_idx]
                T_test = T[test_idx]
                q50_test = df.iloc[test_idx]["dtp_q50_value"].values
                q75_test = df.iloc[test_idx]["dtp_q75_value"].values
                q90_test = df.iloc[test_idx]["dtp_q90_value"].values
                y0_raw_test = df.iloc[test_idx][raw_outcome].values     # >>> NEW

                contrasts = {
                    "event_plus10": {"description": "Observed event state + 10 cm",
                                     "T0": T_test, "T1": T_test + 0.1},
                    "mean_plus10": {"description": "Mean anomaly + 10 cm",
                                    "T0": np.zeros_like(T_test),
                                    "T1": np.zeros_like(T_test) + 0.1},
                    "q50_to_q75": {"description": "Q50 (moderate) to Q75 (high) state",
                                   "T0": q50_test, "T1": q75_test},
                    "q75_to_q90": {"description": "Q75 (high) to Q90 (very high) state",
                                   "T0": q75_test, "T1": q90_test},
                    "q90_plus10": {"description": "Q90 state + 10 cm",
                                   "T0": q90_test, "T1": q90_test + 0.1},
                }

                fold_effects = pd.DataFrame({
                    "fold": fold,
                    "row_index": df.index[test_idx],
                    "catch_id": df.iloc[test_idx][group_col].values,
                    "dtp": df.iloc[test_idx][treatment].values,
                    "dtp_percentile": df.iloc[test_idx]["dtp_percentile"].values,
                    "dtp_state": df.iloc[test_idx]["dtp_state"].values,
                    "dtp_q75_value": q75_test,
                    "dtp_q90_value": q90_test,
                    "delta_q75_q90": q90_test - q75_test,
                    "mean_dtp": df.iloc[test_idx]["mean_dtp"].values,   # >>> NEW
                    "y0_raw": y0_raw_test,                              # >>> NEW
                })

                contrast_metadata = {}
                for contrast_name, contrast_def in contrasts.items():
                    print(f"  Estimating contrast: {contrast_name}")
                    T0, T1 = contrast_def["T0"], contrast_def["T1"]
                    tau = np.asarray(cf.effect(X_test, T0=T0, T1=T1)).ravel()

                    try:
                        tau_lb, tau_ub = cf.effect_interval(X_test, T0=T0, T1=T1, alpha=0.05)
                        tau_lb = np.asarray(tau_lb).ravel()
                        tau_ub = np.asarray(tau_ub).ravel()
                    except Exception as e:
                        print(f"  effect_interval not available for {contrast_name}: {e}")
                        tau_lb = np.full_like(tau, np.nan, dtype=float)
                        tau_ub = np.full_like(tau, np.nan, dtype=float)

                    tau_report, effect_unit = convert_tau_to_report_units(tau, target_type, is_log_outcome)
                    tau_lb_report, _ = convert_tau_to_report_units(tau_lb, target_type, is_log_outcome)
                    tau_ub_report, _ = convert_tau_to_report_units(tau_ub, target_type, is_log_outcome)

                    # >>> NEW: absolute per-event effect (days / volume units / discharge)
                    delta_abs = compute_absolute_effect(tau, y0_raw_test, target_type)

                    fold_effects[f"{contrast_name}_T0"] = T0
                    fold_effects[f"{contrast_name}_T1"] = T1
                    fold_effects[f"{contrast_name}_delta_T"] = T1 - T0
                    fold_effects[f"{contrast_name}_tau"] = tau
                    fold_effects[f"{contrast_name}_tau_report"] = tau_report
                    fold_effects[f"{contrast_name}_tau_lb_report"] = tau_lb_report
                    fold_effects[f"{contrast_name}_tau_ub_report"] = tau_ub_report
                    fold_effects[f"{contrast_name}_delta_abs"] = delta_abs   # >>> NEW

                    contrast_metadata[contrast_name] = {
                        "description": contrast_def["description"],
                        "effect_unit": effect_unit,
                    }

                all_effects.append(fold_effects)

            effects_event_df = pd.concat(all_effects, ignore_index=True)
            print("Finished all folds.")

            # -----------------------------
            # Summaries
            # -----------------------------
            contrast_names = ["event_plus10", "mean_plus10", "q50_to_q75",
                              "q75_to_q90", "q90_plus10"]
            contrast_report_cols = [f"{c}_tau_report" for c in contrast_names]

            overall_contrast_summary_df = pd.DataFrame([{
                "contrast": c, "n": effects_event_df[f"{c}_tau_report"].notna().sum(),
                "mean_effect": effects_event_df[f"{c}_tau_report"].mean(),
                "median_effect": effects_event_df[f"{c}_tau_report"].median(),
                "std_effect": effects_event_df[f"{c}_tau_report"].std(),
                "p05_effect": np.nanpercentile(effects_event_df[f"{c}_tau_report"], 5),
                "p95_effect": np.nanpercentile(effects_event_df[f"{c}_tau_report"], 95),
            } for c in contrast_names])

            fold_contrast_results_df = (effects_event_df
                                        .groupby("fold")[contrast_report_cols]
                                        .mean().reset_index())

            contrast_stability_df = pd.DataFrame([{
                "contrast": c,
                "n_folds": fold_contrast_results_df[f"{c}_tau_report"].notna().sum(),
                "mean_of_fold_means": fold_contrast_results_df[f"{c}_tau_report"].mean(),
                "std_of_fold_means": fold_contrast_results_df[f"{c}_tau_report"].std(),
                "min_fold_mean": fold_contrast_results_df[f"{c}_tau_report"].min(),
                "max_fold_mean": fold_contrast_results_df[f"{c}_tau_report"].max(),
                "positive_folds": int((fold_contrast_results_df[f"{c}_tau_report"] > 0).sum()),
                "negative_folds": int((fold_contrast_results_df[f"{c}_tau_report"] < 0).sum()),
            } for c in contrast_names])
            print(contrast_stability_df)

            # >>> NEW ---------------------------------------------------------
            # Headline summary: extensive vs intensive margin, per contrast.
            #  - occurrence: pp effect + baseline flood rate + relative change
            #  - volume/duration/peak: unconditional % effect (Figure 4),
            #    conditional % and ABSOLUTE effect among flood events
            # ------------------------------------------------------------------
            flood_mask = effects_event_df["y0_raw"] > 0
            baseline_stats = {
                "baseline_flood_rate": float(flood_mask.mean()) if target_type == "occurrence"
                                       else np.nan,
                "mean_y0_all_events": float(effects_event_df["y0_raw"].mean()),
                "mean_y0_flood_events": float(effects_event_df.loc[flood_mask, "y0_raw"].mean())
                                        if flood_mask.any() else np.nan,
            }

            headline_rows = []
            for c in contrast_names:
                rep = effects_event_df[f"{c}_tau_report"]
                fold_rep = effects_event_df.groupby("fold")[f"{c}_tau_report"].mean()

                row = {
                    "contrast": c,
                    "effect_unit": effect_unit,
                    "uncond_mean": fold_rep.mean(),
                    "uncond_fold_std": fold_rep.std(),
                }

                if target_type == "occurrence":
                    base = effects_event_df["y0_raw"].mean()  # mean flood frequency
                    row["baseline_flood_rate_pct"] = 100 * base
                    row["relative_increase_pct"] = 100 * fold_rep.mean() / (100 * base) * 100 \
                        if base > 0 else np.nan
                    # i.e. pp effect / baseline pp * 100
                else:
                    cond = effects_event_df.loc[flood_mask]
                    fold_cond_pct = cond.groupby("fold")[f"{c}_tau_report"].mean()
                    fold_cond_abs = cond.groupby("fold")[f"{c}_delta_abs"].mean()
                    row["cond_mean_pct"] = fold_cond_pct.mean()
                    row["cond_fold_std_pct"] = fold_cond_pct.std()
                    row["cond_mean_abs"] = fold_cond_abs.mean()
                    row["cond_fold_std_abs"] = fold_cond_abs.std()
                    row["abs_unit"] = {"duration": "days",
                                       "volume": "raw_volume_units",
                                       "peak": "raw_discharge_units"}[target_type]
                    row["n_flood_events"] = int(flood_mask.sum())
                    row["mean_y0_flood_events"] = baseline_stats["mean_y0_flood_events"]

                headline_rows.append(row)

            headline_summary_df = pd.DataFrame(headline_rows)
            print("HEADLINE SUMMARY (", target_type, scale, ")")
            print(headline_summary_df.round(3).to_string(index=False))
            # ------------------------------------------------------------------

            state_effect_summary_df = (effects_event_df
                                       .groupby("dtp_state", observed=False)[contrast_report_cols]
                                       .agg(["count", "mean", "median", "std"]))

            # >>> CHANGED: state stability for ALL contrasts (long format)
            state_fold = (effects_event_df
                          .groupby(["fold", "dtp_state"], observed=False)[contrast_report_cols]
                          .mean().reset_index())
            state_stability_frames = []
            for c in contrast_names:
                g = (state_fold.groupby("dtp_state", observed=False)[f"{c}_tau_report"]
                     .agg(n_folds="count", mean_of_fold_means="mean",
                          std_of_fold_means="std", min_fold_mean="min",
                          max_fold_mean="max",
                          positive_folds=lambda x: int((x > 0).sum()))
                     .reset_index())
                g.insert(0, "contrast", c)
                state_stability_frames.append(g)
            event_state_stability_df = pd.concat(state_stability_frames, ignore_index=True)

            save_obj = {
                "target_type": target_type,
                "outcome": outcome,
                "treatment": treatment,
                "scale": scale,
                "effect_unit": effect_unit,
                "antecedent_lag_days": ANTECEDENT_LAG_DAYS,          # >>> NEW
                "confounders": confounders,
                "heterogeneity_features": heterogeneity_features,
                "contrast_metadata": contrast_metadata,
                "baseline_stats": baseline_stats,                    # >>> NEW
                "nuisance_results": results_df,
                "causal_event_results": effects_event_df,
                "overall_contrast_summary": overall_contrast_summary_df,
                "fold_contrast_results": fold_contrast_results_df,
                "contrast_stability": contrast_stability_df,
                "headline_summary": headline_summary_df,             # >>> NEW
                "state_effect_summary": state_effect_summary_df,
                "event_state_fold_results": state_fold,              # >>> CHANGED
                "event_state_stability": event_state_stability_df,   # >>> CHANGED
            }

            out_file = out_dir / f"causal_ml_results_{target_type}_{scale}_all_contrasts.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(save_obj, f)
            print(f"Saved results to: {out_file}")