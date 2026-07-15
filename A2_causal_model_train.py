# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:54:28 2026

@author: juliu
"""
import os
import pickle
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt 
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

# from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from econml.dml import CausalForestDML, NonParamDML
from lightgbm import LGBMRegressor, LGBMClassifier

from A0_detect_events import load_camels_info, load_camels_dynamic, load_camels_static

# geology_features = ['pct_aeolain_sand', 'pct_water_deposit', 'pct_marsh',
#                     'pct_marine_sand', 'pct_beach', 'pct_sandy_till', 'pct_till',
#                     'pct_glaf_sand', 'pct_glal_clay', 'pct_down_sand', 'pct_glam_clay',
#                     'chalk_d', 'uaquifer_t', 'uaquifer_d', 'uclay_t', 'usand_t']
# landuse_features = ['pct_forest_levin_2021', 'pct_agriculture_levin_2021',
#                     'pct_water_levin_2021', 'pct_urban_levin_2021',
#                     'pct_naturedry_levin_2021', 'pct_naturewet_levin_2021',]
# soil_features    = ['root_depth', 'pct_sand', 'pct_silt', 'pct_clay',
#                    'pct_organic', 'pct_gravel', 'tawc', 'bulk_density', 'pct_claynor_30',
#                    'pct_claynor_60', 'pct_claynor_100', 'pct_claynor_200',
#                    'pct_fsandno_30', 'pct_fsandno_60', 'pct_fsandno_100',
#                    'pct_fsandno_200', 'pct_gsandno_30', 'pct_gsandno_60',
#                    'pct_gsandno_100', 'pct_gsandno_200', 'FC', 'HCC', 'KS', 'MRC', 'THS',
#                    'WP']
# topo_features    = ['catch_area', 
#                     'elev_mean', 'elev_max', 'elev_median', 'elev_min', 'slope_mean',
#                     'slope_median', 'slope_max', 'slope_min', 'pct_flat_area']

ealpha_e_features =['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07',
       'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
       'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
       'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
       'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47',
       'A48', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57',
       'A58', 'A59', 'A60', 'A61', 'A62', 'A63',]

# known_vars = [
#     'elev_mean',
#     'slope_mean',
#     'catch_area',
#     'wetland_fraction',
#     'agriculture_fraction',
#     'urban_fraction',
#     'forest_fraction',
#     # soil features
#     'clay_fraction',
#     'sand_fraction',
#     'chalk_fraction',
#     # geological features 
#     'mean_dtp', 
# ]

def sin_day_of_year(camels_ts):
    # Make sure index is datetime
    camels_ts.index = pd.to_datetime(camels_ts.index)
    
    # Day of year: 1–365 or 1–366
    camels_ts['dayofyear'] = camels_ts.index.dayofyear
    
    # Use 365.25 to handle leap years smoothly
    camels_ts['doy_sin'] = np.sin(2 * np.pi * camels_ts['dayofyear'] / 365.25)
    camels_ts['doy_cos'] = np.cos(2 * np.pi * camels_ts['dayofyear'] / 365.25)
    return camels_ts
    
def net_prep(camels_ts):
    camels_ts['net_precip'] = camels_ts['precipitation'] - camels_ts['pet']

    for window in [1, 3, 7, 14, 30, 60, 90]:
        camels_ts[f'net_precip_{window}d'] = (
            camels_ts
            .groupby('catch_id')['net_precip']
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=window).sum())
        )
    return camels_ts
    
    

def load_dataset(scale="catch"):
    
    # Load dynamics
    dynamics = {}
    for DKM_id in range(1, 7):
        with open(rf"data/alphaearth/shp_df_dtp_wcr_DKM{DKM_id}.pkl", "rb") as f:
            dynamics.update(pickle.load(f))

    
    # Load statics
    with open(r"data/alphaearth/shp_df_ee_features.pkl", "rb") as f:
        statics = pickle.load(f)


    # Load events
    all_events = pd.read_csv(r"data/rainfall_runoff_events_all_304_catchments.csv", index_col="Unnamed: 0")
    all_events = all_events[~all_events['catch_id'].isin([82100426.0, 37710002.0, 91500006.0, 91200485.0, 91300007.0])] # remove bornhalm
    all_events["rain_start_date"] = pd.to_datetime(all_events["rain_start_date"])
    all_events["catch_id_float"] = all_events["catch_id"].astype(float)

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
    all_events = all_events.merge(static_df,on="catch_id_float",how="left")

    # -------------------------
    # Dynamic DTP / WCR features
    # -------------------------
    dynamic_rows = []
    
    for catch_id in all_events["catch_id_float"].unique():
        needed_dates = all_events.loc[all_events["catch_id_float"] == catch_id,"rain_start_date"].unique()
    
        # use relative dtp
        dtp_df = dynamics[catch_id][f"dtp_{scale}"].copy()
    
        # relative groundwater anomaly
        dtp_df["dtp"] = dtp_df["dtp"] - dtp_df["dtp"].mean()
    
        # catchment-specific groundwater percentile, higher dtp = shallower groundwater in your data
        dtp_df["dtp_quantile"] = dtp_df["dtp"].rank(pct=True)
    
        # optional groundwater-state class
        dtp_df["dtp_state"] = pd.cut(dtp_df["dtp_quantile"],
                                     bins=[0, 0.50, 0.75, 0.90, 1.00],
                                     labels=["below_Q50", "Q50_Q75", "Q75_Q90", "above_Q90"],
                                     include_lowest=True)
    
        # wcr
        wcr_df = dynamics[catch_id][f"wcr_{scale}"].copy()
        # catchment-specific groundwater percentile, higher dtp = shallower groundwater in your data
        wcr_df["wcr_quantile"] = wcr_df["wcr"].rank(pct=True)
    
        # optional groundwater-state class
        wcr_df["wcr_state"] = pd.cut(wcr_df["wcr_quantile"],
                                     bins=[0, 0.50, 0.75, 0.90, 1.00],
                                     labels=["below_Q50", "Q50_Q75", "Q75_Q90", "above_Q90"],
                                     include_lowest=True)
        
        tmp = pd.DataFrame({"catch_id_float": catch_id,"rain_start_date": needed_dates})
    
        tmp["dtp"]           = dtp_df.reindex(needed_dates)["dtp"].values
        tmp["dtp_quantile"]  = dtp_df.reindex(needed_dates)["dtp_quantile"].values
        tmp["dtp_state"]     = dtp_df.reindex(needed_dates)["dtp_state"].values
        tmp["dtp_q50_value"] = dtp_df["dtp"].quantile(0.50)
        tmp["dtp_q75_value"] = dtp_df["dtp"].quantile(0.75)
        tmp["dtp_q90_value"] = dtp_df["dtp"].quantile(0.90)
        
        tmp["wcr"]           = wcr_df.reindex(needed_dates)["wcr"].values
        tmp["wcr_quantile"]  = wcr_df.reindex(needed_dates)["wcr_quantile"].values
        tmp["wcr_state"]     = wcr_df.reindex(needed_dates)["wcr_state"].values
        tmp["wcr_q50_value"] = wcr_df["wcr"].quantile(0.50)
        tmp["wcr_q75_value"] = wcr_df["wcr"].quantile(0.75)
        tmp["wcr_q90_value"] = wcr_df["wcr"].quantile(0.90)
        
        dynamic_rows.append(tmp)
    
    dynamic_df = pd.concat(dynamic_rows, ignore_index=True)

    all_events = all_events.merge(dynamic_df,on=["catch_id_float", "rain_start_date"],how="left")

    # -------------------------
    # Seasonal day-of-year features
    # -------------------------
    doy = all_events["rain_start_date"].dt.dayofyear
    all_events["sdy"] = np.sin(2 * np.pi * doy / 365.25)
    all_events["cdy"] = np.cos(2 * np.pi * doy / 365.25)

    # -------------------------
    # Climate features: read each CSV once per catchment
    # -------------------------
    datapath = (
        r"\\geodata.geus.dk\DKmodel_users\FloodWarning\LSTM_postprocessing"
        r"\CAMELS-DK\Level_3\Dynamics\Gauged_catchments"
    )

    climate_rows = []

    for catch_id in all_events["catch_id"].unique():
        needed_dates = all_events.loc[all_events["catch_id"] == catch_id,"rain_start_date"].unique()
        camels_ts = pd.read_csv(os.path.join(datapath, f"CAMELS_DK_obs_based_{catch_id}.csv"),index_col="time",parse_dates=True)
        tmp = camels_ts.reindex(needed_dates)[["temperature", "pet"]].reset_index()
        tmp = tmp.rename(columns={
            "time": "rain_start_date",
            "temperature": "tem"})
        tmp["catch_id"] = catch_id
        
        
        # --------------------------------------------------
        # 3-day antecedent precipitation
        # Excludes rain_start_date itself
        # --------------------------------------------------
        precip = camels_ts["precipitation"].fillna(0)
        pre_event_precip_3d = []
        for date in needed_dates:
            date = pd.to_datetime(date).normalize()
            start_date = date - pd.Timedelta(days=3)
            end_date = date - pd.Timedelta(days=1)
            value = precip.loc[start_date:end_date].sum()
            pre_event_precip_3d.append(value)
        
        tmp["pre_event_precip_3d"] = pre_event_precip_3d

        climate_rows.append(tmp)
    climate_df = pd.concat(climate_rows, ignore_index=True)
    all_events = all_events.merge(climate_df,on=["catch_id", "rain_start_date"],how="left")
    all_events = all_events.drop(columns=["catch_id_float"])
    return all_events



def make_model_y(target_type, random_state):
    """
    Outcome nuisance model E[Y | X, W].

    Use classifier for binary flood occurrence.
    Use regressor for peak, volume, duration.
    """

    if target_type == "occurrence":
        return LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
        )   


    else:
        return LGBMRegressor(
                    n_estimators=1000,       # capped by early stopping, not tuned
                    learning_rate=0.03,      # small step size, stabilized by many estimators
                    num_leaves=31,           # LightGBM default; fine at this sample size
                    max_depth=-1,            # unconstrained, regularized via num_leaves instead
                    min_child_samples=50,    # slightly higher than default (20) given n=168k
                    subsample=0.8,           # row subsampling for regularization
                    subsample_freq=1,
                    colsample_bytree=0.8,    # feature subsampling, useful with 70 features
                    reg_lambda=1.0,          # mild L2 regularization
                    reg_alpha=0.0,
                    random_state=random_state,
                )
    
def make_model_t(random_state):
    # return ExtraTreesRegressor(
    #     n_estimators=1000,
    #     min_samples_leaf=10,
    #     max_features=1.0,
    #     random_state=random_state,
    #     n_jobs=42,
    # )
    # return TabPFNRegressor(device="cuda",random_state=random_state)
    return LGBMRegressor(
            n_estimators=1000,       # capped by early stopping, not tuned
            learning_rate=0.03,      # small step size, stabilized by many estimators
            num_leaves=31,           # LightGBM default; fine at this sample size
            max_depth=-1,            # unconstrained, regularized via num_leaves instead
            min_child_samples=50,    # slightly higher than default (20) given n=168k
            subsample=0.8,           # row subsampling for regularization
            subsample_freq=1,
            colsample_bytree=0.8,    # feature subsampling, useful with 70 features
            reg_lambda=1.0,          # mild L2 regularization
            reg_alpha=0.0,
            random_state=random_state,
        )

# Choose output folder
out_dir = Path(r"\\geodata.geus.dk\dkmodel_users\FloodWarning\Literature_survey\Causal_ML\results_pickle")
out_dir.mkdir(parents=True, exist_ok=True)
n_splits=20         

if __name__ == "__main__":
    for scale in ['catch', 'valley', 'downs']: #¤'catch', 'valley', 'downs'
        # 0. load dataset
        dataset = load_dataset(scale=scale) # catch, valley, downs
        dataset = dataset.dropna(subset=["flood_occurrence"]).reset_index(drop=True)

        # -----------------------------
        # Choose target
        # -----------------------------
        for target_type in ["peak", "occurrence", "volume", "duration"]: #"peak", "occurrence", "volume", "duration"
            
            # options: "peak", "occurrence", "volume", "duration"
            if target_type == "peak":
                dataset["peak_discharge_log"] = np.log(dataset["flood_peak"] + 1e-6)
                outcome = "peak_discharge_log"
                raw_outcome = "flood_peak"
                is_binary_outcome = False
                is_log_outcome = True
                zero_heavy = False
            
            elif target_type == "occurrence":
                dataset["flood_occurrence_bool"] = dataset["flood_occurrence"].copy()
                outcome = "flood_occurrence_bool"
                raw_outcome = "flood_occurrence"
                is_binary_outcome = True
                is_log_outcome = False
                zero_heavy = False
            
            elif target_type == "volume":
                dataset["flood_volume_above_q95_log"] = np.log1p(dataset["flood_volume_above_q95"])
                outcome = "flood_volume_above_q95_log"
                raw_outcome = "flood_volume_above_q95"
                is_binary_outcome = False
                is_log_outcome = True
                zero_heavy = True
                print("Percentage of events with flood_volume_above_q95 > 0:",100 * (dataset[raw_outcome] > 0).mean())
            
            elif target_type == "duration":
                dataset["duration_above_q95_log"] = np.log1p(dataset["duration_above_q95"])
                outcome = "duration_above_q95_log"
                raw_outcome = "duration_above_q95"
                is_binary_outcome = False
                is_log_outcome = True
                zero_heavy = True
                print("Percentage of events with duration_above_q95 > 0:",100 * (dataset[raw_outcome] > 0).mean())
                
            #%% nuisance
            # -----------------------------
            # 2. test nuisance model performance
            # -----------------------------
            treatment = 'dtp' # # groundwater depth relative to mean 
            
            # Confounders (W) are variables must control for to remove bias. 
            # Because accumulated net precipitation partly controls groundwater recharge, 
            # long antecedent water-balance indices may absorb part of the groundwater preconditioning signal. 
            # We therefore used short-term event precipitation and seasonality in the main causal 
            # specification and treated longer☻ accumulated net precipitation windows as sensitivity controls. 
            # This allowed us to distinguish the total groundwater preconditioning 
            # effect from the residual groundwater effect after accounting for catchment-scale climate memory. 
            confounders = [ "rain_duration_days", 
                           "event_precip_mm", 
                           "max_1d_precip_mm", 
                           "split_from_long_event", 
                           "wcr", # soil water content / soil moisture 
                           "sdy", # sine day of year 
                           "cdy", # cosine day of year 
                           "tem", # temperature 
                           "pet" # potential evapotranspiration 
                           ] 

            # Heterogeneity features (X) are variables used to explain 
            # where the groundwater effect is stronger or weaker. 
            heterogeneity_features = ealpha_e_features + ["sdy","cdy"]
            extra_state_cols = ["dtp_q75_value", "dtp_q90_value"]
            # cross validation
            group_col = 'catch_id'
            cols = [outcome, raw_outcome, treatment, group_col] + confounders + heterogeneity_features + extra_state_cols
            df = dataset[cols].dropna().copy() 
            Y = df[outcome].values 
            T = df[treatment].values 
            W = df[confounders].values 
            X = df[heterogeneity_features].values 
            groups = df[group_col].values
            
            
            gkf = GroupKFold(n_splits=n_splits)
            
            results = []
            for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), start=1):
                print(f"Evaluating nuisance model fold {fold}")
            
                model_y = make_model_y(target_type=target_type, random_state=fold)
                model_t = make_model_t(random_state=fold + 100)
            
                XW_train = np.column_stack([X[train_idx], W[train_idx]])
                XW_test  = np.column_stack([X[test_idx], W[test_idx]])
            
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                T_train, T_test = T[train_idx], T[test_idx]
            
                model_y.fit(XW_train, Y_train)
                model_t.fit(XW_train, T_train)
            
                # -----------------------------
                # Outcome prediction
                # -----------------------------
                if is_binary_outcome:
                    # predicted probability of flood occurrence
                    Y_pred = model_y.predict_proba(XW_test)[:, 1]
                    Y_pred_clip = np.clip(Y_pred, 1e-6, 1 - 1e-6)
            
                    if len(np.unique(Y_test)) == 2:
                        Y_auc = roc_auc_score(Y_test, Y_pred)
                        Y_pr_auc = average_precision_score(Y_test, Y_pred)
                        Y_log_loss = log_loss(Y_test, Y_pred_clip)
                    else:
                        Y_auc = np.nan
                        Y_pr_auc = np.nan
                        Y_log_loss = np.nan
            
                    Y_brier = brier_score_loss(Y_test, Y_pred)
            
                    Y_R2 = np.nan
                    Y_RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))
                    Y_MAE = mean_absolute_error(Y_test, Y_pred)
            
                else:
                    Y_pred = model_y.predict(XW_test)
            
                    Y_R2 = r2_score(Y_test, Y_pred)
                    Y_RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))
                    Y_MAE = mean_absolute_error(Y_test, Y_pred)
            
                    Y_auc = np.nan
                    Y_pr_auc = np.nan
                    Y_brier = np.nan
                    Y_log_loss = np.nan
            
                    # Extra diagnostics for zero-heavy volume/duration
                    if zero_heavy:
                        raw_test = df.iloc[test_idx][raw_outcome].values
                        has_positive = (raw_test > 0).astype(int)
            
                        if len(np.unique(has_positive)) == 2:
                            Y_positive_auc = roc_auc_score(has_positive, Y_pred)
                            Y_positive_pr_auc = average_precision_score(has_positive, Y_pred)
                        else:
                            Y_positive_auc = np.nan
                            Y_positive_pr_auc = np.nan
            
                        positive_rate = has_positive.mean()
                    else:
                        Y_positive_auc = np.nan
                        Y_positive_pr_auc = np.nan
                        positive_rate = np.nan
            
                # -----------------------------
                # Treatment prediction
                # -----------------------------
                T_pred = model_t.predict(XW_test)
            
                T_R2 = r2_score(T_test, T_pred)
                T_RMSE = np.sqrt(mean_squared_error(T_test, T_pred))
                T_MAE = mean_absolute_error(T_test, T_pred)
            
                row = {
                    "fold": fold,
                    "n_test": len(test_idx),
            
                    # outcome nuisance metrics
                    "Y_R2": Y_R2,
                    "Y_RMSE": Y_RMSE,
                    "Y_MAE": Y_MAE,
            
                    # binary occurrence diagnostics
                    "Y_AUC": Y_auc,
                    "Y_PR_AUC": Y_pr_auc,
                    "Y_Brier": Y_brier,
                    "Y_LogLoss": Y_log_loss,
            
                    # zero-heavy volume/duration diagnostics
                    "positive_rate": positive_rate if not is_binary_outcome else Y_test.mean(),
                    "Y_positive_AUC": Y_positive_auc if (not is_binary_outcome and zero_heavy) else np.nan,
                    "Y_positive_PR_AUC": Y_positive_pr_auc if (not is_binary_outcome and zero_heavy) else np.nan,
            
                    # treatment nuisance metrics
                    "T_R2": T_R2,
                    "T_RMSE": T_RMSE,
                    "T_MAE": T_MAE,
                }
            
                results.append(row)
            
            results_df = pd.DataFrame(results)
            
            print(results_df)
            print(results_df.mean(numeric_only=True))
            
            #%% causal
            # -----------------------------
            # Test stability of causal effects across folds
            # Estimate all contrasts from the same fitted model
            # -----------------------------
            
            all_effects = []
            
            # catchment-relative groundwater percentile based on event-start DTP
            df["dtp_percentile"] = df.groupby(group_col)[treatment].rank(pct=True)
            
            df["dtp_state"] = pd.cut(
                df["dtp_percentile"],
                bins=[0, 0.50, 0.75, 0.90, 1.00],
                labels=["below_Q50", "Q50_Q75", "Q75_Q90", "above_Q90"],
                include_lowest=True
            )
            
            required_quantile_cols = ["dtp_q75_value", "dtp_q90_value"]
            missing_cols = [c for c in required_quantile_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required DTP quantile columns: {missing_cols}. "
                    "Add these to dataset before creating df, and include them in cols."
                )
            
            
            def convert_tau_to_report_units(tau, target_type, is_log_outcome):
                """
                Convert raw causal effect to reporting units.
            
                occurrence: probability effect -> percentage points
                log outcomes: log effect -> percent change
                other outcomes: raw-unit effect
                """
                tau = np.asarray(tau).ravel()
            
                if target_type == "occurrence":
                    return 100 * tau, "percentage_points"
            
                elif is_log_outcome:
                    return (np.exp(tau) - 1) * 100, "percent"
            
                else:
                    return tau, "raw_units"
            
            
            for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), start=1):
                print(f"Evaluating causal model fold {fold}")
            
                model_y_fold = make_model_y(target_type=target_type, random_state=fold)
                model_t_fold = make_model_t(random_state=fold + 100)
            
                cf = CausalForestDML(
                    model_y=model_y_fold,
                    model_t=model_t_fold,
                    n_estimators=1000,
                    min_samples_leaf=20,
                    discrete_treatment=False,
                    discrete_outcome=is_binary_outcome,
                    cv=3,
                    random_state=fold + 200
                )
                # cf = NonParamDML(
                #     model_y=model_y_fold,
                #     model_t=model_t_fold,
                #     model_final=LGBMRegressor(
                #             n_estimators=1000,       # capped by early stopping, not tuned
                #             learning_rate=0.03,      # small step size, stabilized by many estimators
                #             num_leaves=31,           # LightGBM default; fine at this sample size
                #             max_depth=-1,            # unconstrained, regularized via num_leaves instead
                #             min_child_samples=50,    # slightly higher than default (20) given n=168k
                #             subsample=0.8,           # row subsampling for regularization
                #             subsample_freq=1,
                #             colsample_bytree=0.8,    # feature subsampling, useful with 70 features
                #             reg_lambda=1.0,          # mild L2 regularization
                #             reg_alpha=0.0,
                #             random_state=fold+300),
                #     discrete_treatment=False,
                #     discrete_outcome=is_binary_outcome,
                #     cv=3,
                #     random_state=fold + 200)
                
            
                cf.fit(
                    Y=Y[train_idx],
                    T=T[train_idx],
                    X=X[train_idx],
                    W=W[train_idx]
                )
            
                X_test = X[test_idx]
                T_test = T[test_idx]
            
                q75_test = df.iloc[test_idx]["dtp_q75_value"].values
                q90_test = df.iloc[test_idx]["dtp_q90_value"].values
            
                # -----------------------------
                # Define all contrasts here
                # -----------------------------
                contrasts = {
                    "event_plus10": {
                        "description": "Observed event groundwater state to observed state + 10 cm",
                        "T0": T_test,
                        "T1": T_test + 0.1,
                    },
                    "mean_plus10": {
                        "description": "Mean groundwater anomaly to mean + 10 cm",
                        "T0": np.zeros_like(T_test),
                        "T1": np.zeros_like(T_test) + 0.1,
                    },
                    "q75_to_q90": {
                        "description": "Catchment-specific Q75 groundwater state to Q90 state",
                        "T0": q75_test,
                        "T1": q90_test,
                    },
                    "q90_plus10": {
                        "description": "Catchment-specific Q90 groundwater state to Q90 + 10 cm",
                        "T0": q90_test,
                        "T1": q90_test + 0.1,
                    },
                }
            
                # -----------------------------
                # Base event information
                # -----------------------------
                fold_effects = pd.DataFrame({
                    "fold": fold,
                    "row_index": df.index[test_idx],
                    "catch_id": df.iloc[test_idx][group_col].values,
            
                    # observed event-start groundwater state
                    "dtp": df.iloc[test_idx][treatment].values,
                    "dtp_percentile": df.iloc[test_idx]["dtp_percentile"].values,
                    "dtp_state": df.iloc[test_idx]["dtp_state"].values,
            
                    # catchment-specific groundwater quantile thresholds
                    "dtp_q75_value": q75_test,
                    "dtp_q90_value": q90_test,
                    "delta_q75_q90": q90_test - q75_test,
                })
            
                # -----------------------------
                # Estimate every contrast from the same fitted model
                # -----------------------------
                contrast_metadata = {}
            
                for contrast_name, contrast_def in contrasts.items():
                    print(f"  Estimating contrast: {contrast_name}")
            
                    T0 = contrast_def["T0"]
                    T1 = contrast_def["T1"]
                    delta_T = T1 - T0
            
                    tau = cf.effect(X_test, T0=T0, T1=T1)
                    tau = np.asarray(tau).ravel()
                    
                    try:
                        tau_lb, tau_ub = cf.effect_interval(X_test,T0=T0,T1=T1,alpha=0.05)
                        tau_lb = np.asarray(tau_lb).ravel()
                        tau_ub = np.asarray(tau_ub).ravel()
                    except Exception as e:
                        print(f"  effect_interval not available for {contrast_name}: {e}")
                    
                        tau_lb = np.full_like(tau, np.nan, dtype=float)
                        tau_ub = np.full_like(tau, np.nan, dtype=float)
    
                    tau_report, effect_unit = convert_tau_to_report_units(
                        tau,
                        target_type=target_type,
                        is_log_outcome=is_log_outcome
                    )
            
                    tau_lb_report, _ = convert_tau_to_report_units(
                        tau_lb,
                        target_type=target_type,
                        is_log_outcome=is_log_outcome
                    )
            
                    tau_ub_report, _ = convert_tau_to_report_units(
                        tau_ub,
                        target_type=target_type,
                        is_log_outcome=is_log_outcome
                    )
            
                    # Save wide-format columns
                    fold_effects[f"{contrast_name}_T0"] = T0
                    fold_effects[f"{contrast_name}_T1"] = T1
                    fold_effects[f"{contrast_name}_delta_T"] = delta_T
            
                    fold_effects[f"{contrast_name}_tau"] = tau
                    fold_effects[f"{contrast_name}_tau_report"] = tau_report
                    fold_effects[f"{contrast_name}_tau_lb_report"] = tau_lb_report
                    fold_effects[f"{contrast_name}_tau_ub_report"] = tau_ub_report
            
                    contrast_metadata[contrast_name] = {
                        "description": contrast_def["description"],
                        "effect_unit": effect_unit
                    }
            
                all_effects.append(fold_effects)
            
            effects_event_df = pd.concat(all_effects, ignore_index=True)
            
            print("Finished all folds.")
            print(effects_event_df.head())

            # -----------------------------
            # Summaries for all contrasts
            # -----------------------------
            
            contrast_names = ["event_plus10", "mean_plus10", "q75_to_q90", "q90_plus10"]
            contrast_report_cols = [f"{c}_tau_report" for c in contrast_names]
            
            # Overall event-level summary
            overall_contrast_summary = []
            
            for contrast_name in contrast_names:
                col = f"{contrast_name}_tau_report"
                vals = effects_event_df[col].dropna()
            
                overall_contrast_summary.append({
                    "contrast": contrast_name,
                    "n": len(vals),
                    "mean_effect": vals.mean(),
                    "median_effect": vals.median(),
                    "std_effect": vals.std(),
                    "p05_effect": np.percentile(vals, 5),
                    "p25_effect": np.percentile(vals, 25),
                    "p75_effect": np.percentile(vals, 75),
                    "p95_effect": np.percentile(vals, 95),
                })
            
            overall_contrast_summary_df = pd.DataFrame(overall_contrast_summary)
            print(overall_contrast_summary_df)
            
            
            # -----------------------------
            # Fold-level stability for all contrasts
            # -----------------------------
            
            fold_contrast_results_df = (
                effects_event_df
                .groupby("fold")[contrast_report_cols]
                .mean()
                .reset_index()
            )
            
            contrast_stability = []
            
            for contrast_name in contrast_names:
                col = f"{contrast_name}_tau_report"
                vals = fold_contrast_results_df[col].dropna()
            
                contrast_stability.append({
                    "contrast": contrast_name,
                    "n_folds": len(vals),
                    "mean_of_fold_means": vals.mean(),
                    "std_of_fold_means": vals.std(),
                    "min_fold_mean": vals.min(),
                    "max_fold_mean": vals.max(),
                    "median_fold_mean": vals.median(),
                    "positive_folds": int((vals > 0).sum()),
                    "negative_folds": int((vals < 0).sum()),
                })
            
            contrast_stability_df = pd.DataFrame(contrast_stability)
            print(contrast_stability_df)
            
            
            # -----------------------------
            # State-specific summary
            # Most useful for event_plus10
            # -----------------------------
            
            state_effect_summary_df = (
                effects_event_df
                .groupby("dtp_state", observed=False)[contrast_report_cols]
                .agg(["count", "mean", "median", "std"])
            )
            
            print(state_effect_summary_df)
            
            
            # -----------------------------
            # Fold-level state stability
            # Most useful for event_plus10
            # -----------------------------
            
            event_state_fold_results_df = (
                effects_event_df
                .groupby(["fold", "dtp_state"], observed=False)["event_plus10_tau_report"]
                .agg(
                    mean_effect="mean",
                    median_effect="median",
                    n_test="count"
                )
                .reset_index()
            )
            
            event_state_stability_df = (
                event_state_fold_results_df
                .groupby("dtp_state", observed=False)["mean_effect"]
                .agg(
                    n_folds="count",
                    mean_of_fold_means="mean",
                    std_of_fold_means="std",
                    min_fold_mean="min",
                    max_fold_mean="max",
                    median_fold_mean="median",
                    positive_folds=lambda x: int((x > 0).sum()),
                    negative_folds=lambda x: int((x < 0).sum()),
                )
                .reset_index()
            )
            
            print(event_state_stability_df)
                        
            save_obj = {
                "target_type": target_type,
                "outcome": outcome,
                "treatment": treatment,
                "scale": scale,
                "effect_unit": effect_unit,
                "confounders": confounders,
                "heterogeneity_features": heterogeneity_features,
                "contrast_metadata": contrast_metadata,
            
                # nuisance
                "nuisance_results": results_df,
            
                # event-level causal effects, wide format
                "causal_event_results": effects_event_df,
            
                # summaries
                "overall_contrast_summary": overall_contrast_summary_df,
                "fold_contrast_results": fold_contrast_results_df,
                "contrast_stability": contrast_stability_df,
                "state_effect_summary": state_effect_summary_df,
                "event_state_fold_results": event_state_fold_results_df,
                "event_state_stability": event_state_stability_df,
            }
            
            out_file = out_dir / f"causal_ml_results_{target_type}_{scale}_all_contrasts.pkl"
            
            with open(out_file, "wb") as f:
                pickle.dump(save_obj, f)
            
            print(f"Saved results to: {out_file}")




        
