"""Conventional ML baseline for held-out catchments."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from .config import StudyConfig
from .data import prepare_model_data

def compare_predictor_sets(frame, cfg: StudyConfig, include_treatment=True):
    """Train LightGBM on grouped train/test splits; only fold test metrics are reported."""
    from lightgbm import LGBMRegressor, LGBMClassifier
    df, y = prepare_model_data(frame, cfg)
    features = list(dict.fromkeys([*cfg.confounders, *cfg.heterogeneity] +
                                 ([cfg.treatment] if include_treatment else [])))
    cv = GroupKFold(n_splits=cfg.n_splits)
    rows = []
    for fold, (a, b) in enumerate(cv.split(df, y, df[cfg.group]), 1):
        classification = cfg.target == "occurrence"
        model = (LGBMClassifier if classification else LGBMRegressor)(
            n_estimators=400, learning_rate=0.05, random_state=cfg.random_state+fold, verbosity=-1)
        model.fit(df.iloc[a][features], y[a])
        pred = model.predict_proba(df.iloc[b][features])[:, 1] if classification else model.predict(df.iloc[b][features])
        metric = "roc_auc" if classification else "rmse"
        if classification:
            score = float(roc_auc_score(y[b], pred)) if len(np.unique(y[b])) == 2 else np.nan
        else:
            score = float(np.sqrt(mean_squared_error(y[b], pred)))
        rows.append({"fold": fold, "target": cfg.target, "include_groundwater": include_treatment,
                     "metric": metric, "score": score, "n_test": len(b), "n_catchments_test": df.iloc[b][cfg.group].nunique()})
    return pd.DataFrame(rows)
