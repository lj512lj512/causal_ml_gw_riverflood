"""Train conditional causal-effect models (requires causal assumptions)."""
import numpy as np
from sklearn.model_selection import GroupKFold
from .config import StudyConfig
from .data import prepare_model_data

def _factory(cfg, seed):
    from econml.dml import CausalForestDML
    from lightgbm import LGBMRegressor, LGBMClassifier
    is_binary = cfg.target == "occurrence"
    y_model = (LGBMClassifier if is_binary else LGBMRegressor)(
        n_estimators=300, learning_rate=0.05, random_state=seed, verbosity=-1)
    t_model = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                            random_state=seed + 1, verbosity=-1)
    return CausalForestDML(
        model_y=y_model, model_t=t_model, discrete_treatment=False,
        discrete_outcome=is_binary, n_estimators=400, min_samples_leaf=10,
        random_state=seed, cv=3, inference=True)

def fit_causal(df, cfg: StudyConfig):
    df, y = prepare_model_data(df, cfg)
    if len(df) < 50:
        raise ValueError("Too few complete events")
    model = _factory(cfg, cfg.random_state)
    model.fit(y, df[cfg.treatment].to_numpy(), X=df[list(cfg.heterogeneity)].to_numpy(),
              W=df[list(cfg.confounders)].to_numpy())
    return model, df

def evaluate_heldout_catchments(df, cfg: StudyConfig, contrast=0.1):
    """Group-held-out effect summaries, not causal ground-truth validation or CIs."""
    data, y = prepare_model_data(df, cfg)
    n_groups = data[cfg.group].nunique()
    if n_groups < cfg.n_splits:
        raise ValueError(f"Need >= {cfg.n_splits} catchments; found {n_groups}")
    cv = GroupKFold(n_splits=cfg.n_splits)
    out = []
    for fold, (train, test) in enumerate(cv.split(data, groups=data[cfg.group]), 1):
        m = _factory(cfg, cfg.random_state + fold)
        m.fit(y[train], data.iloc[train][cfg.treatment].to_numpy(),
              X=data.iloc[train][list(cfg.heterogeneity)].to_numpy(),
              W=data.iloc[train][list(cfg.confounders)].to_numpy())
        x = data.iloc[test][list(cfg.heterogeneity)].to_numpy()
        t = data.iloc[test][cfg.treatment].to_numpy()
        effect = np.asarray(m.effect(x, T0=t, T1=t + contrast)).reshape(-1)
        for g, val in zip(data.iloc[test][cfg.group].to_numpy(), effect):
            out.append((fold, g, float(val)))
    import pandas as pd
    return pd.DataFrame(out, columns=["fold", cfg.group, "effect"])
