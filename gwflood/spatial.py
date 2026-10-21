"""Transport a fitted conditional effect model to compatible spatial features."""
import numpy as np
import pandas as pd
from .config import StudyConfig

def estimate_spatial_effects(model, spatial, cfg: StudyConfig, t0_col, delta=0.1,
                             id_col="catch_id"):
    """Effect per spatial unit, conditional on supplied X; NOT a validated global causal map."""
    needed = [id_col, t0_col, *cfg.heterogeneity]
    missing = sorted(set(needed) - set(spatial))
    if missing:
        raise ValueError(f"Missing spatial columns: {missing}")
    dat = spatial[needed].copy()
    if dat.isna().any().any():
        raise ValueError("Missing values in spatial inputs; do not extrapolate silently")
    x = dat[list(cfg.heterogeneity)].to_numpy(dtype=float)
    t0 = dat[t0_col].to_numpy(dtype=float)
    eff = np.asarray(model.effect(x, T0=t0, T1=t0+delta)).reshape(-1)
    out = dat[[id_col]].copy()
    out["t0"] = t0
    out["delta_treatment"] = delta
    out["effect_model_scale"] = eff
    if cfg.target == "peak":
        out["change_percent_if_log_peak"] = (np.exp(eff)-1)*100
    return out
