"""Prepare reproducible, event-level tabular inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import OUTCOMES, StudyConfig

def read_events(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def with_seasonality(frame, date_col="rain_start_date"):
    df = frame.copy()
    dates = pd.to_datetime(df[date_col], errors="raise")
    doy = dates.dt.dayofyear
    df["sdy"] = np.sin(2*np.pi*doy/365.25)
    df["cdy"] = np.cos(2*np.pi*doy/365.25)
    return df

def prepare_model_data(frame, cfg: StudyConfig):
    """Return complete rows and correctly transformed outcome; no silent feature creation."""
    if cfg.target not in OUTCOMES:
        raise ValueError(f"Unsupported target {cfg.target!r}; use {tuple(OUTCOMES)}")
    raw, transform = OUTCOMES[cfg.target]
    cols = list(dict.fromkeys([raw, cfg.treatment, cfg.group, *cfg.confounders, *cfg.heterogeneity]))
    missing = sorted(set(cols) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = frame[cols].copy()
    for c in [raw, cfg.treatment, *cfg.confounders, *cfg.heterogeneity]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).reset_index(drop=True)
    if df.empty:
        raise ValueError("No complete events after dropping missing values")
    if transform == "log_eps":
        if (df[raw] < 0).any():
            raise ValueError("Peak discharge cannot be negative")
        y = np.log(df[raw].to_numpy() + 1e-6)
    elif transform == "log1p":
        if (df[raw] < 0).any():
            raise ValueError(f"{raw} cannot be negative")
        y = np.log1p(df[raw].to_numpy())
    else:
        if not set(df[raw].unique()).issubset({0, 1}):
            raise ValueError("Occurrence must be binary (0, 1)")
        y = df[raw].to_numpy(dtype=float)
    return df, y
