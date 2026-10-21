"""Centralized column conventions; edit here instead of across scripts."""
from dataclasses import dataclass
from typing import Tuple

ALPHA_EARTH = tuple(f"A{i:02d}" for i in range(64))
OUTCOMES = {
    "peak": ("flood_peak", "log_eps"),
    "occurrence": ("flood_occurrence", "binary"),
    "volume": ("flood_volume_above_q95", "log1p"),
    "duration": ("duration_above_q95", "log1p"),
}
DEFAULT_CONFOUNDERS = (
    "rain_duration_days", "event_precip_mm", "max_1d_precip_mm",
    "split_from_long_event", "pre_event_precip_3d", "wcr",
    "sdy", "cdy", "tem", "pet",
)

@dataclass(frozen=True)
class StudyConfig:
    treatment: str = "dtp"
    group: str = "catch_id"
    target: str = "peak"
    confounders: Tuple[str, ...] = DEFAULT_CONFOUNDERS
    heterogeneity: Tuple[str, ...] = ALPHA_EARTH + ("sdy", "cdy")
    n_splits: int = 5
    random_state: int = 42
