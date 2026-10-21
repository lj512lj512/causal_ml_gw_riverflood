"""Standalone daily rainfall delineation; not historical event-equivalent."""
import pandas as pd

def detect_rainfall_events(ts, precip_col="precipitation", threshold=3.0,
                           dry_gap=2, min_total_mm=10.0, max_duration=14):
    """Detect contiguous wet spells allowing up to dry_gap-1 consecutive dry days.

    A spell exceeding max_duration is excluded. This deliberately does not
    implement historical main-pulse splitting; see docs/MIGRATION.md.
    Missing precipitation is an error, not a dry day.
    """
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise TypeError("ts must have a DatetimeIndex")
    x = ts.sort_index()
    if x.index.has_duplicates or x.index.to_series().diff().dropna().ne(pd.Timedelta(days=1)).any():
        raise ValueError("Daily timestamps must be unique and continuous")
    p = pd.to_numeric(x[precip_col], errors="raise")
    if p.isna().any() or (p < 0).any():
        raise ValueError("Precipitation contains missing or negative values")
    events = []
    start = last_wet = None
    dry = 0
    def emit(s, e):
        seg = p.iloc[s:e+1]
        if seg.sum() >= min_total_mm and len(seg) <= max_duration:
            events.append(dict(rain_start_date=seg.index[0], rain_end_date=seg.index[-1],
                rain_duration_days=len(seg), event_precip_mm=float(seg.sum()),
                max_1d_precip_mm=float(seg.max()), main_pulse_date=seg.idxmax(),
                split_from_long_event=False))
    for i, amount in enumerate(p.to_numpy()):
        if amount >= threshold:
            if start is None:
                start = i
            last_wet = i
            dry = 0
        elif start is not None:
            dry += 1
            if dry >= dry_gap:
                emit(start, last_wet)
                start = last_wet = None
                dry = 0
    if start is not None:
        emit(start, last_wet)
    return pd.DataFrame(events, columns=["rain_start_date", "rain_end_date",
        "rain_duration_days", "event_precip_mm", "max_1d_precip_mm",
        "main_pulse_date", "split_from_long_event"])
