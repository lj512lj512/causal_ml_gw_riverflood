# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:17:33 2026

@author: juliu
"""
import re
import os
from datetime import datetime
import xarray as xr 
import numpy as np 
import pandas as pd
import sys
sys.path.append(r'\\geodata.geus.dk\DKmodel_users\FloodWarning\LSTM_postprocessing\CAMELS-DK\Level_1\DKM_simulations')
import RS_dfs_mikeio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib as mpl
import geopandas as gpd
import mikeio
from subprocess import Popen, PIPE
import time
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from datetime import timedelta
import rioxarray
import pickle
from tqdm.auto import tqdm
import shutil
from pathlib import Path

HIP_FOLDER = (
    r"\\geodata.geus.dk\Dkmodel-hydro\HIP_Realtid"
    r"\HIP_100m_500m_realtid_historical_1989_2024_newClimTS_precip"
)
CAMELS_FOLDER = (
    r"\\geodata.geus.dk\DKmodel_users\FloodWarning\Literature_survey\GWH_flood\CAMELS-DK"
)
QOBS_FILE = (
    r"\\geodata.geus.dk\DKmodel_users\FloodWarning\Literature_survey"
    r"\GWH_flood\Qdata\Q_Stations2025.dfs0"
)
# get precipitation
def load_climate_grid() -> xr.Dataset:
    """
    Load gridded precipitation and assign CRS.

    Returns
    -------
    xr.Dataset
    """
    precip = RS_dfs_mikeio.extract_dfs_data(
        root=r"\\geodata.geus.dk\Dkmodel-hydro\HIP_Realtid\HIP_100m_500m_realtid_historical_1989_2024_newClimTS_precip\climate\DK_Precip_CorrGEUS_10km_24h_1989_to_today.dfs2",
    ).to_xarray()
    
    temp = RS_dfs_mikeio.extract_dfs_data(
        root=r"\\geodata.geus.dk\Dkmodel-hydro\HIP_Realtid\HIP_100m_500m_realtid_historical_1989_2024_newClimTS_precip\climate\DK_Ta_20km_24h_1989_to_today.dfs2",
    ).to_xarray()
    
    pet = RS_dfs_mikeio.extract_dfs_data(
        root=r"\\geodata.geus.dk\Dkmodel-hydro\HIP_Realtid\HIP_100m_500m_realtid_historical_1989_2024_newClimTS_precip\climate\DK_ETref_0.95VForStorebaelt_20km_24h_1989_to_today.dfs2",
    ).to_xarray()
    
    # Rename 
    precip = precip.rename({'P, GEUS corr  ': "pre"})
    temp   = temp.rename(  {'Ta            ': "tem"})
    pet    = pet.rename(   {'ETref, *0.95 w': "pet"})
    
    return xr.merge([precip, temp, pet])



def load_dkm_dynamics(DKM_id, jumpfilled = False) -> xr.Dataset:
    """
    Load DKM phreatic daily dfs2 output and convert to xarray.

    Parameters
    ----------
    DKM_ID : int
        DKM model ID.

    Returns
    -------
    xr.Dataset
        Daily phreatic dataset.
    """
    filename = os.path.join(
        HIP_FOLDER,
        f"DK{DKM_id}_HIPmodels",
        "Result",
        f"DK{DKM_id}_HIP_500m",
        f"DK{DKM_id}_HIP_500m_2DSZ.dfs2",
    )
    
    if jumpfilled:
        resolution = 500
        dataPath = r'\\geodata.geus.dk\DKmodel_users\FloodWarning\GWH_emulator\Hydro03_E_temporary\juliu\GWH_emulator\Dataset\DKM_sim'
        DKM_dtp_fill = xr.load_dataset(os.path.join(dataPath, f'DK{DKM_id}_HIP_{resolution}m_2DSZ_dtp_jumps_filled_1989_2024.nc'))
    else:
        dtp = RS_dfs_mikeio.extract_dfs_data(
            root=filename,
        ).to_xarray()
        dtp = dtp.rename({'depth to top phreatic surface (negative)': 'dtp'})
    
    wcr = RS_dfs_mikeio.extract_dfs_data(
        root=Path(filename).parent,
        mname=Path(filename).parent.name, 
        var = 'wcr'
    ).to_xarray()
    wcr = wcr.rename({'average water content in the rootzone': 'wcr'})
    
    return dtp, wcr

def load_camels_info():
    """
    Load CAMELS-DK station, catchment, and attribute data.
    
    Returns
    -------
    tuple
        stations, catchments, attributes
    """
    
    stations = gpd.read_file(
        os.path.join(CAMELS_FOLDER, "Shapefile", "CAMELS_DK_304_gauging_stations.shp")
    )
    
    catchments = gpd.read_file(
        os.path.join(
            CAMELS_FOLDER,
            "Shapefile",
            "CAMELS_DK_304_gauging_catchment_boundaries.shp",
        )
    )
    
    topography = pd.read_csv(os.path.join(CAMELS_FOLDER, "Attibutes", "CAMELS_DK_topography.csv"))
    landuse    = pd.read_csv(os.path.join(CAMELS_FOLDER, "Attibutes", "CAMELS_DK_landuse.csv"))[['catch_id', 
                                                                                                 'pct_forest_corine_2018', 
                                                                                                 'pct_agriculture_corine_2018',
                                                                                                 'pct_water_corine_2018', 
                                                                                                 'pct_urban_corine_2018',
                                                                                                 'pct_wetlands_corine_2018']]
    hydrology  = pd.read_csv(os.path.join(CAMELS_FOLDER, "Attibutes", "CAMELS_DK_signature_obs_based.csv"))[['catch_id', 'BFI']]
    
    attributes = topography.merge(landuse, left_on = 'catch_id', right_on='catch_id').merge(hydrology, left_on = 'catch_id', right_on='catch_id')
    
    return stations, catchments, attributes

def load_camels_dynamic(STATION_ID):
    DATAPATH = r'\\geodata.geus.dk\DKmodel_users\FloodWarning\Literature_survey\GWH_flood\CAMELS-DK\Dynamics\Gauged_catchments'
    camels_ts= pd.read_csv(os.path.join(DATAPATH, f'CAMELS_DK_obs_based_{STATION_ID}.csv'), 
                           index_col='time', parse_dates=True)
    return camels_ts

def load_camels_static():
    DATAPATH = r'\\geodata.geus.dk\DKmodel_users\FloodWarning\Literature_survey\GWH_flood\CAMELS-DK\Attibutes'
    geology  = pd.read_csv(os.path.join(DATAPATH, 'CAMELS_DK_geology.csv'))
    landuse  = pd.read_csv(os.path.join(DATAPATH, 'CAMELS_DK_landuse.csv'))
    soil     = pd.read_csv(os.path.join(DATAPATH, 'CAMELS_DK_soil.csv'))
    topo     = pd.read_csv(os.path.join(DATAPATH, 'CAMELS_DK_topography.csv'))
    
    return geology, landuse, soil, topo


def load_observed_discharge(order_station: str) -> pd.DataFrame:
    """
    Load observed discharge from dfs0.

    Parameters
    ----------
    order_station : str
        Station code in the dfs0 file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Qobs', 'time'] and datetime index.
    """
    qobs = mikeio.read(QOBS_FILE)
    qobs = qobs[order_station].to_dataframe()
    qobs = qobs.rename(columns={order_station: "Qobs"})
    qobs["time"] = qobs.index
    return qobs



def lyne_hollick_baseflow(q: pd.Series, alpha: float = 0.925, passes: int = 3) -> pd.Series:
    """
    Lyne-Hollick recursive digital filter for daily streamflow.
    Returns estimated baseflow.

    Parameters
    ----------
    q : pd.Series
        Daily discharge with DatetimeIndex.
    alpha : float
        Filter parameter. Paper uses 0.925 for daily data.
    passes : int
        Number of passes. Paper uses 3 (forward-backward-forward).

    Returns
    -------
    pd.Series
        Baseflow series.
    """
    q = q.astype(float).copy()

    if not isinstance(q.index, pd.DatetimeIndex):
        raise ValueError("q must have a DatetimeIndex.")

    if q.isna().any():
        q = q.interpolate(limit_direction="both")

    def one_pass_baseflow(x: np.ndarray, alpha_: float) -> np.ndarray:
        # First estimate quickflow, then baseflow = total - quickflow
        qf = np.zeros_like(x, dtype=float)
        for i in range(1, len(x)):
            qf[i] = alpha_ * qf[i - 1] + ((1 + alpha_) / 2.0) * (x[i] - x[i - 1])
            qf[i] = max(qf[i], 0.0)
            qf[i] = min(qf[i], x[i])
        bf = x - qf
        bf = np.clip(bf, 0.0, x)
        return bf

    arr = q.values.copy()

    for p in range(passes):
        if p % 2 == 0:
            arr = one_pass_baseflow(arr, alpha)
        else:
            arr = one_pass_baseflow(arr[::-1], alpha)[::-1]

    bf = pd.Series(arr, index=q.index, name="baseflow")
    bf = bf.clip(lower=0.0, upper=q)
    return bf


def compute_mixed_hfs_threshold(
    q: pd.Series,
    quantile: float = 0.99
) -> pd.Series:
    """
    Mixed threshold for HFS:
    threshold(day) = max(fixed quantile threshold, monthly threshold for that month)

    Parameters
    ----------
    q : pd.Series
        Daily discharge with DatetimeIndex.
    quantile : float
        Fixed annual quantile threshold, e.g. 0.99 (more severe) or 0.95 (less severe)

    Returns
    -------
    pd.Series
        Daily mixed threshold.
    """
    q = q.astype(float).copy()

    fixed_thr = q.quantile(quantile)
    monthly_thr = q.groupby(q.index.month).quantile(quantile)
    thr = pd.Series(q.index.month, index=q.index).map(monthly_thr).astype(float)

    mixed = pd.Series(np.maximum(fixed_thr, thr.values), index=q.index, name="Qt")
    return mixed


def detect_high_flow_spells_paper(
    q: pd.Series,
    hfs_quantile: float = 0.99,
    alpha: float = 0.925,
    passes: int = 3,
    baseflow_tolerance: float = 0.0,
    min_duration_days: int = 1,
    compute_volume: bool = True,
    basin_area_km2: float | None = None,
):
    """
    Detect high-flow spells (HFS) following the article logic:

    1) Compute mixed HFS threshold Qt = max(fixed quantile, monthly quantile)
    2) Start date ti = first day Q > Qt
    3) Find te,b = first later day when:
          - Q falls below Qt
          - and Q equals baseflow (or is close to it)
    4) Spell end te,f = search backwards from te,b to the last day Q > Qt

    Notes
    -----
    This is a SPELL detector, not an individual peak detector.
    So multiple nearby peaks may be pooled into one spell by design.

    Parameters
    ----------
    q : pd.Series
        Daily discharge (e.g. m3/s) with DatetimeIndex.
    hfs_quantile : float
        0.99 for more severe threshold, 0.95 for less severe threshold.
    alpha : float
        LH filter parameter, default 0.925.
    passes : int
        LH passes, default 3.
    baseflow_tolerance : float
        Relative tolerance for "Q close to baseflow".
        Example 0.05 means abs(Q - BF)/Q <= 5%.
        The paper notes "equals or, depending on the method, is close to" baseflow.
    min_duration_days : int
        Keep only spells of at least this length.
    compute_volume : bool
        Whether to compute cumulative surplus above threshold and above baseflow.
    basin_area_km2 : float | None
        Optional basin area to express daily runoff depth in mm.

    Returns
    -------
    events : pd.DataFrame
        One row per HFS spell.
    series_out : pd.DataFrame
        Q, baseflow, threshold, and mask columns.
    """
    q = q.sort_index().astype(float).copy()

    if not isinstance(q.index, pd.DatetimeIndex):
        raise ValueError("q must have a DatetimeIndex.")

    if q.isna().any():
        q = q.interpolate(limit_direction="both")

    bf = lyne_hollick_baseflow(q, alpha=alpha, passes=passes)
    Qt = compute_mixed_hfs_threshold(q, quantile=hfs_quantile)

    above = q > Qt
    idx = q.index
    n = len(q)

    events = []
    i = 0

    while i < n:
        if not above.iloc[i]:
            i += 1
            continue

        # Start date ti
        ti = idx[i]

        # Move forward until the first date te,b where:
        # Q <= Qt and Q ~= baseflow
        te_b = None
        j = i + 1
        while j < n:
            qj = q.iloc[j]
            bfj = bf.iloc[j]
            Qtj = Qt.iloc[j]

            close_to_bf = (
                np.isclose(qj, bfj, rtol=baseflow_tolerance, atol=0.0)
                if baseflow_tolerance == 0.0
                else (abs(qj - bfj) / max(qj, 1e-12) <= baseflow_tolerance)
            )

            if (qj <= Qtj) and close_to_bf:
                te_b = idx[j]
                break
            j += 1

        # If never returns to threshold+baseflow condition, stop at series end
        if te_b is None:
            te_b = idx[-1]
            j = n - 1

        # Search backward from te_b to find te,f:
        # the last date where Q remained above threshold
        te_f = None
        for k in range(j, i - 1, -1):
            if q.iloc[k] > Qt.iloc[k]:
                te_f = idx[k]
                break

        # Fallback
        if te_f is None:
            te_f = ti

        event_q = q.loc[ti:te_f]
        event_bf = bf.loc[ti:te_f]
        event_qt = Qt.loc[ti:te_f]

        duration = len(event_q)
        if duration >= min_duration_days:
            row = {
                "start_date": ti,
                "end_date": te_f,
                "duration_days": duration,
                "peak_date": event_q.idxmax(),
                "peak_flow": float(event_q.max()),
                "mean_flow": float(event_q.mean()),
                "mean_baseflow": float(event_bf.mean()),
                "threshold_quantile": hfs_quantile,
            }

            if compute_volume:
                # Surplus above threshold during spell
                excess_thr = (event_q - event_qt).clip(lower=0.0)
                excess_bf = (event_q - event_bf).clip(lower=0.0)

                if basin_area_km2 is None:
                    # volumes in m3 if Q is in m3/s and timestep is daily
                    row["surplus_above_threshold_m3"] = float((excess_thr * 86400.0).sum())
                    row["surplus_above_baseflow_m3"] = float((excess_bf * 86400.0).sum())
                else:
                    # convert daily discharge to runoff depth in mm/day
                    area_m2 = basin_area_km2 * 1_000_000.0
                    q_mm = event_q * 86400.0 / area_m2 * 1000.0
                    bf_mm = event_bf * 86400.0 / area_m2 * 1000.0
                    qt_mm = event_qt * 86400.0 / area_m2 * 1000.0

                    row["surplus_above_threshold_mm"] = float((q_mm - qt_mm).clip(lower=0.0).sum())
                    row["surplus_above_baseflow_mm"] = float((q_mm - bf_mm).clip(lower=0.0).sum())

            # Count threshold crossings within the spell
            spell_above = (event_q > event_qt).astype(int)
            crossings = int((spell_above.diff().abs() == 1).sum())
            row["threshold_crossings_within_spell"] = crossings

            events.append(row)

        # Continue after te_b, not te_f
        # because the paper pools until baseflow recovery date te,b
        i = j + 1

    events = pd.DataFrame(events)

    series_out = pd.DataFrame({
        "Q": q,
        "baseflow": bf,
        "Qt_hfs": Qt,
        "above_threshold": above
    })

    return events, series_out






def estimate_baseflow_simple(q, alpha=0.925, passes=3):
    q = q.astype(float).copy()

    def one_pass(x):
        qf = np.zeros(len(x))
        for i in range(1, len(x)):
            qf[i] = alpha * qf[i - 1] + ((1 + alpha) / 2.0) * (x[i] - x[i - 1])
            qf[i] = max(0.0, min(qf[i], x[i]))
        bf = x - qf
        return np.clip(bf, 0.0, x)

    arr = q.values.copy()
    for p in range(passes):
        if p % 2 == 0:
            arr = one_pass(arr)
        else:
            arr = one_pass(arr[::-1])[::-1]

    return pd.Series(arr, index=q.index, name="baseflow")


def split_event_on_multipeaks(
    q_event: pd.Series,
    min_peak_distance_days: int = 2,
    min_peak_prominence: float | None = None,
    valley_ratio: float = 0.8,
):
    """
    Split one flood event into sub-events if it contains multiple meaningful peaks.

    Parameters
    ----------
    q_event : pd.Series
        Discharge within one event, indexed by date.
    min_peak_distance_days : int
        Minimum separation between peaks for scipy find_peaks.
    min_peak_prominence : float or None
        Minimum prominence for peaks. If None, it is chosen automatically.
    valley_ratio : float
        Split between two peaks if valley flow is below:
            valley_ratio * min(peak1, peak2)

    Returns
    -------
    list of tuple
        [(sub_start, sub_peak, sub_end), ...]
    """
    if len(q_event) < 3:
        peak_date = q_event.idxmax()
        return [(q_event.index[0], peak_date, q_event.index[-1])]

    x = q_event.values.astype(float)

    if min_peak_prominence is None:
        # simple adaptive default for daily data
        min_peak_prominence = max(np.std(x) * 0.25, (x.max() - x.min()) * 0.1)

    peak_idx, props = find_peaks(
        x,
        distance=min_peak_distance_days,
        prominence=min_peak_prominence
    )

    # If only one meaningful peak, keep event unchanged
    if len(peak_idx) <= 1:
        peak_date = q_event.idxmax()
        return [(q_event.index[0], peak_date, q_event.index[-1])]

    # Determine valid split points between adjacent peaks
    split_positions = []
    valid_peaks = [peak_idx[0]]

    for i in range(len(peak_idx) - 1):
        p1 = peak_idx[i]
        p2 = peak_idx[i + 1]

        seg = x[p1:p2 + 1]
        valley_local = np.argmin(seg)
        valley_idx = p1 + valley_local
        valley_q = x[valley_idx]

        q1 = x[p1]
        q2 = x[p2]
        split_threshold = valley_ratio * min(q1, q2)

        if valley_q < split_threshold:
            split_positions.append(valley_idx)
            valid_peaks.append(p2)

    # If no sufficiently deep valley, do not split
    if len(split_positions) == 0:
        peak_date = q_event.idxmax()
        return [(q_event.index[0], peak_date, q_event.index[-1])]

    # Build sub-events
    boundaries = [0] + split_positions + [len(q_event) - 1]
    subevents = []

    for i in range(len(boundaries) - 1):
        s_idx = boundaries[i]
        e_idx = boundaries[i + 1]

        # avoid overlapping on the valley day:
        # first event ends at valley day, next starts next day
        if i > 0:
            s_idx = boundaries[i] + 1

        sub_q = q_event.iloc[s_idx:e_idx + 1]
        if len(sub_q) == 0:
            continue

        sub_start = sub_q.index[0]
        sub_peak = sub_q.idxmax()
        sub_end = sub_q.index[-1]

        subevents.append((sub_start, sub_peak, sub_end))

    return subevents

def detect_flood_events_with_precip_split(
    q: pd.Series,
    p: pd.Series,
    q_thresh_quantile: float = 0.95,
    merge_gap_days: int = 2,
    dry_threshold_mm: float = 1.0,
    dry_spell_days: int = 2,
    max_lookback_days: int = 14,
    use_baseflow_end: bool = True,
    baseflow_tol: float = 0.05,
    split_multi_peaks: bool = True,
    min_peak_distance_days: int = 2,
    min_peak_prominence: float | None = None,
    valley_ratio: float = 0.8,
    adjust_start_by_rain: bool = True,
    max_start_adjust_days: int = 15,
):
    df = pd.concat([q.rename("Q"), p.rename("P")], axis=1).sort_index()
    df["Q"] = df["Q"].interpolate(limit_direction="both")
    df["P"] = df["P"].fillna(0.0)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("q and p must have DatetimeIndex.")

    df["baseflow"] = estimate_baseflow_simple(df["Q"])
    df["quickflow"] = (df["Q"] - df["baseflow"]).clip(lower=0.0)

    qthr = df["Q"].quantile(q_thresh_quantile)
    df["above_thr"] = df["Q"] > qthr

    dates = df.index[df["above_thr"]]
    clusters = []
    if len(dates) > 0:
        s = dates[0]
        prev = dates[0]
        for d in dates[1:]:
            if (d - prev).days <= merge_gap_days + 1:
                prev = d
            else:
                s_adj = max(df.index[0], s - timedelta(days=2))
                clusters.append((s_adj, prev))
                s = d
                prev = d
        s_adj = max(df.index[0], s - timedelta(days=2))
        clusters.append((s_adj, prev))

    events = []

    for s, e in clusters:
        event_q0 = df.loc[s:e, "Q"]
        peak_date0 = event_q0.idxmax()

        if use_baseflow_end:
            end_date0 = df.index[-1]
            for d in df.loc[peak_date0:].index:
                qd = df.at[d, "Q"]
                bd = df.at[d, "baseflow"]
                if qd <= qthr and abs(qd - bd) / max(qd, 1e-9) <= baseflow_tol:
                    end_date0 = d
                    break
        else:
            end_date0 = e
            for d in df.loc[peak_date0:].index:
                if df.at[d, "Q"] <= qthr:
                    end_date0 = d
                    break

        q_event_full = df.loc[s:end_date0, "Q"]

        if split_multi_peaks:
            subevents = split_event_on_multipeaks(
                q_event_full,
                min_peak_distance_days=min_peak_distance_days,
                min_peak_prominence=min_peak_prominence,
                valley_ratio=valley_ratio,
            )
        else:
            subevents = [(q_event_full.index[0], q_event_full.idxmax(), q_event_full.index[-1])]

        for start_date, peak_date, end_date in subevents:
            start_date_discharge = start_date

            # rainfall attribution window based on peak
            rain_end = peak_date
            rain_start = max(df.index[0], peak_date - pd.Timedelta(days=max_lookback_days))

            search_idx = df.loc[rain_start:rain_end].index
            dry_run = 0
            found_start = rain_start

            for d in reversed(search_idx):
                if df.at[d, "P"] < dry_threshold_mm:
                    dry_run += 1
                else:
                    if dry_run >= dry_spell_days:
                        found_start = d
                        break
                    dry_run = 0
                    found_start = d

            rain_start = found_start
            rain_window = df.loc[rain_start:rain_end, "P"]

            # optional start-date adjustment
            start_adjusted = False
            start_adjust_reason = "no adjustment"
            start_rain_gap_days = (
                (rain_start - start_date).days if pd.notna(rain_start) else np.nan
            )

            if (
                adjust_start_by_rain
                and pd.notna(rain_start)
                and rain_start > start_date
                and (rain_start - start_date).days <= max_start_adjust_days
            ):
                start_date = rain_start
                start_adjusted = True
                start_adjust_reason = (
                    f"adjusted start to rain_start_date "
                    f"(gap={(rain_start - start_date_discharge).days} d)"
                )

            # compute final event metrics once, using final start_date
            q_event = df.loc[start_date:end_date, "Q"]
            p_event = df.loc[start_date:end_date, "P"]
            bf_event = df.loc[start_date:end_date, "baseflow"]
            qf_event = df.loc[start_date:end_date, "quickflow"]

            # recompute peak inside final event window just to stay consistent
            peak_date = q_event.idxmax()
            peak_discharge = float(q_event.max())

            # if peak moved, realign rain_end and rain window
            rain_end = peak_date
            rain_start2 = max(df.index[0], peak_date - pd.Timedelta(days=max_lookback_days))

            search_idx = df.loc[rain_start2:rain_end].index
            dry_run = 0
            found_start = rain_start2

            for d in reversed(search_idx):
                if df.at[d, "P"] < dry_threshold_mm:
                    dry_run += 1
                else:
                    if dry_run >= dry_spell_days:
                        found_start = d
                        break
                    dry_run = 0
                    found_start = d

            rain_start = found_start
            rain_window = df.loc[rain_start:rain_end, "P"]

            if rain_window.sum() > 0:
                t = np.arange(len(rain_window))
                centroid_idx = int(round(np.average(t, weights=rain_window.values)))
                rain_centroid_date = rain_window.index[centroid_idx]
                lag_days = (peak_date - rain_centroid_date).days
            else:
                rain_centroid_date = pd.NaT
                lag_days = np.nan

            events.append({
                "start_date_discharge": start_date_discharge,
                "start_date": start_date,
                "start_adjusted": start_adjusted,
                "start_adjust_reason": start_adjust_reason,
                "start_rain_gap_days": start_rain_gap_days,
                "peak_date": peak_date,
                "end_date": end_date,
                "duration_days": len(q_event),
                "peak_discharge": peak_discharge,
                "event_volume_m3": float((q_event * 86400).sum()),
                "flood_volume_above_baseflow_m3": float((qf_event * 86400).sum()),
                "threshold": float(qthr),
                "rain_start_date": rain_start,
                "rain_end_date": rain_end,
                "event_precip_mm": float(rain_window.sum()),
                "antecedent_precip_3d_mm": float(
                    df.loc[start_date - pd.Timedelta(days=3):start_date - pd.Timedelta(days=1), "P"].sum()
                ) if start_date > df.index[0] else np.nan,
                "antecedent_precip_7d_mm": float(
                    df.loc[start_date - pd.Timedelta(days=7):start_date - pd.Timedelta(days=1), "P"].sum()
                ) if start_date > df.index[0] else np.nan,
                "rain_centroid_date": rain_centroid_date,
                "lag_peak_from_rain_centroid_days": lag_days,
            })

    out = pd.DataFrame(events).sort_values(["start_date", "peak_date"]).reset_index(drop=True)
    return out

def plot_precip_dtp_flood_characteristics(
    df: pd.DataFrame,
    x_col: str = "event_precip_mm",
    y_col: str = "DKM_dtp",
    peak_col: str = "peak_discharge",
    volume_col: str = "event_volume_m3",
    duration_col: str = "duration_days",
    event_id_col: str = "event_id",
    figsize=(12, 3.5),
    cmap_peak: str = "viridis",
    cmap_volume: str = "plasma",
    cmap_duration: str = "cividis",
    log_color_volume: bool = True,
    s: float = 50,
    alpha: float = 0.8,
    edgecolor: str = "k",
    linewidth: float = 0.3,
    annotate: bool = True,
    annotate_top_n: int = 10,
    annotate_by: str = "peak_discharge",   # "peak_discharge", "event_volume_m3", "duration_days"
    savepath: str = None,
):
    data = df.copy()

    if event_id_col not in data.columns:
        data[event_id_col] = np.arange(1, len(data) + 1)

    cols_needed = [x_col, y_col, peak_col, volume_col, duration_col, event_id_col]
    data = data[cols_needed].dropna().copy()

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # Panel 1
    sc_peak = axes[0].scatter(
        data[x_col], data[y_col],
        c=data[peak_col], cmap=cmap_peak,
        s=s, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth
    )
    axes[0].set_xlabel("Precipitation volume (mm)")
    axes[0].set_ylabel("Groundwater depth (DTP)")
    axes[0].set_title("Peak discharge")
    fig.colorbar(sc_peak, ax=axes[0]).set_label("Peak discharge")

    # Panel 2
    if log_color_volume:
        positive = data[volume_col] > 0
        norm_volume = colors.LogNorm(
            vmin=data.loc[positive, volume_col].min(),
            vmax=data.loc[positive, volume_col].max()
        ) if positive.any() else None
    else:
        norm_volume = None

    sc_volume = axes[1].scatter(
        data[x_col], data[y_col],
        c=data[volume_col], cmap=cmap_volume, norm=norm_volume,
        s=s, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth
    )
    axes[1].set_xlabel("Precipitation volume (mm)")
    axes[1].set_ylabel("Groundwater depth (DTP)")
    axes[1].set_title("Flood/event volume")
    fig.colorbar(sc_volume, ax=axes[1]).set_label("Event volume (m³)")

    # Panel 3
    sc_duration = axes[2].scatter(
        data[x_col], data[y_col],
        c=data[duration_col], cmap=cmap_duration,
        s=s, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth
    )
    axes[2].set_xlabel("Precipitation volume (mm)")
    axes[2].set_ylabel("Groundwater depth (DTP)")
    axes[2].set_title("Flood duration")
    fig.colorbar(sc_duration, ax=axes[2]).set_label("Duration (days)")

    # Annotate selected events consistently across all panels
    if annotate:
        if annotate_by not in data.columns:
            raise ValueError(f"annotate_by='{annotate_by}' is not a column in df")

        ann = data.nlargest(annotate_top_n, annotate_by)

        for _, row in ann.iterrows():
            label = str(row[event_id_col])
            for ax in axes:
                ax.annotate(
                    label,
                    (row[x_col], row[y_col]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8
                )

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_events_spatial_single(camels_ts, events, events_index):
    ### plot four flood events
    # events_index = 65
    START_DATE = events.loc[events_index, 'start_date'] #datetime(2015, 11, 17)
    END_DATE   = events.loc[events_index, 'end_date'  ] # datetime(2015, 12, 21)
    events1_q = camels_ts.loc[events.loc[events_index, 'start_date'] : events.loc[events_index, 'end_date'], ]["Qdkm"]
    events1_p = camels_ts.loc[events.loc[events_index, 'rain_start_date'] : events.loc[events_index, 'rain_end_date'], ]["precipitation"]
    
    ### plot events spatial
    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [1, 2.5]}  # smaller top, taller bottom
    )
    
    # Top: time series
    ts = camels_ts.loc[START_DATE: END_DATE, 'precipitation'] # climate['pre'].mean(dim=['x', 'y'])
    axes[0].bar(ts.index.values, ts.values, width=1.0)
    axes[0].set_ylabel('Precip (mm/day)')
    axes[0].set_xlabel('')
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=3))   # every 2 months
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    
    
    # Bottom: spatial map
    climate['pre'].sel(time=events.loc[events_index, 'peak_date']).plot(ax=axes[1], cmap='viridis',
                                         cbar_kwargs={"shrink": 0.7,
                                                      "label": "Precip (mm/day)"
                                                      })
    stations[stations['catch_id'] == STATION_ID].plot(ax=axes[1], color='r', markersize=20)
    catchments[catchments['catch_id'] == STATION_ID].plot(ax=axes[1], facecolor = 'none', edgecolor = 'r')
    axes[1].set_aspect('equal')
    axes[1].axis("off")
    plt.show()



    fig, ax = plt.subplots(figsize=(10, 5))
    
    # --- Discharge ---
    ax.plot(events1_q.index, events1_q,color='tab:blue', lw=2, marker='o', ms=4,label='Discharge')
    
    # Highlight peak
    ax.scatter(events.loc[events_index, 'peak_date'],events.loc[events_index, 'peak_discharge'],color='red', s=80, zorder=5, label='Peak')
    
    # Optional: mark start and end date if available
    ax.axvline(events.loc[events_index, 'start_date'],color='green',  ls='--', lw=1.5, label='Start')
    ax.axvline(events.loc[events_index, 'end_date'],  color='purple', ls='--', lw=1.5, label='End')
    
    ax.set_ylabel('Discharge (m³/s)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    
    # --- Precipitation on secondary axis ---
    ax1 = ax.twinx()
    ax1.bar(events1_p.index, events1_p,width=0.8, color='tab:gray', alpha=0.4,label='Precipitation')
    
    # Invert precipitation axis (common in hydrology)
    ax1.invert_yaxis()
    ax1.set_ylabel('Precipitation (mm/day)', color='tab:gray')
    ax1.tick_params(axis='y', labelcolor='tab:gray')
    
    # --- Date formatting ---
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # --- Legends ---
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.tight_layout()
    plt.show()
    
    
    ### plt GWD_precip_Qv
    events_dtp = events.merge(camels_ts[['DKM_dtp']], left_on='start_date', right_index=True)
    df_q_gw_p = events_dtp[['DKM_dtp', 
                            'event_precip_mm', 
                            'flood_volume_above_baseflow_m3', 
                            'peak_discharge', 
                            'duration_days', 
                            'event_volume_m3'
                            ]].dropna(axis=0)
    
    df_q_gw_p = df_q_gw_p.copy()
    df_q_gw_p["event_id"] = np.arange(1, len(df_q_gw_p) + 1)
    
    fig, axes = plot_precip_dtp_flood_characteristics(
        df_q_gw_p,
        annotate=True,
        annotate_top_n=3,
        annotate_by="peak_discharge"
    )
    plt.show()
    

if __name__ == "__main__": 
    stations, catchments, attributes = load_camels_info()
    
    for STATION_ID in stations['catch_id'].to_list():
        id15_id = int(STATION_ID)
        camels_ts = load_camels_dynamic(id15_id)
        area    = attributes[attributes['catch_id'] == id15_id]['catch_area'].to_list()[0]

        # get flood events
        q = camels_ts["Qobs"] / area * 86400 * 1000
        p = camels_ts["precipitation"]
        
        events = detect_flood_events_with_precip_split(
            q,
            p,
            q_thresh_quantile=0.95,
            merge_gap_days=1,
            dry_threshold_mm=5.0,
            dry_spell_days=1,
            max_lookback_days=10,
            use_baseflow_end=True,
            baseflow_tol=0.05)
        
        # save the results
        with open(f"data/Qevents_id15_{id15_id}.pickle", "wb") as handle:
            pickle.dump([camels_ts, events], handle, protocol=pickle.HIGHEST_PROTOCOL)
