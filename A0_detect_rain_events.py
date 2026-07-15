# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:28:39 2026

@author: juliu
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from A0_detect_flood_events import load_camels_info, load_camels_dynamic
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def extract_main_rainfall_pulse(
    ts_event,
    precip_col='precipitation',
    pulse_threshold=3.0,
    min_event_precip=10.0,
    max_pulse_duration=14
):
    """
    Extract the main rainfall pulse from a long rainfall event.

    The main pulse is centered on the day with maximum precipitation.
    It expands backward and forward while precipitation remains
    >= pulse_threshold.

    Parameters
    ----------
    ts_event : pandas.DataFrame
        Subset of time series for one long rainfall event.
    precip_col : str
        Precipitation column.
    pulse_threshold : float
        Minimum daily precipitation used to define the pulse.
    min_event_precip : float
        Minimum total precipitation required to keep the pulse.
    max_pulse_duration : int
        Maximum allowed pulse duration.

    Returns
    -------
    dict or None
        Main-pulse event information, or None if no valid pulse is found.
    """

    ts_event = ts_event.copy().sort_index()
    P = ts_event[precip_col].fillna(0)

    if len(P) == 0:
        return None

    # Day of maximum rainfall
    peak_day = P.idxmax()

    # Expand backward from peak day
    pulse_start = peak_day
    current = peak_day

    while True:
        previous_day = current - pd.Timedelta(days=1)

        if previous_day not in P.index:
            break

        if P.loc[previous_day] >= pulse_threshold:
            pulse_start = previous_day
            current = previous_day
        else:
            break

    # Expand forward from peak day
    pulse_end = peak_day
    current = peak_day

    while True:
        next_day = current + pd.Timedelta(days=1)

        if next_day not in P.index:
            break

        if P.loc[next_day] >= pulse_threshold:
            pulse_end = next_day
            current = next_day
        else:
            break

    pulse_P = P.loc[pulse_start:pulse_end]

    pulse_precip = pulse_P.sum()
    pulse_duration = (pulse_end - pulse_start).days + 1
    max_1d_precip = pulse_P.max()

    if pulse_precip < min_event_precip:
        return None

    if pulse_duration > max_pulse_duration:
        # If still too long, keep a window around the peak day
        half_window = max_pulse_duration // 2
        pulse_start = peak_day - pd.Timedelta(days=half_window)
        pulse_end = pulse_start + pd.Timedelta(days=max_pulse_duration - 1)

        # Clip to available event dates
        pulse_start = max(pulse_start, P.index.min())
        pulse_end = min(pulse_end, P.index.max())

        pulse_P = P.loc[pulse_start:pulse_end]
        pulse_precip = pulse_P.sum()
        pulse_duration = (pulse_end - pulse_start).days + 1
        max_1d_precip = pulse_P.max()

        if pulse_precip < min_event_precip:
            return None

    return {
        'rain_start_date': pulse_start,
        'rain_end_date': pulse_end,
        'rain_duration_days': pulse_duration,
        'event_precip_mm': pulse_precip,
        'max_1d_precip_mm': max_1d_precip,
        'main_pulse_date': peak_day,
        'split_from_long_event': True,
        'original_rain_start': P.index.min(),
        'original_rain_end': P.index.max(),
        'original_rain_duration_days': len(P),
        'original_event_precip_mm': P.sum()
    }


def detect_rainfall_events(
    ts,
    precip_col='precipitation',
    rain_threshold=3.0,
    dry_gap=2,
    min_event_precip=10.0,
    max_event_duration=14,
    use_main_pulse_for_long_events=True,
    pulse_threshold=3.0
):
    """
    Detect rainfall events using threshold and dry-gap separation.

    For events longer than max_event_duration, optionally replace the
    long event with its main rainfall pulse.

    Parameters
    ----------
    ts : pandas.DataFrame
        Daily time series with datetime index.
    precip_col : str
        Name of precipitation column in mm/day.
    rain_threshold : float
        Rainfall threshold in mm/day to start/continue an event.
    dry_gap : int
        Number of consecutive dry days required to end an event.
    min_event_precip : float
        Minimum total event precipitation in mm.
    max_event_duration : int
        Maximum event duration retained as a normal event.
    use_main_pulse_for_long_events : bool
        If True, long events are replaced by their main rainfall pulse.
        If False, long events are excluded.
    pulse_threshold : float
        Rainfall threshold used to define the main pulse.

    Returns
    -------
    events : pandas.DataFrame
        Rainfall-event table.
    """

    ts = ts.copy()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()

    P = ts[precip_col].fillna(0).values
    dates = ts.index.to_numpy()

    events = []
    in_event = False
    start_idx = None
    last_wet_idx = None
    dry_count = 0

    def finalize_event(start_idx, end_idx):
        event_slice = ts.iloc[start_idx:end_idx + 1]
        event_P = event_slice[precip_col].fillna(0)

        event_precip = event_P.sum()
        event_duration = end_idx - start_idx + 1
        max_1d_precip = event_P.max()

        if event_precip < min_event_precip:
            return None

        # Normal event
        if event_duration <= max_event_duration:
            return {
                'rain_start_date': pd.Timestamp(dates[start_idx]),
                'rain_end_date': pd.Timestamp(dates[end_idx]),
                'rain_duration_days': event_duration,
                'event_precip_mm': event_precip,
                'max_1d_precip_mm': max_1d_precip,
                'main_pulse_date': pd.Timestamp(event_P.idxmax()),
                'split_from_long_event': False,
                'original_rain_start': pd.NaT,
                'original_rain_end': pd.NaT,
                'original_rain_duration_days': np.nan,
                'original_event_precip_mm': np.nan
            }

        # Long event
        if not use_main_pulse_for_long_events:
            return None

        pulse = extract_main_rainfall_pulse(
            event_slice,
            precip_col=precip_col,
            pulse_threshold=pulse_threshold,
            min_event_precip=min_event_precip,
            max_pulse_duration=max_event_duration
        )

        return pulse

    for i, p in enumerate(P):

        is_wet = p >= rain_threshold

        if not in_event:
            if is_wet:
                in_event = True
                start_idx = i
                last_wet_idx = i
                dry_count = 0

        else:
            if is_wet:
                last_wet_idx = i
                dry_count = 0
            else:
                dry_count += 1

                if dry_count >= dry_gap:
                    end_idx = last_wet_idx

                    event = finalize_event(start_idx, end_idx)

                    if event is not None:
                        events.append(event)

                    in_event = False
                    start_idx = None
                    last_wet_idx = None
                    dry_count = 0

    # Close event if time series ends during an event
    if in_event:
        end_idx = last_wet_idx

        event = finalize_event(start_idx, end_idx)

        if event is not None:
            events.append(event)

    events = pd.DataFrame(events)

    if len(events) == 0:
        return events

    events = events.sort_values('rain_start_date').reset_index(drop=True)

    return events


def response_lag_from_area(catch_area_km2, min_lag=5, max_lag=21):
    """
    Area-dependent response lag following T = 5 + log(A),
    where A is catchment area in square miles.
    """
    area_mi2 = catch_area_km2 * 0.386102
    lag = int(round(5 + np.log(area_mi2)))
    lag = int(np.clip(lag, min_lag, max_lag))
    return lag

def add_flood_severity_metrics(
    events,
    ts,
    q_col='Qobs',
    catch_area_km2=None,
    q_threshold=None,
    response_lag_days=7,
    use_specific_discharge=True
):
    events = events.copy()
    ts = ts.copy()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()

    events['rain_start_date'] = pd.to_datetime(events['rain_start_date'])
    events['rain_end_date'] = pd.to_datetime(events['rain_end_date'])

    if 'next_rain_start' in events.columns:
        events['next_rain_start'] = pd.to_datetime(events['next_rain_start'])

    if use_specific_discharge:
        if catch_area_km2 is None:
            raise ValueError("catch_area_km2 is required for specific discharge.")
        ts['Q_mmd'] = ts[q_col] * 86.4 / catch_area_km2
        q_use_col = 'Q_mmd'
    else:
        q_use_col = q_col

    if q_threshold is None:
        q_threshold = ts[q_use_col].quantile(0.95)

    output = []

    for _, row in events.iterrows():

        rain_start = row['rain_start_date']
        rain_end = row['rain_end_date']

        response_start = rain_start
        response_end_reg = rain_end + pd.Timedelta(days=response_lag_days)

        next_rain_start = row.get('next_rain_start', pd.NaT)

        if pd.notna(next_rain_start) and next_rain_start <= response_end_reg:
            response_end = next_rain_start - pd.Timedelta(days=1)
            truncated = True
        else:
            response_end = response_end_reg
            truncated = False

        # Safety check: response_end should not be before rain_end
        # If it is, keep at least the rainfall period
        if response_end < rain_end:
            response_end = rain_end
            truncated = True

        q_window = ts.loc[response_start:response_end, q_use_col].dropna()

        result = row.to_dict()
        result['response_start'] = response_start
        result['response_end'] = response_end
        result['response_lag_days'] = response_lag_days
        result['response_truncated_by_next_rain'] = truncated
        result['q95_threshold'] = q_threshold

        if len(q_window) == 0:
            result.update({
                'flood_occurrence': np.nan,
                'flood_peak': np.nan,
                'flood_peak_date': pd.NaT,
                'flood_volume_above_q95': np.nan,
                'days_above_q95': np.nan,
                'duration_above_q95': np.nan,
                'time_to_peak_from_start': np.nan,
                'time_to_peak_from_rain_end': np.nan,
            })
            output.append(result)
            continue

        peak_val = q_window.max()
        peak_day = q_window.idxmax()
        excess = (q_window - q_threshold).clip(lower=0)

        above_q95 = q_window >= q_threshold

        result['flood_occurrence'] = int(above_q95.any())
        result['flood_peak'] = peak_val
        result['flood_peak_date'] = peak_day
        result['flood_volume_above_q95'] = excess.sum()
        result['days_above_q95'] = int(above_q95.sum())

        if above_q95.any():
            above = q_window[above_q95]
            result['duration_above_q95'] = (above.index[-1] - above.index[0]).days + 1
        else:
            result['duration_above_q95'] = 0

        result['time_to_peak_from_start'] = (peak_day - rain_start).days
        result['time_to_peak_from_rain_end'] = (peak_day - rain_end).days

        output.append(result)

    return pd.DataFrame(output)


if __name__ == "__main__": 
    stations, catchments, attributes = load_camels_info()
    
    all_events = []
    for STATION_ID in stations['catch_id'].to_list():
        id15_id = int(STATION_ID)
        camels_ts = load_camels_dynamic(id15_id)
        camels_ts = camels_ts.loc[datetime(1989, 1, 1) : datetime(2019, 12, 31)]
        catch_area_km2 = (attributes.loc[attributes['catch_id'] == id15_id, 'catch_area'].iloc[0] / 1000000)
        
        events = detect_rainfall_events(camels_ts,
                                        precip_col='precipitation',
                                        rain_threshold=3.0,
                                        dry_gap=2,
                                        min_event_precip=10.0,
                                        max_event_duration=14,
                                        use_main_pulse_for_long_events=True,
                                        pulse_threshold=3.0)
        print(events['split_from_long_event'].value_counts())
        
        events['catch_id'] = id15_id
        events = events.sort_values('rain_start_date').copy()
        # Add next valid rainfall event start date
        events['next_rain_start'] = events['rain_start_date'].shift(-1)
        
        # add flood events 
        events = add_flood_severity_metrics(
            events=events,
            ts=camels_ts,
            q_col='Qobs',
            catch_area_km2=catch_area_km2,
            response_lag_days=response_lag_from_area(catch_area_km2),
            use_specific_discharge=True)
        all_events.append(events)
        
    # get all flood events
    all_events = pd.concat(all_events, ignore_index=True)
    all_events.to_csv(r'data/rainfall_runoff_events_all_304_catchments.csv')
    
    # check how many merged events, rainfall contunity
    all_events['rain_end_date'] = pd.to_datetime(all_events['rain_end_date'])
    all_events['response_end'] = pd.to_datetime(all_events['response_end'])
    bad = all_events[all_events['response_end'] < all_events['rain_end_date']] 
    print("Number of bad response windows:", len(bad))
    
    # check how many floods 
    print("Flood occurrence rate:")
    print(all_events['flood_occurrence'].mean()) 
    print("Truncated response windows:")
    print(all_events['response_truncated_by_next_rain'].mean()) 
    print(all_events[
                    ['rain_duration_days', 
                     'response_lag_days',
                     'flood_occurrence', 'days_above_q95',
                     'time_to_peak_from_start', 'time_to_peak_from_rain_end']
                ].describe())
    
#%% plot
id15_id = 12430456

ts_s = datetime(2006, 7, 20)
ts_e = datetime(2007, 8, 1)

stor_river = all_events[all_events['catch_id'] == id15_id]
stor_river = stor_river[(stor_river['rain_start_date'] >ts_s)  & (stor_river['rain_start_date'] < ts_e)]

camels_ts   = load_camels_dynamic(id15_id)
camels_ts_y = camels_ts.loc[ts_s : ts_e]


# Make sure dates are datetime
camels_ts_y = camels_ts_y.copy()
camels_ts_y.index = pd.to_datetime(camels_ts_y.index)

stor_river = stor_river.copy()
stor_river['rain_start_date'] = pd.to_datetime(stor_river['rain_start_date'])
stor_river['rain_end_date']   = pd.to_datetime(stor_river['rain_end_date'])
stor_river['response_end']   = pd.to_datetime(stor_river['response_end'])
# Q95 threshold from full time series or yearly subset
q95 = camels_ts['Qobs'].quantile(0.95)

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 9),
    sharex=True,
    constrained_layout=True
)

ax1, ax2, ax3 = axes

# --------------------------------------------------
# 1. Precipitation
# --------------------------------------------------
ax1.bar(camels_ts_y.index,camels_ts_y['precipitation'],width=1.0,align='center')


# Shade rainfall events
for _, row in stor_river.iterrows():
    ax1.axvspan(row['rain_start_date'],row['rain_end_date'],alpha=0.25)
ax1.set_xlim([ts_s, ts_e])
ax1.set_ylabel('Precipitation (mm/d)')
# --------------------------------------------------
# 2. Discharge
# --------------------------------------------------
ax2.plot(camels_ts_y.index,camels_ts_y['Qobs'],linewidth=1.5,label='Observed discharge')
ax2.axhline(y=q95,linestyle='--',linewidth=1.2,label='Q95 threshold')

# Shade same rainfall events
for _, row in stor_river.iterrows():
    ax2.axvspan(row['rain_start_date'],row['rain_end_date'],alpha=0.15)
    ax2.axvline(x=row['response_end'], color='gray')

ax2.set_ylabel('Discharge (m³/s)')
ax2.legend(loc='upper right', frameon=False)
ax2.set_xlim([ts_s, ts_e])

# --------------------------------------------------
# 3. Groundwater depth
# --------------------------------------------------
ax3.plot(camels_ts_y.index,camels_ts_y['DKM_dtp'],c='tab:blue',   linewidth=1.5,label='Groundwater depth')
ax4 = ax3.twinx()
ax4.plot(camels_ts_y.index,camels_ts_y['DKM_wcr'],c='tab:orange', linewidth=1.5,label='Soil moisture')
# Shade same rainfall events
for _, row in stor_river.iterrows():
    ax3.axvspan(row['rain_start_date'],row['rain_end_date'],alpha=0.15)

ax3.set_ylabel('Groundwater depth (m)')
ax4.set_ylabel('Soil water content')
ax3.set_xlabel('Date')
ax3.legend(loc='upper right', frameon=False)
ax3.set_xlim([ts_s, ts_e])
plt.show()
