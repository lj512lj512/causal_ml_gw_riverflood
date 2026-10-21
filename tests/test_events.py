import pandas as pd
from gwflood.events import detect_rainfall_events

def test_rain_event_threshold_and_gap():
    t = pd.DataFrame({"precipitation":[0,5,8,0,0,12,0,0]},
        index=pd.date_range("2020-01-01",periods=8))
    e=detect_rainfall_events(t,threshold=3,dry_gap=2)
    assert len(e)==2
    assert e["event_precip_mm"].tolist()==[13,12]
