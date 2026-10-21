"""CLI: delineate rainfall from a daily station CSV."""
import argparse
import pandas as pd
from gwflood.events import detect_rainfall_events

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Daily CSV with date and precipitation")
    p.add_argument("--out", required=True)
    p.add_argument("--date", default="date")
    p.add_argument("--precip", default="precipitation")
    args = p.parse_args()
    ts = pd.read_csv(args.data, parse_dates=[args.date]).set_index(args.date)
    detect_rainfall_events(ts, precip_col=args.precip).to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
