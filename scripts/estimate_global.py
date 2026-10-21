"""CLI: transport effect model to spatial features (not validated outside Denmark)."""
import argparse
from pathlib import Path
import joblib
from gwflood.data import read_events
from gwflood.spatial import estimate_spatial_effects

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,help="train_causal model.joblib")
    p.add_argument("--features",required=True,help="CSV/Parquet with same X feature definitions")
    p.add_argument("--out",required=True)
    p.add_argument("--t0-col",default="dtp_q90_value")
    p.add_argument("--delta",type=float,default=0.1)
    p.add_argument("--id-col",default="catch_id")
    a=p.parse_args()
    obj=joblib.load(a.model)
    out=estimate_spatial_effects(obj["model"],read_events(a.features),
        obj["config"],a.t0_col,a.delta,a.id_col)
    target=Path(a.out); target.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(target,index=False)
    print(f"Wrote {len(out)} spatial units to {target}")

if __name__ == "__main__":
    main()
