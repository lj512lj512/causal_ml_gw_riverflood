"""CLI: train causal forest and optionally export catchment-held-out estimates."""
import argparse
from pathlib import Path
import joblib
from gwflood.config import StudyConfig
from gwflood.data import read_events
from gwflood.causal import fit_causal, evaluate_heldout_catchments

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", choices=["peak","occurrence","volume","duration"], default="peak")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--evaluate", action="store_true", help="Fit one additional model per held-out fold")
    a = p.parse_args()
    cfg = StudyConfig(target=a.target, n_splits=a.folds)
    df = read_events(a.data)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, clean = fit_causal(df, cfg)
    joblib.dump({"model":model, "config":cfg}, out/"model.joblib")
    if a.evaluate:
        evaluate_heldout_catchments(df, cfg).to_csv(out/"heldout_effects.csv",index=False)
    print(f"Saved model; {len(clean)} complete event rows.")

if __name__ == "__main__":
    main()
