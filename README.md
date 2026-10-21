# Groundwater controls on river floods

Modular research code for event delineation, causal effect estimation, predictive benchmarks and spatial transfer.

## Start here

1. Install dependencies: `pip install -e ".[models]"`
2. Prepare an **event-level** CSV with one row per catchment/rainfall event. Preserve the original files and event definitions; see `docs/MIGRATION.md`.
3. Run `python scripts/train_causal.py --data PATH.csv --out results/causal`.
4. Run `python scripts/train_predictive.py --data PATH.csv --out results/predictive`.
5. For spatial effects, run `python scripts/estimate_global.py --model results/causal/model.joblib --features PATH.csv --out results/spatial.csv`.

These are new, explicit entry points. See `--help` for options. The historical scripts remain unchanged in `backup/` and are **not** automatically equivalent to the new pipeline. In particular, Earth Engine / MIKE SHE preprocessing and historical figure layouts have not yet been migrated.

## Layout

- `gwflood/config.py`: feature and outcome definitions
- `gwflood/data.py`: event-table loading and validation
- `gwflood/events.py`: rainfall event delineation (reference implementation)
- `gwflood/causal.py`: CausalForestDML training and estimation
- `gwflood/predictive.py`: LightGBM predictive baseline
- `gwflood/spatial.py`: effect estimation for new spatial units
- `gwflood/plots.py`: reusable plotting
- `scripts/`: command-line entry points
- `backup/`: original root-level scripts, preserved byte-for-byte

**Research caution:** This refactor separates code but does not itself validate causal identification. Catchment-held-out validation is distinct from temporal blocking. Check treatment units/sign, confounders, overlap, spatial extrapolation, and uncertainty before using maps or manuscript conclusions.
