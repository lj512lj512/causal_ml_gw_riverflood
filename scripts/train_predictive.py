"""CLI: groundwater ablation with catchment-held-out predictive evaluation."""
import argparse
from pathlib import Path
import pandas as pd
from gwflood.config import StudyConfig
from gwflood.data import read_events
from gwflood.predictive import compare_predictor_sets

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--target",choices=["peak","occurrence","volume","duration"],default="peak")
    p.add_argument("--folds",type=int,default=5)
    a=p.parse_args()
    cfg=StudyConfig(target=a.target,n_splits=a.folds)
    df=read_events(a.data)
    result=pd.concat([compare_predictor_sets(df,cfg,False),
                      compare_predictor_sets(df,cfg,True)],ignore_index=True)
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    result.to_csv(dest/"predictive_ablation.csv",index=False)
    print(result.groupby("include_groundwater")["score"].agg(["mean","std"]))

if __name__ == "__main__":
    main()
