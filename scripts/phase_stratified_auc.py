#!/usr/bin/env python3
"""Absolute headline AUC, full cohort vs Phase-III-only, for all three tasks.

PASS requires Phase III completion, so terminated failures skew to earlier phases and the
full-cohort AUC could in principle ride a phase confound. Restricting to Phase III (where
both PASS and failing drugs are present) tests that directly: if the headline holds within
Phase III, it is not a phase artifact. Complements phase_matched_decomposition.py (which
does the molecular *increment*). Reads the canonical OOF; mean-of-folds, matching the headline.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/production_v8_clean_mort_singlehead_jul6"
df = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
ph = df[["SMILES", "Disease", "Phase"]].drop_duplicates(["SMILES", "Disease"])
P3 = ["Phase 3", "Phase 2/3"]


def mof(sub):
    a = [roc_auc_score(g.y, g.raw_prob) for _, g in sub.groupby(["seed", "fold"]) if g.y.nunique() > 1]
    return float(np.mean(a))


print(f"{'task':9s} {'full':>6s} {'Phase III':>10s} {'drop':>7s}   positive phase-mix")
for task in ["overall", "efficacy", "safety"]:
    oof = pd.read_parquet(BASE / f"oof_{task}.parquet").merge(ph, on=["SMILES", "Disease"], how="left")
    oof["is_p3"] = oof.Phase.isin(P3)
    full, p3 = mof(oof), mof(oof[oof.is_p3])
    g = oof.groupby(["SMILES", "Disease"]).agg(y=("y", "mean"), Phase=("Phase", "first"))
    mix = g[g.y > 0.5].Phase.value_counts().to_dict()
    print(f"{task:9s} {full:6.3f} {p3:10.3f} {full-p3:+7.3f}   {mix}")
