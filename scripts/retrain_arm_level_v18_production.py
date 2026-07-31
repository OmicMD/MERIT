#!/usr/bin/env python3
"""Arm-level v18 (production): SHARED safety head + DILI dose×logP feature.

Replaces v12 stratified safety with a single shared safety model (disease_is_oncology
is an ordinary feature) plus dili_dose_x_logp (validated +0.016). Efficacy + overall
unchanged. Writes results/arm_level_v18/metrics.json.

Produces the published arm-level numbers (Supplementary Table S14: overall 0.710,
safety 0.688, efficacy 0.720).

NOT CURRENTLY RUNNABLE (Jul 2026): reads data/models/dili_dose_logp_feature.csv, which is
absent from the repo. scripts/build_dili_dose_lipophilicity.py takes its output path as
argv[2], so that file was written ad hoc and never committed. The committed
results/arm_level_v18/metrics.json is intact and is what Table S14 reports.
"""
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_calibrated import run_task_cv
from retrain_arm_level import DATA, prepare_arm_dataset
from retrain_arm_level_v12_stratified import merge_tdc, get_feature_cols

OUT = ROOT / "results" / "arm_level_v18"; OUT.mkdir(parents=True, exist_ok=True)


def fold_aucs(oof):
    return [roc_auc_score(g.y, g.raw_prob) for _, g in oof.groupby(["seed", "fold"]) if g.y.nunique() == 2]


def main():
    df = prepare_arm_dataset(); df = merge_tdc(df)
    feat = pd.read_csv(ROOT / "data/models/dili_dose_logp_feature.csv")
    df = df.merge(feat, on=["NCT_ID", "Arm_Label"], how="left")
    df["dili_dose_x_logp"] = df["dili_dose_x_logp"].fillna(df["dili_dose_x_logp"].median())
    feats = get_feature_cols(df)
    print(f"features: {len(feats)} (incl dili_dose_x_logp={'dili_dose_x_logp' in feats})", flush=True)

    def task(mask, pos):
        t = df[df.Corrected_Outcome.isin(mask)].copy()
        t["_y"] = t.Corrected_Outcome.isin(pos).astype(int)
        return t

    res = {}
    print("=== SAFETY (shared + dili) ===", flush=True)
    s = task(["PASS", "FAIL_SAFETY", "FAIL_BOTH"], ["FAIL_SAFETY", "FAIL_BOTH"])
    oof_s, _ = run_task_cv(s, feats, "safety", calibrate=False)
    oof_s["onco"] = oof_s.row_idx.map(s["disease_is_oncology"].fillna(0))
    a = fold_aucs(oof_s)
    g = oof_s.groupby("row_idx").agg(y=("y", "first"), p=("raw_prob", "mean"), onco=("onco", "first")).reset_index()
    res["safety_auc_mean"] = float(np.mean(a)); res["safety_auc_std"] = float(np.std(a))
    res["safety_nononc_pooled"] = float(roc_auc_score(g[g.onco == 0].y, g[g.onco == 0].p))
    res["safety_onc_pooled"] = float(roc_auc_score(g[g.onco == 1].y, g[g.onco == 1].p))

    print("=== EFFICACY ===", flush=True)
    excl = pd.Series(False, index=df.index)
    for c in ("is_anti_pathogen", "is_endogenous"):
        if c in df.columns:
            excl |= df[c] == 1
    e = df[~excl & df.Corrected_Outcome.isin(["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    e["_y"] = e.Corrected_Outcome.isin(["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)
    oof_e, _ = run_task_cv(e, feats, "efficacy", calibrate=False)
    ae = fold_aucs(oof_e); res["efficacy_auc_mean"] = float(np.mean(ae)); res["efficacy_auc_std"] = float(np.std(ae))

    print("=== OVERALL ===", flush=True)
    o = task(["PASS", "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"], ["FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"])
    oof_o, _ = run_task_cv(o, feats, "overall", calibrate=False)
    ao = fold_aucs(oof_o); res["overall_auc_mean"] = float(np.mean(ao)); res["overall_auc_std"] = float(np.std(ao))

    # persist OOF for figure regeneration (arm ROC panel)
    oof_s.to_parquet(OUT / "oof_safety.parquet", index=False)
    oof_e.to_parquet(OUT / "oof_efficacy.parquet", index=False)
    oof_o.to_parquet(OUT / "oof_overall.parquet", index=False)
    res["approach"] = "shared_safety + dili_dose_x_logp"; res["n_features"] = len(feats)
    with open(DATA, "rb") as f:
        res["data_sha256"] = hashlib.sha256(f.read()).hexdigest()
    json.dump(res, open(OUT / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
