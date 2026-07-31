#!/usr/bin/env python3
"""Full-model temporal (leave-future-out) validation for the v8 honest cohort.

Train on OLDER trials (Start_Year < cut), test on NEWER trials (Start_Year >= cut),
DRUG-DISJOINT (any SMILES present in train is removed from test), for all three heads:
  - overall  : single GBM (production overall head)
  - efficacy : calibrated ensemble (production efficacy head)
  - safety   : noisy-OR over per-mechanism detectors (production safety head)

Cohorts/labels/exclusions/protect mirror retrain_calibrated.main() exactly.
Reports mean-over-seeds test AUC per task per cutoff. This is the honest prospective
proxy: every test compound is both temporally future AND structurally absent from train.

Runs on the canonical clean_mort cohort, the same one the production model uses; this is
the run reported in Supplementary Table S4.

Usage:
  python scripts/temporal_leave_future_out.py \
      --cuts 2016 2017 2018 --out results/temporal_v8/leave_future_out.json
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_calibrated import (  # noqa: E402
    prepare_fold, fit_predict, fit_gbm, predict_ensemble, fit_ensemble_eff,
    mechanism_groups, get_features, SEEDS,
)


def build_cohorts(df):
    safety_mask = df["Corrected_Outcome"].isin(["PASS", "FAIL_SAFETY", "FAIL_BOTH"])
    if "is_multi_drug_exclude" in df.columns:
        safety_mask &= df["is_multi_drug_exclude"] != 1
    df_s = df[safety_mask].copy()
    df_s["_y"] = df_s["Corrected_Outcome"].isin(["FAIL_SAFETY", "FAIL_BOTH"]).astype(int)

    eff_excl = pd.Series(False, index=df.index)
    for c in ["is_anti_pathogen", "is_endogenous", "is_mispaired_supportive",
              "is_healthy_volunteer", "is_procedural_exclude", "is_multi_drug_exclude"]:
        if c in df.columns:
            eff_excl |= df[c] == 1
    df_e = df[~eff_excl & df["Corrected_Outcome"].isin(
        ["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    df_e["_y"] = df_e["Corrected_Outcome"].isin(["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)

    df_o = df[df["Corrected_Outcome"].isin(
        ["PASS", "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    df_o["_y"] = df_o["Corrected_Outcome"].isin(
        ["FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)
    return df_s, df_e, df_o


def split_future(dft, cut):
    """train = Start_Year < cut; test = Start_Year >= cut AND SMILES absent from train."""
    tr = dft[dft.Start_Year < cut].copy()
    te = dft[dft.Start_Year >= cut].copy()
    te = te[~te.SMILES.isin(set(tr.SMILES))].copy()
    return tr, te


def disease_vec(d):
    return d["Disease"].fillna("unknown").values


def predict_single(tr, te, feats, task, seed, protect_idx=None):
    Xt, Xe, _, _ = prepare_fold(tr[feats].values, te[feats].values,
                                tr["_y"].values, disease_vec(tr), disease_vec(te),
                                feats, protect_idx=protect_idx)
    return fit_predict(Xt, tr["_y"].values, Xe, task, seed)


def predict_noisy_or(tr, te, feats, protect_cols, seed):
    groups = mechanism_groups(feats)
    cols = []
    for g, cs in groups.items():
        prot = [c for c in protect_cols if c in cs]
        prot_idx = [cs.index(c) for c in prot] if prot else None
        Xt, Xe, _, _ = prepare_fold(tr[cs].values, te[cs].values,
                                    tr["_y"].values, disease_vec(tr), disease_vec(te),
                                    cs, protect_idx=prot_idx)
        mod = fit_gbm(Xt, tr["_y"].values, seed)
        cols.append(mod.predict_proba(Xe)[:, 1])
    M = np.clip(np.array(cols).T, 0, 0.999)
    return 1.0 - np.prod(1.0 - M, axis=1)


def eval_task(tr, te, feats, task, protect_cols=None):
    if te["_y"].nunique() < 2 or len(tr) < 20 or tr["_y"].nunique() < 2:
        return None
    protect_idx = ([feats.index(c) for c in protect_cols if c in feats]
                   if protect_cols else None)
    aucs = []
    for seed in SEEDS:
        if task == "safety":
            p = predict_noisy_or(tr, te, feats, protect_cols or [], seed)
        else:
            p = predict_single(tr, te, feats, task, seed, protect_idx=protect_idx)
        aucs.append(roc_auc_score(te["_y"].values, p))
    return {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "pos_train": int(tr["_y"].sum()), "pos_test": int(te["_y"].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sources/training_dataset_v8_clean_mort.csv")
    ap.add_argument("--cuts", type=int, nargs="+", default=[2016, 2017, 2018])
    ap.add_argument("--out", default="results/temporal_v8/leave_future_out.json")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.data, low_memory=False)
    feats = get_features(df)
    safety_protect = [c for c in feats if c == "logdose" or c.endswith("_xdose")]
    df_s, df_e, df_o = build_cohorts(df)
    print(f"loaded {len(df)} trials, {len(feats)} feats; "
          f"safety n={len(df_s)}/{df_s._y.sum()} eff n={len(df_e)}/{df_e._y.sum()} "
          f"over n={len(df_o)}/{df_o._y.sum()}", flush=True)

    results = {}
    for cut in args.cuts:
        print(f"\n=== cutoff {cut} (train <{cut}, test >={cut}, drug-disjoint) ===", flush=True)
        cut_res = {}
        for task, dft, prot in [("overall", df_o, None),
                                ("safety", df_s, safety_protect),
                                ("efficacy", df_e, None)]:
            tr, te = split_future(dft, cut)
            r = eval_task(tr, te, feats, task, protect_cols=prot)
            cut_res[task] = r
            if r:
                print(f"  {task:9s} AUC={r['auc']:.3f}±{r['auc_sd']:.3f} "
                      f"train n={r['n_train']}/{r['pos_train']} "
                      f"test n={r['n_test']}/{r['pos_test']}", flush=True)
            else:
                print(f"  {task:9s} SKIPPED (insufficient class balance)", flush=True)
        results[str(cut)] = cut_res

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
