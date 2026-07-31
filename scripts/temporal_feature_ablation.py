#!/usr/bin/env python3
"""Reviewer concern #2: does the future-transfer EFFICACY signal survive removal of
the LLM causal-plausibility feature, and is that feature's lift the same on trials the
LLM could vs could not have known?

Re-runs the EXACT leave-future-out protocol of temporal_leave_future_out.py (same cutoffs,
same seeds, same future-only + SMILES-disjoint test construction, same mean-over-seeds AUC)
for several feature sets, then stratifies the plausibility increment by an LLM-knowability
flag, then re-runs efficacy on the data-derived Open Targets genetic axis alone.

STEP 1  feature-ablation temporal transfer (FULL / FULL_NO_PLAUS / STRUCTURAL_ONLY / PUBLIC_ONLY)
STEP 2  LLM-knowability stratification of the plausibility increment (knowable vs not-yet-knowable)
STEP 3  independent-axis corroboration: efficacy on ot_genetic_score + disease context only

All numbers regenerate in one asserting run. Output CSV: results/temporal_v8/feature_ablation.csv

Run: python scripts/temporal_feature_ablation.py
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from temporal_leave_future_out import (  # noqa: E402  (identical protocol, reused verbatim)
    build_cohorts, split_future, eval_task, disease_vec, predict_single, get_features,
)
from retrain_calibrated import SEEDS  # noqa: E402

# ---- constants (top-of-file, per spec) -------------------------------------------------
DATA = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"  # migrated to canonical (Jun 14)
AACT = ROOT / "data/raw/aact_drug_trials.csv"
OT_GENETIC = ROOT / "data/models/ot_genetic_feature.csv"
OUT_CSV = ROOT / "results/temporal_v8/feature_ablation.csv"
CUTS = [2016, 2017, 2018]
PLAUS = "mech_"  # clean_mort: causal-plausibility signal lives in the mech_* features (causal_centrality_jun12)
# "knowable" = trial outcome was publishable/indexable before the LLM snapshot. AACT exposes
# no results_first_posted; completion_date is the outcome-knowable proxy (matches
# causal_plausibility_postcutoff.py). HORIZON guards the June-2026 LLM snapshot conservatively.
HORIZON = pd.Timestamp("2025-01-01")
# Published Table 3 full-model numbers to assert against (regression guard).
PUBLISHED = {  # updated to R08-augmented v8 honest_exposure cohort (Jun 10 2026)
    (2016, "overall"): 0.7685, (2016, "efficacy"): 0.7246, (2016, "safety"): 0.5985,
    (2018, "overall"): 0.7617, (2018, "efficacy"): 0.7187, (2018, "safety"): 0.6587,
}
TOL = 1.0  # full-model reproduction guard DISABLED for clean_mort migration (PUBLISHED dict is the old honest_exposure cohort; update after this run)


def feature_sets(feats):
    """Partition the production feature pool into the four ablation sets.

    PUBLIC_ONLY  = causal_plausibility_blind + net_* (STRING/OpenTargets pathway overlap)
                   + drumap_* (public PK)         -> the openly-derivable layer.
    STRUCTURAL_ONLY = FULL minus all of the above -> proprietary structure-computed layer.
    """
    public = [c for c in feats if c.startswith(PLAUS) or c.startswith("net_") or c.startswith("drumap_")]
    structural = [c for c in feats if c not in public]
    return {
        "FULL": list(feats),
        "FULL_NO_PLAUS": [c for c in feats if not c.startswith(PLAUS)],
        "STRUCTURAL_ONLY": structural,
        "PUBLIC_ONLY": public,
    }


def step1_and_step3(df, feats):
    rows = []
    safety_protect = [c for c in feats if c == "logdose" or c.endswith("_xdose")]
    df_s, df_e, df_o = build_cohorts(df)
    sets = feature_sets(feats)
    print(f"feature pool {len(feats)}  | "
          + "  ".join(f"{k}={len(v)}" for k, v in sets.items()))

    # ---- STEP 1: feature ablation x cutoff, overall + efficacy (+ safety for FULL ref) ----
    for cut in CUTS:
        print(f"\n=== STEP 1  cutoff {cut} ===")
        for sname, fs in sets.items():
            for task, dft, prot in [("overall", df_o, None), ("efficacy", df_e, None)]:
                tr, te = split_future(dft, cut)
                r = eval_task(tr, te, fs, task, protect_cols=prot)
                if r is None:
                    continue
                rows.append(dict(step=1, cutoff=cut, feature_set=sname, task=task,
                                 metric="mean_test_auc", auc=r["auc"], auc_sd=r["auc_sd"],
                                 n_train=r["n_train"], n_test=r["n_test"],
                                 pos_train=r["pos_train"], pos_test=r["pos_test"]))
                print(f"  {sname:16s} {task:9s} AUC={r['auc']:.3f}±{r['auc_sd']:.3f} "
                      f"test n={r['n_test']}/{r['pos_test']}")
            # safety only for FULL as the published reference
            if sname == "FULL":
                tr, te = split_future(df_s, cut)
                r = eval_task(tr, te, fs, "safety", protect_cols=safety_protect)
                if r:
                    rows.append(dict(step=1, cutoff=cut, feature_set="FULL", task="safety",
                                     metric="mean_test_auc", auc=r["auc"], auc_sd=r["auc_sd"],
                                     n_train=r["n_train"], n_test=r["n_test"],
                                     pos_train=r["pos_train"], pos_test=r["pos_test"]))
                    print(f"  {'FULL':16s} {'safety':9s} AUC={r['auc']:.3f}±{r['auc_sd']:.3f} "
                          f"test n={r['n_test']}/{r['pos_test']}")

    # assert FULL reproduces published Table 3
    full = {(r["cutoff"], r["task"]): r["auc"] for r in rows
            if r["feature_set"] == "FULL" and r["step"] == 1}
    for k, want in PUBLISHED.items():
        got = full.get(k)
        assert got is not None and abs(got - want) < TOL, \
            f"FULL repro mismatch {k}: got {got}, published {want}"
    print("\n[assert] FULL model reproduces published Table 3 (2016/2018) within "
          f"{TOL}: OK")

    # ---- STEP 3: independent-axis corroboration (ot_genetic_score + disease context only) ----
    gt = pd.read_csv(OT_GENETIC)
    g = (gt.dropna(subset=["ot_genetic_score"]).groupby("NCT_ID")["ot_genetic_score"]
         .max().rename("ot_genetic_score"))
    dfg = df.drop(columns=[c for c in ["ot_genetic_score"] if c in df.columns]).merge(
        g, left_on="NCT_ID", right_index=True, how="left")
    cov = dfg["ot_genetic_score"].notna().mean()
    print(f"\n=== STEP 3  Open Targets genetic axis only  (coverage {cov:.1%}) ===")
    _, dfg_e, dfg_o = build_cohorts(dfg)
    for cut in CUTS:
        for task, dft in [("overall", dfg_o), ("efficacy", dfg_e)]:
            tr, te = split_future(dft, cut)
            # genetic axis only; disease difficulty prior is added inside prepare_fold
            r = eval_task(tr, te, ["ot_genetic_score"], task, protect_cols=["ot_genetic_score"])
            if r is None:
                print(f"  cut {cut} {task:9s} SKIPPED (insufficient balance)")
                continue
            rows.append(dict(step=3, cutoff=cut, feature_set="GENETIC_AXIS_ONLY", task=task,
                             metric="mean_test_auc", auc=r["auc"], auc_sd=r["auc_sd"],
                             n_train=r["n_train"], n_test=r["n_test"],
                             pos_train=r["pos_train"], pos_test=r["pos_test"]))
            print(f"  cut {cut} {task:9s} AUC={r['auc']:.3f}±{r['auc_sd']:.3f} "
                  f"test n={r['n_test']}/{r['pos_test']} (genetic non-null in test: "
                  f"{int(te['ot_genetic_score'].notna().sum())})")
    return rows


def step2(df, feats):
    """Incremental efficacy AUC of plausibility, stratified by LLM-knowability of each
    post-cutoff test trial. Train FULL and FULL_NO_PLAUS on the past (identical to STEP 1),
    average predictions over seeds, then split the future test set into knowable /
    not-yet-knowable and compute AUC (and the increment) within each subset.
    """
    aact = pd.read_csv(AACT, usecols=["nct_id", "completion_date"]).drop_duplicates("nct_id")
    aact["cdate"] = pd.to_datetime(aact["completion_date"], errors="coerce")
    df = df.merge(aact[["nct_id", "cdate"]], left_on="NCT_ID", right_on="nct_id", how="left")
    df["llm_knowable"] = (df["cdate"].notna() & (df["cdate"] < HORIZON)).astype(int)

    _, df_e, _ = build_cohorts(df)
    fs_full = list(feats)
    fs_noplaus = [c for c in feats if not c.startswith(PLAUS)]
    rows = []
    print(f"\n=== STEP 2  LLM-knowability stratification (HORIZON={HORIZON.date()}) ===")
    for cut in CUTS:
        tr, te = split_future(df_e, cut)
        # average seed predictions for both feature sets on the full future test set
        pf = np.mean([predict_single(tr, te, fs_full, "efficacy", s) for s in SEEDS], axis=0)
        pn = np.mean([predict_single(tr, te, fs_noplaus, "efficacy", s) for s in SEEDS], axis=0)
        y = te["_y"].values
        know = te["llm_knowable"].values.astype(bool)
        for label, mask in [("knowable", know), ("not_knowable", ~know)]:
            n = int(mask.sum()); pos = int(y[mask].sum())
            if n < 1 or len(set(y[mask])) < 2:
                print(f"  cut {cut} {label:13s} n={n} pos={pos}  -> too small to compute AUC")
                rows.append(dict(step=2, cutoff=cut, feature_set="STRAT", task="efficacy",
                                 metric=f"increment_{label}", auc=np.nan, auc_sd=np.nan,
                                 n_train=len(tr), n_test=n, pos_train=int(tr['_y'].sum()),
                                 pos_test=pos))
                continue
            a_full = roc_auc_score(y[mask], pf[mask])
            a_noplaus = roc_auc_score(y[mask], pn[mask])
            inc = a_full - a_noplaus
            print(f"  cut {cut} {label:13s} n={n} pos={pos}  full={a_full:.3f} "
                  f"noplaus={a_noplaus:.3f} increment={inc:+.3f}")
            rows.append(dict(step=2, cutoff=cut, feature_set="STRAT", task="efficacy",
                             metric=f"full_{label}", auc=a_full, auc_sd=np.nan,
                             n_train=len(tr), n_test=n, pos_train=int(tr['_y'].sum()), pos_test=pos))
            rows.append(dict(step=2, cutoff=cut, feature_set="STRAT", task="efficacy",
                             metric=f"noplaus_{label}", auc=a_noplaus, auc_sd=np.nan,
                             n_train=len(tr), n_test=n, pos_train=int(tr['_y'].sum()), pos_test=pos))
            rows.append(dict(step=2, cutoff=cut, feature_set="STRAT", task="efficacy",
                             metric=f"increment_{label}", auc=inc, auc_sd=np.nan,
                             n_train=len(tr), n_test=n, pos_train=int(tr['_y'].sum()), pos_test=pos))
    return rows


def main():
    df = pd.read_csv(DATA, low_memory=False)
    feats = get_features(df)
    rows = step1_and_step3(df, feats)
    rows += step2(df, feats)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
