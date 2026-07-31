#!/usr/bin/env python3
"""Arm-level retrain v12: stratified safety head (oncology vs non-oncology).

Motivation (Jun 4 2026 OOF analysis):
  - 75% of safety TPs are oncology arms (kinase/HDAC/PI3K inhibitors)
  - 80% FN rate; FNs are almost entirely non-oncology (Fasiglifam/DILI, Dexpramipexole/mito)
  - The combined safety model learns "oncology + promiscuous binder = dangerous"
    and doesn't learn the non-oncology signal at all
  - Training separate models lets each head optimize for its own feature space:
      oncology:     binding promiscuity + cell-cycle MoA + epigenetic features
      non-oncology: AMES + CYP1A2 + DILI + hepatic binding + fup

Approach:
  1. Split safety subset by disease_is_oncology flag
  2. Train separate GBM on each half (same hyperparams, same nested top-20 selection)
  3. Combine OOF predictions (oncology preds for oncology arms, non-onco for the rest)
  4. Report combined AUC + per-stratum AUC

Uses TDC features from v11 (must run compute_tdc_safety_scores.py first).
"""
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_calibrated import run_task_cv
from retrain_corrected import get_features
from retrain_arm_level import DATA, prepare_arm_dataset, build_task_subsets

OUT = ROOT / "results" / "arm_level_v12"   # created in main(), not on import: v18 imports
                                           # merge_tdc/get_feature_cols from here and must not
                                           # spawn an empty v12 results dir as a side effect

TDC_FILE   = ROOT / "data/models/tdc_safety_scores_v1.csv"
TDC_FEATS  = ["dili_prob","ames_prob","cyp1a2_prob","cyp3a4_prob","cyp2c9_prob","carcinogen_prob"]

ARM_BLOCKLIST = {
    "n_drugs","n_investigational","is_methodology_study","is_dosing_schedule_trial",
    "net_mech_apoptosis_n","net_mech_apoptosis_frac","net_mech_cell_cycle_n",
    "net_mech_cell_cycle_frac","net_mech_dna_damage_n","net_mech_dna_damage_frac",
    "net_mech_epigenetic_n","net_mech_epigenetic_frac","net_mech_immune_n",
    "net_mech_immune_frac","net_n_disease_in_ppi","net_frac_disease_in_ppi",
}
ARM_BLOCKLIST_PREFIXES = ("n_high_protein_in_",)


def merge_tdc(df):
    if not TDC_FILE.exists():
        print(f"ERROR: {TDC_FILE} missing — run compute_tdc_safety_scores.py first"); sys.exit(1)
    tdc = pd.read_csv(TDC_FILE, usecols=["NCT_ID","Arm_Label"] + TDC_FEATS)
    df = df.merge(tdc, on=["NCT_ID","Arm_Label"], how="left")
    for col in TDC_FEATS:
        df[col] = df[col].fillna(df[col].median())
    return df


def get_feature_cols(df):
    return [c for c in get_features(df)
            if c not in ARM_BLOCKLIST
            and not any(c.startswith(p) for p in ARM_BLOCKLIST_PREFIXES)
            and not (c.startswith("max_") and c.endswith("_interaction"))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = prepare_arm_dataset()
    df = merge_tdc(df)
    feature_cols = get_feature_cols(df)
    print(f"Feature columns: {len(feature_cols)}", flush=True)

    # -------------------------------------------------------------------------
    # Build safety subset
    # -------------------------------------------------------------------------
    safety_mask = df["Corrected_Outcome"].isin(["PASS","FAIL_SAFETY","FAIL_BOTH"])
    df_safety = df[safety_mask].copy()
    df_safety["_y"] = df_safety["Corrected_Outcome"].isin(["FAIL_SAFETY","FAIL_BOTH"]).astype(int)

    onco_mask  = df_safety["disease_is_oncology"].fillna(0).astype(bool)
    df_onco    = df_safety[onco_mask].copy()
    df_nonco   = df_safety[~onco_mask].copy()

    print(f"\nOncology safety:     n={len(df_onco)}  pos={df_onco['_y'].sum()} ({df_onco['_y'].mean():.1%})", flush=True)
    print(f"Non-oncology safety: n={len(df_nonco)} pos={df_nonco['_y'].sum()} ({df_nonco['_y'].mean():.1%})", flush=True)

    # -------------------------------------------------------------------------
    # Train stratified safety heads
    # -------------------------------------------------------------------------
    print("\n=== SAFETY — ONCOLOGY ===", flush=True)
    oof_onco, fm_onco = run_task_cv(df_onco, feature_cols, "safety_onco", calibrate=False)

    print("\n=== SAFETY — NON-ONCOLOGY ===", flush=True)
    oof_nonco, fm_nonco = run_task_cv(df_nonco, feature_cols, "safety_nonco", calibrate=False)

    # -------------------------------------------------------------------------
    # Combined safety AUC (merge OOF preds back)
    # -------------------------------------------------------------------------
    oof_onco["stratum"]  = "oncology"
    oof_nonco["stratum"] = "non_oncology"
    oof_combined = pd.concat([oof_onco, oof_nonco], ignore_index=True)
    oof_combined.to_parquet(OUT / "oof_safety_stratified.parquet", index=False)

    def fold_aucs(oof_df):
        aucs = []
        for (seed, fold), grp in oof_df.groupby(["seed","fold"]):
            if grp["y"].nunique() == 2:
                aucs.append(roc_auc_score(grp["y"], grp["raw_prob"]))
        return aucs

    aucs_onco  = fold_aucs(oof_onco)
    aucs_nonco = fold_aucs(oof_nonco)
    aucs_comb  = fold_aucs(oof_combined)

    print(f"\n=== RESULTS ===")
    print(f"Oncology-only  safety AUC: {np.mean(aucs_onco):.3f} ± {np.std(aucs_onco):.3f}")
    print(f"Non-onco-only  safety AUC: {np.mean(aucs_nonco):.3f} ± {np.std(aucs_nonco):.3f}")
    print(f"Combined       safety AUC: {np.mean(aucs_comb):.3f} ± {np.std(aucs_comb):.3f}")
    print(f"v8 baseline              : 0.613 ± 0.097")
    print(f"v11 (TDC)                : 0.621 ± 0.100")

    # -------------------------------------------------------------------------
    # Also run efficacy + overall with TDC features (same as v11, for completeness)
    # -------------------------------------------------------------------------
    print("\n=== EFFICACY ===", flush=True)
    eff_excl = pd.Series(False, index=df.index)
    for c in ("is_anti_pathogen","is_endogenous"):
        if c in df.columns: eff_excl |= df[c] == 1
    df_eff = df[~eff_excl & df["Corrected_Outcome"].isin(["PASS","FAIL_EFFICACY","FAIL_BOTH"])].copy()
    df_eff["_y"] = df_eff["Corrected_Outcome"].isin(["FAIL_EFFICACY","FAIL_BOTH"]).astype(int)
    oof_e, fm_e = run_task_cv(df_eff, feature_cols, "efficacy", calibrate=False)
    oof_e.to_parquet(OUT / "oof_efficacy.parquet", index=False)

    print("\n=== OVERALL ===", flush=True)
    df_over = df[df["Corrected_Outcome"].isin(["PASS","FAIL_SAFETY","FAIL_EFFICACY","FAIL_BOTH"])].copy()
    df_over["_y"] = df_over["Corrected_Outcome"].isin(["FAIL_SAFETY","FAIL_EFFICACY","FAIL_BOTH"]).astype(int)
    oof_o, fm_o = run_task_cv(df_over, feature_cols, "overall", calibrate=False)
    oof_o.to_parquet(OUT / "oof_overall.parquet", index=False)

    aucs_e = fold_aucs(oof_e)
    aucs_o = fold_aucs(oof_o)

    # -------------------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------------------
    pd.concat([fm_onco, fm_nonco, fm_e, fm_o], ignore_index=True).to_csv(OUT / "fold_metrics.csv", index=False)

    metrics = {
        "safety_auc_mean":         float(np.mean(aucs_comb)),
        "safety_auc_std":          float(np.std(aucs_comb)),
        "safety_oncology_auc":     float(np.mean(aucs_onco)),
        "safety_nononcology_auc":  float(np.mean(aucs_nonco)),
        "safety_n_pos":            int(df_safety["_y"].sum()),
        "efficacy_auc_mean":       float(np.mean(aucs_e)),
        "efficacy_auc_std":        float(np.std(aucs_e)),
        "overall_auc_mean":        float(np.mean(aucs_o)),
        "overall_auc_std":         float(np.std(aucs_o)),
        "n_features":              len(feature_cols),
        "approach":                "stratified_safety_oncology_vs_nononcology",
    }
    with open(DATA,"rb") as f:
        metrics["data_sha256"] = hashlib.sha256(f.read()).hexdigest()

    with open(OUT / "metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {OUT/'metrics.json'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
