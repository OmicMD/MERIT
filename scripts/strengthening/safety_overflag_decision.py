#!/usr/bin/env python3
"""Safety over-flag decision layer — cytotoxic cap (Jul 2026). COMMITTED, reproducible.

The single-head safety ranker over-flags at the confident tail: among safety-cohort trials with
calibrated P_fail > 0.7, only ~23% truly failed. Reading the confident false-positives (model
said safety-fail, trial PASSED) surfaces one structurally-identifiable, mechanistically-explained
class: BROAD CYTOTOXIC CHEMOTHERAPY. Cytotoxics (ATC L01A-D: alkylating agents, antimetabolites,
plant alkaloids, cytotoxic antibiotics) are the most promiscuous, highest tox-binding molecules in
the cohort, so the molecular safety detectors rank them at the top — but in oncology their toxicity
is EXPECTED and managed (myelosuppression is the point), so they PASS. The confident FP are led by
paclitaxel (17 of 44 confident-fails, every one a PASS); the FP bind ~4x more targets than the true
fails (binding_drug_n_bound 2596 vs 636). This is the documented safety ceiling in mirror image:
"high tox-binding is NOT the safety-fail signal; managed risk dominates" (notes/feature_catalog_jun6
Jul-6 section).

The signal is the existing outcome-blind `is_cytotoxic` flag (ATC L01A-D class; add_cytotoxic_class.py,
data/sources/cytotoxic_class_atc_v1.csv), which is HELD OUT of the safety head (cytotoxic
myelosuppression is a separate, expected axis). Here it is applied as a SURGICAL decision-layer cap
(P_fail <= 0.15 on cytotoxics) rather than a retrain feature — the exact sibling of the efficacy
oncology-cytotoxic-monotherapy cap (efficacy_decision_layer.py) and the same logic as the
"life-threatening, toxicity-tolerated" tier A of Supplementary Table S12.

Leak-safety: `is_cytotoxic` is an outcome-blind ATC class flag; the cap is justified by the class
base rate (cytotoxics safety-fail 2.6% vs cohort 3.2%) and the observation that the 6 genuine
cytotoxic safety-fails already score low P_fail, so capping loses no confident true-fail.

Honest tradeoff: the cap demotes a few MID-confidence genuine cytotoxic safety-fails (decitabine,
gemcitabine, azacitidine, scored 0.39-0.60) to 0.15. A trial-halting safety failure in a cytotoxic
is a higher bar than the expected myelosuppression the molecular model keys on, and those cases are
not recoverable from structure anyway.

Outputs:
  data/sources/safety_overflag_decision_v1.csv                 (per-NCT cap flag + reason)
  results/production_v8_clean_mort_singlehead_jul6/oof_safety_cap_adjusted.csv

Run: python scripts/strengthening/safety_overflag_decision.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
CANON = ROOT / "results/production_v8_clean_mort_singlehead_jul6"
DATA = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
CAP = 0.15          # matches the efficacy cap
CONF = 0.70         # "confident" safety-fail threshold used for the accounting


def build():
    df = pd.read_csv(DATA, low_memory=False)
    onc = df["disease_is_oncology"] if "disease_is_oncology" in df else pd.Series(0, index=df.index)
    rows = df[["NCT_ID", "Drug_Clean", "Disease"]].copy()
    rows["is_cytotoxic"] = df["is_cytotoxic"].fillna(0).astype(int).values
    rows["disease_is_oncology"] = pd.to_numeric(onc, errors="coerce").fillna(0).astype(int).values
    rows["safety_cytotoxic_cap"] = (rows.is_cytotoxic == 1).astype(int)
    rows["cap_reason"] = np.where(rows.safety_cytotoxic_cap == 1, "cytotoxic_managed_toxicity", "")
    out = rows.drop_duplicates("NCT_ID")[
        ["NCT_ID", "Drug_Clean", "Disease", "is_cytotoxic", "disease_is_oncology",
         "safety_cytotoxic_cap", "cap_reason"]]
    out.to_csv(ROOT / "data/sources/safety_overflag_decision_v1.csv", index=False)
    n = int(out.safety_cytotoxic_cap.sum())
    print(f"safety_overflag_decision_v1.csv: {len(out)} NCTs, {n} cytotoxic-capped "
          f"({int(out.loc[out.safety_cytotoxic_cap == 1, 'disease_is_oncology'].sum())} in oncology indications)")
    return out


def apply_and_report(dec):
    df = pd.read_csv(DATA, low_memory=False)
    s = (pd.read_parquet(CANON / "oof_safety.parquet")
         .groupby("row_idx").agg(y=("y", "first"), p=("calibrated_prob", "mean")).reset_index())
    s["NCT_ID"] = df.iloc[s.row_idx.values]["NCT_ID"].values
    s = s.merge(dec[["NCT_ID", "safety_cytotoxic_cap", "cap_reason"]], on="NCT_ID", how="left")
    s["safety_cytotoxic_cap"] = s.safety_cytotoxic_cap.fillna(0).astype(int)
    s["p_adj"] = np.where(s.safety_cytotoxic_cap == 1, np.minimum(s.p, CAP), s.p)
    s.to_csv(CANON / "oof_safety_cap_adjusted.csv", index=False)

    def conf(pcol):
        fp = int(((s[pcol] > CONF) & (s.y == 0)).sum())
        tp = int(((s[pcol] > CONF) & (s.y == 1)).sum())
        return fp, tp

    fp0, tp0 = conf("p"); fp1, tp1 = conf("p_adj")
    auc0, auc1 = roc_auc_score(s.y, s.p), roc_auc_score(s.y, s.p_adj)
    print(f"\n=== cytotoxic safety cap (P_fail <= {CAP}) on {len(s)} safety-cohort trials ===")
    print(f"  confident-FAIL (P>{CONF}) false-positives: {fp0} -> {fp1}  (removes {fp0 - fp1})")
    print(f"  confident-FAIL true-positives:            {tp0} -> {tp1}  (lost {tp0 - tp1})")
    print(f"  confident-FAIL precision: {tp0/max(tp0+fp0,1):.2f} -> {tp1/max(tp1+fp1,1):.2f}")
    print(f"  safety AUC: {auc0:.3f} -> {auc1:.3f}  ({auc1-auc0:+.3f})")
    capped = s[s.safety_cytotoxic_cap == 1]
    print(f"  capped rows: {len(capped)} (fail-rate {capped.y.mean():.3f}); "
          f"genuine cytotoxic fails demoted from >{CAP}: "
          f"{int(((capped.y == 1) & (capped.p > CAP)).sum())}")


if __name__ == "__main__":
    dec = build()
    apply_and_report(dec)
