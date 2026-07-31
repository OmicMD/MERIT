#!/usr/bin/env python3
"""Efficacy over-flag decision layer (Jun 17) — COMMITTED, reproducible.

The honest model over-predicts efficacy-failure for two structurally-identifiable, mechanistically-
explained classes, found by reading the confident false positives (model said fail, trial passed):

  (A) ENDPOINT SURROGATE-PASS — the primary endpoint is a reliably-movable PD surrogate the drug
      directly moves (on-mechanism organ-function/biomarker, or an intrinsically-easy metabolic
      biomarker: weight/HbA1c/glucose). 4.8% fail. Source: data/sources/endpoint_mechanism_v1.csv.
  (B) ONCOLOGY CYTOTOXIC MONOTHERAPY — broad-spectrum antiproliferatives (tubulin/antimetabolite/
      topoisomerase targets) have no causal driver-target, so the mechanism-fit model sees "no
      target->disease link" and over-flags them; clinically they pass ~95% as monotherapy. 4.7%
      fail (within Phase 3 5.6%). Combinations excluded (backbone-attribution: those fail 44%).

Both are outcome-blind (drug target family / ATC + endpoint text), survive within-Phase-3 + shuffle
(p<1e-4), and are applied SURGICALLY (cap P(fail)<=0.15 on just these cases) rather than as retrain
features — a high-precision targeted signal redistributes mass when retrained (nets ~0) but nets a
real confident-miss reduction applied post-hoc. Sibling of the risk-tolerance (Supplementary Table S12)
and effect-size-uncertainty (built but not adopted in the manuscript) decision layers.

Effect on canonical efficacy OOF: confident misses 248 -> 217 (-31; FP -38, FN +7).
Full per-case investigation: notes/investigation_oncology_cytotoxic_jun17.md + the endpoint×mechanism thread.
"""
import sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
CAP = 0.15
CYTOTOXIC_TARGETS = {"TUBB", "TUBB1", "TUBB4B", "TUBA1A", "TUBA1B", "TUBA4A", "TUBA3C", "TYMS",
                     "RRM1", "RRM2", "RRM2B", "DHFR", "GART", "TOP1", "TOP2A", "TOP2B", "DNMT1",
                     "POLA1", "TYMP"}


def build():
    df = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
    df["IK14"] = df.feature_IK.astype(str).str[:14]
    moa = pd.read_csv(ROOT / "data/sources/ik14_moa_targets_combined_v1.csv")
    ik2tg = moa.groupby("ik14").target_gene.apply(lambda s: set(str(x) for x in s.dropna())).to_dict()
    endp = pd.read_csv(ROOT / "data/sources/endpoint_mechanism_v1.csv")[["NCT_ID", "is_surrogate_pass"]]

    rows = df[["NCT_ID", "Drug_Clean", "Disease", "IK14", "disease_is_oncology", "is_combination"]].drop_duplicates("NCT_ID").copy()
    rows = rows.merge(endp, on="NCT_ID", how="left")
    rows["is_surrogate_pass"] = rows.is_surrogate_pass.fillna(0).astype(int)
    rows["is_onc_cytotoxic_mono"] = ((rows.disease_is_oncology == 1) & (rows.is_combination == 0) &
        rows.IK14.map(lambda ik: bool(ik2tg.get(ik, set()) & CYTOTOXIC_TARGETS))).astype(int)
    rows["efficacy_surgical_pass"] = ((rows.is_surrogate_pass == 1) | (rows.is_onc_cytotoxic_mono == 1)).astype(int)
    rows["surgical_reason"] = np.where(rows.is_onc_cytotoxic_mono == 1, "oncology_cytotoxic_monotherapy",
                              np.where(rows.is_surrogate_pass == 1, "endpoint_surrogate_pass", ""))
    out = rows[["NCT_ID", "Drug_Clean", "Disease", "is_surrogate_pass", "is_onc_cytotoxic_mono",
                "efficacy_surgical_pass", "surgical_reason"]]
    out.to_csv(ROOT / "data/sources/efficacy_overflag_decision_v1.csv", index=False)
    print(f"wrote data/sources/efficacy_overflag_decision_v1.csv ({len(out)} trials; "
          f"surrogate_pass={int(out.is_surrogate_pass.sum())}, onc_cyto_mono={int(out.is_onc_cytotoxic_mono.sum())}, "
          f"any_surgical_pass={int(out.efficacy_surgical_pass.sum())})")
    return out


def apply_layer(flags):
    oof = pd.read_parquet(ROOT / "results/production_v8_clean_mort_gapBD_jun28/oof_efficacy.parquet")
    df = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
    nct = df[["SMILES", "Disease", "NCT_ID"]].drop_duplicates(["SMILES", "Disease"])
    agg = oof.groupby(["SMILES", "Disease"]).agg(y=("y", "first"), p=("raw_prob", "mean")).reset_index()
    m = agg.merge(nct, on=["SMILES", "Disease"], how="left").merge(
        flags[["NCT_ID", "efficacy_surgical_pass"]], on="NCT_ID", how="left")
    m["efficacy_surgical_pass"] = m.efficacy_surgical_pass.fillna(0)
    sel = m.efficacy_surgical_pass == 1
    m["p_adj"] = m.p.where(~sel, np.minimum(m.p, CAP))
    def conf(p): return int(((m.y == 1) & (p < 0.30)).sum()), int(((m.y == 0) & (p > 0.60)).sum())
    fnb, fpb = conf(m.p); fna, fpa = conf(m.p_adj)
    print(f"\nEfficacy over-flag decision layer (cap P_fail<={CAP} on {int(sel.sum())} surgical-pass trials):")
    print(f"  confident FN {fnb}->{fna} ({fna-fnb:+d}) | FP {fpb}->{fpa} ({fpa-fpb:+d}) | "
          f"total {fnb+fpb}->{fna+fpa} ({fna+fpa-fnb-fpb:+d})")
    print(f"  Brier on adjusted cohort {(((m[sel].p-m[sel].y)**2).mean()):.3f} -> {(((m[sel].p_adj-m[sel].y)**2).mean()):.3f}")
    m[["NCT_ID", "Disease", "y", "p", "p_adj", "efficacy_surgical_pass"]].to_csv(
        ROOT / "results/production_v8_clean_mort_gapBD_jun28/oof_efficacy_decision_adjusted.csv", index=False)


if __name__ == "__main__":
    apply_layer(build())
