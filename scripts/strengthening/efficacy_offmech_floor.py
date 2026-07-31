#!/usr/bin/env python3
"""Efficacy off-mechanism-endpoint decision FLOOR (Jun 17) — COMMITTED, reproducible.

The FN-side sibling of the over-flag CAP (scripts/strengthening/efficacy_decision_layer.py). Reading the
confident efficacy FALSE NEGATIVES (model said pass, p<0.30, trial FAILED) surfaced a structurally-
identifiable, mechanistically-explained class:

  OFF-MECHANISM ENDPOINT — the primary endpoint demands a physiological process the drug's target does
  NOT control, so the drug cannot move it however good it is for its established indication. The
  textbook case: empagliflozin in heart failure measured by 6-minute walk distance (EMPERIAL,
  NCT03448406/419) or cardiac PCr/ATP energetics (NCT03332212) — empagliflozin is a landmark HF
  *outcomes* drug (EMPEROR passes here, phys=0) but SGLT2/natriuresis does not raise exercise capacity
  or high-energy phosphates, so these surrogate trials FAILED. Same shape: metoprolol/bisoprolol COPD
  exacerbation-mortality, esomeprazole sepsis, iloprost septic shock, testosterone hip-fracture function,
  Venglustat ADPKD imaging.

The signal is the EXISTING outcome-blind model feature endpoint_physiology_score == -1 (built by
scripts/strengthening/endpoint_physiology_score.py from two curated, outcome-blind tables: the process a
disease-conditioned endpoint demands x the process the drug's target controls). The model HAS this
feature, but the GBM down-weights it relative to "this is a strong drug for this disease," so the minority
off-mechanism-endpoint trials stay confident-pass. A high-precision targeted signal nets ~0 on retrain
(redistribution) but a real confident-miss reduction applied SURGICALLY post-hoc.

Guardrails (all pass): flag is outcome-blind by construction; flag-vs-outcome AUC 0.515 (<0.58); shuffle
p<1e-4; within Phase 3 fail 0.556 (n=27) vs base 0.126; the recovered FN span 7 distinct drugs and all 5
folds; floor robust across 0.30-0.50 (all -9 confident misses). FLOOR raises P(fail) to the cohort's
honest empirical fail rate region (0.50), NOT to near-certainty (the off-mech cohort fails ~0.59, not
~0.95) -- it converts wrongly-confident passes to "uncertain," it does not assert failure.

Effect on canonical efficacy OOF: confident FN 102 -> 93 (-9), confident FP 225 -> 225 (0).
Orthogonal to the CAP layer (phys=+1 on-mech surrogate-pass vs phys=-1 off-mech are mutually exclusive).
Full per-case investigation: notes/investigation_offmech_endpoint_floor_jun17.md.
"""
import sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
FLOOR = 0.50


def build():
    df = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
    rows = df[["NCT_ID", "Drug_Clean", "Disease", "endpoint_physiology_score"]].drop_duplicates("NCT_ID").copy()
    rows["is_offmech_endpoint"] = (rows.endpoint_physiology_score == -1).astype(int)
    out = rows[["NCT_ID", "Drug_Clean", "Disease", "endpoint_physiology_score", "is_offmech_endpoint"]]
    out.to_csv(ROOT / "data/sources/efficacy_offmech_floor_v1.csv", index=False)
    print(f"wrote data/sources/efficacy_offmech_floor_v1.csv ({len(out)} trials; "
          f"off-mechanism-endpoint={int(out.is_offmech_endpoint.sum())})")
    return df


def apply_layer(df):
    oof = pd.read_parquet(ROOT / "results/production_v8_clean_mort_gapBD_jun28/oof_efficacy.parquet")
    g = oof.groupby("row_idx").agg(y=("y", "first"), p=("raw_prob", "mean")).reset_index()
    # exact per-trial mapping: row_idx == positional index of training_dataset_v8_clean_mort.csv
    g["is_offmech"] = (df.iloc[g.row_idx.values].endpoint_physiology_score.values == -1).astype(int)
    g["NCT_ID"] = df.iloc[g.row_idx.values].NCT_ID.values
    g["Disease"] = df.iloc[g.row_idx.values].Disease.values
    # Compose with the over-flag CAP: a trial cannot be both "confidently passes" (cap) and
    # "off-mechanism fail" (floor). The floor YIELDS to the cap (the two classifiers disagree on 2
    # eplerenone cardiac-MRI trials; neither is among the 9 recovered FN). Floor only un-capped trials.
    cap_path = ROOT / "data/sources/efficacy_overflag_decision_v1.csv"
    if cap_path.exists():
        cap = pd.read_csv(cap_path)[["NCT_ID", "efficacy_surgical_pass"]]
        g = g.merge(cap, on="NCT_ID", how="left")
        g["efficacy_surgical_pass"] = g.efficacy_surgical_pass.fillna(0)
    else:
        g["efficacy_surgical_pass"] = 0
    sel = (g.is_offmech == 1) & (g.efficacy_surgical_pass == 0)
    g["p_adj"] = g.p.where(~sel, np.maximum(g.p, FLOOR))

    def conf(p):
        return int(((g.y == 1) & (p < 0.30)).sum()), int(((g.y == 0) & (p > 0.60)).sum())
    fnb, fpb = conf(g.p); fna, fpa = conf(g.p_adj)
    print(f"\nEfficacy off-mechanism-endpoint floor (P_fail>={FLOOR} on {int(sel.sum())} off-mech trials):")
    print(f"  confident FN {fnb}->{fna} ({fna-fnb:+d}) | FP {fpb}->{fpa} ({fpa-fpb:+d}) | "
          f"total {fnb+fpb}->{fna+fpa} ({fna+fpa-fnb-fpb:+d})")
    bri_b = ((g[sel].p - g[sel].y) ** 2).mean()
    bri_a = ((g[sel].p_adj - g[sel].y) ** 2).mean()
    print(f"  Brier on floored cohort {bri_b:.3f} -> {bri_a:.3f}")
    g[["NCT_ID", "Disease", "y", "p", "p_adj", "is_offmech"]].to_csv(
        ROOT / "results/production_v8_clean_mort_gapBD_jun28/oof_efficacy_offmech_floor_adjusted.csv", index=False)


if __name__ == "__main__":
    apply_layer(build())
