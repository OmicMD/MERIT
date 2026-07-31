#!/usr/bin/env python3
"""Unified decision-layer composition (Jun 18; safety cap added Jul 6) — COMMITTED, reproducible.

Composes the post-hoc decision layers built in isolation over the canonical OOF into ONE coherent
triage layer, with a single combined confident-miss number and a per-layer breakdown. This is
consolidation + interaction-resolution, NOT new feature mining. Every layer stays outcome-blind (no layer
reads Corrected_Outcome / y / raw_prob to DEFINE its class).

Canonical: results/production_v8_clean_mort_singlehead_jul6 (honest, LLM-free, single-GBM safety head,
isotonic-calibrated).

The layers
----------
CORRECTORS (modify P_fail; justified only because the structural cohort fails at an EXTREME rate vs the
17.5% base — the jun17 cap-vs-tier rule):
  1. CAP  endpoint surrogate-pass        (efficacy, FP-side, cohort fails ~4%):  data/sources/endpoint_mechanism_v1.csv
  2. CAP  oncology cytotoxic monotherapy (efficacy, FP-side, cohort fails ~5%):  data/sources/efficacy_overflag_decision_v1.csv
     (caps 1+2 are unioned into `efficacy_surgical_pass` by efficacy_decision_layer.py)   -> P_fail <= 0.15
  3. FLOOR off-mechanism endpoint        (efficacy, FN-side, cohort fails ~61%): endpoint_physiology_score == -1  -> P_fail >= 0.50
  4. CAP  cytotoxic managed-toxicity     (safety,   FP-side, class fails ~2.6%): data/sources/safety_overflag_decision_v1.csv
     (built by safety_overflag_decision.py; is_cytotoxic ATC L01A-D, held OUT of the safety head)  -> P_fail <= 0.15

TIERS (annotate confidence / shift the operating point; do NOT move P_fail — kept in a separate
`confidence_tier` column):
  5. TIER risk-tolerance         (safety,   Supplementary Table S12): data/sources/disease_risk_tolerance_v1.csv
  6. TIER effect-size-uncertainty(efficacy, NOT in the manuscript):   data/sources/effectsize_uncertainty_v1.csv
     (built and validated here, but not adopted into the paper: on the honest LLM-free model the
      readout-noise gradient does not survive, so only the safety risk-tolerance tier reached the
      Supplementary. Do not cite a table for it.)

Composition rules (jun17, verified here — not re-derived)
--------------------------------------------------------
* Per-trial mapping is the EXACT row_idx -> training_dataset_v8_clean_mort.csv iloc map (100% SMILES+Disease
  verified), NOT the lossy SMILES x Disease aggregation the isolated cap scripts used (which conflates e.g.
  EMPEROR-PASS with EMPERIAL-FAIL under one (SMILES,Disease) key).
* Correctors applied in a defined order: caps first (min, FP-side), then the floor (max, FN-side) on
  not-already-capped trials. Caps and the floor are mutually exclusive by construction EXCEPT 2 eplerenone
  cardiac-MRI trials where the surrogate classifier (on-mechanism) and the physiology scorer (phys=-1)
  disagree.
* eplerenone precedence (RESOLVED here, floor yields to cap): cardiac-remodeling MRI endpoints (LV strain,
  LV volume) ARE an on-mechanism pharmacodynamic readout for an aldosterone antagonist (the RALES/EPHESUS
  anti-fibrotic / anti-remodeling effect), so the surrogate classifier is biologically correct and the
  physiology scorer's phys=-1 is the mis-score. The cap wins. Consequence: the PLN-R14del cardiomyopathy
  trial (NCT01857856, FAIL) — already correctly lean-fail at raw 0.67 — is capped to 0.15, manufacturing
  one confident FN. That is one of the cap cohort's expected ~4% effect-size failures (an on-mechanism
  surrogate can still fall short on effect size in a specific severe genetic cardiomyopathy), NOT a
  composition error. Flagged for Gabe as the framing call; net effect on the combined count is 0.
* Tiers stay a separate column; they never move P_fail (AUC unchanged, only the confidence label / operating
  point moves).

Outputs
-------
  results/production_v8_clean_mort_singlehead_jul6/oof_efficacy_triaged.csv
  results/production_v8_clean_mort_singlehead_jul6/oof_safety_triaged.csv
  + one printed summary table (raw -> per-layer cumulative -> combined) and an interaction audit.

Run: python scripts/strengthening/decision_layer_compose.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "results/production_v8_clean_mort_singlehead_jul6"
COH = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"

CAP = 0.15      # surrogate-pass / cytotoxic-mono cap (P_fail upper bound)
FLOOR = 0.50    # off-mechanism-endpoint floor (P_fail lower bound)
FN_THR = 0.30   # confident false negative: y==1 & p < 0.30
FP_THR = 0.60   # confident false positive: y==0 & p > 0.60


def conf_counts(y: np.ndarray, p: np.ndarray) -> tuple[int, int]:
    """(confident FN, confident FP) under the project operating thresholds."""
    fn = int(((y == 1) & (p < FN_THR)).sum())
    fp = int(((y == 0) & (p > FP_THR)).sum())
    return fn, fp


def load_oof_exact(task: str, cols: list[str]) -> pd.DataFrame:
    """Per-trial OOF via the EXACT row_idx -> training-row iloc map (verified 100%)."""
    df = pd.read_csv(COH, low_memory=False)
    df["IK14"] = df.feature_IK.astype(str).str[:14]
    oof = pd.read_parquet(RES / f"oof_{task}.parquet")
    g = oof.groupby("row_idx").agg(y=("y", "first"), p=("raw_prob", "mean"),
                                   sm=("SMILES", "first"), dis=("Disease", "first")).reset_index()
    sub = df.iloc[g.row_idx.values]
    assert (sub.SMILES.values == g.sm.values).all() and (sub.Disease.values == g.dis.values).all(), \
        "row_idx -> iloc mapping broke (SMILES/Disease mismatch)"
    for c in cols:
        g[c] = sub[c].values
    return g.drop(columns=["sm", "dis"])


# ---------------------------------------------------------------- efficacy
def compose_efficacy() -> pd.DataFrame:
    g = load_oof_exact("efficacy", ["NCT_ID", "Disease", "Drug_Clean", "Phase",
                                    "endpoint_physiology_score", "disease_is_oncology", "is_combination"])
    base = g.y.mean()

    # corrector flags (outcome-blind, from committed source CSVs / existing model feature)
    cap = pd.read_csv(ROOT / "data/sources/efficacy_overflag_decision_v1.csv")[
        ["NCT_ID", "is_surrogate_pass", "is_onc_cytotoxic_mono", "efficacy_surgical_pass", "surgical_reason"]]
    g = g.merge(cap, on="NCT_ID", how="left")
    for c in ["is_surrogate_pass", "is_onc_cytotoxic_mono", "efficacy_surgical_pass"]:
        g[c] = g[c].fillna(0).astype(int)
    g["surgical_reason"] = g.surgical_reason.fillna("")
    g["is_offmech"] = (g.endpoint_physiology_score == -1).astype(int)

    # effect-size-uncertainty reliability TIER (does NOT move P_fail)
    esu = pd.read_csv(ROOT / "data/sources/effectsize_uncertainty_v1.csv")[["disease", "effect_size_uncertainty"]]
    g = g.merge(esu, left_on="Disease", right_on="disease", how="left").drop(columns="disease")
    thr = g.effect_size_uncertainty.quantile(2 / 3)
    g["confidence_tier"] = np.where(g.effect_size_uncertainty.isna(), "unscored",
                            np.where(g.effect_size_uncertainty > thr, "low_confidence", "reliable"))

    # ---- apply correctors in defined order: caps (min) then floor (max, only where not capped) ----
    p = g.p.values.copy()
    p_cap = np.where(g.efficacy_surgical_pass == 1, np.minimum(p, CAP), p)
    floor_sel = (g.is_offmech == 1) & (g.efficacy_surgical_pass == 0)   # floor yields to cap
    p_combined = np.where(floor_sel, np.maximum(p_cap, FLOOR), p_cap)
    g["p_cap"] = p_cap
    g["p_adj"] = p_combined
    g["floor_applied"] = floor_sel.astype(int)
    g["cap_applied"] = (g.efficacy_surgical_pass == 1).astype(int)

    # ---- per-layer cumulative confident-miss accounting ----
    y = g.y.values
    layers = [
        ("raw model", g.p.values),
        ("+ CAP surrogate-pass", np.where(g.is_surrogate_pass == 1, np.minimum(g.p.values, CAP), g.p.values)),
        ("+ CAP onc-cytotoxic-mono (both caps)", p_cap),
        ("+ FLOOR off-mechanism (COMBINED)", p_combined),
    ]
    rows = []
    fn0, fp0 = conf_counts(y, g.p.values)
    for name, pv in layers:
        fn, fp = conf_counts(y, pv)
        rows.append(dict(layer=name, conf_FN=fn, conf_FP=fp, total=fn + fp,
                         dFN=fn - fn0, dFP=fp - fp0, dTotal=(fn + fp) - (fn0 + fp0)))
    eff_tbl = pd.DataFrame(rows)

    # ---- interaction audit ----
    both = g[(g.cap_applied == 1) & (g.is_offmech == 1)]                       # touched by cap AND floor-class
    # correctors that flip a previously-CORRECT (non-confident-miss) prediction into a confident miss
    was_miss = ((g.p < FN_THR) & (y == 1)) | ((g.p > FP_THR) & (y == 0))
    now_miss = ((g.p_adj < FN_THR) & (y == 1)) | ((g.p_adj > FP_THR) & (y == 0))
    flipped_bad = g[(~was_miss) & now_miss]
    fixed = g[was_miss & (~now_miss)]

    g.to_csv(RES / "oof_efficacy_triaged.csv", index=False)
    return eff_tbl, g, base, both, flipped_bad, fixed, thr


# ---------------------------------------------------------------- safety
def compose_safety() -> pd.DataFrame:
    g = load_oof_exact("safety", ["NCT_ID", "Disease", "Drug_Clean", "Phase"])
    # risk-tolerance TIER (does NOT move P_fail; a-priori safety operating point per indication)
    rt = pd.read_csv(ROOT / "data/sources/disease_risk_tolerance_v1.csv")[["disease", "risk_tolerance"]]
    g = g.merge(rt, left_on="Disease", right_on="disease", how="left").drop(columns="disease")
    g["risk_tolerance"] = g.risk_tolerance.fillna(50.0)   # unmatched -> Tier B (serious-chronic) default

    def tier(v):
        return "A_tox_tolerated" if v >= 70 else ("B_serious_chronic" if v >= 40 else "C_benign")
    g["confidence_tier"] = g.risk_tolerance.apply(tier)
    # Operating thresholds on the RAW safety ranker (Jun 19 re-tune, Gabe). The original
    # (0.65/0.45/0.25) redistributed by clinical cost but doubled false positives (22% vs 11% flat):
    # the aggressive benign/serious-chronic low thresholds flooded FP because the ranker is weak
    # (AUC 0.74) — C_benign 0.25 fired 247 FP to catch 5 of 7 real fails. Re-tuned to compress the low
    # end while preserving the clinical ordering (more-tolerant indication -> stricter-to-flag):
    # A 0.65 / B 0.45->0.60 / C 0.25->0.50. Cuts total FP 553->297 (-46%) at recall 48%->45% (the 3
    # lost catches are all benign-indication; oncology + serious-chronic recall unchanged).
    # notes/safety_fp_operating_point_jun19.md. AUC unchanged (operating-point only).
    g["tier_threshold"] = g.confidence_tier.map(
        {"A_tox_tolerated": 0.65, "B_serious_chronic": 0.60, "C_benign": 0.50})

    # ---- CORRECTOR: cytotoxic managed-toxicity cap (safety FP-side; sibling of the efficacy cytotoxic-mono
    # cap). is_cytotoxic (ATC L01A-D) is held OUT of the safety head, so the molecular ranker over-flags
    # promiscuous cytotoxics whose oncology toxicity is expected/managed -> PASS. Cap P_fail <= 0.15 on the
    # outcome-blind class flag. Applied to the same raw-prob `p` the composer uses throughout (matches the
    # efficacy caps). See safety_overflag_decision.py / data/sources/safety_overflag_decision_v1.csv.
    cap = pd.read_csv(ROOT / "data/sources/safety_overflag_decision_v1.csv")[
        ["NCT_ID", "safety_cytotoxic_cap", "cap_reason"]]
    g = g.merge(cap, on="NCT_ID", how="left")
    g["safety_cytotoxic_cap"] = g.safety_cytotoxic_cap.fillna(0).astype(int)
    g["cap_reason"] = g.cap_reason.fillna("")
    g["p_adj"] = np.where(g.safety_cytotoxic_cap == 1, np.minimum(g.p, CAP), g.p)
    g["cap_applied"] = g.safety_cytotoxic_cap
    g.to_csv(RES / "oof_safety_triaged.csv", index=False)
    return g


def main():
    eff_tbl, eff, base, both, flipped_bad, fixed, esu_thr = compose_efficacy()
    saf = compose_safety()

    print("=" * 90)
    print("UNIFIED DECISION-LAYER TRIAGE — canonical production_v8_clean_mort_singlehead_jul6")
    print(f"efficacy cohort: {len(eff)} trials, base fail rate {base*100:.1f}%   "
          f"(confident: FN p<{FN_THR}&fail, FP p>{FP_THR}&pass)")
    print("=" * 90)
    print("\n### Efficacy correctors — cumulative confident-miss accounting (exact per-trial mapping)\n")
    print(eff_tbl.to_string(index=False))

    fn0, fp0 = conf_counts(eff.y.values, eff.p.values)
    fnc, fpc = conf_counts(eff.y.values, eff.p_adj.values)
    print(f"\nCOMBINED efficacy: confident misses {fn0+fp0} -> {fnc+fpc} "
          f"(FN {fn0}->{fnc} {fnc-fn0:+d}, FP {fp0}->{fpc} {fpc-fp0:+d}; net {(fnc+fpc)-(fn0+fp0):+d})")

    print("\n### Interaction audit")
    print(f"  trials touched by BOTH a cap and the off-mechanism floor-class (cap wins): {len(both)}")
    for _, r in both.iterrows():
        print(f"    {r.NCT_ID}  {r.Drug_Clean} / {r.Disease[:42]:42}  "
              f"y={int(r.y)} raw={r.p:.3f} -> adj={r.p_adj:.3f}  reason={r.surgical_reason}")
    print(f"  correctors FIXED (was confident-miss -> now correct): {len(fixed)}  "
          f"[FP {int((fixed.y==0).sum())}, FN {int((fixed.y==1).sum())}]")
    print(f"  correctors FLIPPED correct -> confident-miss: {len(flipped_bad)}  "
          f"[FP {int((flipped_bad.y==0).sum())}, FN {int((flipped_bad.y==1).sum())}]")
    for _, r in flipped_bad.iterrows():
        print(f"    {r.NCT_ID}  {r.Drug_Clean} / {r.Disease[:42]:42}  "
              f"y={int(r.y)} raw={r.p:.3f} -> adj={r.p_adj:.3f}")

    print("\n### Safety corrector — cytotoxic managed-toxicity cap (exact per-trial mapping)")
    sfn0, sfp0 = conf_counts(saf.y.values, saf.p.values)
    sfn1, sfp1 = conf_counts(saf.y.values, saf.p_adj.values)
    ncap = int(saf.cap_applied.sum())
    demoted = int(((saf.cap_applied == 1) & (saf.y == 1) & (saf.p > CAP)).sum())
    print(f"  {ncap} cytotoxic trials capped to P_fail <= {CAP} (class fail-rate "
          f"{saf.loc[saf.cap_applied == 1, 'y'].mean():.3f})")
    print(f"  confident FP (p>{FP_THR}&pass): {sfp0} -> {sfp1} ({sfp1-sfp0:+d}); "
          f"confident FN (p<{FN_THR}&fail): {sfn0} -> {sfn1} ({sfn1-sfn0:+d})")
    print(f"  genuine cytotoxic fails demoted from >{CAP}: {demoted} (expected/managed myelosuppression, "
          f"not structurally recoverable)")

    print("\n### Tiers (separate confidence column; P_fail unchanged)")
    print(f"  efficacy effect-size-uncertainty tier (split at q2/3={esu_thr:.0f}):")
    print("   ", eff.confidence_tier.value_counts().to_dict())
    print(f"  safety risk-tolerance tier:")
    print("   ", saf.confidence_tier.value_counts().to_dict())

    print(f"\nwrote {RES.relative_to(ROOT)}/oof_efficacy_triaged.csv ({len(eff)} trials)")
    print(f"wrote {RES.relative_to(ROOT)}/oof_safety_triaged.csv  ({len(saf)} trials)")


if __name__ == "__main__":
    main()
