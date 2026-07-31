#!/usr/bin/env python3
"""
Path C, Stage 7: confidence gate (selective prediction).

Not all forward predictions are equal. We LOCK a prediction only where the model is
demonstrably reliable on held-out data, and ABSTAIN on the uncertain middle. Two
thresholds are DERIVED from the canonical out-of-fold OVERALL predictions (the same
head the lock applies):
  - the WIDEST low-P_fail band whose held-out PASS precision >= TARGET, and
  - the NARROWEST high-P_fail band whose held-out FAIL precision >= TARGET.
A forward pair is a BET only if it falls inside one of those bands; otherwise it is
recorded as ABSTAIN (not scored at readout).

The thresholds are set on held-out COHORT indications; the forward pairs are NOVEL
(out-of-distribution), so whether the reliability transfers is itself part of what the
prospective test measures — disclosed, not assumed.

Inputs : results/benchmark/prereg_C/prereg_C_locked_predictions_final.csv
         results/production_v8_clean_mort_coverage_jun22/oof_overall.parquet
Outputs: results/benchmark/prereg_C/prereg_C_confident_bets.csv (+ .sha256)  <- the registration
         results/benchmark/prereg_C/prereg_C_abstained.csv
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "results/benchmark/prereg_C"
FINAL = P / "prereg_C_locked_predictions_final.csv"
# Gate on the CANONICAL model's OWN out-of-fold predictions — the same model the lock fits-on-all to
# score the forward pairs — so the band thresholds are SELF-CONSISTENT with the scores they gate (Jun 28).
# (Previously borrowed coverage_jun22's OOF, whose calibration was looser: its 0.54 FAIL cutoff reads
# ~90% precision on coverage_jun22 but only ~82-85% under the canonical scoring model, overstating the
# against-base-rate FAIL claims. The two OOF calibrations are otherwise near-identical, so the PASS band
# is unaffected.)
OOF = ROOT / "results/production_v8_clean_mort_gapBD_jun28/oof_overall.parquet"
# Asymmetric, HONEST targets. The PASS band (≈ base rate) cleanly reaches 90% held-out precision. The
# FAIL band is against-base-rate and the calibrated model genuinely cannot reach 90% at a usefully-low
# threshold under self-consistent gating (90% needs P_fail≥0.62, n=20); 85% is the honest achievable
# ceiling for the FAIL calls (P_fail≥~0.51, n≈60). We DERIVE at these targets and REPORT the precision
# actually achieved, rather than claim a precision the scoring model does not deliver.
PASS_TARGET = 0.90
FAIL_TARGET = 0.85
MIN_N = 20      # min held-out support for a band edge to count (guards small-sample 100%s)


def derive_thresholds(g, pass_target=PASS_TARGET, fail_target=FAIL_TARGET, min_n=MIN_N):
    """Widest t_low with PASS-precision(p<=t_low) >= pass_target; narrowest t_high with
    FAIL-precision(p>=t_high) >= fail_target. Both on held-out OOF, with a support floor."""
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    t_low = 0.0
    for t in grid:
        sub = g[g.p <= t]
        if len(sub) >= min_n and (1 - sub.y.mean()) >= pass_target:
            t_low = float(t)
    t_high = 1.0
    for t in grid[::-1]:
        sub = g[g.p >= t]
        if len(sub) >= min_n and sub.y.mean() >= fail_target:
            t_high = float(t)
    return t_low, t_high


def main():
    oof = pd.read_parquet(OOF)
    g = (oof.groupby(["SMILES", "Disease"])
            .agg(y=("y", "first"), p=("calibrated_prob", "mean")).reset_index())
    t_low, t_high = derive_thresholds(g)
    pass_n = int((g.p <= t_low).sum()); pass_prec = 1 - g[g.p <= t_low].y.mean()
    fail_n = int((g.p >= t_high).sum()); fail_prec = g[g.p >= t_high].y.mean()
    print(f"Held-out OOF ({len(g)} rows; canonical gapBD_jun28) — self-consistent bands (min n={MIN_N}):")
    print(f"  confident PASS: P_fail <= {t_low:.2f}  precision {pass_prec:.3f}  "
          f"(target {PASS_TARGET:.0%}; n={pass_n}, {pass_n/len(g)*100:.0f}% coverage)")
    print(f"  confident FAIL: P_fail >= {t_high:.2f}  precision {fail_prec:.3f}  "
          f"(target {FAIL_TARGET:.0%}; n={fail_n}, {fail_n/len(g)*100:.0f}% coverage)")

    f = pd.read_csv(FINAL)
    p = f.P_fail_overall_calibrated
    f["confidence_regime"] = np.where(p <= t_low, "confident_PASS",
                             np.where(p >= t_high, "confident_FAIL", "ABSTAIN"))

    # ---- biology override (Jun 28): force-ABSTAIN a confident bet the model mis-frames on grounds
    # a threshold gate cannot see. Committed, web-evidenced, keyed NCT_ID+drug; the pair stays in the
    # registered novel set but leaves the confident bets. Currently: metoprolol->DMD, a confident FAIL
    # on disease-level mechanism mismatch (ADRB1 is not a dystrophin-pathway driver) whose trial endpoint
    # is LVEF change — beta-blockers are cardioprotective in DMD cardiomyopathy (preserve LVEF; the
    # enalapril+metoprolol RCT program, cohort evidence), so the model's disease-level pessimism
    # overrides the correct endpoint match (endpoint_physiology=+1) and the FAIL is a likely false positive.
    OVERRIDES = ROOT / "data/sources/prereg_C_confident_overrides_v1.csv"
    if OVERRIDES.exists():
        ov = pd.read_csv(OVERRIDES)
        ov_keys = set(zip(ov.NCT_ID.astype(str), ov.drug.str.lower().str.strip()))
        om = [(str(n), str(d).lower().strip()) in ov_keys for n, d in zip(f.NCT_ID, f.drug)]
        n_ov = int(sum(m and r != "ABSTAIN" for m, r in zip(om, f.confidence_regime)))
        f.loc[[m and r.startswith("confident") for m, r in zip(om, f.confidence_regime)],
              "confidence_regime"] = "ABSTAIN_BIOLOGY"
        if n_ov:
            print(f"  biology override: ABSTAIN {n_ov} mis-framed confident bet(s) "
                  f"({', '.join(ov.drug + '->' + ov.indication)})")

    # ---- OOD / coverage handling (jun22) ---------------------------------------------------
    # The confident-FAIL band's reliability is estimated on IN-DISTRIBUTION held-out OOF, but the forward
    # pairs are NOVEL. We treat the two OOD signals differently (deliberate, not symmetric):
    #
    #  (1) mechanism coverage ABSENT  -> ABSTAIN.  When the disease has no resolved gene module OR the drug
    #      has no mapped target, the mechanism-fit features that would justify "won't work" are STRUCTURALLY
    #      zero — not evidence (the missing=bad conflation). A confident FAIL with no mechanistic basis is
    #      the model guessing, so it is abstained outright.
    #
    #  (2) raw score in the SPARSE extrapolation region -> FLAG, do NOT abstain.  Essentially ALL forward
    #      FAILs land at raw P_fail in the OOF upper tail (forward raw>0.9 is ~14.7% vs ~3.1% OOF) — the raw
    #      GBM is over-confident on novel pairs, but the isotonic calibration already COMPRESSES that
    #      (raw ~0.97 -> calibrated ~0.6). Whether that OOF->forward mapping transfers is exactly what the
    #      prospective test measures, so we DISCLOSE the exposure with a `raw_extrapolation` flag rather than
    #      abstain it away (a hard raw cut is untunable: 99th pct catches ~none, 95th pct abstains them all).
    # Applied to confident_FAIL only (the against-base-rate claims); confident_PASS ~ base rate, left as is.
    raw95 = float(np.quantile(oof.raw_prob, 0.95))
    # Map IK14 by DRUG NAME (not by NCT_ID): a multi-drug trial shares one NCT, so an
    # NCT_ID join assigns the wrong drug's IK14 on combination trials (e.g. NCT07136987 =
    # acetylcysteine + metformin) and corrupts the per-pair mechanism merge. Mirrors novelty_filter.
    drug2ik = (pd.read_csv(ROOT / "results/benchmark/prereg_C_trials.csv")
               .assign(IK14=lambda x: x.IK14.astype(str).str[:14])
               .drop_duplicates("Drug_Clean").set_index("Drug_Clean").IK14.to_dict())
    f["IK14"] = f["drug"].map(drug2ik)
    mech = pd.read_csv(ROOT / "data/sources/mechanism_dataderived_prereg_C.csv")
    covcols = [c for c in ["coverage_disease", "coverage_drug", "direct_target_max"]
               if c in mech.columns]
    f = f.merge(
        mech[["IK14", "Disease"] + covcols].drop_duplicates(["IK14", "Disease"]),
        left_on=["IK14", "novel_indication"], right_on=["IK14", "Disease"], how="left")
    # LOUD per-prediction feature-completeness flag (visible in every output sheet). All biological
    # sub-blocks are RECOMPUTED per pair: OT/topology/KEGG mechanism + direct-target engagement
    # (prereg_C_build_mech.py) AND Mendelian/ClinVar, DepMap, mechanism-impact genetics
    # (prereg_C_build_genetics.py). Honest 3-way completeness (prereg_C_completeness):
    # COMPLETE / N/A (drug has no protein target — cytotoxic-DNA or metabolic MOA, mechanism-fit
    # not applicable) / INCOMPLETE (resolvable data gap: disease module not in OT, or drug targets
    # not yet OT-mapped). "Targetless" is never asserted.
    from prereg_C_completeness import label as _completeness, summary as _csum
    f["feature_completeness"] = _completeness(f, ik_col="IK14")
    # ABSTAIN_OOD (below) fires for any confident-FAIL lacking FULL mechanism coverage — both
    # INCOMPLETE (data gap) and N/A (no protein target): in either case the overall head's
    # confident FAIL rests on no mechanism signal, so it must not be trusted.
    cov_absent = ~f["feature_completeness"].str.startswith("COMPLETE")
    print(f"  feature completeness: {_csum(f['feature_completeness'])}")
    is_fail = f.confidence_regime == "confident_FAIL"
    n_cov = int((is_fail & cov_absent).sum())
    f.loc[is_fail & cov_absent, "confidence_regime"] = "ABSTAIN_OOD"
    # transparency flag (does NOT change the bet): raw score in OOF upper-tail extrapolation region
    f["raw_extrapolation"] = (f.P_fail_overall_raw > raw95).astype(int)
    print(f"\nOOD/coverage handling (confident-FAIL only):")
    print(f"  ABSTAIN {n_cov} confident-FAIL with mechanism coverage ABSENT (structural-zero, not evidence)")
    print(f"  FLAG raw_extrapolation (raw P_fail > OOF 95th pct {raw95:.3f}): disclosed, NOT abstained")
    # ----------------------------------------------------------------------------------------

    # ---- PREDICTION CONFIDENCE SCORE + support/stability gate (jun22) -----------------------
    # The reliability bands above gate on the P_fail POINT ESTIMATE; they say nothing about how
    # much EVIDENCE backs each estimate (epistemic uncertainty). Isotonic calibration fixes
    # in-distribution probabilities (aleatoric) but cannot know the model has barely seen a
    # disease. Diagnosis of the fenofibrate->diabetic-retinopathy false FAIL (an overall-head
    # call at raw 0.985, mislabeled "safety", uncorroborated by either sub-head, on a disease
    # with ~1 training trial) showed 15/17 confident-FAIL bets rest on <=4 training trials.
    # We compute three leak-safe, outcome-blind confidence signals per prediction and ABSTAIN a
    # confident-FAIL only when it is BOTH thin-support AND mechanistically uncorroborated -- i.e.
    # the overall head is extrapolating disease-level pessimism with nothing to stand on.
    MIN_SUPPORT = 10    # training trials in the disease for "adequate support"
    CORROB = 0.50       # a sub-head (efficacy/safety) independently leaning fail corroborates
    coh = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
    dis_counts = coh.Disease.value_counts()
    f["n_disease_train"] = f.novel_indication.map(dis_counts).fillna(0).astype(int)
    f["subhead_max"] = f[["P_fail_efficacy_calibrated", "P_fail_safety_calibrated"]].max(axis=1)
    f["extrap_gap"] = (f.P_fail_overall_raw - f.P_fail_overall_calibrated).clip(lower=0)
    # DIRECTION-AWARE corroboration: does a mechanistic sub-head agree with the overall call?
    # For a FAIL call a sub-head should also lean fail (subhead_max high); for a PASS call the
    # sub-heads should agree it passes (subhead_max low). So corroboration = subhead_max for a
    # fail call and (1 - subhead_max) for a pass call. Without this, every PASS (which correctly
    # has low sub-head fail-prob) would be mislabeled "uncorroborated".
    fail_call = f.P_fail_overall_calibrated >= 0.5
    f["corroboration"] = np.where(fail_call, f.subhead_max, 1.0 - f.subhead_max).round(3)
    # transparent composite (higher = more trustworthy); s_calib is ~uniformly low for the
    # forward set (all bets are OOD extrapolations), which is the honest disclosure, so the gate
    # below keys on the two DISCRIMINATING signals (support, corroboration), not s_calib.
    s_support = np.minimum(1.0, f.n_disease_train / 20.0)
    s_corrob = np.minimum(1.0, f.corroboration / 0.60)
    s_calib = 1.0 - np.minimum(1.0, f.extrap_gap / 0.40)
    f["confidence_score"] = (0.45 * s_support + 0.45 * s_corrob + 0.10 * s_calib).round(3)
    thin = f.n_disease_train < MIN_SUPPORT
    uncorrob = f.corroboration < CORROB
    f["confidence_tier"] = np.where(~thin & ~uncorrob, "high",
                            np.where(thin & uncorrob, "low", "medium"))
    is_fail = f.confidence_regime == "confident_FAIL"   # coverage abstains already moved out
    low_fail = is_fail & thin & uncorrob
    n_low = int(low_fail.sum())
    f.loc[low_fail, "confidence_regime"] = "ABSTAIN_LOWCONF"
    print(f"\nConfidence gate (confident-FAIL only; support<{MIN_SUPPORT} trials AND no sub-head >= {CORROB}):")
    print(f"  ABSTAIN {n_low} confident-FAIL as LOW-confidence (thin disease support AND uncorroborated)")
    for r in f[low_fail].sort_values("confidence_score").itertuples():
        print(f"    {r.drug:18s} -> {r.novel_indication:42s} "
              f"(n_dis={r.n_disease_train}, subhead_max={r.subhead_max:.2f}, conf={r.confidence_score:.2f})")
    # ----------------------------------------------------------------------------------------

    f["bet"] = f.confidence_regime.map({"confident_PASS": "PASS", "confident_FAIL": "FAIL"}).fillna("")
    bets = f[f.bet != ""].copy().sort_values("P_fail_overall_calibrated", ascending=False)
    abst = f[f.bet == ""].copy()

    bets_out, abst_out = P / "prereg_C_confident_bets.csv", P / "prereg_C_abstained.csv"
    bets.to_csv(bets_out, index=False); abst.to_csv(abst_out, index=False)
    sha = hashlib.sha256(bets_out.read_bytes()).hexdigest()
    (P / "prereg_C_confident_bets.sha256").write_text(sha + "  prereg_C_confident_bets.csv\n")

    # ---- PRIMARY artifact (Jun 28): the FULL registered+scored set ---------------------------
    # Rather than locking only the confident subset and discarding the rest, we REGISTER every
    # cleanly-attributable novel pair with its calibrated P_fail + pre-registered confidence stratum,
    # and SCORE THE WHOLE SET at readout (ROC-AUC + calibration). The confidence stratum is a
    # PRE-REGISTERED stratifier, not a selection filter: the headline hypothesis is that the
    # high-confidence subset out-performs the uncertain middle (grounded on held-out OOF below).
    # This removes the cherry-picking critique and tests the model's calibration over the full forward
    # distribution + its ability to self-identify where it is reliable. The confident bets remain the
    # most-actionable highlight; ABSTAIN_BIOLOGY (metoprolol) is scored but disclosed as a known
    # endpoint-mismatch caveat and held out of the highlight only.
    f["predicted_label"] = np.where(f.P_fail_overall_calibrated >= 0.5, "FAIL", "PASS")
    f["confidence_stratum"] = np.where(
        f.confidence_regime.isin(["confident_PASS", "confident_FAIL"]), "high_confidence",
        np.where(f.confidence_regime == "ABSTAIN_BIOLOGY", "disclosed_caveat", "uncertain_middle"))
    reg_cols = ["NCT_ID", "drug", "novel_indication", "overall_status", "predicted_label",
                "P_fail_overall_calibrated", "P_fail_efficacy_calibrated", "P_fail_safety_calibrated",
                "confidence_score", "confidence_tier", "confidence_stratum", "confidence_regime",
                "n_disease_train", "corroboration", "raw_extrapolation", "feature_completeness"]
    reg = f[[c for c in reg_cols if c in f.columns]].sort_values(
        "P_fail_overall_calibrated", ascending=False)
    reg_out = P / "prereg_C_registered_predictions.csv"
    reg.to_csv(reg_out, index=False)
    reg_sha = hashlib.sha256(reg_out.read_bytes()).hexdigest()
    (P / "prereg_C_registered_predictions.sha256").write_text(
        reg_sha + "  prereg_C_registered_predictions.csv\n")

    # held-out grounding for the pre-registered stratification hypothesis
    conf_oof = g[(g.p <= t_low) | (g.p >= t_high)]
    mid_oof = g[(g.p > t_low) & (g.p < t_high)]
    oof_acc = lambda d: ((d.p >= 0.5).astype(int) == d.y).mean()
    from sklearn.metrics import roc_auc_score as _auc
    print(f"\n=== PRIMARY: full registered+scored set ({len(f)} cleanly-attributable novel pairs) ===")
    print(f"  predicted @0.5: {int((f.predicted_label=='PASS').sum())} PASS / "
          f"{int((f.predicted_label=='FAIL').sum())} FAIL  |  strata: "
          f"{f.confidence_stratum.value_counts().to_dict()}")
    print(f"  readout scoring: full-set ROC-AUC + calibration, stratified by confidence.")
    print(f"  PRE-REGISTERED stratifier grounded on held-out OOF:")
    print(f"    high-confidence band  acc {oof_acc(conf_oof):.3f} (n={len(conf_oof)})  vs  "
          f"uncertain middle acc {oof_acc(mid_oof):.3f} (n={len(mid_oof)})  | full-set AUC {_auc(g.y,g.p):.3f}")
    print(f"  registered SHA-256 {reg_sha}")

    n_pass, n_fail = int((bets.bet == "PASS").sum()), int((bets.bet == "FAIL").sum())
    print(f"\nHIGH-CONFIDENCE highlight (subset of the {len(f)}): {len(bets)} "
          f"({n_pass} PASS, {n_fail} FAIL) | uncertain-middle (still scored) {len(abst)}")
    print(f"expected accuracy at readout (held-out): PASS-band ~{pass_prec:.0%}, FAIL-band ~{fail_prec:.0%}")
    n_flag = int(bets[bets.bet == "FAIL"].raw_extrapolation.sum())
    print(f"SHA-256 {sha}")
    print(f"  ({n_flag}/{n_fail} FAIL-bets carry raw_extrapolation=1 — in OOF upper-tail, calibration extrapolated)")
    print(f"\nFAIL-bets (model predicts these NOVEL indications fail; the striking, against-base-rate calls):")
    print(f"  [* = raw_extrapolation: raw P_fail in sparse OOF region, calibrated value is an extrapolation]")
    for r in bets[bets.bet == "FAIL"].itertuples():
        print(f"  {r.P_fail_overall_calibrated:.2f}{'*' if r.raw_extrapolation else ' '} {r.drug:22s} -> {r.novel_indication}")


if __name__ == "__main__":
    main()
