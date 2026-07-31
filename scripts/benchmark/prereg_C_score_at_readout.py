#!/usr/bin/env python3
"""FROZEN readout-scoring protocol for the prospective registration (committed Jun 28 2026, BEFORE
any registered trial reports — this is the whole point: the scoring code is fixed with the predictions
so the metric cannot be tuned after outcomes are seen).

It scores the registered novel-pair predictions (results/benchmark/prereg_C/
prereg_C_registered_predictions.csv) against the realized trial outcomes, exactly as the manuscript
pre-commits: the ENTIRE registered set is scored (no post-hoc selection), and the pre-registered
confidence stratifier is tested (high-confidence band vs uncertain middle).

USAGE
  python scripts/benchmark/prereg_C_score_at_readout.py <realized_outcomes.csv>
  python scripts/benchmark/prereg_C_score_at_readout.py --self-test     # validate now, no outcomes

REALIZED-OUTCOMES SCHEMA (the file Gabe fills in at readout; one row per registered NCT_ID):
  NCT_ID,realized_outcome
  NCT05843643,FAIL          # realized_outcome in {PASS, FAIL}  (FAIL = trial did NOT meet its
  NCT06545526,PASS          #   primary efficacy endpoint / was a safety/efficacy failure, scored
  ...                       #   under the SAME label-audit protocol as the training cohort)
A 0/1 column (1 = FAIL) is also accepted. Rows not present in the registration are ignored; registered
rows with no outcome yet are reported as still-pending and excluded from scoring.

METRICS (all pre-committed):
  1. Full-set discrimination: ROC-AUC of P_fail_overall_calibrated vs realized FAIL, with a
     DRUG-CLUSTERED bootstrap 95% CI (resample drugs with replacement — the same clustering the
     headline CI uses, scripts/strengthening/bootstrap_ci.py).
  2. Full-set calibration: Brier score + expected calibration error (ECE, 10 equal-width bins) +
     a reliability table.
  3. Pre-registered stratifier test: accuracy in the high-confidence band vs the uncertain middle,
     with a two-proportion z-test of the difference (H1: high > middle). The disclosed-caveat row
     (metoprolol→DMD) is reported separately, not in either stratum.
  4. Actionable highlight: PASS-band and FAIL-band precision (the high-confidence subset).
The held-out expectation these will be compared against (canonical gapBD OOF, frozen at registration):
full-set ranking AUC ~0.785, high-confidence band ~90% accurate vs ~63% in the uncertain middle.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "results/benchmark/prereg_C/prereg_C_registered_predictions.csv"
OOF = ROOT / "results/production_v8_clean_mort_gapBD_jun28/oof_overall.parquet"
PCOL = "P_fail_overall_calibrated"


def _auc_clustered_ci(df, ycol="y", pcol=PCOL, cluster="drug", n_boot=5000, seed=0):
    """Point AUC + drug-clustered bootstrap 95% CI (resample whole drugs with replacement).
    Index-based for speed: precompute each cluster's row positions, then concatenate position
    arrays per bootstrap draw rather than rebuilding DataFrames."""
    df = df.reset_index(drop=True)
    y = df[ycol].to_numpy(); p = df[pcol].to_numpy()
    point = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    units = df[cluster].dropna().unique()
    pos = {u: np.flatnonzero((df[cluster] == u).to_numpy()) for u in units}
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(units, size=len(units), replace=True)
        ix = np.concatenate([pos[u] for u in samp])
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        boots.append(roc_auc_score(yy, p[ix]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return point, lo, hi, len(units)


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, n = p[m].mean(), y[m].mean(), int(m.sum())
        e += (n / len(y)) * abs(conf - acc)
        rows.append((f"[{edges[b]:.1f},{edges[b+1]:.1f})", n, round(conf, 3), round(acc, 3)))
    return e, rows


def _two_prop_z(k1, n1, k2, n2):
    """One-sided z-test that p1 (high) > p2 (middle)."""
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    from math import erf
    p_one_sided = 1 - 0.5 * (1 + erf(z / np.sqrt(2)))
    return z, p_one_sided


def score(reg, outcomes):
    """outcomes: DataFrame with NCT_ID + realized FAIL indicator y in {0,1}. One outcome per TRIAL
    (NCT_ID) applies to every registered drug-pair of that trial; dedup so the join cannot multiply."""
    outcomes = outcomes.drop_duplicates("NCT_ID")
    d = reg.merge(outcomes[["NCT_ID", "y"]], on="NCT_ID", how="left")
    pending = d[d.y.isna()]
    d = d[d.y.notna()].copy()
    d["y"] = d.y.astype(int)
    d["correct"] = ((d[PCOL] >= 0.5).astype(int) == d.y).astype(int)
    print(f"registered {len(reg)} | scored {len(d)} | still pending {len(pending)}")
    if not len(d):
        print("no realized outcomes yet — nothing to score."); return
    print(f"realized FAIL rate among scored: {d.y.mean():.3f}\n")

    print("=== 1. FULL-SET discrimination (the primary, no selection) ===")
    if d.y.nunique() == 2:
        auc, lo, hi, nu = _auc_clustered_ci(d)
        print(f"  ROC-AUC {auc:.3f}  drug-clustered 95% CI [{lo:.3f}, {hi:.3f}]  (n_drugs={nu})")
    else:
        print("  (only one outcome class present — AUC undefined yet)")
    print(f"  accuracy @0.5: {d.correct.mean():.3f}")

    print("\n=== 2. FULL-SET calibration ===")
    brier = brier_score_loss(d.y, d[PCOL]) if d.y.nunique() == 2 else float("nan")
    ece, rel = _ece(d.y.values, d[PCOL].values)
    print(f"  Brier {brier:.3f} | ECE {ece:.3f}")
    print("  reliability (bin, n, mean P_fail, realized fail-rate):")
    for r in rel:
        print(f"    {r[0]:>11s}  n={r[1]:3d}  pred={r[2]:.3f}  actual={r[3]:.3f}")

    print("\n=== 3. PRE-REGISTERED stratifier: high-confidence vs uncertain middle ===")
    hi_ = d[d.confidence_stratum == "high_confidence"]
    mid = d[d.confidence_stratum == "uncertain_middle"]
    cav = d[d.confidence_stratum == "disclosed_caveat"]
    for name, s in [("high_confidence", hi_), ("uncertain_middle", mid)]:
        if len(s):
            print(f"  {name:16s} n={len(s):3d}  accuracy={s.correct.mean():.3f}")
    if len(hi_) and len(mid):
        z, pz = _two_prop_z(hi_.correct.sum(), len(hi_), mid.correct.sum(), len(mid))
        verdict = "CONFIRMED" if (pz < 0.05) else "not confirmed"
        print(f"  H1 (high > middle): z={z:.2f}, one-sided p={pz:.4f} -> {verdict}")
    if len(cav):
        print(f"  disclosed_caveat   n={len(cav):3d}  accuracy={cav.correct.mean():.3f} "
              f"(reported separately; e.g. metoprolol→DMD endpoint-mismatch flag)")

    print("\n=== 4. Actionable highlight: high-confidence band precision ===")
    band = d[d.confidence_regime.isin(["confident_PASS", "confident_FAIL"])]
    pb = band[band.confidence_regime == "confident_PASS"]
    fb = band[band.confidence_regime == "confident_FAIL"]
    if len(pb):
        print(f"  PASS-band precision {1 - pb.y.mean():.3f}  (n={len(pb)}; pre-registered ~0.91)")
    if len(fb):
        print(f"  FAIL-band precision {fb.y.mean():.3f}  (n={len(fb)}; pre-registered ~0.85)")


def self_test():
    """Validate the frozen protocol NOW (no real outcomes): confirm the registration loads and the
    metric functions run, by scoring the held-out canonical OOF — the same numbers the manuscript
    pre-commits as the expectation."""
    reg = pd.read_csv(REG)
    print(f"registration loads: {len(reg)} predictions | strata "
          f"{reg.confidence_stratum.value_counts().to_dict()}")
    print(f"  predicted @0.5: {reg.predicted_label.value_counts().to_dict()}")
    print("\n--- protocol self-test on held-out canonical OOF (proves the frozen code runs) ---")
    g = pd.read_parquet(OOF)
    g = g.groupby(["SMILES", "Disease"]).agg(y=("y", "first"),
                                             **{PCOL: ("calibrated_prob", "mean")}).reset_index()
    g = g.rename(columns={"SMILES": "drug"})
    auc, lo, hi, nu = _auc_clustered_ci(g, n_boot=2000)
    print(f"  full-set ROC-AUC {auc:.3f}  drug-clustered 95% CI [{lo:.3f}, {hi:.3f}]")
    tl, th = 0.20, 0.51
    g["correct"] = ((g[PCOL] >= 0.5).astype(int) == g.y).astype(int)
    band = g[(g[PCOL] <= tl) | (g[PCOL] >= th)]
    mid = g[(g[PCOL] > tl) & (g[PCOL] < th)]
    z, pz = _two_prop_z(band.correct.sum(), len(band), mid.correct.sum(), len(mid))
    print(f"  high-confidence band accuracy {band.correct.mean():.3f} (n={len(band)}) vs "
          f"uncertain middle {mid.correct.mean():.3f} (n={len(mid)}); z={z:.1f}, p={pz:.1e}")
    print("\nfrozen protocol OK. At readout: run with the realized-outcomes CSV.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("--self-test", "-h", "--help"):
        self_test(); return
    reg = pd.read_csv(REG)
    oc = pd.read_csv(sys.argv[1])
    if "realized_outcome" in oc.columns:
        oc["y"] = (oc.realized_outcome.astype(str).str.upper().str.startswith("FAIL")
                   | oc.realized_outcome.astype(str).isin(["1"])).astype(int)
    elif "y" not in oc.columns:
        sys.exit("outcomes file needs a 'realized_outcome' (PASS/FAIL) or 'y' (1=FAIL) column")
    score(reg, oc)


if __name__ == "__main__":
    main()
