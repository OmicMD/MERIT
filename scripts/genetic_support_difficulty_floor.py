#!/usr/bin/env python3
"""Failure rate by target-disease genetic support within difficult indications (Discussion).

SOURCE OF THE MAIN-TEXT CONTRAST. The Discussion states: "Among compounds in the most
difficult quartile of indications, those whose target is genetically associated with the
disease failed 45% of the time, against 62% for those without such an association"
(MERIT_manuscript_JCIM_revised.md, Discussion). No script for it was ever committed. This
file is a RECONSTRUCTION written 2026-09-28, not the original analysis.

  RECONSTRUCTED: 44.7% (n=123) with genetic association vs 62.2% (n=614) without.
  PUBLISHED (submitted version): ranges of 41-47% and 58-80% "across thresholds".
  The associated range reproduces (40.8-44.7% over genetic cutoffs 0.0-0.5). The
  non-associated group sits at 61.5-62.3% across those same cutoffs, so the published
  58-80% upper bound does not arise from the genetic cutoff alone and was probably
  produced by varying the difficulty cut as well. The main text now reports the single
  top-quartile contrast rather than a range.

Why it matters: genetic support for the target reduces but does not remove the failure
risk carried by a difficult indication, which bounds how much a strong mechanism match
can be trusted on its own.

Definitions used here:
  cohort       PASS + FAIL_EFFICACY + FAIL_BOTH, deduplicated on NCT_ID (2,892 trials)
  difficulty   per-indication failure rate, Bayesian-smoothed with SMOOTH = 3 toward the
               cohort rate, matching scripts/disease_complexity_asof.py. Computed over the
               whole cohort, not within CV folds: this is a descriptive contrast, not a
               model feature, so no fold structure applies.
  difficult    indication in the top quartile of that rate
  associated   mech_ot_genetic_assoc > 0 (Open Targets genetic association channel)

CAVEATS, both material to how the number should be read:
  - Sensitive to the difficulty cut. The top tertile shows no separation (27.0% vs 27.4%);
    the top quartile shows a large one. The contrast is a property of the most difficult
    indications, not a monotone trend across difficulty.
  - mech_ot_genetic_assoc is 0 for 82.5% of trials, so "no association" mostly means the
    target-disease pair is absent from Open Targets rather than measured as unassociated.

Run: python scripts/genetic_support_difficulty_floor.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COH = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
SMOOTH = 3
DIFFICULT_Q = 0.75


def efficacy_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Efficacy-task trials with a per-indication smoothed failure rate attached."""
    e = df[df.Corrected_Outcome.isin(["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    e["y"] = e.Corrected_Outcome.isin(["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)
    e = e.drop_duplicates("NCT_ID")
    rate = e.y.mean()
    g = e.groupby("Disease").y.agg(n="size", k="sum")
    g["difficulty"] = (g.k + SMOOTH * rate) / (g.n + SMOOTH)
    return e.merge(g[["difficulty"]], left_on="Disease", right_index=True)


def contrast(e: pd.DataFrame, cutoff: float = 0.0) -> dict:
    """Failure rate with and without genetic association, within difficult indications."""
    hi = e[e.difficulty >= e.difficulty.quantile(DIFFICULT_Q)]
    a = hi[hi.mech_ot_genetic_assoc > cutoff]
    b = hi[hi.mech_ot_genetic_assoc <= cutoff]
    return dict(n_difficult=len(hi), n_assoc=len(a), fail_assoc=100 * a.y.mean(),
                n_none=len(b), fail_none=100 * b.y.mean())


def main():
    e = efficacy_cohort(pd.read_csv(COH, low_memory=False))
    print(f"efficacy trials {len(e)}  failure rate {e.y.mean():.1%}")

    r = contrast(e)
    print(f"\nMost difficult quartile of indications (n = {r['n_difficult']}):")
    print(f"  target genetically associated : n {r['n_assoc']:4d}  failed {r['fail_assoc']:.1f}%")
    print(f"  no genetic association        : n {r['n_none']:4d}  failed {r['fail_none']:.1f}%")

    print("\nSensitivity to the genetic-association cutoff:")
    for c in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        r = contrast(e, c)
        print(f"  > {c:.1f}  associated n {r['n_assoc']:4d} {r['fail_assoc']:5.1f}%  |  "
              f"none n {r['n_none']:4d} {r['fail_none']:5.1f}%")

    print("\nSensitivity to the difficulty cut (genetic cutoff 0):")
    for q in [0.667, 0.75, 0.90]:
        hi = e[e.difficulty >= e.difficulty.quantile(q)]
        a = hi[hi.mech_ot_genetic_assoc > 0]
        b = hi[hi.mech_ot_genetic_assoc == 0]
        print(f"  top {1 - q:.0%} n {len(hi):5d}  associated {a.y.mean():5.1%} (n {len(a):4d})  |  "
              f"none {b.y.mean():5.1%} (n {len(b):4d})")


if __name__ == "__main__":
    main()
