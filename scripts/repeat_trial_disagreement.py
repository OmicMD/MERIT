#!/usr/bin/env python3
"""Repeat-trial disagreement among drug-disease pairs (the "irreducible band", Results).

SOURCE OF THE MAIN-TEXT 18% CLAIM. The manuscript states: "nearly one in five disagree
(18%, 95% CI: 14-22%), and most drug-disease pairs that failed once passed on another
attempt" (MERIT_manuscript_JCIM_revised.md, Results, "Confident errors fall into two
classes outside molecular features"). No script for it was ever committed. This file is a
RECONSTRUCTION written 2026-09-25, not the original analysis.

  RECONSTRUCTED VALUE: 15.8% (95% CI 12.1-19.4%), 61 discordant of 387 repeated pairs.
  PUBLISHED VALUE:     18%   (95% CI 14-22%).
  The CIs overlap but the point estimates differ. The published definition is unknown.

Why it matters: a molecular model assigns one score to every trial of a drug in a given
disease, so discordant repeats are outcomes no molecular feature can separate. The rate
bounds what the efficacy head can reach on this cohort.

Definitions used here (each is a guess where the original is unrecoverable):
  cohort     PASS + FAIL_EFFICACY rows, deduplicated on NCT_ID x pair
  pair       (SMILES, Disease); Drug_Clean x Disease gives identical results (1:1 mapping)
  repeated   pair with more than one trial
  discordant repeated pair holding at least one PASS and at least one failure
  CI         normal approximation on the proportion

Settings already tested that do NOT close the gap to 18% (do not retry):
  - counting FAIL_BOTH as an efficacy failure: unchanged, 15.8%
  - keying pairs on Drug_Clean instead of SMILES: unchanged, 15.8%
Untested candidates: the 2,639-trial efficacy cohort used elsewhere, a Wilson CI, phase
restriction. The published CI is 8 points wide against 7.3 here, so the original
denominator was probably smaller than 387.

UNVERIFIED SIBLING CLAIM: the same passage says "MERIT's residual efficacy error falls
largely inside this irreducible band". That could mean the error is bounded by this rate,
or that it is concentrated in these discordant pairs. This script does not test either.

Run: python scripts/repeat_trial_disagreement.py
"""
from pathlib import Path
from statistics import NormalDist

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COH = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
Z = NormalDist().inv_cdf(0.975)


def disagreement(df: pd.DataFrame, key: list[str], fails: list[str]) -> dict:
    """Discordance rate among repeated drug-disease pairs.

    key   columns identifying a drug-disease pair
    fails outcome labels counted as an efficacy failure
    """
    d = df[df.Corrected_Outcome.isin(["PASS"] + fails)].copy()
    d["fail"] = d.Corrected_Outcome.isin(fails).astype(int)
    d = d.drop_duplicates(subset=["NCT_ID"] + key)
    g = d.groupby(key).fail.agg(n="size", k="sum")
    rep = g[g.n > 1]
    disc = rep[(rep.k > 0) & (rep.k < rep.n)]
    p = len(disc) / len(rep)
    se = (p * (1 - p) / len(rep)) ** 0.5
    failed = rep[rep.k > 0]
    also_passed = failed[failed.k < failed.n]
    return dict(trials=len(d), pairs=len(g), repeated=len(rep), discordant=len(disc),
                rate=100 * p, lo=100 * (p - Z * se), hi=100 * (p + Z * se),
                failed=len(failed), also_passed=len(also_passed),
                also_passed_pct=100 * len(also_passed) / len(failed))


def main():
    df = pd.read_csv(COH, low_memory=False)
    for label, key, fails in [
        ("SMILES x Disease (headline)", ["SMILES", "Disease"], ["FAIL_EFFICACY"]),
        ("+ FAIL_BOTH as failure", ["SMILES", "Disease"], ["FAIL_EFFICACY", "FAIL_BOTH"]),
        ("Drug_Clean x Disease", ["Drug_Clean", "Disease"], ["FAIL_EFFICACY"]),
    ]:
        r = disagreement(df, key, fails)
        print(f"\n{label}")
        print(f"  trials {r['trials']}  pairs {r['pairs']}  repeated {r['repeated']}  "
              f"discordant {r['discordant']}")
        print(f"  disagreement {r['rate']:.1f}% (95% CI {r['lo']:.1f}-{r['hi']:.1f}%)")
        print(f"  failed at least once and also passed: {r['also_passed']}/{r['failed']} "
              f"({r['also_passed_pct']:.0f}%)")
    print("\nPublished: 18% (95% CI 14-22%). Reconstruction does not reproduce it exactly.")


if __name__ == "__main__":
    main()
