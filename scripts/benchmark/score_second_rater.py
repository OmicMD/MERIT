#!/usr/bin/env python3
"""
Score the second-rater label audit: Cohen's kappa (6-way + headline binary),
percent agreement, and a confusion matrix vs the first rater.

Reads:  data/review/second_rater/audit_sample_RATED.csv  (rater2_category filled)
        data/review/second_rater/audit_key_HIDDEN.csv     (rater1_category)
Writes: results/benchmark/second_rater_kappa.json
Kappa is implemented directly (no sklearn dependency).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data/review/second_rater"
RATED = DIR / "audit_sample_RATED.csv"
KEY = DIR / "audit_key_HIDDEN.csv"
OUT = ROOT / "results/benchmark/second_rater_kappa.json"

GENUINE = {"FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"}


def cohen_kappa(a, b):
    """Unweighted Cohen's kappa for two equal-length label sequences."""
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    k = len(labels)
    m = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        m[idx[x]][idx[y]] += 1
    po = sum(m[i][i] for i in range(k)) / n
    row = [sum(m[i]) for i in range(k)]
    col = [sum(m[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(k))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return kappa, po, labels, m


def main():
    rated_xlsx = DIR / "audit_sample_RATED.xlsx"
    if rated_xlsx.exists():
        rated = pd.read_excel(rated_xlsx, sheet_name="Rate here")
    elif RATED.exists():
        rated = pd.read_csv(RATED)
    else:
        raise SystemExit(
            f"Not found: {rated_xlsx} or {RATED}\nFill rater2_category in the "
            "'Rate here' tab of audit_sample_BLINDED.xlsx (or the CSV) and save as "
            "audit_sample_RATED.xlsx (or .csv) first."
        )
    key = pd.read_csv(KEY)
    df = key.merge(rated[["audit_id", "rater2_category"]], on="audit_id", how="inner")
    df["rater2_category"] = df["rater2_category"].astype(str).str.strip().str.upper()

    unrated = df["rater2_category"].isin(["", "NAN", "NONE"])
    if unrated.any():
        print(f"WARNING: {unrated.sum()} rows unrated — excluded from kappa.")
        df = df[~unrated]
    # UNCERTAIN excluded from the agreement statistic (reported separately)
    n_uncertain = int((df["rater2_category"] == "UNCERTAIN").sum())
    scored = df[df["rater2_category"] != "UNCERTAIN"]

    r1 = scored["rater1_category"].tolist()
    r2 = scored["rater2_category"].tolist()

    k6, po6, labels, cm = cohen_kappa(r1, r2)

    b1 = ["GENUINE_FAIL" if x in GENUINE else "EXCLUDE" for x in r1]
    b2 = ["GENUINE_FAIL" if x in GENUINE else "EXCLUDE" for x in r2]
    kb, pob, blabels, bcm = cohen_kappa(b1, b2)

    out = {
        "n_rated": int(len(df)),
        "n_uncertain_excluded": n_uncertain,
        "n_scored": int(len(scored)),
        "six_way": {"cohen_kappa": round(k6, 4), "percent_agreement": round(po6, 4),
                    "labels": labels, "confusion_matrix_rater1_rows": cm},
        "binary_genuine_vs_exclude": {"cohen_kappa": round(kb, 4),
                                      "percent_agreement": round(pob, 4),
                                      "labels": blabels, "confusion_matrix_rater1_rows": bcm},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "six_way"}, indent=2))
    print(f"\nHeadline binary kappa = {kb:.3f}  (n={len(scored)}); full report -> {OUT}")


if __name__ == "__main__":
    main()
