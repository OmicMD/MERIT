#!/usr/bin/env python3
"""Tier-1 inflation exhibit, STAR feature set, CLEAN apples-to-apples folds.

Identical in spirit to matched_star_phase3_split_compare.py, but uses the SAME clean
2,787 unique-NCT fold partitions HINT is evaluated on, so the STAR holdout/blind bars are
directly comparable to hint_holdout_clean_full.csv / hint_blind_full.csv:

  HOLDOUT (structure-based compound-holdout): data/sources/hint_folds_holdout_clean.csv —
    IK14-grouped, each NCT in exactly ONE fold per seed (no within-NCT duplication, no
    compound spanning train and test).
  BLIND (structure-blind): data/sources/hint_our_cohort_folds_blind.csv — the SAME 2,787
    NCTs randomly assigned to 5 folds per seed (same file HINT's blind arm used), so the
    same compound's other trials can sit in training.

The cohort d is restricted to the 2,787 NCTs present in the clean fold set. We keep our
native trial-indication scoring grain: each row is assigned its NCT's fold, so a
multi-indication NCT's rows all land in the same fold (no within-NCT split).

Everything else (features, model, imputation, nested selection, 5x5 repeats, seeds) is held
constant. holdout-minus-blind is the evaluation inflation for the STAR set.

The HOLDOUT arm here is also the apples-to-apples matched STAR comparator for Supplementary Table S1
(vs HINT 0.626 on the identical 2,787-NCT IK14-grouped folds).

Output: results/benchmark/star_phase3_split_compare_clean.csv (per-fold, both arms) + summary.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "strengthening"))
from _pipeline import get_features, fit_predict_fold  # noqa: E402

OUR = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
HOLDOUT_FOLDS = ROOT / "data/sources/hint_folds_holdout_clean.csv"
BLIND_FOLDS = ROOT / "data/sources/hint_our_cohort_folds_blind.csv"
OUT = ROOT / "results/benchmark"; OUT.mkdir(parents=True, exist_ok=True)


def prep():
    d = pd.read_csv(OUR, low_memory=False)
    d = d[d.Corrected_Outcome.isin(["PASS", "FAIL_EFFICACY", "FAIL_SAFETY", "FAIL_BOTH"])].copy()
    d = d[d.Phase.isin(["Phase 3", "Phase 2/3"])].copy().reset_index(drop=True)
    feature_cols = get_features(d)              # before adding numeric label
    d["nct"] = d.NCT_ID.astype(str)
    d["ik14"] = d.feature_IK.astype(str).str[:14]
    d["label"] = (d.Corrected_Outcome == "PASS").astype(int)
    # restrict to the clean 2,787-NCT universe so both arms (and HINT) share the cohort
    universe = set(pd.read_csv(HOLDOUT_FOLDS).nctid.astype(str))
    d = d[d.nct.isin(universe)].reset_index(drop=True)
    return d, feature_cols


def run_fold(d, feature_cols, te_mask, tr_mask, s):
    X = d[feature_cols].values
    y = d.label.values
    dis = d.Disease.astype(str).values
    if len(np.unique(y[te_mask])) < 2:
        return None
    proba, _ = fit_predict_fold(X[tr_mask], y[tr_mask], X[te_mask], dis[tr_mask], dis[te_mask],
                                feature_cols, seed=s)
    return roc_auc_score(y[te_mask], proba)


def fold_arm(d, feature_cols, folds_path, arm):
    """Returns per-fold rows plus the summed count of test-compounds-also-in-train
    (measured in OUR feature_IK space). The holdout folds are IK14-grouped in HINT's
    InChIKey space (built from HINT cohort SMILES); a residual leak count here reflects
    cross-source InChIKey disagreement, NOT a within-NCT split (every NCT sits in exactly
    one fold). The caller asserts blind >> holdout, which is the inflation signal."""
    folds = pd.read_csv(folds_path)
    rows, leaked_total = [], 0
    for s in sorted(folds.seed.unique()):
        fs = folds[folds.seed == s]
        for f in sorted(fs.test_fold.unique()):
            test_ncts = set(fs[fs.test_fold == f].nctid.astype(str))
            train_ncts = set(fs[fs.test_fold != f].nctid.astype(str)) - test_ncts
            te = d.nct.isin(test_ncts).values
            tr = d.nct.isin(train_ncts).values
            shared = set(d.ik14[te]) & set(d.ik14[tr])
            leaked_total += len(shared)
            auc = run_fold(d, feature_cols, te, tr, int(s))
            if auc is None:
                continue
            rows.append({"arm": arm, "seed": int(s), "fold": int(f), "roc_auc": auc,
                         "n_train": int(tr.sum()), "n_test": int(te.sum()),
                         "n_test_pos": int(d.label.values[te].sum()),
                         "n_leaked_compounds": len(shared)})
    print(f"[leak check] {arm} arm: {leaked_total} test-compound-also-in-train instances "
          f"(our feature_IK space, summed over folds)")
    return rows, leaked_total


def main():
    d, feature_cols = prep()
    print(f"{len(d)} phase-3 trial-indication rows (clean 2,787-NCT universe), "
          f"{len(feature_cols)} STAR features, PASS rate {d.label.mean():.3f}, "
          f"{d.ik14.nunique()} unique compounds, {d.nct.nunique()} unique NCTs")
    ho, ho_leak = fold_arm(d, feature_cols, HOLDOUT_FOLDS, "holdout")
    bl, bl_leak = fold_arm(d, feature_cols, BLIND_FOLDS, "blind")
    assert bl_leak > ho_leak, (
        f"blind arm ({bl_leak}) does not leak more than holdout ({ho_leak}) — "
        f"the structure-blind split is not actually blinder")
    res = pd.DataFrame(ho + bl)
    res.to_csv(OUT / "star_phase3_split_compare_clean.csv", index=False)
    h = res[res.arm == "holdout"].roc_auc
    b = res[res.arm == "blind"].roc_auc
    print(f"\nSTAR features, clean 2,787-NCT cohort, same model/protocol, split varied:")
    print(f"  structure-HOLDOUT AUC : {h.mean():.4f} ± {h.std(ddof=1):.4f}  ({len(h)} folds)")
    print(f"  structure-BLIND   AUC : {b.mean():.4f} ± {b.std(ddof=1):.4f}  ({len(b)} folds)")
    print(f"  inflation (blind - holdout): {b.mean()-h.mean():+.4f}")
    print(f"\nwrote {OUT/'star_phase3_split_compare_clean.csv'}")


if __name__ == "__main__":
    main()
