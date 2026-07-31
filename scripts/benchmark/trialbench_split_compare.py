#!/usr/bin/env python3
"""Tier-1 inflation exhibit, TrialBench feature sets: structure-blind vs structure-holdout.

Reuses the exact loaders/merge of compare_trialbench_features.py (TrialBench approval label
on the NCT intersection with our cohort), and evaluates each feature set under TWO splits
that differ ONLY in fold assignment:

  HOLDOUT: StratifiedGroupKFold grouped by IK14 (no compound spans train and test).
  BLIND:   StratifiedKFold ignoring compound identity (compounds leak across folds).

Same classifier (HistGradientBoosting), same seeds, same 5 folds. The holdout-minus-blind
difference is the evaluation inflation for each feature set.

Output: results/benchmark/trialbench_split_compare.csv + summary.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
from compare_trialbench_features import load_trialbench, load_ours  # noqa: E402

OUT = ROOT / "results/benchmark"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
NSPLIT = 5


def fit_auc(X, y, tr, te, s):
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                       random_state=s)
    m.fit(X[tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])


def cv(X, y, groups, blind):
    aucs = []
    leaked = 0
    for s in SEEDS:
        if blind:
            splitter = StratifiedKFold(n_splits=NSPLIT, shuffle=True, random_state=s)
            it = splitter.split(X, y)
        else:
            splitter = StratifiedGroupKFold(n_splits=NSPLIT, shuffle=True, random_state=s)
            it = splitter.split(X, y, groups)
        for tr, te in it:
            if len(np.unique(y[te])) < 2:
                continue
            if blind:
                leaked += len(set(groups[te]) & set(groups[tr]))
            aucs.append(fit_auc(X, y, tr, te, s))
    return np.mean(aucs), np.std(aucs, ddof=1), len(aucs), leaked


def main():
    tb, tb_feats = load_trialbench()
    ours, star_feats = load_ours()
    j = ours.merge(tb, on="nct", how="inner").drop_duplicates("nct")
    y = j.outcome.astype(int).values
    g = j.ik14.values
    print(f"overlap n={len(j)}, compounds={j.ik14.nunique()}, approval_rate={y.mean():.3f}")

    # Leak-free scopes (Jul 2026 principled regen; benchmark_final_v2.py rationale).
    # Our side: biology only, excluding trial-design / trial-context features (module 9,
    # drug-count, combination, healthy-volunteer, cleaning flags). Their side: legitimate
    # trial-design only, after removing the enrollment leak AND establishment metadata
    # (sponsor / oversight / regulatory / data-sharing) that acts as a reverse-causation proxy.
    fl = pd.read_csv(ROOT / "data/sources/feature_list_v8.csv")
    mod9 = set(fl[fl.module_family == "9_endpoint_population_design_leverage"].feature)
    our_ctx = mod9 | {"trial_n_drugs", "is_combination", "is_healthy_volunteer",
                      "is_multi_drug_exclude", "is_procedural_exclude",
                      "is_mispaired_supportive", "attr_misindexed"}
    bio = [c for c in star_feats if c not in our_ctx]
    enroll_cols = [c for c in tb_feats if "enroll" in c.lower()]
    estab = [c for c in tb_feats if any(k in c.lower() for k in
             ["enroll", "sponsor", "agency", "oversight", "dmc", "fda_regulated",
              "sharing_ipd", "collaborator", "responsible", "funded"])]
    feat_sets = {
        "their design (all)": tb_feats,
        "their design minus enrollment": [c for c in tb_feats if c not in enroll_cols],
        "their legit design": [c for c in tb_feats if c not in estab],
        "our biology": bio,
    }
    rows = []
    for name, feats in feat_sets.items():
        X = j[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        hm, hsd, hk, _ = cv(X, y, g, blind=False)
        bm, bsd, bk, leaked = cv(X, y, g, blind=True)
        assert leaked > 0, f"BLIND ARM NOT BLIND for {name}: no compound spans train/test"
        rows.append(dict(feature_set=name, n_feats=len(feats),
                         holdout_auc=hm, holdout_sd=hsd, blind_auc=bm, blind_sd=bsd,
                         inflation=bm - hm, n_folds=hk, blind_leaked_compounds=leaked))
        print(f"  {name:32s}  holdout {hm:.3f}±{hsd:.3f} | blind {bm:.3f}±{bsd:.3f} | "
              f"inflation {bm-hm:+.4f}  ({len(feats)} feats)")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "trialbench_split_compare.csv", index=False)
    print(f"\nwrote {OUT/'trialbench_split_compare.csv'}")


if __name__ == "__main__":
    main()
