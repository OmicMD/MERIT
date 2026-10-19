#!/usr/bin/env python3
"""Safety-head score against trial termination and regulatory toxicity labels.

At the drug level, computes the AUROC of the safety head's mean out-of-fold score for
three labels: FAIL_SAFETY (trial terminated for safety), ChEMBL black-box warning and
ChEMBL withdrawn flag. Also compares the black-box rate of confident false positives
(PASS drugs with max score >= 0.8) with all PASS drugs (Fisher exact test).

Writes results/safety_toxicity_vs_termination/result.json.
Usage: python scripts/safety_toxicity_vs_termination.py
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
SAF = ROOT / "results/production_v8_clean_mort_singlehead_jul6/oof_safety.parquet"
CHEMBL = ROOT / "data/sources/03_chembl_compound_results.csv"
OUT = ROOT / "results/safety_toxicity_vs_termination"


def _bin(x):
    return pd.to_numeric(x, errors="coerce").fillna(0).clip(0, 1).astype(int)


def main():
    df = pd.read_csv(DATA, low_memory=False)
    df["IK14"] = df.feature_IK.astype(str).str[:14]
    oof = pd.read_parquet(SAF)
    g = (oof.groupby("row_idx").agg(p=("raw_prob", "mean"), y=("y", "first")).reset_index())
    m = g.merge(df[["IK14"]], left_on="row_idx", right_index=True)
    ch = pd.read_csv(CHEMBL)
    ch["IK14"] = ch.InchiKey.astype(str).str[:14]
    tox = ch.groupby("IK14").agg(bbw=("black_box_warning", lambda s: _bin(s).max()),
                                 wd=("withdrawn_flag", lambda s: _bin(s).max()))
    d = (m.groupby("IK14").agg(pmax=("p", "max"), pmean=("p", "mean"),
                               failsafety=("y", "max")).reset_index()
         .merge(tox, on="IK14", how="left"))
    match = float(d.IK14.isin(set(ch.IK14)).mean())
    d[["bbw", "wd"]] = d[["bbw", "wd"]].fillna(0)

    res = {"n_drugs": len(d), "chembl_match": match,
           "pos": {"failsafety": int(d.failsafety.sum()), "bbw": int(d.bbw.sum()),
                   "withdrawn": int(d.wd.sum())}, "auc": {}}
    print(f"drugs={len(d)}  ChEMBL IK14 match={match:.2f}  "
          f"(FAIL_SAFETY {int(d.failsafety.sum())}, black-box {int(d.bbw.sum())}, "
          f"withdrawn {int(d.wd.sum())})")
    for tgt, lab in [("failsafety", "FAIL_SAFETY (termination)"),
                     ("bbw", "black-box (intrinsic toxicity)"),
                     ("wd", "withdrawn (post-market safety)")]:
        auc = float(roc_auc_score(d[tgt], d.pmean))
        res["auc"][tgt] = auc
        print(f"  safety-head -> {lab:32s} AUROC={auc:.3f}")

    po = d[d.failsafety == 0]
    hi = po[po.pmax >= 0.8]
    lo = po[po.pmax < 0.8]
    tab = [[int(hi.bbw.sum()), len(hi) - int(hi.bbw.sum())],
           [int(lo.bbw.sum()), len(lo) - int(lo.bbw.sum())]]
    orr, pval = fisher_exact(tab)
    res["confident_fp"] = {"n": len(hi), "bbw_rate": float(hi.bbw.mean()),
                           "pass_base_bbw": float(po.bbw.mean()),
                           "fisher_or": float(orr), "fisher_p": float(pval)}
    print(f"\nconfident FP (PASS, P_fail>=0.8, n={len(hi)}): black-box rate "
          f"{hi.bbw.mean():.2f} vs PASS base {po.bbw.mean():.2f} "
          f"(Fisher OR={orr:.2f}, p={pval:.3f})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT/'result.json'}")


if __name__ == "__main__":
    main()
