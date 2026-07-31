#!/usr/bin/env python3
"""Clean signal decomposition for the v8 honest cohort (manuscript option B).

Drops the RETRACTED ATC therapeutic-class prior entirely (novelty leak, Jun 6)
and drops has_black_box (survivorship). Reports the honest block decomposition:

  F   = FLAGS only            (disease/vulnerability/trial-design flags)
  D   = FLAGS + DENC          (+ within-fold disease-difficulty prior)
  DM  = FLAGS + DENC + MOL    (+ full molecular profile, nested top-20)
  M   = MOL only              (molecular mechanism alone)

Single-GBM block models (fit_predict), 5x5 StratifiedGroupKFold by SMILES,
mean-of-folds AUC — matches the manuscript decomposition methodology. The
production head (noisy-OR safety) is reported separately; this isolates WHERE
the signal lives, not the head architecture.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_calibrated import fit_predict
from retrain_corrected import nested_feature_selection, compute_disease_encoding, get_features

DATA = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"  # migrated to canonical cohort (Jun 14)
SEEDS = [42, 123, 456, 789, 2024]
# has_black_box dropped (survivorship, inverted); class prior dropped (retracted leak).
# Disease/trial CONTEXT block. MOL (molecular mechanism) = get_features minus this list minus
# MOL_EXCLUDE, so CONTEXT must capture every NON-mechanism feature, else outcome-correlated context
# leaks into "molecular mechanism". Jun 14 reconciliation found two leaks the published decompose
# missed: (1) design_* trial-design primitives (single-arm/randomization/comparator) are outcome-
# correlated (single-arm trials cannot register an efficacy failure) and inflated MOL so badly that
# molecular-alone safety beat the full model (DM 0.824 > 0.740) — they are trial-design CONTEXT, not
# mechanism; (2) disease_mortality_* is a raw disease base-rate proxy that memorizes per-disease rates
# under compound-holdout. design_* moved to CONTEXT here; mortality excluded entirely (redundant with
# the within-fold prior). net_*_disease_* pathway-overlap features ARE molecular mechanism (Fig. 3b) and
# stay in MOL. This makes MOL pure compound mechanism; it diverges from the published partition (which
# left design_* in MOL) — a correction, flagged for Gabe.
FLAGS = ['disease_is_oncology', 'disease_is_infectious', 'disease_is_cns',
         'disease_is_cardiac', 'disease_is_autoimmune', 'disease_is_metastatic',
         'disease_is_transplant', 'disease_is_severe', 'is_combination',
         'trial_n_drugs']
# Apples-to-apples with the PUBLISHED partition: design_* stays in MOL (published "flags alone"=0.617
# matches design NOT in flags). Only two principled deviations, both production-consistent: (1) exclude
# the NEW disease_mortality_* (disease base-rate proxy absent from the published table); (2) the safety
# task additionally excludes mech_* (done per-task in main(), matching retrain_calibrated L598 survivorship
# guard). net_*_disease_* pathway features stay in MOL (mechanism, Fig. 3b).
MOL_EXCLUDE = ['disease_mortality_1y', 'disease_mortality_5y']
VARIANTS = ["M", "F", "D", "DM"]


def run_task(df_task, mol_cols, task):
    flags = [c for c in FLAGS if c in df_task.columns]
    X = df_task[mol_cols].values
    F = df_task[flags].values.astype(float)
    y = df_task["_y"].values
    grp = df_task["SMILES"].values
    dis = df_task["Disease"].fillna("unknown").values
    perfold = {v: {} for v in VARIANTS}
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X, y, grp)):
            if len(np.unique(y[te])) < 2:
                continue
            imp = SimpleImputer(strategy="median")
            Xtr = imp.fit_transform(X[tr]); Xte = imp.transform(X[te])
            fimp = SimpleImputer(strategy="median")
            Ftr = fimp.fit_transform(F[tr]); Fte = fimp.transform(F[te])
            top = nested_feature_selection(Xtr, y[tr], mol_cols)
            Mtr, Mte = Xtr[:, top], Xte[:, top]
            de_tr, de_te = compute_disease_encoding(dis[tr], y[tr], dis[te])
            de_tr, de_te = de_tr.reshape(-1, 1), de_te.reshape(-1, 1)
            blocks = {
                "M":  ([Mtr], [Mte]),
                "F":  ([Ftr], [Fte]),
                "D":  ([de_tr, Ftr], [de_te, Fte]),
                "DM": ([de_tr, Ftr, Mtr], [de_te, Fte, Mte]),
            }
            for v, (ctr, cte) in blocks.items():
                pred = fit_predict(np.column_stack(ctr), y[tr],
                                   np.column_stack(cte), task, seed)
                perfold[v][(seed, fold)] = roc_auc_score(y[te], pred)
    return {v: np.mean(list(d.values())) for v, d in perfold.items()}


def main():
    df = pd.read_csv(DATA, low_memory=False)
    mol_cols = [c for c in get_features(df) if c not in FLAGS and c not in MOL_EXCLUDE]
    print(f"DATA={DATA.name} | molecular pool {len(mol_cols)} cols | flags {len([c for c in FLAGS if c in df.columns])}")

    tasks = {}
    saf = df[df["Corrected_Outcome"].isin(["PASS", "FAIL_SAFETY", "FAIL_BOTH"])].copy()
    if "is_multi_drug_exclude" in saf.columns:
        saf = saf[saf["is_multi_drug_exclude"] != 1]
    saf["_y"] = saf["Corrected_Outcome"].isin(["FAIL_SAFETY", "FAIL_BOTH"]).astype(int)
    tasks["safety"] = saf

    excl = pd.Series(False, index=df.index)
    for c in ("is_anti_pathogen", "is_endogenous", "is_mispaired_supportive",
              "is_healthy_volunteer", "is_procedural_exclude", "is_multi_drug_exclude"):
        if c in df.columns:
            excl |= df[c] == 1
    de = df[~excl & df["Corrected_Outcome"].isin(["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    de["_y"] = de["Corrected_Outcome"].isin(["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)
    tasks["efficacy"] = de

    do = df[df["Corrected_Outcome"].isin(["PASS", "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    do["_y"] = do["Corrected_Outcome"].isin(["FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)
    tasks["overall"] = do

    res = {}
    for tname, dft in tasks.items():
        # Match the PRODUCTION per-task feature treatment (retrain_calibrated.py L598): the safety head
        # excludes mech_* (causal-centrality) features because they are a SURVIVORSHIP proxy for safety
        # (clean for efficacy/overall). Without this the safety molecular block is inflated above the full
        # model (DM 0.82 > 0.74). Efficacy/overall keep mech_* exactly as production does.
        # Safety additionally excludes design_* (outcome-correlated trial metadata: single-arm trials
        # cannot register a failure) which leak under compound-holdout and pushed safety DM above the full
        # model; overall/efficacy keep design_* in MOL exactly as the published table did.
        if tname == 'safety':
            task_mol = [c for c in mol_cols if not c.startswith('mech_') and not c.startswith('design_')]
        else:
            task_mol = mol_cols
        print(f"\n=== {tname.upper()} (n={len(dft)}, pos={int(dft['_y'].sum())}, mol={len(task_mol)}) ===", flush=True)
        r = run_task(dft, task_mol, tname)
        res[tname] = r
        for v in VARIANTS:
            print(f"  {v:3s} {r[v]:.4f}")
        print(f"  -- disease-difficulty over flags (D - F)  = {r['D']-r['F']:+.4f}")
        print(f"  -- molecular over disease       (DM - D)  = {r['DM']-r['D']:+.4f}")

    print("\n======== v8 DECOMPOSITION SUMMARY (mean-of-folds AUC, single-GBM) ========")
    print(f"{'task':9s} {'M':>7s} {'F':>7s} {'D':>7s} {'DM':>7s}  {'D-F':>8s} {'DM-D':>8s}")
    for tname in tasks:
        r = res[tname]
        print(f"{tname:9s} {r['M']:7.4f} {r['F']:7.4f} {r['D']:7.4f} {r['DM']:7.4f}  "
              f"{r['D']-r['F']:+8.4f} {r['DM']-r['D']:+8.4f}")


if __name__ == "__main__":
    main()
