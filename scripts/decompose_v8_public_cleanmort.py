#!/usr/bin/env python3
"""MAJOR 3 robustness: does the efficacy molecular increment survive with
PUBLIC-derivable molecular features only (no proprietary binding/tissue/tox)?

Mirrors decompose_v8_noclass.py exactly (same FLAGS, single-GBM, 5x5 SGKF by
SMILES, nested top-20, within-fold disease encoding) but swaps the molecular
pool for several public-only subsets. Reports D (flags+disease prior) and
DM (+molecular) and the molecular increment DM-D, for efficacy and overall.

Public subsets (clean_mort: the blind causal-plausibility signal lives in the mech_* columns,
which replaced the single causal_plausibility_blind feature dropped at build time):
  FULL          : all molecular features (reproduces the manuscript decomposition)
  PUBLIC        : mech_* + net_* (STRING/OpenTargets pathway) + drumap_* (public PK)
  PUB_NO_PK     : mech_* + net_*
  PLAUS_ONLY    : mech_* only

NOT YET RE-RUN ON clean_mort (Jul 2026). This script supersedes decompose_v8_public.py
(deleted), which read the honest_exposure cohort and filtered for causal_plausibility_blind
-- a column build_v8_honest_exposure.py DROPS at build time, so its plausibility subsets were
silently empty. The committed values still quoted in notebooks/03_supporting_analyses.ipynb
(efficacy public-only +0.119 vs full +0.113; plausibility-only +0.080) come from that OLD
script on the OLD cohort and are NOT this script's output. Re-run to obtain current numbers.

No manuscript number depends on them: the paper claims only that a public-only feature set
"recovers most of the efficacy increment" (Results; Code availability), with no figure quoted.
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

DATA = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"  # clean_mort migration (Jun 15)
SEEDS = [42, 123, 456, 789, 2024]
FLAGS = ['disease_is_oncology', 'disease_is_infectious', 'disease_is_cns',
         'disease_is_cardiac', 'disease_is_autoimmune', 'disease_is_metastatic',
         'disease_is_transplant', 'disease_is_severe', 'is_combination',
         'trial_n_drugs']


def public_subsets(mol_cols):
    # clean_mort: the name-blinded causal-plausibility signal lives in the 4 mech_* columns
    # (causal_centrality_jun12 blind determination), replacing single causal_plausibility_blind.
    plaus = [c for c in mol_cols if c.startswith("mech_")]
    net = [c for c in mol_cols if c.startswith("net_")]
    pk = [c for c in mol_cols if c.startswith("drumap_")]
    return {
        "FULL": mol_cols,
        "PUBLIC": plaus + net + pk,
        "PUB_NO_PK": plaus + net,
        "PLAUS_ONLY": plaus,
    }


def run_task(df_task, mol_cols, task):
    flags = [c for c in FLAGS if c in df_task.columns]
    subsets = public_subsets(mol_cols)
    F = df_task[flags].values.astype(float)
    y = df_task["_y"].values
    grp = df_task["SMILES"].values
    dis = df_task["Disease"].fillna("unknown").values
    # per subset: D (flags+denc) is identical; track DM per subset, D once
    pf_D = {}
    pf_DM = {s: {} for s in subsets}
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(F, y, grp)):
            if len(np.unique(y[te])) < 2:
                continue
            fimp = SimpleImputer(strategy="median")
            Ftr = fimp.fit_transform(F[tr]); Fte = fimp.transform(F[te])
            de_tr, de_te = compute_disease_encoding(dis[tr], y[tr], dis[te])
            de_tr, de_te = de_tr.reshape(-1, 1), de_te.reshape(-1, 1)
            predD = fit_predict(np.column_stack([de_tr, Ftr]), y[tr],
                                np.column_stack([de_te, Fte]), task, seed)
            pf_D[(seed, fold)] = roc_auc_score(y[te], predD)
            for sname, cols in subsets.items():
                if not cols:
                    continue
                X = df_task[cols].values
                imp = SimpleImputer(strategy="median")
                Xtr = imp.fit_transform(X[tr]); Xte = imp.transform(X[te])
                top = nested_feature_selection(Xtr, y[tr], cols)
                Mtr, Mte = Xtr[:, top], Xte[:, top]
                pred = fit_predict(np.column_stack([de_tr, Ftr, Mtr]), y[tr],
                                   np.column_stack([de_te, Fte, Mte]), task, seed)
                pf_DM[sname][(seed, fold)] = roc_auc_score(y[te], pred)
    D = np.mean(list(pf_D.values()))
    DM = {s: (np.mean(list(d.values())) if d else float("nan")) for s, d in pf_DM.items()}
    return D, DM, {s: len(c) for s, c in subsets.items()}


def main():
    df = pd.read_csv(DATA, low_memory=False)
    mol_cols = [c for c in get_features(df) if c not in FLAGS]

    excl = pd.Series(False, index=df.index)
    for c in ("is_anti_pathogen", "is_endogenous", "is_mispaired_supportive",
              "is_healthy_volunteer", "is_procedural_exclude", "is_multi_drug_exclude"):
        if c in df.columns:
            excl |= df[c] == 1
    de = df[~excl & df["Corrected_Outcome"].isin(["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    de["_y"] = de["Corrected_Outcome"].isin(["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)

    do = df[df["Corrected_Outcome"].isin(["PASS", "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    do["_y"] = do["Corrected_Outcome"].isin(["FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)

    for tname, dft in [("efficacy", de), ("overall", do)]:
        D, DM, sizes = run_task(dft, mol_cols, tname)
        print(f"\n=== {tname.upper()} (n={len(dft)}, pos={int(dft['_y'].sum())}) ===")
        print(f"  D (flags+disease prior) = {D:.4f}")
        print(f"  {'subset':12s} {'ncols':>5s} {'DM':>8s} {'DM-D':>8s}")
        for s in ["FULL", "PUBLIC", "PUB_NO_PK", "PLAUS_ONLY"]:
            print(f"  {s:12s} {sizes[s]:5d} {DM[s]:8.4f} {DM[s]-D:+8.4f}")


if __name__ == "__main__":
    main()
