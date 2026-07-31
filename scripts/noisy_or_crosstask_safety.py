#!/usr/bin/env python3
"""Does a noisy-OR safety head improve the CROSS-TASK (overall-cohort) safety axis
used by the risk-zone triage?

Production triage scores safety on the full overall cohort by training a single
GBM on the safety subset of each outer-train fold and predicting on the full
outer-test (run_overall_crosstask in retrain_calibrated.py) -> safety AUC 0.648,
below the per-task noisy-OR head (0.716). This script swaps ONLY that head:
within the identical overall-cohort folds it trains, on the safety subset of
outer-train, (a) the production single GBM and (b) a noisy-OR over per-mechanism
detectors, predicts both on the full outer-test, and compares the safety-axis AUC
(evaluated on the safety-cohort test rows, label y_safety).

Does NOT modify the locked production pipeline; imports its helpers verbatim.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrain_calibrated import (mechanism_groups, prepare_fold, fit_predict)
from retrain_corrected import get_features, SEEDS

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    'data/sources/training_dataset_v8_honest_exposure.csv'


def noisy_or_fit_predict(Xtr_raw, ytr, dtr, Xte_raw, dte, feature_cols,
                         protect_cols, seed):
    """Fit per-mechanism GBM detectors on (Xtr,ytr), predict on Xte, combine by
    noisy-OR. Mirrors noisy_or_safety_oof's per-group construction (each detector
    gets its own top-k selection + disease encoding + within-group exposure
    protection), but in fit/predict mode for cross-task scoring."""
    groups = mechanism_groups(feature_cols)
    probs = []
    for g, cs in groups.items():
        col_idx = [feature_cols.index(c) for c in cs]
        prot_in_g = [cs.index(c) for c in protect_cols if c in cs]
        Xt, Xe, _, _ = prepare_fold(
            Xtr_raw[:, col_idx], Xte_raw[:, col_idx], ytr, dtr, dte, cs,
            protect_idx=(prot_in_g or None))
        probs.append(fit_predict(Xt, ytr, Xe, 'safety', seed))
    M = np.clip(np.column_stack(probs), 0, 0.999)
    return 1.0 - np.prod(1.0 - M, axis=1)


def single_fit_predict(Xtr_raw, ytr, dtr, Xte_raw, dte, feature_cols,
                       protect_cols, seed, protect=False):
    pidx = ([feature_cols.index(c) for c in protect_cols if c in feature_cols]
            if protect else None)
    Xt, Xe, _, _ = prepare_fold(Xtr_raw, Xte_raw, ytr, dtr, dte, feature_cols,
                                protect_idx=pidx)
    return fit_predict(Xt, ytr, Xe, 'safety', seed)


def main():
    df = pd.read_csv(DATA, low_memory=False)
    feature_cols = get_features(df)
    print(f'{DATA}\nTrials={len(df)} Drugs={df.SMILES.nunique()} '
          f'Feats={len(feature_cols)}', flush=True)

    df_over = df[df['Corrected_Outcome'].isin(
        ['PASS', 'FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH'])].copy().reset_index(drop=True)
    # safety-cohort membership (mirrors retrain_calibrated main): PASS/SAFETY/BOTH,
    # minus multi-drug-exclude rows.
    df_over['_is_safety'] = df_over['Corrected_Outcome'].isin(
        ['PASS', 'FAIL_SAFETY', 'FAIL_BOTH']).astype(int)
    if 'is_multi_drug_exclude' in df_over.columns:
        df_over.loc[df_over['is_multi_drug_exclude'] == 1, '_is_safety'] = 0
    df_over['_y'] = df_over['Corrected_Outcome'].isin(
        ['FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH']).astype(int)
    df_over['_y_safety'] = df_over['Corrected_Outcome'].isin(
        ['FAIL_SAFETY', 'FAIL_BOTH']).astype(int)

    X = df_over[feature_cols].values
    y_all = df_over['_y'].values
    groups = df_over['SMILES'].values
    diseases = df_over['Disease'].fillna('unknown').values
    is_saf = df_over['_is_safety'].values.astype(bool)
    y_saf = df_over['_y_safety'].values
    protect = [c for c in feature_cols if c == 'logdose' or c.endswith('_xdose')]
    print(f'overall n={len(df_over)} pos={y_all.sum()} | '
          f'safety-cohort rows={is_saf.sum()} safety-pos={y_saf[is_saf].sum()} | '
          f'protect={protect}', flush=True)

    fold_rows = []
    pooled = {k: [] for k in ['idx', 'y', 'single', 'single_prot', 'nor']}
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X, y_all, groups)):
            tr_s = tr[is_saf[tr]]
            yts = y_saf[tr_s]
            if len(tr_s) < 10 or len(np.unique(yts)) < 2:
                continue
            te_s = te[is_saf[te]]
            if len(te_s) < 5 or len(np.unique(y_saf[te_s])) < 2:
                continue
            p_single = single_fit_predict(X[tr_s], yts, diseases[tr_s], X[te],
                                          diseases[te], feature_cols, protect, seed)
            p_sprot = single_fit_predict(X[tr_s], yts, diseases[tr_s], X[te],
                                         diseases[te], feature_cols, protect, seed,
                                         protect=True)
            p_nor = noisy_or_fit_predict(X[tr_s], yts, diseases[tr_s], X[te],
                                         diseases[te], feature_cols, protect, seed)
            # evaluate on safety-cohort test rows
            sel = is_saf[te]
            ys = y_saf[te][sel]
            a_single = roc_auc_score(ys, p_single[sel])
            a_sprot = roc_auc_score(ys, p_sprot[sel])
            a_nor = roc_auc_score(ys, p_nor[sel])
            fold_rows.append(dict(seed=seed, fold=fold, n=int(sel.sum()),
                                  pos=int(ys.sum()), single=a_single,
                                  single_prot=a_sprot, noisy_or=a_nor))
            pooled['idx'].extend(te[sel]); pooled['y'].extend(ys)
            pooled['single'].extend(p_single[sel]); pooled['single_prot'].extend(p_sprot[sel])
            pooled['nor'].extend(p_nor[sel])
            print(f'  seed={seed} fold={fold} n={int(sel.sum())} pos={int(ys.sum())} '
                  f'single={a_single:.3f} single_prot={a_sprot:.3f} noisyOR={a_nor:.3f}',
                  flush=True)

    fm = pd.DataFrame(fold_rows)
    out = Path('results/noisy_or_crosstask_safety.csv'); out.parent.mkdir(exist_ok=True)
    fm.to_csv(out, index=False)
    py = np.array(pooled['y'])
    print('\n==== CROSS-TASK SAFETY AXIS (overall cohort) ====')
    print(f'folds={len(fm)}  safety positives (pooled OOF rows)={py.sum()}/{len(py)}')
    for col, lab in [('single', 'single GBM (production crosstask)'),
                     ('single_prot', 'single GBM + exposure protect'),
                     ('noisy_or', 'noisy-OR mechanism detectors')]:
        print(f'  {lab:38s}: mean-of-folds {fm[col].mean():.3f} ± {fm[col].std():.3f} '
              f'| pooled-OOF {roc_auc_score(py, np.array(pooled["single" if col=="single" else ("single_prot" if col=="single_prot" else "nor")])):.3f}')
    d = fm['noisy_or'] - fm['single']
    w = stats.wilcoxon(fm['noisy_or'], fm['single']).pvalue
    print(f'\n  noisy-OR vs single: mean-of-folds delta={d.mean():+.3f} '
          f'(median {d.median():+.3f}), Wilcoxon p={w:.3f}, '
          f'{(d>0).sum()}/{len(d)} folds favor noisy-OR')
    print('saved', out)


if __name__ == '__main__':
    main()
