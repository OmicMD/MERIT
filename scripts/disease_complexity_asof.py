#!/usr/bin/env python3
"""
Reviewer-requested check on the disease-complexity feature: rebuild it so that
each trial's indication failure rate uses ONLY trials that started strictly
before that trial began, in addition to the existing training-fold restriction.

The canonical feature is already fold-internal (no test trial informs it), but
a training-fold trial may still post-date the test trial it helps score. This
script re-runs the production protocol with an as-of-date encoding and reports
the AUROC change.

Outputs results/strengthening/disease_complexity_asof.csv.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrain_corrected import (
    get_features, compute_disease_encoding, nested_feature_selection,
    SEEDS, DATA_PATH, BASE_DIR,
)

OUT = BASE_DIR / 'results' / 'strengthening'
OUT.mkdir(parents=True, exist_ok=True)
SMOOTH = 3


def asof_disease_encoding(d_train, y_train, yr_train, d_test, yr_test):
    """Disease failure rate from training trials that began strictly earlier.

    Each row is encoded using only training-fold trials in the same indication
    whose start year precedes that row's own start year, so no future trial
    contributes to any row's complexity value.
    """
    global_rate = y_train.mean()
    order = np.argsort(yr_train, kind='stable')
    d_sorted, y_sorted, yr_sorted = d_train[order], y_train[order], yr_train[order]

    def encode(diseases, years):
        enc = np.empty(len(diseases))
        for i, (d, yr) in enumerate(zip(diseases, years)):
            prior = (d_sorted == d) & (yr_sorted < yr)
            n = int(prior.sum())
            if n == 0:
                enc[i] = global_rate
            else:
                raw = y_sorted[prior].mean()
                enc[i] = (n * raw + SMOOTH * global_rate) / (n + SMOOTH)
        return enc

    return encode(d_train, yr_train), encode(d_test, yr_test)


def run(X, y, groups, diseases, years, feature_names, mode):
    aucs = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, groups):
            imp = SimpleImputer(strategy='median')
            Xtr = imp.fit_transform(X[tr])
            Xte = imp.transform(X[te])
            top = nested_feature_selection(Xtr, y[tr], feature_names)
            if mode == 'canonical':
                e_tr, e_te = compute_disease_encoding(diseases[tr], y[tr], diseases[te])
            else:
                e_tr, e_te = asof_disease_encoding(diseases[tr], y[tr], years[tr],
                                                   diseases[te], years[te])
            Xtr = np.column_stack([Xtr[:, top], e_tr])
            Xte = np.column_stack([Xte[:, top], e_te])
            m = GradientBoostingClassifier(n_estimators=500, max_depth=3,
                                           learning_rate=0.05, subsample=0.8,
                                           random_state=seed)
            m.fit(Xtr, y[tr], sample_weight=compute_sample_weight('balanced', y[tr]))
            aucs.append(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1]))
    return np.array(aucs)


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    feature_cols = get_features(df)

    eff_excl = pd.Series(False, index=df.index)
    for col in ['is_anti_pathogen', 'is_endogenous', 'is_mispaired_supportive',
                'is_healthy_volunteer', 'is_procedural_exclude', 'is_multi_drug_exclude']:
        if col in df.columns:
            eff_excl |= df[col] == 1

    tasks = {
        'overall': (df[df['Corrected_Outcome'].isin(
            ['PASS', 'FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH'])],
            ['FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH']),
        'efficacy': (df[~eff_excl & df['Corrected_Outcome'].isin(
            ['PASS', 'FAIL_EFFICACY', 'FAIL_BOTH'])],
            ['FAIL_EFFICACY', 'FAIL_BOTH']),
    }

    rows = []
    for task, (d, fails) in tasks.items():
        d = d[d['Start_Year'].notna()].copy()
        y = d['Corrected_Outcome'].isin(fails).astype(int).values
        X = d[feature_cols].values
        g = d['SMILES'].values
        dis = d['Disease'].fillna('unknown').values
        yr = d['Start_Year'].values.astype(float)
        print(f'{task}: {len(d)} trials, {y.sum()} failures', flush=True)
        for mode in ['canonical', 'asof']:
            a = run(X, y, g, dis, yr, feature_cols, mode)
            rows.append(dict(task=task, encoding=mode, auc_mean=a.mean(),
                             auc_sd=a.std(), n_folds=len(a)))
            print(f'  {mode}: {a.mean():.4f} +/- {a.std():.4f}', flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'disease_complexity_asof.csv', index=False)
    print(out.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
