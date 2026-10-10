#!/usr/bin/env python3
"""
Traditional ML baselines (random forest, logistic regression) against the
production gradient-boosting head, under the identical compound-holdout
protocol: same cohorts, labels, folds, imputation, nested feature selection
and within-fold disease encoding. Only the classifier changes.

Outputs results/strengthening/baseline_models.csv.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, \
    precision_score, recall_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrain_corrected import (
    get_features, compute_disease_encoding, nested_feature_selection,
    SEEDS, DATA_PATH, BASE_DIR,
)

OUT = BASE_DIR / 'results' / 'strengthening'
OUT.mkdir(parents=True, exist_ok=True)


def make_model(name: str, seed: int):
    if name == 'gradient_boosting':
        return GradientBoostingClassifier(n_estimators=500, max_depth=3,
                                          learning_rate=0.05, subsample=0.8,
                                          random_state=seed), False
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=500, max_depth=None,
                                      min_samples_leaf=2, n_jobs=-1,
                                      class_weight='balanced',
                                      random_state=seed), False
    if name == 'logistic_regression':
        return LogisticRegression(max_iter=5000, class_weight='balanced',
                                  random_state=seed), True
    raise ValueError(name)


def run(name, X, y, groups, diseases, feature_names):
    """Cross-validated scores for one classifier on one task."""
    rows = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
            assert not (set(groups[tr]) & set(groups[te])), 'Drug leakage!'
            imp = SimpleImputer(strategy='median')
            Xtr = imp.fit_transform(X[tr])
            Xte = imp.transform(X[te])
            top = nested_feature_selection(Xtr, y[tr], feature_names)
            enc_tr, enc_te = compute_disease_encoding(diseases[tr], y[tr], diseases[te])
            Xtr = np.column_stack([Xtr[:, top], enc_tr])
            Xte = np.column_stack([Xte[:, top], enc_te])

            model, needs_scaling = make_model(name, seed)
            if needs_scaling:
                sc = StandardScaler()
                Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
            if name == 'gradient_boosting':
                model.fit(Xtr, y[tr], sample_weight=compute_sample_weight('balanced', y[tr]))
            else:
                model.fit(Xtr, y[tr])
            p = model.predict_proba(Xte)[:, 1]

            thr = y[tr].mean()
            yp = (p >= thr).astype(int)
            rows.append(dict(
                model=name, seed=seed, fold=fold,
                auc=roc_auc_score(y[te], p), pr_auc=average_precision_score(y[te], p),
                accuracy=accuracy_score(y[te], yp),
                balanced_accuracy=balanced_accuracy_score(y[te], yp),
                f1=f1_score(y[te], yp, zero_division=0),
                precision=precision_score(y[te], yp, zero_division=0),
                recall=recall_score(y[te], yp, zero_division=0),
                mcc=matthews_corrcoef(y[te], yp)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    feature_cols = get_features(df)

    eff_excl = pd.Series(False, index=df.index)
    for col in ['is_anti_pathogen', 'is_endogenous', 'is_mispaired_supportive',
                'is_healthy_volunteer', 'is_procedural_exclude', 'is_multi_drug_exclude']:
        if col in df.columns:
            eff_excl |= df[col] == 1

    tasks = {
        'overall': df[df['Corrected_Outcome'].isin(
            ['PASS', 'FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH'])],
        'efficacy': df[~eff_excl & df['Corrected_Outcome'].isin(
            ['PASS', 'FAIL_EFFICACY', 'FAIL_BOTH'])],
        'safety': df[df['Corrected_Outcome'].isin(
            ['PASS', 'FAIL_SAFETY', 'FAIL_BOTH'])],
    }
    fail_labels = {
        'overall': ['FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH'],
        'efficacy': ['FAIL_EFFICACY', 'FAIL_BOTH'],
        'safety': ['FAIL_SAFETY', 'FAIL_BOTH'],
    }

    out = []
    for task, d in tasks.items():
        y = d['Corrected_Outcome'].isin(fail_labels[task]).astype(int).values
        X = d[feature_cols].values
        g = d['SMILES'].values
        dis = d['Disease'].fillna('unknown').values
        print(f'{task}: {len(d)} trials, {y.sum()} failures', flush=True)
        for name in ['gradient_boosting', 'random_forest', 'logistic_regression']:
            r = run(name, X, y, g, dis, feature_cols)
            r['task'] = task
            out.append(r)
            print(f'  {name}: AUROC {r.auc.mean():.3f} +/- {r.auc.std():.3f}', flush=True)

    folds = pd.concat(out, ignore_index=True)
    folds.to_csv(OUT / 'baseline_models_folds.csv', index=False)

    metrics = ['auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'f1',
               'precision', 'recall', 'mcc']
    summary = folds.groupby(['task', 'model'])[metrics].agg(['mean', 'std'])
    summary.to_csv(OUT / 'baseline_models.csv')
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
