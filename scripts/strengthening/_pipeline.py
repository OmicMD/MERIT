"""Shared pipeline utilities for manuscript strengthening analyses.

Mirrors scripts/retrain_corrected.py but:
- Returns out-of-fold predictions (needed for calibration, PR, risk zones).
- Uses GBM-only across all tasks (no XGB/LGBM ensemble) for speed and
  cross-analysis consistency. Documented in SUMMARY.md.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / 'data' / 'sources' / 'training_dataset_v5_unified.csv'
OUT_ROOT = BASE_DIR / 'results' / 'strengthening'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]

META_COLS = {'SMILES', 'NCT_ID', 'Drug_Clean', 'Disease', 'Phase',
             'Final_Outcome', 'Corrected_Outcome', 'Unnamed: 0',
             'reconciled_label', 'confidence', 'n_sources', 'label_source',
             'outcome', 'reason', 'chembl_id', 'first_approval',
             'max_phase_chembl', 'withdrawn_flag', 'black_box_warning',
             'pref_name', 'FDA_APPROVED', 'CT_TOX',
             'pass_votes', 'fail_safety_votes', 'fail_efficacy_votes',
             'fail_unknown_votes', 'chembl_label', 'repodb_label',
             'AMES', 'BBB_Martins', 'DILI', 'HIA_Hou', 'PAMPA_NCATS',
             'Pgp_Broccatelli', 'hERG', 'Caco2_Wang', 'LD50_Zhu',
             'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ',
             'PPBR_AZ', 'VDss_Lombardo',
             'CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith',
             'CYP2D6_Veith', 'CYP3A4_Veith',
             'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels',
             'CYP3A4_Substrate_CarbonMangels',
             'NR-AR-LBD', 'NR-AR', 'NR-AhR', 'NR-Aromatase',
             'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma',
             'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
             'OpenTargets_N_Targets', 'Disease_N_Targets',
             'clintox_label', 'Start_Year', 'Enrollment',
             'is_anti_pathogen', 'is_endogenous', 'is_mispaired_supportive',
             'is_healthy_volunteer', 'is_procedural_exclude',
             'is_multi_drug_exclude', 'Source', 'feature_IK', 'Is_Biologic',
             'Drug', 'Confidence', 'matched_approved_name', 'Why_Stopped',
             'Trial_Title'}


def load_dataset():
    return pd.read_csv(DATA_PATH, low_memory=False)


def get_features(df):
    feats = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if 'drugbank_approved_percentile' in c or 'median_' in c or 'max_phase' in c:
            continue
        if df[c].dtype in ('float64', 'float32', 'int64', 'int32'):
            feats.append(c)
    return feats


def task_subset(df, task):
    """Return df_task, y for one of {'overall','safety','efficacy'}.

    Mirrors retrain_corrected.py exclusions exactly.
    """
    if task == 'overall':
        m = df['Corrected_Outcome'].isin(['PASS', 'FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH'])
        df_t = df[m].copy()
        y = df_t['Corrected_Outcome'].isin(['FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH']).astype(int).values
    elif task == 'safety':
        m = df['Corrected_Outcome'].isin(['PASS', 'FAIL_SAFETY', 'FAIL_BOTH'])
        if 'is_multi_drug_exclude' in df.columns:
            m &= df['is_multi_drug_exclude'] != 1
        df_t = df[m].copy()
        y = df_t['Corrected_Outcome'].isin(['FAIL_SAFETY', 'FAIL_BOTH']).astype(int).values
    elif task == 'efficacy':
        excl = pd.Series(False, index=df.index)
        for col in ['is_anti_pathogen', 'is_endogenous', 'is_mispaired_supportive',
                    'is_healthy_volunteer', 'is_procedural_exclude', 'is_multi_drug_exclude']:
            if col in df.columns:
                excl |= df[col] == 1
        m = (~excl) & df['Corrected_Outcome'].isin(['PASS', 'FAIL_EFFICACY', 'FAIL_BOTH'])
        df_t = df[m].copy()
        y = df_t['Corrected_Outcome'].isin(['FAIL_EFFICACY', 'FAIL_BOTH']).astype(int).values
    else:
        raise ValueError(task)
    return df_t, y


def disease_encoding(d_train, y_train, d_test, smoothing_n=3):
    g = float(np.mean(y_train))
    rates = {}
    for d, v in zip(d_train, y_train):
        rates.setdefault(d, []).append(v)
    enc_train = np.zeros(len(d_train))
    for i, d in enumerate(d_train):
        v = rates[d]
        n = len(v)
        enc_train[i] = (n * np.mean(v) + smoothing_n * g) / (n + smoothing_n)
    enc_test = np.zeros(len(d_test))
    for i, d in enumerate(d_test):
        if d in rates:
            v = rates[d]
            n = len(v)
            enc_test[i] = (n * np.mean(v) + smoothing_n * g) / (n + smoothing_n)
        else:
            enc_test[i] = g
    return enc_train, enc_test


def select_top_k(X_train, y_train, top_k=20):
    aucs = np.full(X_train.shape[1], 0.5)
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        if np.std(col) == 0:
            continue
        try:
            a = roc_auc_score(y_train, col)
            aucs[j] = max(a, 1 - a)
        except ValueError:
            pass
    return np.argsort(aucs)[-top_k:]


def fit_predict_fold(X_train, y_train, X_test, d_train, d_test, feature_names,
                     seed, top_k=20, use_disease_encoding=True):
    """Return test probs and selected feature names for one fold."""
    imp = SimpleImputer(strategy='median')
    X_tr = imp.fit_transform(X_train)
    X_te = imp.transform(X_test)
    top_idx = select_top_k(X_tr, y_train, top_k=top_k)
    X_tr_s = X_tr[:, top_idx]
    X_te_s = X_te[:, top_idx]
    if use_disease_encoding:
        enc_tr, enc_te = disease_encoding(d_train, y_train, d_test)
        X_tr_f = np.column_stack([X_tr_s, enc_tr])
        X_te_f = np.column_stack([X_te_s, enc_te])
    else:
        X_tr_f, X_te_f = X_tr_s, X_te_s
    w = compute_sample_weight('balanced', y_train)
    gbm = GradientBoostingClassifier(n_estimators=500, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=seed)
    gbm.fit(X_tr_f, y_train, sample_weight=w)
    proba = gbm.predict_proba(X_te_f)[:, 1]
    return proba, [feature_names[i] for i in top_idx]


def cv_oof(X, y, groups, diseases, feature_names, seeds=SEEDS, top_k=20,
           use_disease_encoding=True, verbose=False):
    """Run StratifiedGroupKFold over multiple seeds; return per-seed OOF probs.

    Returns dict with:
      seed_oof: dict[seed] -> np.array of OOF probs (len = len(y))
      mean_oof: np.array averaged across seeds
      fold_aucs: list of all per-fold AUCs (len = len(seeds)*5)
      selected_features: list[list[str]] per fold
    """
    n = len(y)
    seed_oof = {}
    fold_aucs = []
    selected = []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.full(n, np.nan)
        for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
            assert len(set(groups[tr]) & set(groups[te])) == 0
            proba, feats = fit_predict_fold(
                X[tr], y[tr], X[te], diseases[tr], diseases[te],
                feature_names, seed, top_k=top_k,
                use_disease_encoding=use_disease_encoding)
            oof[te] = proba
            try:
                a = roc_auc_score(y[te], proba)
            except ValueError:
                a = np.nan
            fold_aucs.append(a)
            selected.append(feats)
            if verbose:
                print(f'    seed={seed} fold={fold} auc={a:.3f}', flush=True)
        seed_oof[seed] = oof
    mean_oof = np.nanmean(np.stack(list(seed_oof.values())), axis=0)
    return {
        'seed_oof': seed_oof,
        'mean_oof': mean_oof,
        'fold_aucs': fold_aucs,
        'selected_features': selected,
    }


def evaluate_holdout(X_train, y_train, g_train, d_train,
                    X_test, y_test, d_test, feature_names,
                    seeds=SEEDS, top_k=20, use_disease_encoding=True):
    """Train on (X_train, y_train) over multiple seeds, average preds on test."""
    preds = []
    for seed in seeds:
        proba, _ = fit_predict_fold(
            X_train, y_train, X_test, d_train, d_test, feature_names,
            seed, top_k=top_k, use_disease_encoding=use_disease_encoding)
        preds.append(proba)
    mean_pred = np.mean(preds, axis=0)
    try:
        auc = roc_auc_score(y_test, mean_pred)
    except ValueError:
        auc = float('nan')
    return mean_pred, auc, preds
