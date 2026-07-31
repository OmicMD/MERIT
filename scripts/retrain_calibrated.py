#!/usr/bin/env python3
"""
Calibrated production pipeline (v2).

Adds to retrain_corrected.py:
  1. Isotonic calibration via nested 3-fold CV, applied to overall + efficacy only
     (safety left raw; Analysis 2 showed ECE already acceptable).
  2. Cross-task prediction: within overall's outer folds, safety and efficacy
     models are also trained (on their task-specific subsets of outer-train)
     and predicted on the full outer-test. Fixes Analysis 3 intersection bug.

Outputs under results/production_v2/:
  oof_overall.parquet, oof_safety.parquet, oof_efficacy.parquet
  oof_overall_crosstask.parquet  (overall cohort rows, all three probs)
  metrics.json, fold_metrics.csv

The original retrain_corrected.py is left untouched for reviewer reproduction.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# Reuse helpers from the production script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrain_corrected import (  # noqa: E402
    get_features, compute_disease_encoding, nested_feature_selection,
    SEEDS, DATA_PATH, BASE_DIR,
)

OUT_DIR = BASE_DIR / 'results' / 'production_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model fit / predict primitives
# ---------------------------------------------------------------------------
def fit_gbm(X, y, seed):
    w = compute_sample_weight('balanced', y)
    m = GradientBoostingClassifier(n_estimators=500, max_depth=3,
                                   learning_rate=0.05, subsample=0.8,
                                   random_state=seed)
    m.fit(X, y, sample_weight=w)
    return m


def fit_ensemble_eff(X, y, seed):
    """GBM + XGB + LGBM ensemble used for efficacy in production."""
    models = [fit_gbm(X, y, seed)]
    w = compute_sample_weight('balanced', y)
    if HAS_XGB:
        try:
            xgb = XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05,
                                subsample=0.8, random_state=seed,
                                eval_metric='logloss', tree_method='hist',
                                device='cuda')
            xgb.fit(X, y, sample_weight=w)
        except Exception:
            xgb = XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05,
                                subsample=0.8, random_state=seed,
                                eval_metric='logloss')
            xgb.fit(X, y, sample_weight=w)
        models.append(xgb)
    if HAS_LGBM:
        try:
            lgbm = LGBMClassifier(n_estimators=500, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=seed, verbose=-1,
                                  device='gpu')
            lgbm.fit(X, y, sample_weight=w)
        except Exception:
            lgbm = LGBMClassifier(n_estimators=500, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=seed, verbose=-1)
            lgbm.fit(X, y, sample_weight=w)
        models.append(lgbm)
    return models


def predict_ensemble(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def prepare_fold(X_train_raw, X_test_raw, y_train, d_train, d_test, feature_names,
                 protect_idx=None):
    """Impute + top-k + disease encoding, applied within a training subset.

    protect_idx: column indices (into feature_names) that bypass top-k selection and
    are always retained — for weak-univariate / high-interaction mechanistic features
    (e.g. exposure: logdose, off-target*dose) that top-20 selection would otherwise drop.
    """
    imputer = SimpleImputer(strategy='median')
    X_tr = imputer.fit_transform(X_train_raw)
    X_te = imputer.transform(X_test_raw)
    top_idx = nested_feature_selection(X_tr, y_train, feature_names)
    if protect_idx is not None and len(protect_idx):
        top_idx = np.union1d(top_idx, np.asarray(protect_idx, dtype=int))
    X_tr_sel = X_tr[:, top_idx]
    X_te_sel = X_te[:, top_idx]
    enc_tr, enc_te = compute_disease_encoding(d_train, y_train, d_test)
    X_tr_final = np.column_stack([X_tr_sel, enc_tr])
    X_te_final = np.column_stack([X_te_sel, enc_te])
    return X_tr_final, X_te_final, imputer, top_idx


def fit_predict(X_tr, y_tr, X_te, task, seed):
    if task == 'efficacy':
        mods = fit_ensemble_eff(X_tr, y_tr, seed)
        return predict_ensemble(mods, X_te)
    mod = fit_gbm(X_tr, y_tr, seed)
    return mod.predict_proba(X_te)[:, 1]


def fit_predict_gbm_only(X_tr, y_tr, X_te, seed):
    """GBM-only variant (for efficacy AUC-gap diagnostic)."""
    mod = fit_gbm(X_tr, y_tr, seed)
    return mod.predict_proba(X_te)[:, 1]


# ---------------------------------------------------------------------------
# Nested-CV isotonic calibrator
# ---------------------------------------------------------------------------
def fit_isotonic_nested(X_raw, y, groups, diseases, feature_names, task, seed,
                        n_inner=3):
    """Fit IsotonicRegression on OOF predictions from a 3-fold GroupKFold
    over the outer-training fold. Returns the fitted calibrator."""
    inner_cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True,
                                    random_state=seed + 10000)
    oof_raw = np.zeros(len(y))
    oof_y = np.zeros(len(y), dtype=int)
    mask = np.zeros(len(y), dtype=bool)
    for tr, te in inner_cv.split(X_raw, y, groups):
        Xt, Xe, _, _ = prepare_fold(X_raw[tr], X_raw[te], y[tr],
                                    diseases[tr], diseases[te], feature_names)
        oof_raw[te] = fit_predict(Xt, y[tr], Xe, task, seed)
        oof_y[te] = y[te]
        mask[te] = True
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_raw[mask], oof_y[mask])
    return iso


# ---------------------------------------------------------------------------
# Per-task OOF producer (reproduces published numbers)
# ---------------------------------------------------------------------------
def run_task_cv(df_task, feature_cols, task, calibrate, protect_cols=None):
    X_raw = df_task[feature_cols].values
    y = df_task['_y'].values
    groups = df_task['SMILES'].values
    diseases = df_task['Disease'].fillna('unknown').values
    idx_out = df_task.index.values
    protect_idx = ([feature_cols.index(c) for c in protect_cols if c in feature_cols]
                   if protect_cols else None)
    if protect_idx:
        print(f'  [{task}] protecting {len(protect_idx)} features from selection: '
              f'{[c for c in protect_cols if c in feature_cols]}', flush=True)

    rows = []
    fold_metrics = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X_raw, y, groups)):
            assert not (set(groups[tr]) & set(groups[te])), 'drug leakage'
            X_tr, X_te, _, _ = prepare_fold(X_raw[tr], X_raw[te], y[tr],
                                            diseases[tr], diseases[te],
                                            feature_cols, protect_idx=protect_idx)
            raw_te = fit_predict(X_tr, y[tr], X_te, task, seed)
            if calibrate:
                iso = fit_isotonic_nested(X_raw[tr], y[tr], groups[tr],
                                          diseases[tr], feature_cols,
                                          task, seed)
                cal_te = iso.transform(raw_te)
            else:
                cal_te = raw_te.copy()
            for j, pos in enumerate(te):
                rows.append((idx_out[pos], int(y[pos]), raw_te[j], cal_te[j],
                             seed, fold, groups[pos], diseases[pos]))
            try:
                fa_raw = roc_auc_score(y[te], raw_te)
                fa_cal = roc_auc_score(y[te], cal_te)
            except ValueError:
                fa_raw = fa_cal = np.nan
            fold_metrics.append({'task': task, 'seed': seed, 'fold': fold,
                                 'n': len(te), 'auc_raw': fa_raw,
                                 'auc_cal': fa_cal})
            print(f'  {task} seed={seed} fold={fold} AUC_raw={fa_raw:.3f} '
                  f'AUC_cal={fa_cal:.3f}', flush=True)

    oof = pd.DataFrame(rows, columns=['row_idx', 'y', 'raw_prob',
                                      'calibrated_prob', 'seed', 'fold',
                                      'SMILES', 'Disease'])
    return oof, pd.DataFrame(fold_metrics)


# ---------------------------------------------------------------------------
# Noisy-OR mechanism-detector safety head (canonical location; arm-level imports
# mechanism_groups from here). Decomposes the safety feature set into biological
# mechanism detectors, trains each as its own GBM (so the dominant promiscuity
# signal can't wash out minority detectors), and combines parameter-free by
# noisy-OR P=1-prod(1-p_m) ("fail if ANY mechanism trips"). Lifts safety AUC
# +0.020 over the single GBM, replicated in two independent cohorts (arm-level
# 144 pos, 4/5 seeds; trial-level 97 pos, 4/5 seeds). See
# notes/safety_figured_out_jun7.md and notes/metab_dili_redirect_jun7.md.
# ---------------------------------------------------------------------------
def mechanism_groups(feature_cols):
    groups = {
        'promiscuity': [c for c in feature_cols
                        if c.startswith('binding_')
                        or (c.startswith('tox_') and 'hepatic' not in c and 'cardiac' not in c)
                        or c.startswith('essential_')
                        or c in ('tdc_ames', 'tdc_carcinogen')],  # external genotox classifiers
        'hepatic_dili': [c for c in feature_cols if 'hepatic' in c or c.startswith('drumap_')
                         or c == 'tdc_dili'        # external structure->DILI classifier (TDC ground truth)
                         or c.startswith('dili_')   # dose x lipophilicity iDILI axis (rule-of-two, Chen)
                         or c.startswith('rmet_')],  # reactive-metabolite/haptenation structural alerts (jun14)
        'cardiac': [c for c in feature_cols if 'cardiac' in c],
        'network': [c for c in feature_cols if c.startswith('net_')],
        'tissue': [c for c in feature_cols
                   if c.endswith('_interaction') or c.startswith('weighted_score_')
                   or c.startswith('mean_') or c.startswith('min_')],
    }
    assigned = set().union(*groups.values()) if groups else set()
    groups['context'] = [c for c in feature_cols if c not in assigned]
    return {g: cs for g, cs in groups.items() if cs}


def noisy_or_safety_oof(df_safety, feature_cols, protect_cols=None,
                        return_detail=False):
    """Safety OOF via noisy-OR over per-mechanism detectors.

    Returns (oof, fold_metrics) in the SAME schema as run_task_cv so callers are
    unchanged. raw_prob = noisy-OR combined probability; calibrated_prob mirrors
    raw_prob (safety is uncalibrated). Each detector protects whichever exposure
    (dose-interaction) features fall into its group. With return_detail, also
    returns a per-detector OOF DataFrame for audit.
    """
    groups = mechanism_groups(feature_cols)
    exposure = protect_cols or []
    print(f'  [safety] noisy-OR over {len(groups)} mechanism detectors: '
          f'{ {g: len(cs) for g, cs in groups.items()} }', flush=True)
    key = ['row_idx', 'seed', 'fold']
    dets, meta = {}, None
    for g, cs in groups.items():
        prot = [c for c in exposure if c in cs] or None
        oof_g, _ = run_task_cv(df_safety.copy(), cs, 'safety',
                               calibrate=False, protect_cols=prot)
        if meta is None:
            meta = oof_g[['row_idx', 'y', 'seed', 'fold',
                          'SMILES', 'Disease']].copy()
        dets[g] = oof_g.set_index(key)['raw_prob'].rename(g)
        det_auc = np.mean([roc_auc_score(gg.y, gg.raw_prob)
                           for _, gg in oof_g.groupby(['seed', 'fold'])
                           if gg.y.nunique() == 2])
        print(f'    detector {g:13s} ({len(cs):3d} feats) AUC {det_auc:.4f}',
              flush=True)
    M = pd.concat(list(dets.values()), axis=1)
    nor = (1.0 - (1.0 - M.clip(0, 0.999)).prod(axis=1)).rename('raw_prob').reset_index()
    oof = meta.merge(nor, on=key, how='left')

    # Honest-probability combiner (Jun 14, notes/calibration_result_jun14.md): the noisy-OR
    # (raw_prob) assumes detector INDEPENDENCE, but the 6 detectors are correlated (all read the
    # same molecule) -> it multiplies moderate risks into false certainty (the elagolix p=1.00
    # cry-wolf; 49% of the 181 cry-wolf had no single detector >0.7). A leak-aware cross-fold
    # LOGISTIC stacker over the per-detector OOF gives calibrated probabilities (caps ~0.5 at the
    # 3% base rate, cry-wolf-free) at ~-0.012 AUC. raw_prob KEEPS noisy-OR for the ranking/AUC
    # headline; calibrated_prob = logistic stack (use for any probability/threshold).
    from sklearn.linear_model import LogisticRegression
    detcols = list(M.columns)
    ML = M.reset_index().merge(oof[['row_idx', 'seed', 'fold', 'y']], on=key, how='left')
    ML['_stk'] = np.nan
    for sd, gs in ML.groupby('seed'):
        for fl in gs['fold'].unique():
            tr = gs[gs['fold'] != fl]
            te = (ML['seed'] == sd) & (ML['fold'] == fl)
            if tr['y'].nunique() < 2:
                continue
            lr = LogisticRegression(max_iter=1000).fit(tr[detcols].values, tr['y'].values)
            ML.loc[te, '_stk'] = lr.predict_proba(ML.loc[te, detcols].values)[:, 1]
    oof = oof.merge(ML[key + ['_stk']], on=key, how='left')
    oof['calibrated_prob'] = oof['_stk'].fillna(oof['raw_prob'])
    oof = oof[['row_idx', 'y', 'raw_prob', 'calibrated_prob',
               'seed', 'fold', 'SMILES', 'Disease']]
    fold_rows = []
    for (seed, fold), grp in oof.groupby(['seed', 'fold']):
        try:
            fa = roc_auc_score(grp['y'], grp['raw_prob'])
            fc = roc_auc_score(grp['y'], grp['calibrated_prob'])
        except ValueError:
            fa = fc = np.nan
        fold_rows.append({'task': 'safety', 'seed': seed, 'fold': fold,
                          'n': len(grp), 'auc_raw': fa, 'auc_cal': fc})
    fold_metrics = pd.DataFrame(fold_rows)
    if return_detail:
        detail = M.reset_index().merge(
            oof[['row_idx', 'seed', 'fold', 'y', 'raw_prob']]
            .rename(columns={'raw_prob': 'noisy_or'}), on=key, how='left')
        return oof, fold_metrics, detail
    return oof, fold_metrics


# ---------------------------------------------------------------------------
# Cross-task: overall CV folds, predict all three tasks on full outer-test
# ---------------------------------------------------------------------------
def run_overall_crosstask(df_overall, df, feature_cols,
                          safety_mask_full, efficacy_mask_full):
    """
    Within each fold of the overall cohort, train:
      - overall GBM on outer-train
      - safety GBM on (outer-train ∩ safety subset)
      - efficacy ensemble on (outer-train ∩ efficacy subset)
    and predict all three on the full outer-test.
    Also fit nested-CV isotonic for overall & efficacy.
    """
    X_raw = df_overall[feature_cols].values
    y_all = df_overall['_y'].values          # overall label
    groups = df_overall['SMILES'].values
    diseases = df_overall['Disease'].fillna('unknown').values
    idx_out = df_overall.index.values

    # Full-df masks mapped onto df_overall rows (for safety/eff subset within fold).
    is_safety_row = df_overall['_is_safety'].values.astype(bool)
    is_eff_row = df_overall['_is_efficacy'].values.astype(bool)
    y_safety_full = df_overall['_y_safety'].values
    y_eff_full = df_overall['_y_efficacy'].values

    rows = []
    fold_metrics = []
    skip_log = {'safety': 0, 'efficacy': 0, 'total': 0}
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X_raw, y_all, groups)):
            # --- Overall (reuse overall labels) ---
            X_tr_o, X_te_o, _, _ = prepare_fold(
                X_raw[tr], X_raw[te], y_all[tr],
                diseases[tr], diseases[te], feature_cols)
            raw_o = fit_predict(X_tr_o, y_all[tr], X_te_o, 'overall', seed)
            iso_o = fit_isotonic_nested(X_raw[tr], y_all[tr], groups[tr],
                                        diseases[tr], feature_cols,
                                        'overall', seed)
            cal_o = iso_o.transform(raw_o)

            # --- Safety: train on safety subset of outer-train ---
            tr_s = tr[is_safety_row[tr]]
            y_tr_s = y_safety_full[tr_s]
            fb_s = False
            if len(tr_s) < 10 or len(np.unique(y_tr_s)) < 2:
                base = float(y_safety_full[tr].mean()) if len(tr) else 0.0
                raw_s = np.full(len(te), base)
                fb_s = True
                skip_log['safety'] += 1
                print(f'    [crosstask] safety fold seed={seed} fold={fold} '
                      f'n_tr_s={len(tr_s)} pos={int(y_tr_s.sum())} '
                      f'-> fallback base={base:.4f}', flush=True)
            else:
                X_tr_s, X_te_s, _, _ = prepare_fold(
                    X_raw[tr_s], X_raw[te], y_tr_s,
                    diseases[tr_s], diseases[te], feature_cols)
                raw_s = fit_predict(X_tr_s, y_tr_s, X_te_s, 'safety', seed)
            cal_s = raw_s.copy()  # Safety: no calibration

            # --- Efficacy: train on efficacy subset of outer-train ---
            tr_e = tr[is_eff_row[tr]]
            y_tr_e = y_eff_full[tr_e]
            fb_e = False
            if len(tr_e) < 10 or len(np.unique(y_tr_e)) < 2:
                base = float(y_eff_full[tr].mean()) if len(tr) else 0.0
                raw_e = np.full(len(te), base)
                cal_e = raw_e.copy()
                fb_e = True
                skip_log['efficacy'] += 1
                print(f'    [crosstask] efficacy fold seed={seed} fold={fold} '
                      f'n_tr_e={len(tr_e)} pos={int(y_tr_e.sum())} '
                      f'-> fallback base={base:.4f}', flush=True)
            else:
                X_tr_e, X_te_e, _, _ = prepare_fold(
                    X_raw[tr_e], X_raw[te], y_tr_e,
                    diseases[tr_e], diseases[te], feature_cols)
                raw_e = fit_predict(X_tr_e, y_tr_e, X_te_e, 'efficacy', seed)
                iso_e = fit_isotonic_nested(X_raw[tr_e], y_tr_e,
                                            groups[tr_e], diseases[tr_e],
                                            feature_cols, 'efficacy', seed)
                cal_e = iso_e.transform(raw_e)

            skip_log['total'] += 1
            for j, pos in enumerate(te):
                rows.append((
                    idx_out[pos], int(y_all[pos]),
                    int(is_safety_row[pos]), int(y_safety_full[pos]),
                    int(is_eff_row[pos]), int(y_eff_full[pos]),
                    raw_o[j], cal_o[j],
                    raw_s[j], cal_s[j],
                    raw_e[j], cal_e[j],
                    int(fb_s), int(fb_e),
                    seed, fold, groups[pos], diseases[pos],
                ))
            fold_metrics.append({
                'seed': seed, 'fold': fold, 'n_te': len(te),
                'auc_o_raw': roc_auc_score(y_all[te], raw_o),
                'auc_o_cal': roc_auc_score(y_all[te], cal_o),
            })
            print(f'  crosstask seed={seed} fold={fold} '
                  f'n_te={len(te)} AUC_o_raw={fold_metrics[-1]["auc_o_raw"]:.3f}',
                  flush=True)

    oof = pd.DataFrame(rows, columns=[
        'row_idx', 'y_overall',
        'in_safety_cohort', 'y_safety',
        'in_efficacy_cohort', 'y_efficacy',
        'raw_prob_overall', 'calibrated_prob_overall',
        'raw_prob_safety', 'calibrated_prob_safety',
        'raw_prob_efficacy', 'calibrated_prob_efficacy',
        'fallback_safety', 'fallback_efficacy',
        'seed', 'fold', 'SMILES', 'Disease',
    ])
    return oof, pd.DataFrame(fold_metrics), skip_log


# ---------------------------------------------------------------------------
# ECE helper
# ---------------------------------------------------------------------------
def expected_calibration_error(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def summarize(y, p):
    return {
        'auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
        'brier': float(brier_score_loss(y, p)),
        'ece': expected_calibration_error(y, p),
        'n': int(len(y)),
        'pos_rate': float(np.mean(y)),
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def write_provenance(path, inputs, note=''):
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    h = hashlib.sha256()
    with open(DATA_PATH, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    prov = {
        'git_sha': sha,
        'data_path': str(DATA_PATH),
        'data_sha256': h.hexdigest(),
        'seeds': SEEDS,
        'pipeline': 'retrain_calibrated.py v1',
        'has_xgb': HAS_XGB,
        'has_lgbm': HAS_LGBM,
        'inputs': inputs,
        'note': note,
    }
    Path(path).write_text(json.dumps(prov, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibrate', choices=['none', 'isotonic'],
                    default='isotonic')
    ap.add_argument('--skip-crosstask', action='store_true')
    ap.add_argument('--safety-head', choices=['single', 'noisy_or'],
                    default='single',
                    help='single GBM (default, reproduces published headlines) or '
                         'noisy-OR over per-mechanism detectors (+0.020 safety AUC).')
    ap.add_argument('--eff-diag', action='store_true',
                    help='Also compute GBM-only efficacy OOF for AUC-gap diag.')
    ap.add_argument('--data', default=None,
                    help='Override training CSV (default: v5_unified -> production_v2).')
    ap.add_argument('--out', default=None,
                    help='Override output dir (default: results/production_v2).')
    ap.add_argument('--sanity', action='store_true',
                    help='Assert mean-of-folds match the published 0.837/0.772/0.828.')
    ap.add_argument('--include-endogenous', action='store_true',
                    help='Include is_endogenous trials in the efficacy cohort. They act on HUMAN targets '
                         'and the mechanism (causal_centrality) feature already represents them leak-free '
                         '(CV 0.703 vs binding-only 0.505); the original exclusion was over-conservative.')
    args = ap.parse_args()

    # Allow running an alternate dataset/out-dir (e.g. v8) without touching defaults.
    global DATA_PATH, OUT_DIR
    if args.data:
        DATA_PATH = Path(args.data).resolve()
    if args.out:
        OUT_DIR = Path(args.out).resolve()
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Loading {DATA_PATH}', flush=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    feature_cols = get_features(df)
    print(f'Trials={len(df)} Drugs={df.SMILES.nunique()} Feats={len(feature_cols)}',
          flush=True)

    # --- Build per-task cohorts (mirrors retrain_corrected.py) ---
    safety_mask = df['Corrected_Outcome'].isin(['PASS', 'FAIL_SAFETY', 'FAIL_BOTH'])
    if 'is_multi_drug_exclude' in df.columns:
        safety_mask &= df['is_multi_drug_exclude'] != 1
    df_safety = df[safety_mask].copy()
    df_safety['_y'] = df_safety['Corrected_Outcome'].isin(
        ['FAIL_SAFETY', 'FAIL_BOTH']).astype(int)

    eff_excl_flags = ['is_anti_pathogen', 'is_endogenous', 'is_mispaired_supportive',
                      'is_healthy_volunteer', 'is_procedural_exclude',
                      'is_multi_drug_exclude']
    if getattr(args, 'include_endogenous', False):
        eff_excl_flags.remove('is_endogenous')
    eff_excl = pd.Series(False, index=df.index)
    for c in eff_excl_flags:
        if c in df.columns:
            eff_excl |= df[c] == 1
    df_eff = df[~eff_excl & df['Corrected_Outcome'].isin(
        ['PASS', 'FAIL_EFFICACY', 'FAIL_BOTH'])].copy()
    df_eff['_y'] = df_eff['Corrected_Outcome'].isin(
        ['FAIL_EFFICACY', 'FAIL_BOTH']).astype(int)

    df_over = df[df['Corrected_Outcome'].isin(
        ['PASS', 'FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH'])].copy()
    df_over['_y'] = df_over['Corrected_Outcome'].isin(
        ['FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH']).astype(int)
    # Add per-row task membership for cross-task run.
    df_over['_is_safety'] = df_over.index.isin(df_safety.index).astype(int)
    df_over['_is_efficacy'] = df_over.index.isin(df_eff.index).astype(int)
    df_over['_y_safety'] = df_over['Corrected_Outcome'].isin(
        ['FAIL_SAFETY', 'FAIL_BOTH']).astype(int)
    df_over['_y_efficacy'] = df_over['Corrected_Outcome'].isin(
        ['FAIL_EFFICACY', 'FAIL_BOTH']).astype(int)

    print(f'safety n={len(df_safety)} pos={df_safety._y.sum()}', flush=True)
    print(f'efficacy n={len(df_eff)} pos={df_eff._y.sum()}', flush=True)
    print(f'overall n={len(df_over)} pos={df_over._y.sum()}', flush=True)

    all_fold_metrics = []
    metrics = {}
    calibrate = args.calibrate == 'isotonic'

    # --- Per-task CV (reproduces published headline) ---
    # Exposure axis (logdose + off-target*dose) is protected for SAFETY only — it is
    # weak-univariate (top-20 selection drops it) but real in interaction, and it HURTS
    # efficacy/overall. Auto-detected so default v5_unified runs are unaffected.
    # tdc_* = external structure->toxicity classifiers (DILI/AMES/carcinogen, trained on TDC
    # ground truth, NOT our labels). Weak-univariate vs top-20 binding feats but orthogonal
    # (r<0.20) and the only signal on binding-invisible idiosyncratic DILI. Protected for
    # SAFETY only (auto-detected; absent -> no-op). See notes/tdc_safety_dili_jun13.md.
    safety_protect = [c for c in feature_cols
                      if c == 'logdose' or c.endswith('_xdose') or c.startswith('tdc_')
                      or c.startswith('dili_')
                      # aeclass_ = target->AE-class on-target organ-tox priors (jun14): TESTED & NOT
                      # BANKED (0/32 FN flipped, pooled safety -0.0028; notes/target_ae_class_result_jun14.md).
                      # No-op guard kept (columns absent from the canonical build) so a re-test inherits
                      # safety-only protection; same pattern as the mech_ot_/tdc_ tested-not-banked guards.
                      or c.startswith('aeclass_')
                      or c.startswith('rmet_')]   # reactive-metabolite/haptenation alerts (jun14, testing)

    # mech_* (causal-centrality) is an EFFICACY mechanism-fitness construct: clean for efficacy
    # (availability gate 0.502, shuffle-validated) but a SURVIVORSHIP proxy for SAFETY. Two leaks
    # (Jun 12): (a) availability — mech was enumerated over the efficacy cohort so FAIL_SAFETY pairs
    # were unscored -> "has real mech" ≈ "is PASS" (gate 0.948; FIXED by scoring the 85 safety-fail
    # pairs, now 0.500); (b) VALUE — mech correlates with drug establishment/approval (approved score
    # ~10pts higher AND fail safety 1% vs 86%), so mech_population_fit still trips the safety value gate
    # (0.699 > 0.58) even with complete coverage. Within approved-only drugs mech is ~chance for safety
    # (0.53-0.57): there is no real safety signal to lose. Exclude mech_* from the safety head only
    # (kept for efficacy/overall). notes/mech_safety_leak_jun12.md.
    # NB (Jun 14): tested excluding design_* from the safety head (hypothesis: they leak into the context
    # detector and drive the elagolix p=1.0 cry-wolf). FALSIFIED — exclusion LOWERED safety mof 0.734->0.725
    # and did NOT resolve the cry-wolf. design_* carry real signal in the safety context detector; kept.
    # The elagolix cry-wolf is the noisy-OR independence artifact (already mitigated by calibrated_prob).
    # xseff_* = cross-species EFFICACY-translation axes (disease-model fidelity, target species
    # divergence; notes/cross_species_translation_axis_jun19.md) — efficacy-only, excluded from the
    # safety head like mech_*. xssafe_* (reactive-metabolite iDILI, HLA) stays in the safety head.
    # No-op on datasets without these columns (canonical reproducibility preserved).
    # is_cytotoxic (ATC L01A-D class flag, Jun 2026) is an EFFICACY class signal — cytotoxics
    # pass in their proliferative-malignancy indications (real antiproliferative MOA the
    # target->disease mechanism module is blind to: these rows have mech_*=0). Leak-safe
    # (value AUC 0.540, shuffle Δ~0; outcome-blind ChEMBL ATC; class transfer under group folds).
    # Held out of the SAFETY head (cytotoxic myelosuppression is a separate axis, unaudited here)
    # so safety stays byte-identical. No-op when the column is absent (canonical reproducibility).
    # endpoint_cvevent_match (Gap B) and precedent_* (Gap D) are EFFICACY-specific signals
    # (CV-event mechanism-match and negative efficacy-precedent); held out of the safety head so
    # safety stays byte-identical, matching is_cytotoxic and the mechanism-out-of-safety design.
    safety_feature_cols = [c for c in feature_cols
                           if not c.startswith('mech_') and not c.startswith('xseff_')
                           and not c.startswith('precedent_')
                           and c not in ('is_cytotoxic', 'endpoint_cvevent_match')]

    if args.safety_head == 'noisy_or':
        print('\n=== SAFETY (noisy-OR mechanism detectors) ===', flush=True)
        oof_s, fm_s, detail_s = noisy_or_safety_oof(
            df_safety, safety_feature_cols, protect_cols=safety_protect,
            return_detail=True)
        detail_s.to_parquet(OUT_DIR / 'safety_mechanism_detail.parquet', index=False)
    else:
        print('\n=== SAFETY ===', flush=True)
        oof_s, fm_s = run_task_cv(df_safety, safety_feature_cols, 'safety', calibrate=False,
                                  protect_cols=safety_protect)
    oof_s.to_parquet(OUT_DIR / 'oof_safety.parquet', index=False)
    all_fold_metrics.append(fm_s)

    # Trial-design primitives (design_*) are weak-univariate (AUC 0.52-0.60) but add
    # orthogonal leak-safe signal that top-20 selection would otherwise drop. Protected
    # for efficacy/overall only (the efficacy-bar signal is design-context, not safety).
    # Auto-detected: runs on datasets without design_* columns are unaffected.
    design_protect = [c for c in feature_cols if c.startswith('design_')]

    # mech_ot_* (zone-scoped leak-free mechanism) is auto-protected here IF present — but the Jun 13
    # test showed the leak-free signal is too weak in CNS/immune to flip the misses, so the feature
    # was reverted in build_v8_honest_exposure.py. This filter is a no-op on the current dataset and
    # keeps the protect-on-presence behavior for any future leak-free zone-mechanism feature.
    # endpoint_directness_signed (Jun 16): leak-safe endpoint-anchored directness (+1 drug can move the
    # measured biomarker / -1 cannot / 0 N/A). Weak-univariate (non-zero on ~10% of trials) but real in
    # interaction; availability AUC 0.482, shuffle p=0.0000, residual coef +1.26 p=0.037. Protected for
    # efficacy/overall only (SAFETY-neutral). Auto-detected: absent -> no-op.
    efficacy_protect = (design_protect + [c for c in feature_cols if c.startswith('mech_ot_')]
                        + [c for c in feature_cols if c.startswith('endpoint_')]
                        + [c for c in feature_cols if c.startswith('population_leverage')]
                        + [c for c in feature_cols if c.startswith('xseff_')]
                        # mech_coverage_* are weak-univariate by design (leak-safe, avail AUC ~0.51); their
                        # value is the INTERACTION coverage=0 -> discount missing-mechanism zeros, so they
                        # must bypass top-k selection. efficacy/overall only (mechanism stays out of safety).
                        + [c for c in feature_cols if c.startswith('mech_coverage_')]
                        # is_cytotoxic: ATC L01A-D efficacy class flag (weak-univariate, real in
                        # interaction; +0.009 efficacy on top of canonical, 22/25 folds, leak-clean).
                        # efficacy/overall only. No-op when absent (canonical reproducibility).
                        + [c for c in feature_cols if c == 'is_cytotoxic']
                        # precedent_neg_class (Gap D, Jun 28): as-of-date BINARY flag = a same-MOA-class
                        # agent already had a Phase-3 EFFICACY failure in this indication before this trial
                        # started. Leak-safe (as-of-date, negative-only, binary): shuffle p=0.004,
                        # availability AUC 0.511, NON-redundant vs difficulty (coef 0.65 p=0.007). The
                        # efficacy precedent stays OUT of the noisy-OR safety head. No-op when absent.
                        + [c for c in feature_cols if c.startswith('precedent_')])

    print('\n=== EFFICACY (ensemble, calibrated) ===', flush=True)
    oof_e, fm_e = run_task_cv(df_eff, feature_cols, 'efficacy', calibrate=calibrate,
                              protect_cols=efficacy_protect)
    oof_e.to_parquet(OUT_DIR / 'oof_efficacy.parquet', index=False)
    all_fold_metrics.append(fm_e)

    print('\n=== OVERALL (GBM, calibrated) ===', flush=True)
    oof_o, fm_o = run_task_cv(df_over, feature_cols, 'overall', calibrate=calibrate,
                              protect_cols=efficacy_protect)
    oof_o.to_parquet(OUT_DIR / 'oof_overall.parquet', index=False)
    all_fold_metrics.append(fm_o)

    # --- Summaries ---
    # Mean-of-folds is primary (matches published headline + per-fold SDs);
    # pooled-OOF is secondary (right metric for calibration: Brier/ECE).
    for name, oof, fm in [('overall', oof_o, fm_o),
                          ('safety', oof_s, fm_s),
                          ('efficacy', oof_e, fm_e)]:
        mof_raw = float(np.nanmean(fm['auc_raw']))
        mof_raw_sd = float(np.nanstd(fm['auc_raw']))
        mof_cal = float(np.nanmean(fm['auc_cal']))
        mof_cal_sd = float(np.nanstd(fm['auc_cal']))
        metrics[name] = {
            'raw': summarize(oof.y.values, oof.raw_prob.values),
            'calibrated': summarize(oof.y.values, oof.calibrated_prob.values),
            'mean_of_folds': {
                'auc_raw': mof_raw, 'auc_raw_sd': mof_raw_sd,
                'auc_cal': mof_cal, 'auc_cal_sd': mof_cal_sd,
                'n_folds': int(len(fm)),
            },
        }
        print(f'{name}: mean-of-folds raw={mof_raw:.4f}±{mof_raw_sd:.4f} '
              f'cal={mof_cal:.4f}±{mof_cal_sd:.4f} | '
              f'pooled raw AUC={metrics[name]["raw"]["auc"]:.4f} '
              f'ECE={metrics[name]["raw"]["ece"]:.3f} | '
              f'pooled cal AUC={metrics[name]["calibrated"]["auc"]:.4f} '
              f'ECE={metrics[name]["calibrated"]["ece"]:.3f}', flush=True)

    # --- Efficacy GBM-only diagnostic (AUC-gap reconciliation) ---
    if args.eff_diag:
        print('\n=== EFFICACY DIAGNOSTIC: GBM-only ===', flush=True)
        df_eff_g = df_eff.copy()
        # Run CV with same calibration setting but fit GBM-only.
        # Cheapest implementation: swap fit_predict call path via a flag.
        rows = []
        X_raw = df_eff_g[feature_cols].values
        y = df_eff_g['_y'].values
        groups = df_eff_g['SMILES'].values
        diseases = df_eff_g['Disease'].fillna('unknown').values
        for seed in SEEDS:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (tr, te) in enumerate(cv.split(X_raw, y, groups)):
                Xt, Xe, _, _ = prepare_fold(X_raw[tr], X_raw[te], y[tr],
                                            diseases[tr], diseases[te], feature_cols)
                raw = fit_predict_gbm_only(Xt, y[tr], Xe, seed)
                for j, pos in enumerate(te):
                    rows.append((df_eff_g.index.values[pos], int(y[pos]),
                                 float(raw[j]), seed, fold))
        oof_eg = pd.DataFrame(rows, columns=['row_idx', 'y', 'raw_prob',
                                             'seed', 'fold'])
        oof_eg.to_parquet(OUT_DIR / 'oof_efficacy_gbmonly.parquet', index=False)
        metrics['efficacy_gbmonly'] = {
            'raw': summarize(oof_eg.y.values, oof_eg.raw_prob.values),
        }
        print(f'efficacy GBM-only: AUC={metrics["efficacy_gbmonly"]["raw"]["auc"]:.4f}'
              f' ECE={metrics["efficacy_gbmonly"]["raw"]["ece"]:.3f}', flush=True)

    # --- Cross-task OOF on overall cohort ---
    if not args.skip_crosstask:
        print('\n=== CROSS-TASK (overall folds, all three probs) ===', flush=True)
        oof_ct, fm_ct, skip_log = run_overall_crosstask(
            df_over, df, feature_cols,
            df_over['_is_safety'].values,
            df_over['_is_efficacy'].values)
        oof_ct.to_parquet(OUT_DIR / 'oof_overall_crosstask.parquet', index=False)

        # --- Skip-rate decision rule (auto-applied) ---
        total = max(skip_log['total'], 1)
        rate_s = skip_log['safety'] / total
        rate_e = skip_log['efficacy'] / total
        print(f'\n[crosstask] skip rate: safety={skip_log["safety"]}/{total} '
              f'({rate_s:.1%}) efficacy={skip_log["efficacy"]}/{total} '
              f'({rate_e:.1%})', flush=True)
        metrics['crosstask_skip'] = {
            'safety_folds_skipped': int(skip_log['safety']),
            'efficacy_folds_skipped': int(skip_log['efficacy']),
            'total_folds': int(total),
            'safety_skip_rate': rate_s,
            'efficacy_skip_rate': rate_e,
        }
        # Per-task disease-flag crosstab on fallback rows
        disease_flags = [c for c in ['is_oncology', 'is_infectious', 'is_cns',
                                     'is_cardiac', 'is_autoimmune']
                         if c in df.columns]
        ct_merged = oof_ct.merge(
            df[disease_flags].reset_index().rename(columns={'index': 'row_idx'}),
            on='row_idx', how='left') if disease_flags else oof_ct
        for task_fb in ['fallback_safety', 'fallback_efficacy']:
            fb = ct_merged[ct_merged[task_fb] == 1]
            if len(fb) == 0:
                continue
            print(f'\n  {task_fb} rows: n={len(fb)}', flush=True)
            for flag in disease_flags:
                fb_rate = fb[flag].mean()
                all_rate = ct_merged[flag].mean()
                print(f'    {flag}: fallback={fb_rate:.3f} vs '
                      f'overall={all_rate:.3f}', flush=True)

        worst = max(rate_s, rate_e)
        if worst > 0.15:
            print(f'\n[crosstask] SKIP RATE {worst:.1%} EXCEEDS 15% — '
                  f'stopping before Tasks 2–4.', flush=True)
            Path(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2))
            return
        elif worst >= 0.05:
            print(f'\n[crosstask] skip rate {worst:.1%} in 5–15% band — '
                  f'will be flagged in edit file.', flush=True)
            metrics['crosstask_skip']['flag_in_methods'] = True
        else:
            metrics['crosstask_skip']['flag_in_methods'] = False
        # Restrict to in-cohort rows for fair comparison with per-task AUCs.
        for task, key_in, key_y, key_raw, key_cal in [
            ('overall_ct', None, 'y_overall', 'raw_prob_overall', 'calibrated_prob_overall'),
            ('safety_ct', 'in_safety_cohort', 'y_safety', 'raw_prob_safety', 'calibrated_prob_safety'),
            ('efficacy_ct', 'in_efficacy_cohort', 'y_efficacy', 'raw_prob_efficacy', 'calibrated_prob_efficacy'),
        ]:
            sub = oof_ct if key_in is None else oof_ct[oof_ct[key_in] == 1]
            metrics[task] = {
                'raw': summarize(sub[key_y].values, sub[key_raw].values),
                'calibrated': summarize(sub[key_y].values, sub[key_cal].values),
            }
            print(f'{task}: raw AUC={metrics[task]["raw"]["auc"]:.4f}'
                  f' cal AUC={metrics[task]["calibrated"]["auc"]:.4f}'
                  f' n={metrics[task]["raw"]["n"]}', flush=True)
        fm_ct['task'] = 'crosstask'
        all_fold_metrics.append(fm_ct)

    fold_df = pd.concat(all_fold_metrics, ignore_index=True)
    fold_df.to_csv(OUT_DIR / 'fold_metrics.csv', index=False)
    Path(OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    for f in ['oof_overall.parquet', 'oof_safety.parquet', 'oof_efficacy.parquet',
              'oof_overall_crosstask.parquet', 'metrics.json', 'fold_metrics.csv']:
        p = OUT_DIR / f
        if p.exists():
            write_provenance(str(p) + '.provenance.json',
                             inputs=[str(DATA_PATH)],
                             note=f'retrain_calibrated.py --calibrate={args.calibrate}')
    print(f'\nSaved to {OUT_DIR}', flush=True)
    print('\n=== Mean-of-folds vs published production_v2 (0.837/0.772/0.828) ===', flush=True)
    targets = {'overall': (0.837, 'auc_raw'),
               'safety':  (0.772, 'auc_raw'),
               'efficacy':(0.828, 'auc_raw')}
    tol = 0.005
    all_pass = True
    for name, (tgt, key) in targets.items():
        got = metrics[name]['mean_of_folds'][key]
        ok = abs(got - tgt) <= tol
        all_pass &= ok
        print(f'  {name} mean-of-folds {key}: {got:.4f}  v2={tgt:.3f}  '
              f'Δ={got-tgt:+.4f}  {"==" if ok else "MOVED"}', flush=True)
    if args.sanity:
        print(f'\nSanity check (must reproduce v2): '
              f'{"PASS" if all_pass else "FAIL"}', flush=True)
        assert all_pass, 'v2 reproduction failed'


if __name__ == '__main__':
    main()
