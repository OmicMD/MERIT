#!/usr/bin/env python3
"""
Retrain models with corrected trial-level labels and ALL features.

Uses training_dataset_trial_level_all_features.csv which includes:
- 79 tissue interaction features (from STAR_complete.csv)
- 36 network enrichment features (STRING-DB, the dominant signal)
- 24 Binding specificity features
- 8 Binding targeted binding features (0-filled)
- 22 drug network enrichment features

Training:
- TRIAL-LEVEL (not drug-level) — each trial is a data point
- StratifiedGroupKFold with SMILES grouping (zero drug leakage)
- 5 seeds × 5-fold = 25 folds
- GBM for safety, Ensemble (GBM+XGB+LGBM) for efficacy
- FAIL_BOTH counted as failure for both safety and efficacy tasks
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
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

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR.parent / 'data' / 'sources').exists():
    BASE_DIR = SCRIPT_DIR.parent
    DATA_PATH = BASE_DIR / 'data' / 'sources' / 'training_dataset_v5_unified.csv'
else:
    BASE_DIR = SCRIPT_DIR
    DATA_PATH = BASE_DIR / 'training_dataset_v5_unified.csv'

OUTPUT_DIR = BASE_DIR / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]

# Columns that are NOT features — explicit blocklist of all non-biological columns
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
             'is_healthy_volunteer', 'is_procedural_exclude', 'is_multi_drug_exclude',
             'Source', 'feature_IK', 'Is_Biologic'}


def get_features(df):
    """All numeric columns except metadata, outcome labels, and ADMET."""
    feats = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if 'drugbank_approved_percentile' in c or 'median_' in c or 'max_phase' in c:
            continue
        # tb_* = three-body anti-pathogen axes; consumed ONLY by the dedicated anti-pathogen efficacy
        # head, NEVER the canonical small-molecule model. They exist only for is_anti_pathogen rows, so
        # admitting them here would act as a cohort indicator (leak). Keeps the canonical model unchanged.
        if c.startswith('tb_'):
            continue
        if df[c].dtype in ('float64', 'float32', 'int64', 'int32'):
            feats.append(c)
    return feats


def compute_disease_encoding(diseases_train, y_train, diseases_test, smoothing_n=3):
    """Compute Bayesian-smoothed disease failure rate within training fold."""
    global_rate = y_train.mean()
    disease_rates = {}
    for d, y_val in zip(diseases_train, y_train):
        if d not in disease_rates:
            disease_rates[d] = []
        disease_rates[d].append(y_val)

    train_encoded = np.zeros(len(diseases_train))
    for i, d in enumerate(diseases_train):
        vals = disease_rates[d]
        n = len(vals)
        raw_rate = np.mean(vals)
        train_encoded[i] = (n * raw_rate + smoothing_n * global_rate) / (n + smoothing_n)

    test_encoded = np.zeros(len(diseases_test))
    for i, d in enumerate(diseases_test):
        if d in disease_rates:
            vals = disease_rates[d]
            n = len(vals)
            raw_rate = np.mean(vals)
            test_encoded[i] = (n * raw_rate + smoothing_n * global_rate) / (n + smoothing_n)
        else:
            test_encoded[i] = global_rate

    return train_encoded, test_encoded


def nested_feature_selection(X_train, y_train, feature_names, top_k=20):
    """Select top-k features by univariate AUC within training fold."""
    aucs = []
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        if np.std(col) == 0:
            aucs.append(0.5)
            continue
        try:
            auc = roc_auc_score(y_train, col)
            aucs.append(max(auc, 1 - auc))  # Handle inverted features
        except ValueError:
            aucs.append(0.5)
    top_indices = np.argsort(aucs)[-top_k:]
    return top_indices


def train_cv(X, y, groups, diseases, feature_names, task_name, seeds=SEEDS):
    """Train with nested feature selection + disease encoding within each fold."""
    all_fold_aucs = []

    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_aucs = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            # Verify zero leakage
            train_drugs = set(groups[train_idx])
            test_drugs = set(groups[test_idx])
            assert len(train_drugs & test_drugs) == 0, "Drug leakage!"

            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            d_train = diseases[train_idx]
            d_test = diseases[test_idx]

            # Impute within fold
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train_raw)
            X_test_imp = imputer.transform(X_test_raw)

            # Nested feature selection within fold
            top_idx = nested_feature_selection(X_train_imp, y_train, feature_names)
            X_train_sel = X_train_imp[:, top_idx]
            X_test_sel = X_test_imp[:, top_idx]

            # Disease encoding within fold
            enc_train, enc_test = compute_disease_encoding(d_train, y_train, d_test)
            X_train_final = np.column_stack([X_train_sel, enc_train])
            X_test_final = np.column_stack([X_test_sel, enc_test])

            weights = compute_sample_weight('balanced', y_train)

            # GBM
            gbm = GradientBoostingClassifier(
                n_estimators=500, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=seed)
            gbm.fit(X_train_final, y_train, sample_weight=weights)
            proba_gbm = gbm.predict_proba(X_test_final)[:, 1]

            if task_name == 'efficacy' and (HAS_XGB or HAS_LGBM):
                probas = [proba_gbm]
                if HAS_XGB:
                    try:
                        xgb = XGBClassifier(
                            n_estimators=500, max_depth=3, learning_rate=0.05,
                            subsample=0.8, random_state=seed, eval_metric='logloss',
                            tree_method='hist', device='cuda')
                        xgb.fit(X_train_final, y_train, sample_weight=weights)
                        probas.append(xgb.predict_proba(X_test_final)[:, 1])
                    except Exception:
                        xgb = XGBClassifier(
                            n_estimators=500, max_depth=3, learning_rate=0.05,
                            subsample=0.8, random_state=seed, eval_metric='logloss')
                        xgb.fit(X_train_final, y_train, sample_weight=weights)
                        probas.append(xgb.predict_proba(X_test_final)[:, 1])
                if HAS_LGBM:
                    try:
                        lgbm = LGBMClassifier(
                            n_estimators=500, max_depth=3, learning_rate=0.05,
                            subsample=0.8, random_state=seed, verbose=-1,
                            device='gpu')
                        lgbm.fit(X_train_final, y_train, sample_weight=weights)
                        probas.append(lgbm.predict_proba(X_test_final)[:, 1])
                    except Exception:
                        lgbm = LGBMClassifier(
                            n_estimators=500, max_depth=3, learning_rate=0.05,
                            subsample=0.8, random_state=seed, verbose=-1)
                        lgbm.fit(X_train_final, y_train, sample_weight=weights)
                        probas.append(lgbm.predict_proba(X_test_final)[:, 1])
                proba = np.mean(probas, axis=0)
            else:
                proba = proba_gbm

            auc = roc_auc_score(y_test, proba)
            fold_aucs.append(auc)
            selected_names = [feature_names[i] for i in top_idx]
            print(f'    Fold {fold}: AUC={auc:.3f}, top features: {selected_names[:5]}...',
                  flush=True)

        seed_auc = np.mean(fold_aucs)
        print(f'  Seed {seed}: AUC = {seed_auc:.4f} (folds: {[f"{a:.3f}" for a in fold_aucs]})',
              flush=True)
        all_fold_aucs.extend(fold_aucs)

    mean_auc = np.mean(all_fold_aucs)
    std_auc = np.std(all_fold_aucs)
    print(f'  {task_name.upper()} OVERALL: {mean_auc:.4f} ± {std_auc:.4f} ({len(all_fold_aucs)} folds)',
          flush=True)

    return mean_auc, std_auc, all_fold_aucs


def train_disease_only(y, diseases, task_name, seeds=SEEDS):
    """Disease-encoding-only baseline (no molecular features)."""
    groups_placeholder = diseases  # Can't group by SMILES without features
    all_fold_aucs = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
            y_train, y_test = y[train_idx], y[test_idx]
            d_train, d_test = diseases[train_idx], diseases[test_idx]
            enc_train, enc_test = compute_disease_encoding(d_train, y_train, d_test)
            auc = roc_auc_score(y_test, enc_test)
            all_fold_aucs.append(auc)
    mean_auc = np.mean(all_fold_aucs)
    std_auc = np.std(all_fold_aucs)
    print(f'  {task_name.upper()} DISEASE-ONLY: {mean_auc:.4f} ± {std_auc:.4f}', flush=True)
    return mean_auc, std_auc


def train_molecular_only(X, y, groups, diseases, feature_names, task_name, seeds=SEEDS):
    """Molecular-features-only model (no disease encoding)."""
    all_fold_aucs = []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train_raw)
            X_test_imp = imputer.transform(X_test_raw)
            top_idx = nested_feature_selection(X_train_imp, y_train, feature_names)
            X_train_sel = X_train_imp[:, top_idx]
            X_test_sel = X_test_imp[:, top_idx]
            weights = compute_sample_weight('balanced', y_train)
            gbm = GradientBoostingClassifier(
                n_estimators=500, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=seed)
            gbm.fit(X_train_sel, y_train, sample_weight=weights)
            proba = gbm.predict_proba(X_test_sel)[:, 1]
            auc = roc_auc_score(y_test, proba)
            all_fold_aucs.append(auc)
    mean_auc = np.mean(all_fold_aucs)
    std_auc = np.std(all_fold_aucs)
    print(f'  {task_name.upper()} MOLECULAR-ONLY: {mean_auc:.4f} ± {std_auc:.4f}', flush=True)
    return mean_auc, std_auc


def run_permutation_test(X, y, groups, diseases, feature_names, task_name, n_shuffles=50, seeds=SEEDS):
    """Drug-level label permutation test."""
    perm_aucs = []
    for shuffle_i in range(n_shuffles):
        rng = np.random.RandomState(shuffle_i)
        unique_drugs = np.unique(groups)
        drug_labels = {}
        for drug in unique_drugs:
            mask = groups == drug
            drug_labels[drug] = y[mask][0]
        shuffled_labels = rng.permutation(list(drug_labels.values()))
        drug_label_map = dict(zip(unique_drugs, shuffled_labels))
        y_perm = np.array([drug_label_map[g] for g in groups])
        # Single seed, single fold for speed
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_perm, groups)):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y_perm[train_idx], y_perm[test_idx]
            d_train, d_test = diseases[train_idx], diseases[test_idx]
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train_raw)
            X_test_imp = imputer.transform(X_test_raw)
            top_idx = nested_feature_selection(X_train_imp, y_train, feature_names)
            X_train_sel = X_train_imp[:, top_idx]
            X_test_sel = X_test_imp[:, top_idx]
            enc_train, enc_test = compute_disease_encoding(d_train, y_train, d_test)
            X_train_final = np.column_stack([X_train_sel, enc_train])
            X_test_final = np.column_stack([X_test_sel, enc_test])
            weights = compute_sample_weight('balanced', y_train)
            gbm = GradientBoostingClassifier(
                n_estimators=500, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42)
            gbm.fit(X_train_final, y_train, sample_weight=weights)
            proba = gbm.predict_proba(X_test_final)[:, 1]
            try:
                auc = roc_auc_score(y_test, proba)
            except ValueError:
                auc = 0.5
            fold_aucs.append(auc)
        perm_aucs.append(np.mean(fold_aucs))
        if (shuffle_i + 1) % 10 == 0:
            print(f'    Permutation {shuffle_i+1}/{n_shuffles}: mean={np.mean(perm_aucs):.3f}', flush=True)

    mean_perm = np.mean(perm_aucs)
    std_perm = np.std(perm_aucs)
    max_perm = np.max(perm_aucs)
    print(f'  {task_name.upper()} PERMUTATION: mean={mean_perm:.4f} ± {std_perm:.4f}, max={max_perm:.4f}', flush=True)
    return mean_perm, std_perm, max_perm, perm_aucs


def run_ablation(df, feature_cols):
    """Run all ablation baselines: disease-only, molecular-only, permutation."""
    groups = df['SMILES'].values
    diseases = df['Disease'].fillna('unknown').values
    ablation_results = []

    for task_name, outcome_col_vals in [
        ('safety', ['FAIL_SAFETY', 'FAIL_BOTH']),
        ('efficacy', ['FAIL_EFFICACY', 'FAIL_BOTH'])
    ]:
        df_task = df[df['Corrected_Outcome'].isin(['PASS'] + outcome_col_vals)].copy()
        y = df_task['Corrected_Outcome'].isin(outcome_col_vals).astype(int).values
        X = df_task[feature_cols].values
        g = df_task['SMILES'].values
        d = df_task['Disease'].fillna('unknown').values

        print(f'\n{"="*60}', flush=True)
        print(f'ABLATION: {task_name.upper()}', flush=True)
        print(f'{"="*60}', flush=True)
        print(f'Trials: {len(df_task)} (PASS: {(y==0).sum()}, FAIL: {(y==1).sum()})', flush=True)

        # Disease-only
        d_auc, d_std = train_disease_only(y, d, task_name)
        ablation_results.append({
            'task': task_name, 'model': 'disease_only',
            'auc_mean': d_auc, 'auc_std': d_std
        })

        # Molecular-only
        m_auc, m_std = train_molecular_only(X, y, g, d, feature_cols, task_name)
        ablation_results.append({
            'task': task_name, 'model': 'molecular_only',
            'auc_mean': m_auc, 'auc_std': m_std
        })

        # Permutation test
        p_mean, p_std, p_max, _ = run_permutation_test(X, y, g, d, feature_cols, task_name)
        ablation_results.append({
            'task': task_name, 'model': 'permutation',
            'auc_mean': p_mean, 'auc_std': p_std, 'auc_max': p_max
        })

    results_df = pd.DataFrame(ablation_results)
    out_path = OUTPUT_DIR / 'ablation_v3_results.csv'
    results_df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}', flush=True)
    return results_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation baselines: disease-only, molecular-only, permutation test')
    args = parser.parse_args()

    print('Loading training data...', flush=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    feature_cols = get_features(df)
    print(f'Trials: {len(df)}, Drugs: {df["SMILES"].nunique()}', flush=True)
    print(f'Features: {len(feature_cols)}', flush=True)
    print(f'Outcomes: {df["Corrected_Outcome"].value_counts().to_dict()}', flush=True)
    drumap_feats = [c for c in feature_cols if c.startswith('drumap_')]
    print(f'DruMAP features: {drumap_feats}', flush=True)
    print(f'Feature columns: {feature_cols[:10]}...', flush=True)

    if args.ablation:
        run_ablation(df, feature_cols)
        return

    groups = df['SMILES'].values
    diseases = df['Disease'].fillna('unknown').values

    # === SAFETY: PASS vs (FAIL_SAFETY + FAIL_BOTH) ===
    print(f'\n{"="*60}', flush=True)
    print('SAFETY TASK: PASS vs FAIL_SAFETY/FAIL_BOTH', flush=True)
    print(f'{"="*60}', flush=True)
    safety_mask = df['Corrected_Outcome'].isin(['PASS', 'FAIL_SAFETY', 'FAIL_BOTH'])
    if 'is_multi_drug_exclude' in df.columns:
        safety_mask &= df['is_multi_drug_exclude'] != 1
        print(f'  Excluding {(df["is_multi_drug_exclude"]==1).sum()} multi-drug trials from safety', flush=True)
    df_safety = df[safety_mask].copy()
    y_safety = df_safety['Corrected_Outcome'].isin(['FAIL_SAFETY', 'FAIL_BOTH']).astype(int).values
    X_safety = df_safety[feature_cols].values
    g_safety = df_safety['SMILES'].values
    d_safety = df_safety['Disease'].fillna('unknown').values
    print(f'Trials: {len(df_safety)} (PASS: {(y_safety==0).sum()}, FAIL: {(y_safety==1).sum()})', flush=True)
    s_auc, s_std, s_folds = train_cv(X_safety, y_safety, g_safety, d_safety, feature_cols, 'safety')

    # === EFFICACY: PASS vs (FAIL_EFFICACY + FAIL_BOTH) ===
    # Exclude anti-pathogen drugs (target pathogen proteins, pipeline can't capture efficacy)
    # and mispaired supportive care trials (wrong drug-disease pairing)
    print(f'\n{"="*60}', flush=True)
    print('EFFICACY TASK: PASS vs FAIL_EFFICACY/FAIL_BOTH', flush=True)
    print(f'{"="*60}', flush=True)
    eff_excl_mask = pd.Series(False, index=df.index)
    if 'is_anti_pathogen' in df.columns:
        eff_excl_mask |= df['is_anti_pathogen'] == 1
        print(f'  Excluding {(df["is_anti_pathogen"]==1).sum()} anti-pathogen trials from efficacy', flush=True)
    if 'is_endogenous' in df.columns:
        eff_excl_mask |= df['is_endogenous'] == 1
        print(f'  Excluding {(df["is_endogenous"]==1).sum()} endogenous hormone trials', flush=True)
    if 'is_mispaired_supportive' in df.columns:
        eff_excl_mask |= df['is_mispaired_supportive'] == 1
        print(f'  Excluding {(df["is_mispaired_supportive"]==1).sum()} mispaired supportive care trials', flush=True)
    if 'is_healthy_volunteer' in df.columns:
        eff_excl_mask |= df['is_healthy_volunteer'] == 1
        print(f'  Excluding {(df["is_healthy_volunteer"]==1).sum()} healthy volunteer trials', flush=True)
    if 'is_procedural_exclude' in df.columns:
        eff_excl_mask |= df['is_procedural_exclude'] == 1
        print(f'  Excluding {(df["is_procedural_exclude"]==1).sum()} procedural/no-disease trials', flush=True)
    if 'is_multi_drug_exclude' in df.columns:
        eff_excl_mask |= df['is_multi_drug_exclude'] == 1
        print(f'  Excluding {(df["is_multi_drug_exclude"]==1).sum()} multi-drug trials', flush=True)
    df_efficacy = df[~eff_excl_mask & df['Corrected_Outcome'].isin(['PASS', 'FAIL_EFFICACY', 'FAIL_BOTH'])].copy()
    y_efficacy = df_efficacy['Corrected_Outcome'].isin(['FAIL_EFFICACY', 'FAIL_BOTH']).astype(int).values
    X_efficacy = df_efficacy[feature_cols].values
    g_efficacy = df_efficacy['SMILES'].values
    d_efficacy = df_efficacy['Disease'].fillna('unknown').values
    print(f'Trials: {len(df_efficacy)} (PASS: {(y_efficacy==0).sum()}, FAIL: {(y_efficacy==1).sum()})', flush=True)
    e_auc, e_std, e_folds = train_cv(X_efficacy, y_efficacy, g_efficacy, d_efficacy, feature_cols, 'efficacy')

    # === OVERALL: PASS vs FAIL (any type) ===
    print(f'\n{"="*60}', flush=True)
    print('OVERALL TASK: PASS vs FAIL (any type)', flush=True)
    print(f'{"="*60}', flush=True)
    df_overall = df[df['Corrected_Outcome'].isin(['PASS', 'FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH'])].copy()
    y_overall = df_overall['Corrected_Outcome'].isin(['FAIL_SAFETY', 'FAIL_EFFICACY', 'FAIL_BOTH']).astype(int).values
    X_overall = df_overall[feature_cols].values
    g_overall = df_overall['SMILES'].values
    d_overall = df_overall['Disease'].fillna('unknown').values
    print(f'Trials: {len(df_overall)} (PASS: {(y_overall==0).sum()}, FAIL: {(y_overall==1).sum()})', flush=True)
    o_auc, o_std, o_folds = train_cv(X_overall, y_overall, g_overall, d_overall, feature_cols, 'overall')

    # === RESULTS ===
    print(f'\n{"="*60}', flush=True)
    print('RESULTS (corrected labels, drug-level, all features)', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'Overall: {o_auc:.4f} ± {o_std:.4f}', flush=True)
    print(f'Safety:  {s_auc:.4f} ± {s_std:.4f}', flush=True)
    print(f'Efficacy: {e_auc:.4f} ± {e_std:.4f}', flush=True)
    print(f'Features: {len(feature_cols)}', flush=True)
    print(f'Trials: overall={len(df_overall)}, safety={len(df_safety)}, efficacy={len(df_efficacy)}', flush=True)
    print(f'Drugs: overall={df_overall["SMILES"].nunique()}, safety={df_safety["SMILES"].nunique()}, efficacy={df_efficacy["SMILES"].nunique()}', flush=True)

    results = pd.DataFrame({
        'task': ['overall', 'safety', 'efficacy'],
        'auc_mean': [o_auc, s_auc, e_auc],
        'auc_std': [o_std, s_std, e_std],
        'n_trials': [len(df_overall), len(df_safety), len(df_efficacy)],
        'n_drugs': [df_overall['SMILES'].nunique(), df_safety['SMILES'].nunique(), df_efficacy['SMILES'].nunique()],
        'n_features': [len(feature_cols)] * 3,
        'n_folds': [len(o_folds), len(s_folds), len(e_folds)],
    })
    retrain_dir = OUTPUT_DIR / 'retrain'
    retrain_dir.mkdir(exist_ok=True)
    results.to_csv(retrain_dir / 'retrain_corrected_results.csv', index=False)

    fold_df = pd.DataFrame({
        'overall_fold_aucs': o_folds,
        'safety_fold_aucs': s_folds,
        'efficacy_fold_aucs': e_folds,
    })
    fold_df.to_csv(retrain_dir / 'retrain_corrected_fold_aucs.csv', index=False)

    print(f'\nSaved: results/retrain/retrain_corrected_results.csv', flush=True)
    print(f'Saved: results/retrain/retrain_corrected_fold_aucs.csv', flush=True)


if __name__ == '__main__':
    main()
