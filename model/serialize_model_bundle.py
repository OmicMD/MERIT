#!/usr/bin/env python3
"""
Serialize pre-trained models and metadata into a single bundle for deployment.

Run once on a machine with access to the training data:
    python model/serialize_model_bundle.py

Produces: model/model_bundle.pkl
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
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

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / 'data' / 'cache'

# ── Dynamic imports from existing pipeline ──────────────────────────────────
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'manuscript' / 'scripts'))

from generate_publication_figures import (
    load_pair_data,
    get_pretrial_features,
    deduplicate_features,
)


def build_bundle():
    """Train models on full data and package everything for deployment."""
    print("=" * 60)
    print("SERIALIZING MODEL BUNDLE")
    print("=" * 60)

    # ── 1. Load training data ───────────────────────────────────────────────
    print("\n[1/5] Loading pair-level training data...")
    df_train, pair_outcomes, pair_feat, drug_names = load_pair_data()
    print(f"  Loaded {len(pair_outcomes)} pairs, {pair_feat.shape[1]} raw features")

    # ── 2. Load caches ──────────────────────────────────────────────────────
    print("\n[2/5] Loading gene/disease caches...")
    with open(CACHE_DIR / 'gene_to_enst_mapping.json') as f:
        gene_to_enst = json.load(f)
    with open(CACHE_DIR / 'disease_targets_cache.json') as f:
        disease_targets_cache = json.load(f)
    print(f"  gene_to_enst: {len(gene_to_enst)} mappings")
    print(f"  disease_targets_cache: {len(disease_targets_cache)} keys")

    bundle = {
        'gene_to_enst': gene_to_enst,
        'disease_targets_cache': disease_targets_cache,
    }

    # ── 3. Train and serialize each task ────────────────────────────────────
    for task in ['safety', 'efficacy']:
        print(f"\n[3/5] Training {task} model...")

        # Filter pairs
        if task == 'safety':
            mask = pair_outcomes.isin(['PASS_APPROVED', 'FAIL_SAFETY'])
            pos_label = 'FAIL_SAFETY'
        else:
            mask = pair_outcomes.isin(['PASS_APPROVED', 'FAIL_EFFICACY'])
            pos_label = 'FAIL_EFFICACY'

        pairs = pair_outcomes[mask].index
        common = sorted(set(pairs) & set(pair_feat.index))
        y = (pair_outcomes.loc[common] == pos_label).astype(int)

        # Select features
        feat_cols = get_pretrial_features(pair_feat)
        X = pair_feat.loc[common, feat_cols]

        # Impute
        imp = SimpleImputer(strategy='median')
        X_imp = pd.DataFrame(
            imp.fit_transform(X), columns=X.columns, index=X.index
        )

        # Deduplicate
        clean_cols = deduplicate_features(X_imp)
        X_clean = X_imp[clean_cols]

        print(f"  {len(X_clean)} pairs, {y.sum()} positive, {len(clean_cols)} features")

        # Train
        sw = compute_sample_weight('balanced', y)

        if task == 'safety':
            gbm = GradientBoostingClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.01,
                subsample=0.8, random_state=42,
            )
            gbm.fit(X_clean, y, sample_weight=sw)
            bundle['safety_model'] = gbm
            bundle['safety_imputer'] = imp
            bundle['safety_feature_cols'] = clean_cols
            bundle['safety_imputer_cols'] = list(X.columns)

        else:  # efficacy ensemble
            gbm = GradientBoostingClassifier(
                n_estimators=1000, max_depth=3, learning_rate=0.01,
                subsample=0.8, random_state=42,
            )
            gbm.fit(X_clean, y, sample_weight=sw)
            bundle['efficacy_gbm'] = gbm

            if HAS_XGB:
                xgb = XGBClassifier(
                    n_estimators=1000, max_depth=3, learning_rate=0.01,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric='logloss', verbosity=0,
                )
                xgb.fit(X_clean, y, sample_weight=sw)
                bundle['efficacy_xgb'] = xgb

            if HAS_LGBM:
                lgbm = LGBMClassifier(
                    n_estimators=1000, max_depth=3, learning_rate=0.01,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    verbose=-1,
                )
                lgbm.fit(X_clean, y, sample_weight=sw)
                bundle['efficacy_lgbm'] = lgbm

            bundle['efficacy_imputer'] = imp
            bundle['efficacy_feature_cols'] = clean_cols
            bundle['efficacy_imputer_cols'] = list(X.columns)

    # ── 4. Store versions ───────────────────────────────────────────────────
    print("\n[4/5] Recording versions...")
    import sklearn, xgboost, lightgbm
    bundle['versions'] = {
        'sklearn': sklearn.__version__,
        'xgboost': xgboost.__version__,
        'lightgbm': lightgbm.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
    }
    print(f"  {bundle['versions']}")

    # ── 5. Save ─────────────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent / 'model_bundle.pkl'
    print(f"\n[5/5] Saving bundle to {out_path}...")
    with open(out_path, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Bundle size: {size_mb:.1f} MB")

    # Verify roundtrip
    print("\n  Verifying roundtrip...")
    with open(out_path, 'rb') as f:
        loaded = pickle.load(f)
    assert set(loaded.keys()) == set(bundle.keys()), "Key mismatch!"
    assert len(loaded['safety_feature_cols']) == len(bundle['safety_feature_cols'])
    assert len(loaded['efficacy_feature_cols']) == len(bundle['efficacy_feature_cols'])
    print(f"  OK: safety={len(loaded['safety_feature_cols'])} features, "
          f"efficacy={len(loaded['efficacy_feature_cols'])} features")
    print("\nDone.")


if __name__ == '__main__':
    build_bundle()
