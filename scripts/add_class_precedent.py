#!/usr/bin/env python3
"""Gap D (Jun 28): precedent_neg_class — a leak-safe, as-of-date, NEGATIVE class-precedent flag.

precedent_neg_class = 1 iff, BEFORE this trial started, a DIFFERENT drug that shares a MOA target
(same molecular class) already had a Phase-3 EFFICACY failure in the SAME indication; else 0.

This encodes specific, transferable prior knowledge ("the class has already failed here"), distinct
from generic disease difficulty. Proof case: upadacitinib -> SLE — JAK inhibitors baricitinib
(SLE-BRAVE, FAIL 2018) and GSK2586184 (FAIL 2013) failed Phase-3 SLE before upadacitinib's 2023
trial; the decisive negative signal is the class precedent, not a target->disease genetic-driver
argument. (The model already predicts this pair FAIL; the feature supplies the correct rationale and
generalises the signal.)

Leak controls (the inClinico precedent caution, CLAUDE.md #8 — precedent can be an establishment/
crowdedness proxy):
  - AS-OF-DATE: only precedents with Start_Year strictly earlier than the index trial count
    (a failure that had not happened yet cannot be known) -> genuine prior information.
  - NEGATIVE-ONLY: counts only prior FAILURES, never "class has been tried" (which would proxy
    establishment/success). A targetless drug or a never-before-tried class scores 0.
  - BINARY, not count: the raw count is noise-diluted (a class with 7 prior MDD failures can still
    pass) and is REDUNDANT with endpoint_difficulty_tier (count coef p=0.21). The binary flag is
    NON-redundant: logit fail ~ has_np + endpoint_difficulty_tier(+oncology+mech_coverage) gives
    has_np coef ~0.65, p=0.007-0.012.
Leak battery (efficacy cohort): WITH-precedent fail 0.278 vs 0.162, shuffle p=0.004, survives
within-Phase-3 (0.26 vs 0.13), availability AUC 0.511 (<0.58). Protected for efficacy/overall via
the precedent_ prefix in retrain_calibrated / prereg_C_lock (NOT in the noisy-OR safety head).
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_MOA = pd.read_csv(ROOT / 'data/sources/ik14_moa_targets_combined_v1.csv')
IK2TG = _MOA.groupby('ik14').target_gene.apply(lambda s: set(x for x in s.dropna())).to_dict()
_FAIL = ('FAIL_EFFICACY', 'FAIL_BOTH')


def _historical_failures(cohort):
    """List of (ik14, disease, start_year, target_set) for every completed Phase-3 efficacy
    failure in the training cohort — the as-of-date precedent pool. Outcome-blind to the index
    trial (only earlier-starting OTHER drugs are ever matched in _flag)."""
    eff = cohort[cohort['Corrected_Outcome'].isin(_FAIL)].drop_duplicates('NCT_ID')
    out = []
    for r in eff.itertuples():
        yr = getattr(r, 'Start_Year', None)
        if pd.isna(yr):
            continue
        ik = str(r.feature_IK)[:14]
        out.append((ik, str(r.Disease), yr, IK2TG.get(ik, set())))
    return out


def _flag(ik14, disease, start_year, fails):
    if pd.isna(start_year):
        return 0
    tg = IK2TG.get(ik14, set())
    if not tg:
        return 0                                   # targetless -> no class precedent asserted
    dis = str(disease)
    for fik, fdis, fyr, ftg in fails:
        if fik != ik14 and fdis == dis and fyr < start_year and (tg & ftg):
            return 1
    return 0


def add_class_precedent_columns(df, cohort=None):
    """Append precedent_neg_class. The precedent pool defaults to df itself (canonical build);
    for the prereg forward pairs pass the training cohort so the pool is the completed trials."""
    fails = _historical_failures(cohort if cohort is not None else df)
    ik14 = df['feature_IK'].astype(str).str[:14]
    df['precedent_neg_class'] = [
        _flag(k, d, y, fails) for k, d, y in zip(ik14, df['Disease'], df.get('Start_Year'))]
    return df


def main():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    inp = ROOT / 'data/sources/training_dataset_v8_clean_mort.csv'
    df = pd.read_csv(inp, low_memory=False)
    df = add_class_precedent_columns(df)
    print(f'precedent_neg_class dist: {df.precedent_neg_class.value_counts().to_dict()}')
    eff = df[df.Corrected_Outcome.isin(['PASS', *_FAIL])].drop_duplicates('NCT_ID').copy()
    eff['fail'] = (eff.Corrected_Outcome != 'PASS').astype(int)
    print('[discrimination]')
    print(eff.groupby('precedent_neg_class').agg(n=('fail', 'size'), fail=('fail', 'mean')).round(3).to_string())
    rng = np.random.default_rng(0)
    hp = eff.precedent_neg_class.values
    real = eff.fail.values[hp == 1].mean() - eff.fail.values[hp == 0].mean()
    null = np.array([eff.fail.values[rng.permutation(hp) == 1].mean()
                     - eff.fail.values[rng.permutation(hp) == 0].mean() for _ in range(5000)])
    print(f'[shuffle] fail-gap {real:+.3f} | p={(np.abs(null) >= abs(real)).mean():.4f}')
    print(f'[availability] has-precedent vs fail AUC: '
          f'{roc_auc_score(eff.fail, hp):.3f} (want <0.58)')


if __name__ == '__main__':
    main()
