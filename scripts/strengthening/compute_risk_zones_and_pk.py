"""Compute risk zones (Pass A) and precision@k for Supplementary Table S2.

Jul 2026: repointed from the retired production_v2 crosstask parquet to the single-head
canonical's per-task OOF parquets (oof_overall / oof_safety / oof_efficacy). The safety head
is now a single joint GBM (not the prior noisy-OR). Precision@k ranks within each task's own
cohort, so it reproduces exactly from the separate parquets; the Pass-B joint safety x efficacy
zones require cross-task predictions on every trial (the canonical is trained --skip-crosstask,
so a FAIL_EFFICACY trial has no safety score) and are therefore approximate here (a trial
outside a task's cohort is treated as low-risk for that task). S2 depends only on precision@k.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CANON = ROOT / 'results/production_v8_clean_mort_singlehead_jul6'
BASE = CANON

def _per_trial(task):
    d = pd.read_parquet(CANON / f'oof_{task}.parquet')
    return d.groupby('row_idx').agg(y=('y', 'first'), p=('calibrated_prob', 'mean'))

# Average each task's calibrated probability across seeds per trial, then join on row_idx.
# in_safety / in_efficacy mark whether the trial is in that task's cohort (has a score).
agg = (_per_trial('overall').rename(columns={'y': 'y_overall', 'p': 'p_overall'})
       .join(_per_trial('safety').rename(columns={'y': 'y_safety', 'p': 'p_safety'}), how='left')
       .join(_per_trial('efficacy').rename(columns={'y': 'y_efficacy', 'p': 'p_efficacy'}), how='left')
       .reset_index())
agg['in_safety'] = agg['p_safety'].notna()
agg['in_efficacy'] = agg['p_efficacy'].notna()

# ---------- Pass A: binned overall calibrated probability ----------
edges = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.01]
labels = ['[0,0.05)', '[0.05,0.10)', '[0.10,0.20)', '[0.20,0.30)', '[0.30,0.50)', '[0.50,1]']
agg['pass_a_zone'] = pd.cut(agg['p_overall'], bins=edges, labels=labels, right=False, include_lowest=True)
rows_a = []
for z in labels:
    sub = agg[agg['pass_a_zone'] == z]
    if len(sub) == 0:
        continue
    rows_a.append({
        'zone_name': z,
        'n_trials': len(sub),
        'n_fail': int(sub['y_overall'].sum()),
        'observed_fail_rate': float(sub['y_overall'].mean()),
        'mean_pred_prob': float(sub['p_overall'].mean()),
    })
pass_a = pd.DataFrame(rows_a)
pass_a.to_csv(BASE / 'risk_zones.csv', index=False)

# ---------- Pass B: joint safety x efficacy zones ----------
SAFETY_T = 0.10
EFFICACY_T = 0.15
def assign_joint(r):
    hs = r['p_safety'] >= SAFETY_T
    he = r['p_efficacy'] >= EFFICACY_T
    if hs and he:
        return 'RED (dual-risk)'
    if he and not hs:
        return 'YELLOW (high-efficacy-risk)'
    if hs and not he:
        return 'ORANGE (high-safety-risk-only)'
    return 'GREEN (low-risk)'
agg['pass_b_zone'] = agg.apply(assign_joint, axis=1)
order = ['GREEN (low-risk)', 'ORANGE (high-safety-risk-only)', 'YELLOW (high-efficacy-risk)', 'RED (dual-risk)']
rows_b = []
for z in order:
    sub = agg[agg['pass_b_zone'] == z]
    if len(sub) == 0:
        continue
    rows_b.append({
        'zone_name': z,
        'n_trials': len(sub),
        'n_fail': int(sub['y_overall'].sum()),
        'observed_fail_rate': float(sub['y_overall'].mean()),
        'observed_safety_fail_rate': float(sub.loc[sub['in_safety'].astype(bool), 'y_safety'].mean()) if sub['in_safety'].any() else float('nan'),
        'observed_efficacy_fail_rate': float(sub.loc[sub['in_efficacy'].astype(bool), 'y_efficacy'].mean()) if sub['in_efficacy'].any() else float('nan'),
        'mean_pred_prob': float(sub['p_overall'].mean()),
        'mean_p_safety': float(sub['p_safety'].mean()),
        'mean_p_efficacy': float(sub['p_efficacy'].mean()),
    })
pass_b = pd.DataFrame(rows_b)
pass_b.to_csv(BASE / 'risk_zones_joint.csv', index=False)

# ---------- Precision@k ----------
ks = [10, 25, 50, 100, 250, 500]
rows_pk = []
tasks = [
    ('overall', agg, 'p_overall', 'y_overall', None),
    ('safety', agg[agg['in_safety'].astype(bool)].copy(), 'p_safety', 'y_safety', None),
    ('efficacy', agg[agg['in_efficacy'].astype(bool)].copy(), 'p_efficacy', 'y_efficacy', None),
]
for name, sub, pcol, ycol, _ in tasks:
    sub = sub.sort_values(pcol, ascending=False)
    base_rate = float(sub[ycol].mean())
    n = len(sub)
    for k in ks:
        k_eff = min(k, n)
        top = sub.head(k_eff)
        n_pos = int(top[ycol].sum())
        rows_pk.append({
            'task': name,
            'k': k,
            'precision': n_pos / k_eff,
            'n_positives_in_top_k': n_pos,
            'base_rate': base_rate,
        })
pk = pd.DataFrame(rows_pk)
pk.to_csv(BASE / 'precision_at_k.csv', index=False)

print("Pass A:")
print(pass_a.to_string(index=False))
print("\nPass B:")
print(pass_b.to_string(index=False))
print("\nPrecision@k:")
print(pk.to_string(index=False))
