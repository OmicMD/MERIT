#!/usr/bin/env python3
"""Endpoint-PHYSIOLOGY scorer + cardiopulmonary-domain evaluation (Jun 16).

File A (endpoint_physiology_v1): (endpoint, DISEASE) -> demanded rate-limiting physiological process.
File B (target_physiology_v1):   target gene -> controlled physiological process. Both outcome-blind.
Score per trial: +1 the drug's target controls the process the endpoint demands (MATCH) / -1 (MISMATCH) /
0 (endpoint demand unresolved OR drug has no curated target). Disease-conditioned: the SAME endpoint
(6MWD) demands pulmonary_vascular_tone in PAH but cardiac_contractile_function in HF.

Evaluates the cardiopulmonary cluster (the first domain) with the leak gates: availability AUC, shuffle,
within-phase, and NON-REDUNDANCY vs the model (does it disagree with the model where the model is wrong).
"""
import json, sys, re
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
po = json.load(open(ROOT / 'data/cache/ctgov_primary_outcomes_protocol.json'))
A = pd.read_csv(ROOT / 'data/sources/endpoint_physiology_v1.csv').fillna({'disease_keywords': ''})
B = pd.read_csv(ROOT / 'data/sources/target_physiology_v1.csv')
t2p = {r.gene: set(str(r.physiological_process).split(';')) for _, r in B.iterrows()}
moa = pd.read_csv(ROOT / 'data/sources/ik14_moa_targets_combined_v1.csv')
tgtmap = moa.groupby('ik14').target_gene.apply(lambda s: sorted(set(s.dropna()))).to_dict()


def ptitle(nct):
    o = po.get(nct)
    return ' '.join(x.get('title') or '' for x in o if isinstance(x, dict)).lower() if isinstance(o, list) else ''


def demand(nct, disease):
    t, d = ptitle(nct), str(disease).lower()
    for _, r in A.iterrows():
        if not re.search(r.match_keywords, t):
            continue
        if r.disease_keywords and not re.search(r.disease_keywords, d):
            continue
        return r.endpoint_phys, set(str(r.demand_process).split(';'))
    return None, set()


def score_row(nct, disease, ik14):
    ep, dem = demand(nct, disease)
    procs = set().union(*[t2p.get(t, set()) for t in tgtmap.get(ik14, [])]) if tgtmap.get(ik14) else set()
    if not dem or not procs:
        return ep, 0, set()
    ov = procs & dem
    return ep, (1 if ov else -1), ov


def main():
    df = pd.read_csv(ROOT / 'data/sources/training_dataset_v8_clean_mort.csv', low_memory=False)
    eff = df[df.Corrected_Outcome.isin(['PASS', 'FAIL_EFFICACY', 'FAIL_BOTH'])].drop_duplicates('NCT_ID').copy()
    eff['fail'] = (eff.Corrected_Outcome != 'PASS').astype(int)
    eff['ik14'] = eff.feature_IK.astype(str).str[:14]
    eff['tl'] = eff.NCT_ID.map(ptitle)
    CP = (r'6 ?-?minute walk|six.minute walk|6mwd|exercise capacity|peak vo2|cardiopulmonary|'
          r'ejection fraction|cardiac output|ventricular function|ventricular strain|filling pressure|'
          r'probnp|\bbnp\b|fev1|fvc|forced expiratory|forced vital|lung function|spiromet|'
          r'diffusing capacity|peak expiratory|peak flow|pulmonary vascular resist|\bpvr\b')
    cp = eff[eff.tl.str.contains(CP, regex=True, na=False)].copy()
    sc = [score_row(r.NCT_ID, r.Disease, r.ik14) for _, r in cp.iterrows()]
    cp['ep'] = [s[0] for s in sc]; cp['score'] = [s[1] for s in sc]; cp['ov'] = [','.join(s[2]) for s in sc]

    oof = pd.read_parquet(ROOT / 'results/production_v8_clean_mort_biomech_endog_jun16/oof_efficacy.parquet')
    ds = pd.read_csv(ROOT / 'data/sources/training_dataset_v8_clean_mort.csv', low_memory=False)
    ds['row_idx'] = range(len(ds))
    g = oof.groupby('row_idx').agg(p=('raw_prob', 'mean')).reset_index().merge(
        ds[['row_idx', 'NCT_ID']], on='row_idx')
    cp = cp.merge(g[['NCT_ID', 'p']], on='NCT_ID', how='left')

    print(f'CARDIOPULMONARY cluster: {len(cp)} trials | fail {cp.fail.mean():.2f}')
    print(f'  coverage (score != 0): {(cp.score!=0).sum()}/{len(cp)} = {(cp.score!=0).mean():.0%}')
    act = cp[cp.score != 0]
    print('\n=== discrimination (match vs mismatch) ===')
    print(act.groupby('score').agg(n=('fail', 'size'), fail_rate=('fail', 'mean')).round(3).to_string())
    # gates
    rng = np.random.default_rng(42)
    real = act[act.score == -1].fail.mean() - act[act.score == 1].fail.mean()
    null = np.array([act.fail.values[rng.permutation(act.score.values) == -1].mean()
                     - act.fail.values[rng.permutation(act.score.values) == 1].mean() for _ in range(3000)])
    print(f'\n[shuffle] real mismatch-minus-match fail gap {real:+.3f} | null {null.mean():+.3f}+-{null.std():.3f} '
          f'| p={(np.abs(null) >= abs(real)).mean():.4f}')
    cp['has_tgt'] = cp.ik14.map(lambda k: 1 if tgtmap.get(k) else 0)
    cov = cp[cp.score != 0]
    print(f'[availability] has-curated-demand+target vs fail AUC: '
          f'{roc_auc_score(cp.fail, (cp.score!=0).astype(int)):.3f} (want <0.58)')
    print('[within-phase] (Phase 3 only):')
    p3 = act[act.Phase.astype(str).str.contains('3')]
    if len(p3):
        print(p3.groupby('score').agg(n=('fail', 'size'), fail=('fail', 'mean')).round(3).to_string())
    # non-redundancy vs model: does score disagree with the model where the model errs?
    print('\n[non-redundancy vs model] mean model_p by score (model already separate them?):')
    print(act.groupby('score').agg(n=('p', 'size'), model_p=('p', 'mean'), actual_fail=('fail', 'mean')).round(3).to_string())
    print('\n=== per-case (all scored) ===')
    pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 30)
    out = act.sort_values(['score', 'fail'])[['NCT_ID', 'Drug_Clean', 'Disease', 'ep', 'score', 'ov', 'fail', 'p']]
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
