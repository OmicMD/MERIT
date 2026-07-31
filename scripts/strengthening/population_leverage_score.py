#!/usr/bin/env python3
"""Population leverage: is a targeted drug tested in a population where its target is the driver? (Jun 17)

For targeted drug classes (data/sources/drug_target_population_v1.csv), score each trial:
  +1 ON-target population  = the drug's responsive biomarker is REQUIRED in eligibility/title/conditions,
                             OR the indication is the canonical driver disease for that target.
  -1 OFF-target population = a targeted drug given to an all-comers / wrong-driver population.
   0 not a covered targeted drug.

Leak-safe: drug pharmacology (the map) + trial protocol text (eligibility/title/conditions from
data/cache/ctgov_elig/), both pre-trial and blind to outcome. Validated (within-phase gate): Phase-3
off-target 24% fail (n=17) vs on-target 9% (n=43); flag AUC 0.706 vs model 0.584 on the 69 targeted trials.
gating_strictness column marks classes whose benefit is strictly biomarker-gated (EGFR/ALK/BRAF/KRAS/FLT3/
HER2) vs softly enriched (PARP/CDK4-6/JAK) -- the off-target failure signal is sharpest for strict classes.
"""
import json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAP = pd.read_csv(ROOT / 'data/sources/drug_target_population_v1.csv')
ED = ROOT / 'data/cache/ctgov_elig'


def _fields(nct):
    f = ED / f'{nct}.json'
    if not f.exists():
        return ''
    try:
        d = json.load(open(f)).get('protocolSection', {})
        e = d.get('eligibilityModule', {}).get('eligibilityCriteria') or ''
        t = d.get('identificationModule', {}).get('briefTitle') or ''
        c = ' '.join(d.get('conditionsModule', {}).get('conditions', []) or [])
        return (e + ' ' + t + ' ' + c).lower()
    except Exception:
        return ''


def population_leverage(drug_name, disease, nct, text=None):
    """Return (score in {-1,0,1}, drug_class or None). 0 = not a covered targeted drug."""
    dl = str(drug_name).lower()
    for _, r in MAP.iterrows():
        if re.search(r.drug_regex, dl):
            tx = _fields(nct) if text is None else text
            on = bool(re.search(r.biomarker_regex, tx)) or (
                isinstance(r.driver_indication_regex, str)
                and r.driver_indication_regex != '(?!)'
                and bool(re.search(r.driver_indication_regex, str(disease).lower())))
            return (1 if on else -1), r.drug_class
    return 0, None


def add_population_leverage_columns(df):
    out = [population_leverage(d, dis, n) for d, dis, n in zip(df['Drug_Clean'], df['Disease'], df['NCT_ID'])]
    df['population_leverage'] = [o[0] for o in out]
    return df


def main():
    df = pd.read_csv(ROOT / 'data/sources/training_dataset_v8_clean_mort.csv', low_memory=False)
    eff = df[df.Corrected_Outcome.isin(['PASS', 'FAIL_EFFICACY', 'FAIL_BOTH'])].drop_duplicates('NCT_ID').copy()
    eff['fail'] = (eff.Corrected_Outcome != 'PASS').astype(int)
    eff = add_population_leverage_columns(eff)
    cov = eff[eff.population_leverage != 0]
    print(f'covered targeted trials: {len(cov)}')
    print(cov.groupby('population_leverage').agg(n=('fail', 'size'), fail=('fail', 'mean')).round(3).to_string())
    p3 = cov[cov.Phase.astype(str).str.contains('3')]
    print('within Phase 3:')
    print(p3.groupby('population_leverage').agg(n=('fail', 'size'), fail=('fail', 'mean')).round(3).to_string())


if __name__ == '__main__':
    main()
