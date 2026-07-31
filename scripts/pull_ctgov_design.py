#!/usr/bin/env python3
"""Pull ct.gov v2 DESIGN-MODULE primitives for the efficacy cohort (leak-safe,
design-time-fixed trial-context features). Reading the confident efficacy misses
showed they are right-drug/wrong-CONTEXT failures (adult vs adolescent, placebo
vs active-comparator superiority, open-label vs RCT, salvage vs maintenance) —
the molecular features are identical across a drug's pass and fail trials, so the
missing signal is trial design, not mechanism. These are STRUCTURED fields
(arm group types, allocation, intervention model, primary-outcome count/timeframe),
i.e. domain primitives, NOT keyword extraction.

Output: data/cache/ctgov_design.json  {nct: {n_arms, has_active_comparator,
  has_placebo, allocation, intervention_model, primary_purpose, n_primary_outcomes,
  primary_timeframe_text}}  (resumable)
"""
import json, sys, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data/cache/ctgov_design_v2.json'
API='https://clinicaltrials.gov/api/v2/studies/'

def _age_years(s):
    if not isinstance(s,str): return None
    p=s.split()
    try: n=float(p[0])
    except Exception: return None
    u=p[1].lower() if len(p)>1 else 'years'
    return n*{'year':1,'years':1,'month':1/12,'months':1/12,'week':1/52,'weeks':1/52,'day':1/365,'days':1/365}.get(u,1)

def parse_design(ps):
    """Build the design-primitive dict from a ct.gov protocolSection. Shared by the
    live fetch() and any cache-sourced path (e.g. the prospective lock parsing the
    committed ctgov_ongoing_p3.json), so both produce byte-identical design features."""
    dm=ps.get('designModule',{})
    om=ps.get('outcomesModule',{})
    em=ps.get('eligibilityModule',{})
    arms=ps.get('armsInterventionsModule',{}).get('armGroups',[]) or []
    prim=om.get('primaryOutcomes',[]) or []
    sec=om.get('secondaryOutcomes',[]) or []
    di=dm.get('designInfo',{})
    mask=(di.get('maskingInfo') or {})
    types=[ (a.get('type') or '').upper() for a in arms ]
    crit=em.get('eligibilityCriteria') or ''
    # criteria restrictiveness as a COUNT of bullet/numbered items (measured quantity, not classification)
    n_crit=sum(1 for ln in crit.splitlines() if ln.strip().startswith(('*','-','•')))
    return {
        'n_arms': len(arms),
        'has_active_comparator': int(any('ACTIVE_COMPARATOR' in t for t in types)),
        'has_placebo': int(any('PLACEBO' in t for t in types)),
        'has_experimental': int(any('EXPERIMENTAL' in t for t in types)),
        'allocation': di.get('allocation'),
        'intervention_model': di.get('interventionModel'),
        'primary_purpose': di.get('primaryPurpose'),
        'masking': mask.get('masking'),
        'n_masked': len(mask.get('whoMasked') or []),
        'n_primary_outcomes': len(prim),
        'n_secondary_outcomes': len(sec),
        'primary_timeframe': (prim[0].get('timeFrame') if prim else None),
        'min_age_y': _age_years(em.get('minimumAge')),
        'max_age_y': _age_years(em.get('maximumAge')),
        'sex': em.get('sex'),
        'healthy_volunteers': int(bool(em.get('healthyVolunteers'))),
        'n_elig_criteria': n_crit,
        'elig_len': len(crit),
    }

def fetch(nct):
    url=f'{API}{nct}?fields=DesignModule,ArmsInterventionsModule,OutcomesModule,EligibilityModule'
    req=urllib.request.Request(url,headers={'User-Agent':'dt-research/1.0'})
    with urllib.request.urlopen(req,timeout=30) as r:
        d=json.load(r)
    return parse_design(d.get('protocolSection',{}))

def main():
    cache=json.load(open(OUT)) if OUT.exists() else {}
    ncts=set()
    for p in ['data/sources/training_dataset_v8_honest_exposure.csv',
              'data/sources/training_dataset_arm_level.csv']:
        fp=ROOT/p
        if fp.exists():
            ncts |= set(pd.read_csv(fp,low_memory=False)['NCT_ID'].dropna().unique())
    ncts=[n for n in ncts if n not in cache]
    print(f'{len(ncts)} to pull ({len(cache)} cached)',flush=True)
    for i,nct in enumerate(ncts):
        try:
            cache[nct]=fetch(nct)
        except urllib.error.HTTPError as e:
            cache[nct]={'error':e.code}
        except Exception as e:
            cache[nct]={'error':str(e)[:50]}
        if (i+1)%200==0:
            json.dump(cache,open(OUT,'w'))
            print(f'  {i+1}/{len(ncts)}',flush=True)
    json.dump(cache,open(OUT,'w'))
    print('done',len(cache),flush=True)

if __name__=='__main__': main()
