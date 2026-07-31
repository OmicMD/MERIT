#!/usr/bin/env python3
"""Append leak-safe ct.gov trial-DESIGN primitives to a training dataset, keyed by NCT_ID.

Motivation (notes/efficacy_signal_is_trial_design_jun11.md): the efficacy ceiling is
substantially trial-CONTEXT, not mechanism — 17% of efficacy fails are the same drug+disease
as a pass, so molecular features are byte-identical across a drug's pass/fail trials. Trial-design
primitives (comparator type, blinding rigor, eligibility restrictiveness, arms, endpoint count)
add +0.020 drug-grouped over the full production model.

These are STRUCTURED ct.gov fields + counts (domain primitives), NOT keyword classification.
The detection-confounded d_placebo (jun11 posted-p-value triage over-sampled placebo-RCTs) is
EXCLUDED from the wired set; it is computed but not written unless --with-placebo is passed.

Usage: python3 scripts/append_design_features.py --data <csv> [--with-placebo]
Re-runnable: drops any existing design_* columns before re-appending.
Cache: data/cache/ctgov_design_v2.json (pull via scripts/pull_ctgov_design.py).
"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'data/cache/ctgov_design_v2.json'

def tf_weeks(s):
    if not isinstance(s,str): return np.nan
    m=re.search(r'(\d+\.?\d*)\s*(week|month|year|day)',s.lower())
    if not m: return np.nan
    return float(m.group(1))*{'day':1/7,'week':1,'month':4.345,'year':52.14}[m.group(2)]

# Wired de-confounded set (excludes design_has_placebo by default)
def build_row(v):
    if not isinstance(v,dict) or 'error' in v:
        return {k:np.nan for k in COLS}
    im=v.get('intervention_model')
    return {
        'design_n_arms':            v.get('n_arms'),
        'design_active_comparator': v.get('has_active_comparator'),
        'design_randomized':        (1.0 if v.get('allocation')=='RANDOMIZED' else (0.0 if v.get('allocation') else np.nan)),
        'design_single_group':      (1.0 if im=='SINGLE_GROUP' else (0.0 if im else np.nan)),
        'design_n_masked':          v.get('n_masked'),
        'design_n_primary':         v.get('n_primary_outcomes'),
        'design_n_secondary':       v.get('n_secondary_outcomes'),
        'design_primary_weeks':     tf_weeks(v.get('primary_timeframe')),
        'design_max_age':           v.get('max_age_y'),
        'design_n_elig_criteria':   v.get('n_elig_criteria'),
        'design_has_placebo':       v.get('has_placebo'),  # detection-confounded; kept only with --with-placebo
    }
COLS=['design_n_arms','design_active_comparator','design_randomized','design_single_group',
      'design_n_masked','design_n_primary','design_n_secondary','design_primary_weeks',
      'design_max_age','design_n_elig_criteria','design_has_placebo']

# Combination/add-on intensity (arm-level semantics; Jun 12, notes/efficacy_FN_audit_jun12.md).
# The trial-context FN bucket is dominated by "incremental benefit over a strong backbone is hard"
# (ibrutinib+R-CHOP PHOENIX -> 5 combo partners; niraparib+chemo). These count the OTHER active drugs
# co-administered with the indexed drug in its own arm(s) — a structured arm-intervention quantity,
# NOT keyword text. Leak-clean (availability gate 0.500, value orthogonal to outcome). Single-drug
# trials (absent from the multi-drug arm cache) are monotherapy by definition -> 0 partners. The
# comparator_strength ordinal was EVALUATED and DROPPED: its signal is the placebo level, which
# re-imports the excluded design_has_placebo detection-confound. ARMS cache: ctgov_arms_v1.json.
ARMS_CACHE=ROOT/'data/cache/ctgov_arms_v1.json'
COMBO_COLS=['design_n_combo_partners','design_is_monotherapy']
_PLAC=re.compile(r'placebo|sham|matching|vehicle')

def _combo_row(nct, drug, n_drugs, arms):
    rec=arms.get(nct)
    if not rec or 'arms' not in rec or 'error' in rec:
        # single-drug trials are monotherapy by definition; multi-drug w/o arm data -> unknown
        if n_drugs is not None and n_drugs<=1:
            return {'design_n_combo_partners':0.0,'design_is_monotherapy':1.0}
        return {'design_n_combo_partners':np.nan,'design_is_monotherapy':np.nan}
    dn=re.sub(r'[^a-z0-9]',' ',str(drug).lower())
    dtok=dn.split()[0] if dn.split() else ''
    partners=set()
    matched=False
    for a in rec['arms']:
        act=[i for i in a.get('interventions',[]) if not _PLAC.search(i.lower())]
        if dtok and any(dtok in re.sub(r'[^a-z0-9]',' ',i.lower()) for i in act):
            matched=True
            for i in act:
                ni=re.sub(r'[^a-z0-9]',' ',i.lower())
                if dtok not in ni and i.lower().startswith(('drug','biolog')):
                    partners.add(ni)
    if not matched:
        return {'design_n_combo_partners':np.nan,'design_is_monotherapy':np.nan}
    return {'design_n_combo_partners':float(len(partners)),'design_is_monotherapy':float(len(partners)==0)}

def add_design_columns(df, with_placebo=False, cache=None, add_combo=False):
    """Append design_* primitives to a DataFrame in-place-safe; returns new df.
    Re-runnable: drops any existing design_* columns first. d_has_placebo is
    detection-confounded and excluded unless with_placebo=True. Also appends the
    arm-level combination/add-on intensity primitives (design_n_combo_partners,
    design_is_monotherapy) when the arm cache is present."""
    des=cache if cache is not None else json.load(open(CACHE))
    df=df.drop(columns=[c for c in COLS+COMBO_COLS if c in df.columns])
    feat=pd.DataFrame([build_row(des.get(n)) for n in df['NCT_ID']],index=df.index)
    if not with_placebo:
        feat=feat.drop(columns=['design_has_placebo'])
    out=pd.concat([df,feat],axis=1)
    # design_n_combo_partners / design_is_monotherapy: EVALUATED Jun 12, NEUTRAL -> NOT wired.
    # Leak-clean (availability gate 0.500) but too BLUNT: it pushes ALL combination trials toward
    # FAIL (+0.036 on add-on FNs but +0.019 on add-on PASSes), so efficacy is unchanged (0.786->0.785)
    # and the key PHOENIX ibrutinib+R-CHOP case moved the WRONG way (-0.094). "Is it a combination"
    # does not predict "did the combination fail" — that depends on the specific endpoint/comparator,
    # not the partner count. Set add_combo=True only for diagnostics. notes/efficacy_FN_audit_jun12.md.
    if add_combo and ARMS_CACHE.exists():
        arms=json.load(open(ARMS_CACHE))
        nd=out['trial_n_drugs'] if 'trial_n_drugs' in out.columns else pd.Series([None]*len(out),index=out.index)
        combo=pd.DataFrame([_combo_row(r_nct,r_drug,r_nd,arms)
                            for r_nct,r_drug,r_nd in zip(out['NCT_ID'],out['Drug_Clean'],nd)],index=out.index)
        out=pd.concat([out,combo],axis=1)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--with-placebo',action='store_true')
    a=ap.parse_args()
    df=pd.read_csv(a.data,low_memory=False)
    out=add_design_columns(df,with_placebo=a.with_placebo)
    cov=out['design_n_arms'].notna().mean()
    out.to_csv(a.data,index=False)
    print(f'{a.data}: appended design cols, coverage {cov:.3f}, n={len(out)}')

if __name__=='__main__': main()
