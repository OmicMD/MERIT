#!/usr/bin/env python3
"""Append the endpoint-PHYSIOLOGY score (structural category) to the clean_mort cohort (Jun 16).

endpoint_physiology_score in {-1,0,+1}: for STRUCTURAL-category endpoints (narrow mechanism-specific
surrogates in multi-system diseases: cardiopulmonary 6MWD/FEV1/FVC/PVR/cardiac-fn, renal ADPKD-volume/RPF,
cardiac HCM-obstruction/thrombus, ophthalmic IOP) -> +1 if the drug's target controls the physiological
process the endpoint demands, -1 if not, 0 elsewhere. Restricted to the structural category because the
convergent-symptom domains (pain/GI/psych) are flat (chance) -- the mismatch is only real where the
endpoint is narrow AND decoupled from the disease-benefit pathway.

Files (outcome-blind): data/sources/endpoint_physiology_v1.csv (endpoint,disease->demand) +
data/sources/target_physiology_v1.csv (target->process). Validated (scripts/strengthening/
endpoint_physiology_score.py): structural category 80 trials, mismatch 59% vs match 10% fail, shuffle
p=0.0000, availability AUC 0.507, model_p AUC 0.454 (chance) -> +score 0.775, residual coef -1.36 p<1e-4.
Protected for EFFICACY/overall via the endpoint_ prefix in retrain_calibrated.py.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts/strengthening'))
import endpoint_physiology_score as S  # noqa

STRUCT = ('6mwd', 'fev1', 'fvc', 'pulm_vascular', 'cardiac_function', 'cardiac_energetics',
          'hcm', 'lv_thrombus', 'adpkd', 'renal_plasma', 'bladder', 'intraocular',
          'copd_asthma_exacerbation', 'organ_failure_sofa')
# NOTE: the atherothrombotic/cardioembolic EVENT-RATE category ('cv_event_*', curated in
# endpoint_physiology_v1.csv) is deliberately NOT folded into endpoint_physiology_score: a +1
# STRUCT match (fail 0.11) and a +1 CV-event match (fail 0.22) carry different risk, so overloading
# one column breaks its monotone meaning. CV-event match/mismatch lives in its own protected feature
# endpoint_cvevent_match (scripts/add_cvevent_match.py), built off the same File A/File B curation.


def add_endpoint_physiology_columns(df):
    ik14 = df['feature_IK'].astype(str).str[:14]
    vals = []
    for nct, dis, k in zip(df['NCT_ID'], df['Disease'], ik14):
        ep, score, _ = S.score_row(nct, dis, k)
        vals.append(score if (isinstance(ep, str) and ep.startswith(STRUCT)) else 0)
    df['endpoint_physiology_score'] = vals
    return df


def main():
    inp = ROOT / 'data/sources/training_dataset_v8_clean_mort.csv'
    out = ROOT / 'data/sources/training_dataset_v8_clean_mort_physio.csv'
    df = pd.read_csv(inp, low_memory=False)
    df = add_endpoint_physiology_columns(df)
    print(f'endpoint_physiology_score dist: {df.endpoint_physiology_score.value_counts().to_dict()}')
    df.to_csv(out, index=False)
    print(f'wrote {out.name} (candidate; canonical untouched)')


if __name__ == '__main__':
    main()
