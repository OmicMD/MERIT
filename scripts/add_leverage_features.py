#!/usr/bin/env python3
"""Append the two leak-clean leverage-matching features to the canonical clean_mort cohort (Jun 17).

Runs AFTER add_disease_mortality.py in the reproduce chain. Appends, in place:
- endpoint_physiology_score  {-1,0,+1}: does the drug's mechanism move the physiological substrate the
  trial's endpoint measures, for the structural-surrogate category (6 organ systems). Files:
  data/sources/endpoint_physiology_v1.csv + target_physiology_v1.csv. See add_endpoint_physiology.py.
- population_leverage        {-1,0,+1}: is a targeted drug tested in a population where its target is the
  driver (biomarker-required-in-eligibility OR canonical-driver-indication). File:
  data/sources/drug_target_population_v1.csv + data/cache/ctgov_elig/. See population_leverage_score.py.

Both leak-safe (drug pharmacology + pre-trial protocol text, outcome-blind), gated (availability AUC<0.58,
shuffle p<1e-4, survive within-Phase-3), and protected for EFFICACY/overall in retrain_calibrated.py via
the endpoint_ / population_leverage prefixes. They identify the small set where the model is otherwise
INVERTED (leverage-mismatch trials, model AUC 0.35). notes/leverage_matching_framework_jun17.md.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts/strengthening'))
from add_endpoint_physiology import add_endpoint_physiology_columns  # noqa: E402
from add_cvevent_match import add_cvevent_match_columns  # noqa: E402
from add_class_precedent import add_class_precedent_columns  # noqa: E402
import population_leverage_score as PL  # noqa: E402
from endpoint_difficulty import add_endpoint_difficulty_columns  # noqa: E402


def main():
    f = ROOT / 'data/sources/training_dataset_v8_clean_mort.csv'
    df = pd.read_csv(f, low_memory=False)
    n0 = df.shape[1]
    df = add_endpoint_physiology_columns(df)
    df = add_cvevent_match_columns(df)  # Gap B: CV-event mechanism-match (endpoint_ prefix, protected)
    df = add_class_precedent_columns(df)  # Gap D: as-of-date negative class-precedent (precedent_ prefix)
    df = PL.add_population_leverage_columns(df)
    df = add_endpoint_difficulty_columns(df)
    assert {'endpoint_physiology_score', 'population_leverage', 'endpoint_difficulty_tier',
            'endpoint_cvevent_match', 'precedent_neg_class'} <= set(df.columns)
    df.to_csv(f, index=False)
    ep = df.endpoint_physiology_score.value_counts().to_dict()
    pl = df.population_leverage.value_counts().to_dict()
    print(f'appended leverage features to clean_mort (+{df.shape[1]-n0} cols)')
    print(f'  endpoint_physiology_score {ep} | population_leverage {pl}')
    print(f'  any leverage signal: {((df.endpoint_physiology_score!=0)|(df.population_leverage!=0)).sum()} trials')


if __name__ == '__main__':
    main()
