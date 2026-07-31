#!/usr/bin/env python3
"""Append leak-free published disease-mortality features (disease_mortality_1y/5y) to the clean cohort.
Source: data/sources/disease_mortality_1y5y_v1.csv (web-verified SEER/GBD/WHO/registry 1y & 5y mortality
per indication, scored blind to any drug/trial/outcome; 100% cohort coverage). Leak-free + non-circular
(external epidemiology). Banked for EFFICACY/overall (shuffle-clean +0.005); SAFETY-neutral (redundant
with disease_is_*); also the decision-layer risk-tolerance basis. notes/disease_mortality_feature_jun14.md
"""
import pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
df=pd.read_csv(ROOT/'data/sources/training_dataset_v8_clean.csv',low_memory=False)
m=pd.read_csv(ROOT/'data/sources/disease_mortality_1y5y_v1.csv')
df['disease_mortality_1y']=df.Disease.map(dict(zip(m.disease,m.mortality_1y)))
df['disease_mortality_5y']=df.Disease.map(dict(zip(m.disease,m.mortality_5y)))
assert df[['disease_mortality_1y','disease_mortality_5y']].notna().all().all(), 'incomplete mortality coverage'
df.to_csv(ROOT/'data/sources/training_dataset_v8_clean_mort.csv',index=False)
print('wrote training_dataset_v8_clean_mort.csv (mortality features, 100% coverage)')
