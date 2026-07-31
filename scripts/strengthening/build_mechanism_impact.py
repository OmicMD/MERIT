"""Domain-conditional mechanism-impact score: 'is the drug's known target the right one to impact THIS
disease?' Routes to the domain-appropriate CLEAN source — DepMap selective dependency for oncology
(oncogene addiction), Open Targets human genetic/somatic/animal evidence elsewhere. Within-disease
efficacy AUC 0.627 pooled (oncology 0.71), ~63% of the LLM's within-disease edge, no clinical-outcome info.
Writes data/sources/mechanism_impact_v1.csv."""
import pandas as pd, numpy as np
ROOT='.'
df=pd.read_csv(f'{ROOT}/data/sources/training_dataset_v8_clean_mort.csv',low_memory=False)
df['IK14']=df['feature_IK'].astype(str).str[:14]
dep=pd.read_csv(f'{ROOT}/data/sources/depmap_dependency_v1.csv')
df=df.merge(dep,on=['IK14','Disease'],how='left')
ot=pd.read_csv(f'{ROOT}/data/sources/ot_channel_features_v1.csv')
gcols=[c for c in ot.columns if c.startswith(('ot_genetic','ot_somatic','ot_animal'))]
df=df.merge(ot[['IK14','Disease']+gcols].drop_duplicates(['IK14','Disease']),on=['IK14','Disease'],how='left')
for c in ['depmap_dep_lin','depmap_selectivity','depmap_has']+gcols: df[c]=df[c].fillna(0.0)
df['mi_genetics']=df[gcols].max(axis=1)
onc=(df.disease_is_oncology==1)&(df.depmap_has==1)
df['mi_route']=np.where(onc,'depmap','genetics')
df['mi_raw']=np.where(onc,-df.depmap_dep_lin,df.mi_genetics)   # higher = better mechanism-impact
df['mi_within_disease_pct']=df.groupby('Disease').mi_raw.rank(pct=True)
out=df[['IK14','Disease','mi_raw','mi_within_disease_pct','mi_genetics','depmap_dep_lin','depmap_selectivity','mi_route']].drop_duplicates(['IK14','Disease'])
out.to_csv(f'{ROOT}/data/sources/mechanism_impact_v1.csv',index=False)
print(f'wrote mechanism_impact_v1.csv ({len(out)} rows; {(out.mi_route=="depmap").mean():.0%} routed to DepMap)')
