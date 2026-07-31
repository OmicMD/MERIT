import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df=pd.read_csv('data/sources/training_dataset_v8_clean_mort.csv',low_memory=False)
df['IK14']=df['feature_IK'].astype(str).str[:14]
e=df[df.Corrected_Outcome.isin(['PASS','FAIL_EFFICACY','FAIL_BOTH'])].copy()
e['y']=e.Corrected_Outcome.isin(['FAIL_EFFICACY','FAIL_BOTH']).astype(int)
dep=pd.read_csv('data/sources/depmap_dependency_v1.csv'); e=e.merge(dep,on=['IK14','Disease'],how='left')
ot=pd.read_csv('data/sources/ot_channel_features_v1.csv')
gcols=[c for c in ot.columns if c.startswith('ot_genetic') or c.startswith('ot_somatic') or c.startswith('ot_animal')]
e=e.merge(ot[['IK14','Disease']+gcols].drop_duplicates(['IK14','Disease']),on=['IK14','Disease'],how='left')
for c in ['depmap_dep_lin','depmap_has']+gcols: e[c]=e[c].fillna(0.0)
e['ot_impact']=e[gcols].max(axis=1)
# DOMAIN-CONDITIONAL routing: oncology+has-DepMap -> -dep_lin (stronger dep=higher=better); else -> OT genetics
onc=(e.disease_is_oncology==1)&(e.depmap_has==1)
e['impact_raw']=np.where(onc, -e.depmap_dep_lin, e.ot_impact)
# rank-normalize WITHIN disease so scales uniform across domains
e['impact']=e.groupby('Disease').impact_raw.rank(pct=True)
big=e.groupby('Disease').filter(lambda g:g.y.nunique()==2 and len(g)>=6)
# pooled within-disease AUC (impact already disease-rank-normalized; higher impact => pass => lower y)
auc=roc_auc_score(big.y, -big.impact)
print(f'DOMAIN-CONDITIONAL mechanism-impact, pooled within-disease AUC: {auc:.3f}  (naive combined was 0.582; per-domain 0.6-0.74; LLM 0.724)')
# mean of per-disease AUC (weighted by n)
rows=[]
for dis,g in big.groupby('Disease'):
    try: rows.append((dis,len(g),roc_auc_score(g.y,-g.impact_raw)))
    except: pass
pd=__import__('pandas'); R=pd.DataFrame(rows,columns=['dis','n','auc'])
print(f'mean per-disease AUC (n-weighted): {np.average(R.auc,weights=R.n):.3f} across {len(R)} diseases')
# bootstrap
rng=np.random.RandomState(0); a=[]; drugs=big.IK14.unique()
for _ in range(400):
    s=rng.choice(drugs,len(drugs),replace=True); bb=pd.concat([big[big.IK14==d] for d in s])
    if bb.y.nunique()==2: a.append(roc_auc_score(bb.y,-bb.impact))
print(f'drug-bootstrap 95% CI [{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}]')
for lab,m in [('oncology',big.disease_is_oncology==1),('non-onc',big.disease_is_oncology!=1)]:
    s=big[m]
    if s.y.nunique()==2: print(f'  {lab}: {roc_auc_score(s.y,-s.impact):.3f} (n={len(s)})')
