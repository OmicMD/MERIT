import pandas as pd, numpy as np, json
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
df=pd.read_csv('data/sources/training_dataset_v8_clean_mort.csv',low_memory=False)
df['IK14']=df['feature_IK'].astype(str).str[:14]
e=df[df.Corrected_Outcome.isin(['PASS','FAIL_EFFICACY','FAIL_BOTH'])].copy()
base=e[['IK14','Disease','mech_centrality','mech_population_fit','mech_effect_ceiling','mech_fitness']].copy()
def addcsv(path,prefs):
    x=pd.read_csv(path); c=[col for col in x.columns for p in prefs if col.startswith(p)]; c=list(dict.fromkeys(c))
    on=['IK14','Disease'] if 'Disease' in x.columns else ['IK14']
    global base; base=base.merge(x[on+c].drop_duplicates(on),on=on,how='left'); return c
src={}
src['OT-association']=addcsv('data/sources/ot_channel_features_v1.csv',['ot_genetic','ot_somatic','ot_affected','ot_animal','ot_rna'])
src['KEGG-pathway']=addcsv('data/sources/kegg_pathway_features_v1.csv',['kegg_'])
src['Topology(OmniPath)']=addcsv('data/sources/topology_features_v1.csv',['frac_','net_','target_'])
src['Directional-signed']=addcsv('data/sources/directional_mech_v1.csv',['sgn_'])
src['DepMap-dependency']=addcsv('data/sources/depmap_dependency_v1.csv',['depmap_'])
# ATC, GtoPdb, ClinGen, OncoKB inline
moa=pd.read_csv('data/sources/ik14_moa_targets_combined_v1.csv'); ikt=moa.groupby('ik14')['target_gene'].apply(lambda s:[str(x) for x in s]).to_dict()
atc=pd.read_csv('data/sources/cohort_atc_v1.csv'); oh=pd.crosstab(atc.IK14,atc.level1_description).clip(upper=1); oh.columns=[f'atc_{i}' for i in range(oh.shape[1])]
base=base.merge(oh.reset_index(),on='IK14',how='left'); src['ATC-drugclass']=list(oh.columns)
g=pd.read_csv('data/cache/gtopdb_targets_families.csv',skiprows=1,low_memory=False); s2t=dict(zip(g['HGNC symbol'].dropna(),g['Type']))
TYPES=['GPCR','Enzyme','Catalytic receptor','Ion channel','NHR','Transporter']
gc=[]
for t in TYPES: cc=f'gtp_{t[:5]}'; base[cc]=base.IK14.map(lambda ik:int(any(s2t.get(x)==t for x in ikt.get(ik,[])))); gc.append(cc)
src['GtoPdb-physiology']=gc
ok={r['hugoSymbol']:r for r in json.load(open('data/cache/oncokb_cancer_genes.json'))}
base['onc_occ']=base.IK14.map(lambda ik:max([ok.get(x,{}).get('occurrenceCount',0) for x in ikt.get(ik,[])],default=0)); src['OncoKB-driver']=['onc_occ']
# gnomAD constraint (target essentiality/intolerance): LOEUF (oe_lof_upper; lower=more constrained), pLI, lof_z
import subprocess
gn=pd.read_csv('data/cache/gnomad_constraint.txt.bgz',sep='\t',compression='gzip',low_memory=False)[['gene','pLI','oe_lof_upper','lof_z']]
g_pli=dict(zip(gn.gene,pd.to_numeric(gn.pLI,errors='coerce'))); g_loeuf=dict(zip(gn.gene,pd.to_numeric(gn.oe_lof_upper,errors='coerce'))); g_lofz=dict(zip(gn.gene,pd.to_numeric(gn.lof_z,errors='coerce')))
def gnf(ik):
    gs=ikt.get(ik,[]); plis=[g_pli[x] for x in gs if x in g_pli and pd.notna(g_pli[x])]; loeufs=[g_loeuf[x] for x in gs if x in g_loeuf and pd.notna(g_loeuf[x])]; lofzs=[g_lofz[x] for x in gs if x in g_lofz and pd.notna(g_lofz[x])]
    return pd.Series(dict(gnomad_pli_max=max(plis) if plis else np.nan, gnomad_loeuf_min=min(loeufs) if loeufs else np.nan, gnomad_lofz_max=max(lofzs) if lofzs else np.nan))
base[['gnomad_pli_max','gnomad_loeuf_min','gnomad_lofz_max']]=base.IK14.apply(gnf)
src['gnomAD-constraint']=['gnomad_pli_max','gnomad_loeuf_min','gnomad_lofz_max']
# Disease-side pathophysiology: disease class + molecular-driver richness (is the disease modifiable?)
dflags=[c for c in df.columns if c.startswith('disease_is_')]
base=base.merge(e[['IK14','Disease']+dflags].drop_duplicates(['IK14','Disease']),on=['IK14','Disease'],how='left')
dtc=json.load(open('data/cache/disease_targets_cache.json'))
name2id={k[7:]:v for k,v in dtc.items() if k.startswith('search:') and isinstance(v,str)}
id2g={v['disease_id']:v.get('targets',[]) for k,v in dtc.items() if k.startswith('targets:') and isinstance(v,dict)}
def drich(dis):
    efo=name2id.get(str(dis).lower()); ts=id2g.get(efo,[]) if efo else []
    scores=[t.get('score',0) for t in ts]
    return pd.Series(dict(dis_n_drivers=len(ts), dis_max_assoc=max(scores) if scores else 0.0, dis_top5_assoc=np.mean(sorted(scores,reverse=True)[:5]) if scores else 0.0))
base[['dis_n_drivers','dis_max_assoc','dis_top5_assoc']]=base.Disease.apply(drich)
src['Disease-pathophys']=dflags+['dis_n_drivers','dis_max_assoc','dis_top5_assoc']
allc=sum(src.values(),[])
for c in allc: base[c]=pd.to_numeric(base[c],errors='coerce').fillna(0.0)
grp=base.IK14.values
def cvr2(cols,y):
    X=base[cols].values; pr=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,grp):
        m=HistGradientBoostingRegressor(max_iter=150,max_depth=3,learning_rate=0.06,random_state=0);m.fit(X[tr],y[tr]);pr[te]=m.predict(X[te])
    return r2_score(y,pr)
axes={'centrality':'mech_centrality','population_fit':'mech_population_fit','effect_ceiling':'mech_effect_ceiling','fitness':'mech_fitness'}
print('R^2 (compound-grouped CV) explaining each LLM axis, by data source:\n')
hdr='%-22s'%'SOURCE'+''.join('%14s'%a for a in axes); print(hdr); print('-'*len(hdr))
for s,cols in src.items():
    row='%-22s'%s+''.join('%14.3f'%cvr2(cols,base[ax].values) for ax in axes.values()); print(row)
print('-'*len(hdr))
print('%-22s'%'ALL COMBINED'+''.join('%14.3f'%cvr2(allc,base[ax].values) for ax in axes.values()))

# ---- WITHIN-DISEASE recovery: residualize each axis on disease-mean, recover the DRUG-level component ----
# (this is the hard part: telling drugs apart WITHIN a disease; disease-side features can't help by construction)
print('\n\nWITHIN-DISEASE R^2 (axis residualized on disease-mean; the drug-resolution component):\n')
drug_src={k:v for k,v in src.items() if k!='Disease-pathophys'}  # disease-side can't recover within-disease
resid={ax: base[col].values - base.groupby('Disease')[col].transform('mean').values for ax,col in axes.items()}
print(hdr); print('-'*len(hdr))
for s,cols in drug_src.items():
    print('%-22s'%s+''.join('%14.3f'%cvr2(cols,resid[ax]) for ax in axes))
print('-'*len(hdr))
drugall=sum(drug_src.values(),[])
print('%-22s'%'ALL DRUG-SIDE'+''.join('%14.3f'%cvr2(drugall,resid[ax]) for ax in axes))
