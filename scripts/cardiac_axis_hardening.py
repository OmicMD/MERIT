#!/usr/bin/env python3
"""Harden the cardiac liability axis for manuscript use.

(1) hERG prediction by the cardiac binding panel, CONTROLLED for promiscuity and
    lipophilicity (Caco-2 permeability, brain Kp), drug-resampled bootstrap 95% CI,
    and a residualized-feature test (cardiac signal ABOVE the confounds).
(2) SIDER clinical cardiac-AE cross-check: does the cardiac axis track the count of
    MedDRA Cardiac-disorders terms per drug, MORE than it tracks hepatic AEs
    (organ-specificity), across the cohort.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

RNG = np.random.default_rng(13)
df = pd.read_csv("data/sources/training_dataset_v8_honest_exposure.csv", low_memory=False)
df["IK14"] = df["feature_IK"].astype(str).str[:14]
CARD = ["tox_cardiac_burden", "tox_cardiac_max_bind", "tox_cardiac_n_bound", "tox_cardiac_mean_bind"]
CONF = ["binding_drug_n_bound", "drumap_papp_caco2", "drumap_kpbrain"]   # promiscuity + lipophilicity
drug = df.groupby("IK14")[CARD + CONF].mean()

def imp(X):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]; col[np.isnan(col)] = np.nanmedian(col)
    return X

# ---- hERG join via RDKit ----
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
hg = pd.read_csv("data/herg.tab", sep="\t")
def ik14(s):
    try:
        m = Chem.MolFromSmiles(str(s).strip('"')); return MolToInchiKey(m)[:14] if m else None
    except Exception: return None
hg["IK14"] = hg["Drug"].map(ik14)
hg = hg.dropna(subset=["IK14"]).drop_duplicates("IK14")
hg["y"] = (hg["Y"] > 0.5).astype(int)
d = drug.join(hg.set_index("IK14")["y"], how="inner").dropna(subset=["y"])
y = d["y"].values.astype(int)
print(f"=== (1) hERG validation  (n={len(d)}, hERG+={y.sum()}) ===")

Xc = imp(d[CARD].values); Xf = imp(d[CONF].values)
Xcf = np.hstack([Xf, Xc])

def cv_auc(X, y, reps=30):
    from sklearn.model_selection import StratifiedKFold
    aucs = []
    for r in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=r)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return np.mean(aucs), np.std(aucs)

a_f, s_f = cv_auc(Xf, y); a_cf, s_cf = cv_auc(Xcf, y)
print(f"  confounds only (promiscuity+lipophilicity): CV-AUC {a_f:.3f} ± {s_f:.3f}")
print(f"  confounds + cardiac panel:                  CV-AUC {a_cf:.3f} ± {s_cf:.3f}")
print(f"  cardiac panel ADDS: {a_cf-a_f:+.3f} AUC over confounds")

# residualized single feature: tox_cardiac_n_bound minus its confound-prediction
from sklearn.linear_model import LinearRegression
res = d["tox_cardiac_n_bound"].values - LinearRegression().fit(Xf, d["tox_cardiac_n_bound"].values).predict(Xf)
auc_res = roc_auc_score(y, res)
# bootstrap CI (resample drugs) for residualized-cardiac AUC and raw cardiac-panel AUC
boot_res, boot_panel = [], []
for _ in range(2000):
    idx = RNG.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2: continue
    boot_res.append(roc_auc_score(y[idx], res[idx]))
    # panel: use confound+cardiac OOF proxy via single-split logistic for speed -> use raw best feature
    boot_panel.append(roc_auc_score(y[idx], imp(d[CARD].values)[idx, 2]))  # n_bound
ci = lambda b: (np.percentile(b, 2.5), np.percentile(b, 97.5))
print(f"  tox_cardiac_n_bound raw AUC {roc_auc_score(y, imp(d[CARD].values)[:,2]):.3f}  "
      f"95% CI [{ci(boot_panel)[0]:.3f}, {ci(boot_panel)[1]:.3f}]")
print(f"  cardiac residual (confounds removed) AUC {auc_res:.3f}  "
      f"95% CI [{ci(boot_res)[0]:.3f}, {ci(boot_res)[1]:.3f}]  "
      f"({'EXCLUDES' if ci(boot_res)[0] > 0.5 else 'includes'} 0.5)")

# ---- (2) SIDER clinical cardiac-AE concordance ----
print("\n=== (2) SIDER cardiac-AE concordance ===")
sd = pd.read_csv("data/processed/sider_data.csv")
sd["IK14"] = sd["smiles"].map(ik14)
sd = sd.dropna(subset=["IK14"]).drop_duplicates("IK14")
# MedDRA Cardiac-disorders SOC preferred terms (controlled-vocabulary subset, not free-text regex)
CARDIAC_PT = {"arrhythmia", "atrial fibrillation", "bradycardia", "cardiac arrest",
    "cardiac failure", "cardiac failure congestive", "myocardial infarction",
    "tachycardia", "ventricular tachycardia", "ventricular fibrillation",
    "palpitations", "long qt syndrome", "electrocardiogram qt prolonged",
    "torsade de pointes", "cardiomyopathy", "myocarditis", "angina pectoris",
    "atrioventricular block", "cardiac flutter", "supraventricular tachycardia"}
HEPATIC_PT = {"hepatitis", "hepatic failure", "jaundice", "cholestasis",
    "hepatic enzyme increased", "hepatic function abnormal", "hepatotoxicity",
    "hyperbilirubinaemia", "liver disorder", "hepatic necrosis"}
def count_terms(s, vocab):
    if pd.isna(s): return 0
    terms = {t.strip().lower() for t in str(s).split("|")}
    return len(terms & vocab)
sd["card_ae"] = sd["all_side_effects"].map(lambda s: count_terms(s, CARDIAC_PT))
sd["hep_ae"] = sd["all_side_effects"].map(lambda s: count_terms(s, HEPATIC_PT))
m = drug.join(sd.set_index("IK14")[["card_ae", "hep_ae"]], how="inner")
print(f"  cohort drugs with SIDER: {len(m)}")
rc, pc = spearmanr(m["tox_cardiac_burden"], m["card_ae"], nan_policy="omit")
rh, ph = spearmanr(m["tox_cardiac_burden"], m["hep_ae"], nan_policy="omit")
print(f"  cardiac axis vs SIDER CARDIAC AEs:  rho {rc:+.3f} (p={pc:.3f})")
print(f"  cardiac axis vs SIDER HEPATIC AEs:  rho {rh:+.3f} (p={ph:.3f})  [specificity control]")
print("  organ-specific if cardiac-AE rho > hepatic-AE rho.")
