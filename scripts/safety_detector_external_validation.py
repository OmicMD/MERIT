#!/usr/bin/env python3
"""Properly-powered test of whether the per-mechanism safety AXES capture real
ORGAN-SPECIFIC toxicity, validated against EXTERNAL ground truth across the whole
profiled cohort (not the 11 rare idiosyncratic trial failures).

Hepatic axis  vs DILIrank (known hepatotoxins).
Cardiac axis  vs hERG blockers.
Key test = DOUBLE DISSOCIATION + specificity: hepatic features should predict DILI
better than cardiac features do, and cardiac features should predict hERG better
than hepatic features do. That grounds the disjunctive decomposition biologically.

Feeds Supplementary Table S3.

NOT CURRENTLY RUNNABLE (Jul 2026): the hepatic half reads data/processed/dilirank_data.csv,
which is absent from the repo and is written by nothing. It cannot be rebuilt from
data/sources/04_dilirank_raw.csv: the join to the cohort is on InChIKey, and the raw
DILIrank export carries CompoundName only, with no structure column. The name ->
InChIKey resolution that produced the processed file was an external step that was never
committed, so the published DILIrank values (Table S3: 0.53 / 0.61, n = 213) cannot be
regenerated or refined to 3 dp. The cardiac (hERG) half reads data/herg.tab and is intact.

The cohort was migrated to the canonical clean_mort dataset (Jul 2026, matching the rest of
the pipeline); it previously read the superseded training_dataset_v8_honest_exposure.csv.
That change is UNTESTED because of the missing DILIrank file above.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
df["IK14"] = df["feature_IK"].astype(str).str[:14]
HEP = ["tox_hepatic_burden", "tox_hepatic_max_bind", "tox_hepatic_n_bound", "tox_hepatic_mean_bind"]
CAR = ["tox_cardiac_burden", "tox_cardiac_max_bind", "tox_cardiac_n_bound", "tox_cardiac_mean_bind"]
PROM = [c for c in df.columns if c.startswith("binding_") and "n_bound" in c][:1]  # promiscuity proxy
keep = [c for c in HEP + CAR + PROM if c in df.columns]
drug = df.groupby("IK14")[keep].mean()
print(f"cohort unique drugs: {len(drug)}  promiscuity proxy: {PROM}\n")

def auc_safe(y, x):
    m = ~(pd.isna(x) | pd.isna(y))
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return np.nan, int(m.sum())
    return roc_auc_score(y[m], x[m]), int(m.sum())

# ---------- HEPATIC: DILIrank ----------
dr = pd.read_csv("data/processed/dilirank_data.csv")
dr["IK14"] = dr["inchikey"].astype(str).str[:14]
concern = {"vMost-DILI-concern": 1, "vLess-DILI-concern": 1, "vNo-DILI-concern": 0}
dr["dili"] = dr["dili_concern"].map(concern)
dr = dr.dropna(subset=["dili"]).drop_duplicates("IK14")[["IK14", "dili"]]
h = drug.join(dr.set_index("IK14"), how="inner")
print(f"=== HEPATIC axis vs DILIrank ===  (n={len(h)}, DILI+={int(h.dili.sum())})")
for f in HEP + CAR + PROM:
    if f in h:
        a, n = auc_safe(h["dili"].values, h[f].values)
        tag = "hepatic" if f in HEP else ("cardiac" if f in CAR else "promisc")
        print(f"  [{tag:7s}] {f:24s} AUC(DILI) = {a:.3f}")

# ---------- CARDIAC: hERG (needs RDKit for SMILES->IK14) ----------
print("\n=== CARDIAC axis vs hERG blockers ===")
try:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchiKey
    hg = pd.read_csv("data/herg.tab", sep="\t")
    def ik14(smi):
        try:
            m = Chem.MolFromSmiles(str(smi).strip('"'))
            return MolToInchiKey(m)[:14] if m else None
        except Exception:
            return None
    hg["IK14"] = hg["Drug"].map(ik14)
    hg = hg.dropna(subset=["IK14"]).drop_duplicates("IK14")
    hg["herg"] = (hg["Y"] > 0.5).astype(int)
    c = drug.join(hg.set_index("IK14")[["herg"]], how="inner")
    print(f"  (n={len(c)}, hERG+={int(c.herg.sum()) if len(c) else 0})")
    if len(c) >= 20:
        for f in CAR + HEP + PROM:
            if f in c:
                a, n = auc_safe(c["herg"].values, c[f].values)
                tag = "cardiac" if f in CAR else ("hepatic" if f in HEP else "promisc")
                print(f"  [{tag:7s}] {f:24s} AUC(hERG) = {a:.3f}")
    else:
        print("  too few cohort drugs overlap the hERG set for a powered test")
except ImportError:
    print("  rdkit unavailable; skipping hERG join")

print("\nDouble dissociation = hepatic feats win on DILI AND cardiac feats win on hERG => "
      "axes are organ-specific (disjunctive decomposition biologically grounded).")
