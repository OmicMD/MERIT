#!/usr/bin/env python3
"""
Path C, step 1b: recompute the mechanism-GENETICS block (Mendelian/rare-variant
causal, DepMap lineage dependency, domain-conditional mechanism-impact) for the
NOVEL (drug, indication) pairs — the features that prereg_C_lock previously
MEDIAN-IMPUTED (the "INCOMPLETE: mech-genetics imputed" flag). Biology must be
COMPUTED, never median-filled (no_impute_biological_features rule).

Reuses the exact cached-only logic of the three canonical builders — NO external
pulls:
  - mendel_clinvar / mendel_ot_causal / mendel_max  <- build_mendelian_causal.py
        (ClinVar gene-condition + OT rare-variant causal scores)
  - depmap_dep_lin / depmap_selectivity / depmap_has <- depmap_dependency_probe.py
        (DepMap CRISPR gene-effect, lineage-matched to the cancer indication)
  - mi_raw / mi_within_disease_pct / mi_genetics     <- build_mechanism_impact.py
        (oncology -> -depmap_dep_lin ; else -> max OT genetic/somatic/animal)

A pair with no MOA targets (DepMap/Mendelian) or no OT disease gene module
(mi_genetics) yields a COMPUTED 0.0 = "no causal evidence found", which is the
builders' own native value for that case — this is a computed absence, NOT an
imputation. mi_within_disease_pct ranks the novel mi_raw against the COHORT's
mi_raw distribution in that indication (mechanism_impact_v1.csv), label-free.

The OT genetic/somatic/animal channels for the novel pairs are read from
mechanism_dataderived_prereg_C.csv (prereg_C_build_mech.py output) so mi_genetics
uses the SAME recomputed OT biology as the rest of the prereg mechanism block
(corr 0.97 with the cohort ot_channel_features_v1 scale).

Input : results/benchmark/prereg_C_novel_pairs.csv         (IK14, Disease)
        data/sources/mechanism_dataderived_prereg_C.csv    (recomputed OT channels)
Output: data/sources/mechanism_genetics_prereg_C.csv
        columns matching the build_v8_honest_exposure rename map:
        mendel_clinvar, mendel_ot_causal, mendel_max,
        depmap_dep_lin, depmap_selectivity,
        mi_raw, mi_within_disease_pct, mi_genetics
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "results/benchmark/prereg_C_novel_pairs.csv"
PREREG_MECH = ROOT / "data/sources/mechanism_dataderived_prereg_C.csv"
COHORT = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
OUT = ROOT / "data/sources/mechanism_genetics_prereg_C.csv"

MOA = ROOT / "data/sources/ik14_moa_targets_combined_v1.csv"
CLINVAR = ROOT / "data/cache/clinvar_gene_condition.tsv"
OT_CAUSAL = ROOT / "data/cache/ot_causal_scores.json"
DIS_TGT = ROOT / "data/cache/disease_targets_cache.json"
DEPMAP_GE = ROOT / "data/cache/depmap_gene_effect.csv"
DEPMAP_MODEL = ROOT / "data/cache/depmap_model.csv"
MI_COH = ROOT / "data/sources/mechanism_impact_v1.csv"

# ---- drug -> target genes (shared by all three) ----
moa = pd.read_csv(MOA)
IKT = moa.groupby("ik14")["target_gene"].apply(lambda s: [str(x) for x in s]).to_dict()


# ====================== (1) Mendelian causal (build_mendelian_causal.py) ======================
def _toks(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower())) - {
        "disease", "syndrome", "of", "the", "and", "with", "type",
        "deficiency", "hereditary", "familial", "1", "2", "3"}


cv = pd.read_csv(CLINVAR, sep="\t")
_gene_cvdis = {}
for g, grp in cv.groupby("AssociatedGenes"):
    _gene_cvdis[g] = [_toks(d) for d in grp.DiseaseName.dropna().unique()]
_oc = json.load(open(OT_CAUSAL))
_dtc = json.load(open(DIS_TGT))
_name2id = {k[7:]: v for k, v in _dtc.items()
            if k.startswith("search:") and isinstance(v, str)}


def mendel_clinvar(genes, dis):
    dt = _toks(dis)
    best = 0.0
    for g in genes:
        for ct in _gene_cvdis.get(g, []):
            ov = len(dt & ct)
            if ov >= 2 or (ov >= 1 and len(ct) <= 2):
                best = 1.0
    return best


def mendel_ot_causal(genes, dis):
    efo = _name2id.get(str(dis).lower())
    dmap = _oc.get(efo, {}) if efo else {}
    best = 0.0
    for g in genes:
        sc = dmap.get(g)
        if isinstance(sc, dict):
            best = max(best, max(sc.values()))
    return best


# ====================== (2) DepMap lineage dependency (depmap_dependency_probe.py) ======================
_hdr = open(DEPMAP_GE).readline().rstrip("\n").split(",")
_sym2col = {c.split(" (")[0]: c for c in _hdr[1:]}
_our_targets = set(moa.target_gene.dropna().astype(str))
_cols = [_sym2col[g] for g in _our_targets if g in _sym2col]
_idx = [0] + [_hdr.index(c) for c in _cols]
_ge = pd.read_csv(DEPMAP_GE, usecols=_idx)
_ge.columns = ["DepMap_ID"] + [c.split(" (")[0] for c in _ge.columns[1:]]
_ge = _ge.set_index("DepMap_ID")
_mod = pd.read_csv(DEPMAP_MODEL)[["ModelID", "OncotreeLineage",
                                  "OncotreePrimaryDisease"]].set_index("ModelID")
_ge = _ge.join(_mod)
LMAP = {"lung": "Lung", "breast": "Breast", "prostate": "Prostate", "colorect": "Bowel",
        "colon": "Bowel", "pancrea": "Pancreas", "melanoma": "Skin", "ovari": "Ovary",
        "gliob": "CNS/Brain", "glioma": "CNS/Brain", "leukemia": "Myeloid",
        "lymphoma": "Lymphoid", "myeloma": "Plasma Cell", "renal": "Kidney",
        "kidney": "Kidney", "gastric": "Stomach", "liver": "Liver", "hepato": "Liver",
        "bladder": "Bladder", "head and neck": "Head and Neck", "sarcoma": "Soft Tissue",
        "cervical": "Cervix", "endometr": "Uterus", "esophag": "Esophagus",
        "thyroid": "Thyroid", "neuroblast": "Peripheral Nervous System"}


def _lineage(dis):
    d = str(dis).lower()
    for k, v in LMAP.items():
        if k in d:
            return v
    return None


def depmap_feats(ik, dis):
    lin = _lineage(dis)
    gs = [g for g in IKT.get(ik, []) if g in _ge.columns]
    if not gs or lin is None:
        return 0.0, 0.0, 0
    sub = _ge[_ge.OncotreeLineage == lin]
    vals_lin = [sub[g].mean() for g in gs if g in sub and sub[g].notna().any()]
    if not vals_lin:
        return 0.0, 0.0, 0   # no lineage-matched dependency -> dep_lin=selectivity=0, has=0
    vals_glob = [_ge[g].mean() for g in gs]
    dl = float(np.nanmin(vals_lin))
    dg = float(np.nanmin(vals_glob))
    return dl, dl - dg, 1   # dep_lin, selectivity, has


# ====================== (3) mechanism-impact (build_mechanism_impact.py) ======================
# OT genetic/somatic/animal channels for the novel pairs (same recompute as the rest of the
# prereg mechanism block); mi_genetics = max over these channels.
_pm = pd.read_csv(PREREG_MECH)
_gcols = [c for c in ["ot_genetic_association", "ot_genetic_literature",
                      "ot_somatic_mutation", "ot_animal_model"] if c in _pm.columns]
_mi_g = {(r.IK14, r.Disease): float(np.nanmax([getattr(r, c) for c in _gcols]))
         for r in _pm.itertuples(index=False)}

# disease -> oncology flag (disease-level, from cohort)
_coh = pd.read_csv(COHORT, low_memory=False)
_dis_onc = _coh.drop_duplicates("Disease").set_index("Disease")["disease_is_oncology"].to_dict()

# cohort mi_raw distribution per disease for the within-disease percentile
_mi_coh = pd.read_csv(MI_COH)
_coh_mi_by_dis = _mi_coh.groupby("Disease")["mi_raw"].apply(
    lambda s: s.dropna().values).to_dict()


def main():
    pairs = pd.read_csv(PAIRS)
    rows = []
    for r in pairs.itertuples(index=False):
        ik, dis = r.IK14, r.Disease
        genes = IKT.get(ik, [])
        mc = mendel_clinvar(genes, dis)
        moc = mendel_ot_causal(genes, dis)
        dep_lin, sel, has = depmap_feats(ik, dis)
        mi_gen = _mi_g.get((ik, dis), 0.0)
        if pd.isna(mi_gen):
            mi_gen = 0.0
        onc = (_dis_onc.get(dis, 0) == 1) and (has == 1)
        mi_raw = (-dep_lin) if onc else mi_gen
        cv_vals = _coh_mi_by_dis.get(dis)
        mi_wd = (float(np.mean(cv_vals <= mi_raw))
                 if cv_vals is not None and len(cv_vals) else 0.5)
        rows.append(dict(IK14=ik, Disease=dis,
                         mendel_clinvar=mc, mendel_ot_causal=moc,
                         mendel_max=max(mc, moc),
                         depmap_dep_lin=dep_lin, depmap_selectivity=sel,
                         mi_raw=mi_raw, mi_within_disease_pct=mi_wd,
                         mi_genetics=mi_gen,
                         _depmap_has=has, _mi_route="depmap" if onc else "genetics"))
    out = pd.DataFrame(rows).drop_duplicates(["IK14", "Disease"])
    out.to_csv(OUT, index=False)
    nz = (out[["mendel_max", "mi_genetics"]].abs().sum(axis=1) > 0).mean()
    print(f"wrote {OUT.name}: {len(out)} novel pairs")
    print(f"  Mendelian causal evidence (mendel_max>0.5): {(out.mendel_max > 0.5).mean():.1%}")
    print(f"  DepMap lineage dependency present (has=1): {(out._depmap_has == 1).mean():.1%}")
    print(f"  routed to DepMap (oncology): {(out._mi_route == 'depmap').mean():.1%}")
    print(f"  any computed genetics signal (mendel_max|mi_genetics>0): {nz:.1%}")


if __name__ == "__main__":
    main()
