#!/usr/bin/env python3
"""LEAD EXPERIMENT (Jun 6): does the leak-free OpenTargets genetic-association score
separate PASS from FAIL_EFFICACY on the NOVEL-TARGET subset, where survivorship has
NOT compressed the variance?

Population result was null (feature_catalog). Hypothesis (3-layer framing, L1):
established-target trials are survivorship-selected (their targets are genetically
validated by construction -> no variance -> AUC~0.5); novel/unproven targets retain
real variance, so genetic causality should discriminate there.

Three disciplined tests, all leak-controlled, NO peeking at outcome to define subsets:
  A. DECOMPRESSION: univariate genetic AUC on novel- vs established-target arms.
  B. ORTHOGONALITY: within-disease — does genetic separate after the disease
     base-rate (difficulty prior) is removed? (the production null's redundancy q)
  C. PROXY CHECK: coverage/availability AUC < 0.58 ?
Target novelty = # distinct approved (max_phase 4) molecules hitting the drug's
target gene(s) in ChEMBL. Lower = more novel/unproven target.
"""
from __future__ import annotations
import json, sqlite3, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_arm_level import prepare_arm_dataset
from compute_opentargets_assoc import load_ot, drug_target_genes
CHEMBL = ROOT / "data/cache/chembl_36/chembl_36_sqlite/chembl_36.db"


def auc(y, x):
    if len(set(y)) < 2 or np.nanstd(x) == 0:
        return float("nan")
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


def build():
    conn = sqlite3.connect(CHEMBL)
    mech = json.load(open(ROOT / "data/cache/chembl_mechanisms.json"))
    lk = pd.read_csv(ROOT / "data/sources/chembl_smiles_lookup.csv")
    n2c = {}
    for _, r in lk.iterrows():
        for k in ("Drug_Clean", "chembl_pref_name"):
            n = str(r.get(k) or "").strip().lower()
            if n and pd.notna(r.get("chembl_id")) and n not in n2c:
                n2c[n] = r["chembl_id"]
    resolve = drug_target_genes(conn, mech, n2c)
    # gene -> # distinct approved (max_phase=4) molecules (target establishedness)
    gene_napp = {}
    for g, n in conn.execute(
        "SELECT cs.component_synonym, COUNT(DISTINCT dm.molregno) "
        "FROM component_synonyms cs JOIN target_components tc ON cs.component_id=tc.component_id "
        "JOIN target_dictionary td ON tc.tid=td.tid JOIN drug_mechanism dm ON dm.tid=td.tid "
        "JOIN molecule_dictionary md ON dm.molregno=md.molregno "
        "WHERE cs.syn_type='GENE_SYMBOL' AND md.max_phase=4 GROUP BY cs.component_synonym"):
        gene_napp[g] = n
    df = prepare_arm_dataset()
    feat = pd.read_csv(ROOT / "data/models/ot_genetic_feature.csv")
    df = df.merge(feat, on=["NCT_ID", "Arm_Label"], how="left")
    tc = {}
    nap = []
    for _, r in df.iterrows():
        inv = r["Investigational_Drugs"]
        if inv not in tc:
            tc[inv] = resolve(inv)
        genes = tc[inv]
        nap.append(max((gene_napp.get(g, 0) for g in genes), default=np.nan) if genes else np.nan)
    df["target_n_approved"] = nap
    conn.close()
    return df


def main():
    df = build()
    excl = pd.Series(False, index=df.index)
    for c in ("is_anti_pathogen", "is_endogenous"):
        if c in df:
            excl |= df[c] == 1
    e = df[~excl & df.Corrected_Outcome.isin(["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    e["_y"] = e.Corrected_Outcome.isin(["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)

    # ---- PROXY CHECK (C): is coverage itself outcome-correlated?
    e["_covered"] = e.ot_genetic_score.notna().astype(int)
    print(f"=== C. PROXY CHECK ===\ncoverage availability AUC: {auc(e._y, e._covered):.3f} "
          f"(covered fail {e[e._covered==1]._y.mean():.1%} vs uncovered {e[e._covered==0]._y.mean():.1%})")

    cov = e[e.ot_genetic_score.notna() & e.target_n_approved.notna()].copy()
    nonco = cov[cov.get("disease_is_oncology", 0) != 1].copy()
    print(f"\ncovered+novelty arms: {len(cov)} (pos {int(cov._y.sum())}); non-onco {len(nonco)} (pos {int(nonco._y.sum())})")
    print(f"target_n_approved dist: {cov.target_n_approved.describe()[['25%','50%','75%','max']].to_dict()}")

    # ---- DECOMPRESSION (A): genetic AUC by target-novelty tier (non-onco, leak-free univariate)
    print("\n=== A. DECOMPRESSION — univariate genetic AUC by target novelty (NON-ONCO) ===")
    for lab, mask in [("NOVEL  target_n_approved<=1", nonco.target_n_approved <= 1),
                      ("MID    2-5",  nonco.target_n_approved.between(2, 5)),
                      ("ESTAB  >5",   nonco.target_n_approved > 5)]:
        s = nonco[mask]
        a = auc(s._y, s.ot_genetic_score)
        z = s[s.ot_genetic_score == 0]; p = s[s.ot_genetic_score > 0]
        print(f"  {lab:28s} n={len(s):4d} pos={int(s._y.sum()):3d}  genetic-AUC={a:.3f}"
              f"  | fail g==0 {z._y.mean():.1%} (n{len(z)}) vs g>0 {p._y.mean():.1%} (n{len(p)})")

    # ---- ORTHOGONALITY (B): within-disease — control for disease base rate
    print("\n=== B. ORTHOGONALITY — within-disease (removes disease-difficulty prior) ===")
    # diseases with >=15 covered non-onco arms and mixed genetic
    g = nonco.groupby("Disease")
    keep = [d for d, x in g if len(x) >= 15 and (x.ot_genetic_score > 0).sum() >= 3 and (x.ot_genetic_score == 0).sum() >= 3]
    wd = nonco[nonco.Disease.isin(keep)].copy()
    print(f"  {len(keep)} diseases qualify, {len(wd)} arms")
    # within-disease centered genetic score & outcome -> pooled within AUC
    wd["_gc"] = wd.groupby("Disease").ot_genetic_score.transform(lambda s: s - s.mean())
    print(f"  within-disease-centered genetic AUC: {auc(wd._y, wd._gc):.3f}")
    # per-disease fail rate g==0 vs g>0
    rows = []
    for d in keep:
        x = wd[wd.Disease == d]
        rows.append((d, len(x), x[x.ot_genetic_score == 0]._y.mean(), x[x.ot_genetic_score > 0]._y.mean()))
    rep = pd.DataFrame(rows, columns=["disease", "n", "fail_g0", "fail_gpos"])
    rep["delta"] = rep.fail_g0 - rep.fail_gpos
    print(rep.sort_values("delta", ascending=False).to_string(index=False))
    print(f"  mean within-disease delta (g0 - gpos): {rep.delta.mean():+.3f} "
          f"({(rep.delta>0).sum()}/{len(rep)} diseases favor genetic)")

if __name__ == "__main__":
    main()
