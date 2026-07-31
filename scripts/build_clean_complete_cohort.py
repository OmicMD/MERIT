#!/usr/bin/env python3
"""Build a 100%-molecular-feature-complete training cohort (Gabe's standing rule:
feedback_100pct_feature_complete — no silent median-imputation of molecular features).

Steps (all verifiable, reproducible):
  1. Recover network/STRING features for drugs whose raw STAR network TSVs existed but were never
     joined (the full-IK-vs-ik14 merge bug; recovered consistently via
     scripts/append_network_features_new_drugs.py -> network_enrichment_features_v5.csv).
  2. Apply the 2 audited efficacy label corrections (model-was-right mislabels:
     baricitinib/SLE = failed BRAVE-II; dydrogesterone p=0.538).
  3. DROP every row whose compound is still missing ANY molecular feature (binding/tissue/network/
     DruMAP) — these need server re-runs (DruMAP 'alien' server / STAR Binding) and are excluded
     explicitly with a flag rather than silently imputed. (The 4 Binding-crash = BioTransformer-overflow
     molecules with no usable core; the 33 DruMAP-missing = never-run PK.)
Out: data/sources/training_dataset_v8_clean.csv (+ exclusion manifest).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/sources/training_dataset_v8_honest_exposure.csv"
NETV5 = ROOT / "data/models/network_enrichment_features_v5.csv"
OUT = ROOT / "data/sources/training_dataset_v8_clean.csv"
CORR = ROOT / "data/sources/efficacy_label_corrections_fpreaudit_jun14.csv"


def ik14(s):
    m = Chem.MolFromSmiles(str(s))
    return inchi.MolToInchiKey(m)[:14] if m else None


def main():
    df = pd.read_csv(SRC, low_memory=False)
    df["_ik14"] = df["SMILES"].map(ik14)
    netcols = [c for c in df.columns if c.startswith("net_")]
    binding = [c for c in df.columns if (c.startswith("binding_") or c.startswith("tox_")) and not c.endswith("_xdose")]
    tissue = [c for c in df.columns if c.startswith(("weighted_score_", "mean_", "min_"))]
    drumap = [c for c in df.columns if c.startswith("drumap_")]

    # --- 1. recover network from the (updated) v5 table for drugs missing net_ but present there ---
    v5 = pd.read_csv(NETV5)
    v5["_ik14"] = v5["InChIKey"].astype(str).str[:14]
    v5net = [c for c in netcols if c in v5.columns]
    v5map = v5.dropna(subset=v5net, how="all").drop_duplicates("_ik14").set_index("_ik14")[v5net]
    need = df[df[netcols].isna().any(axis=1) & df[binding].notna().all(axis=1)]
    recov = [ik for ik in need["_ik14"].dropna().unique() if ik in v5map.index]
    n_patched = 0
    for ik in recov:
        rowmask = (df["_ik14"] == ik)
        vals = v5map.loc[ik]
        for c in v5net:
            df.loc[rowmask, c] = vals[c]
        n_patched += int(rowmask.sum())
    print(f"network recovered: {len(recov)} compounds, {n_patched} rows patched from v5 table")

    # --- 2. apply audited label corrections (efficacy FP re-audit + safety attribution) ---
    corr_files = [CORR, ROOT / "data/sources/safety_label_corrections_attribution_jun14.csv"]
    cm = {}
    for cf in corr_files:
        if cf.exists():
            cm.update(dict(pd.read_csv(cf)[["NCT_ID", "new_outcome"]].values))
    if cm:
        m = df["NCT_ID"].isin(cm)
        for nct, out in cm.items():
            df.loc[df["NCT_ID"] == nct, "Corrected_Outcome"] = out
        print(f"label corrections applied: {m.sum()} rows ({list(cm)})")

    # --- 3. drop rows still missing ANY molecular feature (explicit exclusion, no imputation) ---
    molcols = binding + tissue + netcols + drumap
    incomplete = df[molcols].isna().any(axis=1)
    excl = df[incomplete].copy()
    manifest = (excl.assign(
        missing=excl.apply(lambda r: ";".join(
            g for g, cs in [("binding", binding), ("tissue", tissue), ("network", netcols), ("drumap", drumap)]
            if r[cs].isna().any()), axis=1))
        [["NCT_ID", "Drug_Clean", "Corrected_Outcome", "missing"]])
    manifest.to_csv(ROOT / "data/sources/excluded_incomplete_jun14.csv", index=False)
    clean = df[~incomplete].drop(columns=["_ik14"]).copy()
    clean.to_csv(OUT, index=False)

    print(f"excluded (still molecularly-incomplete): {excl['SMILES'].nunique()} compounds, {len(excl)} rows")
    print(f"  excluded safety pos: {excl.Corrected_Outcome.isin(['FAIL_SAFETY','FAIL_BOTH']).sum()} | "
          f"efficacy pos: {excl.Corrected_Outcome.isin(['FAIL_EFFICACY','FAIL_BOTH']).sum()}")
    # verify 100% molecular complete
    assert not clean[molcols].isna().any().any(), "still incomplete!"
    print(f"CLEAN cohort: {len(clean)} rows, {clean.SMILES.nunique()} compounds — 100% molecular-complete (verified)")
    print(f"  safety pos {clean.Corrected_Outcome.isin(['FAIL_SAFETY','FAIL_BOTH']).sum()} | "
          f"efficacy pos {clean.Corrected_Outcome.isin(['FAIL_EFFICACY','FAIL_BOTH']).sum()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
