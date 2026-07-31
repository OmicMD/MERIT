#!/usr/bin/env python3
"""Finalize step: apply the committed *_label_corrections_*.csv glob to the clean_mort cohort.

The corrections are also glob-applied upstream in build_v8_dataset.py, so a full notebook-00 rebuild
produces the identical result; this step lets the canonical clean_mort carry them without re-running the
whole chain. It is provably equivalent because the only effect is setting corrected NCTs to their new
outcome (EXCLUDE rows are then filtered by retrain_calibrated), and no baked clean_mort feature depends on
those trials' outcomes (disease_mortality is external epidemiology; disease-difficulty is computed in-fold).

Idempotent. Writes a .bak once, updates the provenance sidecar. Run: python scripts/apply_label_corrections_to_clean_mort.py
"""
import glob, hashlib, json, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COH = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"

df = pd.read_csv(COH, low_memory=False)
fix = {}
for f in sorted(glob.glob(str(ROOT / "data/sources/*_label_corrections_*.csv"))):
    c = pd.read_csv(f)
    if {"NCT_ID", "new_outcome"}.issubset(c.columns):
        fix.update(dict(zip(c.NCT_ID.astype(str), c.new_outcome)))

changed = 0
for nct, new in fix.items():
    m = df.NCT_ID.astype(str) == nct
    if m.any() and (df.loc[m, "Corrected_Outcome"] != new).any():
        df.loc[m, "Corrected_Outcome"] = new
        changed += int(m.sum())

if changed:
    bak = COH.with_suffix(".csv.prelabelfix.bak")
    if not bak.exists():
        shutil.copy(COH, bak)
    df.to_csv(COH, index=False)
    sha = hashlib.sha256(COH.read_bytes()).hexdigest()
    json.dump({"applied_corrections": len(fix), "rows_changed": changed, "sha256": sha,
               "note": "committed *_label_corrections_*.csv glob applied to clean_mort (equiv. to build_v8_dataset glob)"},
              open(str(COH) + ".labelcorr.provenance.json", "w"), indent=1)
    print(f"applied {len(fix)} corrections, {changed} rows changed. new clean_mort sha {sha[:16]}")
else:
    print("no changes (already applied / idempotent)")
