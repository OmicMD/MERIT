#!/usr/bin/env python3
"""Apply audited safety-label corrections to the v8 trial-level datasets.

Auditable (CLAUDE rule #4): reads an explicit corrections CSV, logs every change,
and writes a provenance sidecar. NOT an opaque overwrite. Corrections come from
the blind symmetric safety-label audit (notes/safety_label_audit_jun7.md):
3 trials whose FAIL_SAFETY label is not a genuine drug-attributable in-trial
safety failure (COVID-funding, investigator-death, post-marketing-temporal) are
relabeled EXCLUDE_NONDRUG_STOP so they drop from ALL task cohorts (no isin match).

The durable fix lives in build_v8_dataset.py (applied to v5_unified -> v8); this
script patches the already-built v8 / v8_honest / v8_honest_exposure CSVs in place
so the current headline reflects the correction without re-running the chain.

Usage: python scripts/apply_safety_label_corrections.py
"""
import json
import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/sources"
# All audited label-correction files (safety + efficacy + future). Glob so new
# audits just drop a `*_label_corrections_*.csv` file.
CORR_GLOB = "*_label_corrections_*.csv"
TARGETS = [
    "training_dataset_v8.csv",
    "training_dataset_v8_honest.csv",
    "training_dataset_v8_honest_exposure.csv",
    "training_dataset_arm_level.csv",  # arm-level companion (multiple arms/NCT)
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    files = sorted(SRC.glob(CORR_GLOB))
    fix = {}
    for f in files:
        c = pd.read_csv(f)
        fix.update(dict(zip(c["NCT_ID"], c["new_outcome"])))
    print(f"Loaded {len(fix)} corrections from {[f.name for f in files]}", flush=True)
    prov = {"corrections_files": [str(f) for f in files],
            "n_corrections": len(fix), "applied": {}}
    for t in TARGETS:
        p = SRC / t
        if not p.exists():
            print(f"  SKIP {t} (missing)")
            continue
        df = pd.read_csv(p, low_memory=False)
        before = df.loc[df.NCT_ID.isin(fix), ["NCT_ID", "Corrected_Outcome"]].copy()
        n = 0
        for nct, new in fix.items():
            mask = df.NCT_ID == nct
            if mask.any():
                df.loc[mask, "Corrected_Outcome"] = new
                n += int(mask.sum())
        df.to_csv(p, index=False)
        prov["applied"][t] = {"rows_changed": n, "new_sha256": sha256(p)}
        print(f"  {t}: changed {n} rows; was "
              f"{before.Corrected_Outcome.value_counts().to_dict()}", flush=True)
    (SRC / "safety_label_corrections_jun7.applied.json").write_text(
        json.dumps(prov, indent=2))
    print("Wrote provenance: safety_label_corrections_jun7.applied.json", flush=True)


if __name__ == "__main__":
    main()
