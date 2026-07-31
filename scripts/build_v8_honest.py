#!/usr/bin/env python3
"""Build the attribution-fixed honest trial-level dataset (v8 -> v8_honest).

Closes a prior reproducibility gap: training_dataset_v8_honest.csv used to be an
unscripted artifact (it entered the repo in commit eb09531 with no build script).

The "honest" step is a drug-ATTRIBUTION fix: keep only the (NCT, drug, disease) rows
where the indexed drug is the agent the trial is actually about (drop SOC-backbone /
comparator / mis-indexed rows), and annotate each kept row with its attribution role.

That attribution DETERMINATION (which rows are kept + their attr_role / attr_misindexed /
attr_true_modality) was a one-time pass; it is committed as data in
  data/sources/attribution_determination_v8.csv
so this script can re-apply it deterministically. Separating the one-time determination
(committed data) from its application (this script) makes the whole chain reproducible:

  v5_unified --build_v8_dataset.py--> v8.csv  (label corrections baked in via glob)
            --build_v8_honest.py (THIS)-----> v8_honest.csv
            --build_v8_honest_exposure.py----> v8_honest_exposure.csv  (model input)

Because corrections are applied upstream in build_v8_dataset.py, they propagate here
automatically (this script only filters rows + adds attribution columns).

Run: python scripts/build_v8_honest.py
"""
from __future__ import annotations
import json, hashlib, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/sources/training_dataset_v8.csv"
DET = ROOT / "data/sources/attribution_determination_v8.csv"
OUT = ROOT / "data/sources/training_dataset_v8_honest.csv"
KEY = ["NCT_ID", "Drug_Clean", "Disease"]
ATTR = ["attr_role", "attr_misindexed", "attr_true_modality"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    df = pd.read_csv(SRC, low_memory=False)
    det = pd.read_csv(DET, low_memory=False)
    n0 = len(df)

    # restrict to the attribution-kept rows and attach attribution columns
    for k in KEY:
        df[k] = df[k].astype(str)
        det[k] = det[k].astype(str)
    keep = det[KEY + ATTR].drop_duplicates(KEY)
    out = df.merge(keep, on=KEY, how="inner")

    missing = len(det) - len(out)
    print(f"v8 rows={n0}  attribution-kept={len(det)}  -> honest rows={len(out)}"
          f"  (dropped {n0 - len(out)} mis-attributed; {missing} determination keys absent from v8)")
    print(f"  attr_role: {out['attr_role'].value_counts(dropna=False).to_dict()}")
    print(f"  outcomes:  {out['Corrected_Outcome'].value_counts().to_dict()}")

    out.to_csv(OUT, index=False)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    prov = {
        "built_by": "scripts/build_v8_honest.py", "git_sha": sha,
        "source": str(SRC), "source_sha256": sha256(SRC),
        "determination": str(DET), "determination_sha256": sha256(DET),
        "output": str(OUT), "output_sha256": sha256(OUT),
        "rows_in": n0, "rows_out": len(out),
        "note": "attribution fix re-applied from committed determination; label corrections inherited from v8",
    }
    Path(str(OUT) + ".provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"wrote {OUT} ({len(out)} rows, {len(out.columns)} cols) + provenance sidecar")


if __name__ == "__main__":
    main()
