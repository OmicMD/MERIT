#!/usr/bin/env python3
"""Build the v8 trial-level training dataset from v5_unified by composing the three
Jun-6 validated hygiene/feature changes, so the canonical production head picks them
up as standing columns. v5_unified is left untouched (production_v2 stays reproducible).

Composed changes (each independently A/B-validated against the clean production_v2 head):
  L1c  anti-pathogen extension  -> set is_anti_pathogen=1 for verified direct-antipathogen
       drugs (build_antipathogen_extension_jun6 rules). Drops their efficacy rows
       (pathogen target invisible to the human-target pipeline). antipathogen_extension_ab.py.
  L3a  combo/comparator attribution cleanup -> drop the 18 web-verified label-noise rows
       where the indexed drug is a backbone/comparator and the FAIL belongs to an
       unmodeled partner. l3a_cleanup_ab.py.
  feat causal_plausibility_blind -> NAME-BLINDED pre-trial "does this target causally
       address THIS disease" score (0-100, NaN where unscored). The first leak-free
       mechanistic efficacy predictor. NEVER the named score (drug-name outcome recall).
       causal_plausibility_blind_compare.py / _wire_confirm.py.

get_features() is a denylist over numeric columns, so causal_plausibility_blind becomes a
standing feature automatically; per-fold SimpleImputer(median) fills it inside each fold.

Run:  python scripts/build_v8_dataset.py
Out:  data/sources/training_dataset_v8.csv  (+ .provenance.json)
"""
from __future__ import annotations
import json, hashlib, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_antipathogen_extension_jun6 import ALL_INDICATIONS, INDICATION_SPECIFIC  # noqa: E402

SRC = ROOT / "data/sources/training_dataset_v5_unified.csv"
OUT = ROOT / "data/sources/training_dataset_v8.csv"
L3A = ROOT / "data/sources/l3a_combo_attribution_exclusions_jun6.csv"
BLIND_SCORES = ROOT / "results/phase1/causal_plausibility_blind_scores.jsonl"
BLIND_MAP = ROOT / "data/sources/causal_plausibility_blind_map.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_l1c(df):
    """Set is_anti_pathogen=1 for verified direct-antipathogen drugs (overwrite in place)."""
    before = int(df["is_anti_pathogen"].fillna(0).sum())
    new = df["is_anti_pathogen"].fillna(0).astype(int).copy()
    dn = df["Drug_Clean"].astype(str).str.lower()
    new[dn.isin({k.lower() for k in ALL_INDICATIONS})] = 1
    for drug, (_, rule) in INDICATION_SPECIFIC.items():
        new[(dn == drug.lower()) & df["Disease"].map(rule).fillna(False)] = 1
    df["is_anti_pathogen"] = new
    after = int(new.sum())
    return df, before, after


def apply_l3a(df):
    ex = pd.read_csv(L3A)
    exkey = set(zip(ex["NCT_ID"], ex["Drug_Clean"]))
    mask = df.apply(lambda r: (r["NCT_ID"], r["Drug_Clean"]) in exkey, axis=1)
    missing = exkey - set(zip(df.loc[mask, "NCT_ID"], df.loc[mask, "Drug_Clean"]))
    return df[~mask].copy(), int(mask.sum()), len(ex), missing


def attach_plausibility(df):
    bl = pd.DataFrame([json.loads(l) for l in open(BLIND_SCORES)])
    bl = bl.rename(columns={"id": "blind_id", "plausibility": "p"})[["blind_id", "p"]].drop_duplicates("blind_id")
    mp = pd.read_csv(BLIND_MAP)
    pair = mp.merge(bl, on="blind_id", how="left")[["Drug_Clean", "Disease", "p"]].drop_duplicates(["Drug_Clean", "Disease"])
    pair = pair.rename(columns={"p": "causal_plausibility_blind"})
    df = df.merge(pair, on=["Drug_Clean", "Disease"], how="left")
    return df


def proxy_guard(df):
    """Missingness must NOT be a leak (availability AUC<0.58); score orthogonal to base rate."""
    e = df[df["Corrected_Outcome"].isin(["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    e["_y"] = e["Corrected_Outcome"].str.startswith("FAIL").astype(int)
    avail = e["causal_plausibility_blind"].notna().astype(int)
    a = roc_auc_score(e["_y"], 1 - avail) if avail.nunique() > 1 else float("nan")
    br = e.groupby("Disease")["_y"].transform("mean")
    r = spearmanr(e["causal_plausibility_blind"], br, nan_policy="omit").correlation
    return a, r, e["causal_plausibility_blind"].notna().mean()


def main():
    df = pd.read_csv(SRC, low_memory=False)
    n0, c0 = len(df), len(df.columns)
    print(f"v5_unified: {n0} rows, {c0} cols")

    df, ap_before, ap_after = apply_l1c(df)
    print(f"[L1c] is_anti_pathogen {ap_before} -> {ap_after} drugs/rows flagged (+{ap_after-ap_before})")

    df = attach_plausibility(df)
    a, r, cov = proxy_guard(df)
    print(f"[feat] causal_plausibility_blind merged; efficacy-cohort coverage {cov:.1%}")
    print(f"       PROXY availability AUC (missing->fail) {a:.3f} (<0.58 required)  "
          f"Spearman(score,baserate) {r:+.3f}")
    assert a < 0.58, f"availability is a leak (AUC {a:.3f})"

    df, n_drop, n_listed, missing = apply_l3a(df)
    print(f"[L3a] dropped {n_drop}/{n_listed} listed combo-attribution label-noise rows")
    if missing:
        print(f"      WARNING unmatched L3a keys: {missing}")

    # --- Safety-label corrections (blind symmetric audit, Jun 7) ---
    # 3 trials whose FAIL_SAFETY label is not a genuine drug-attributable in-trial
    # safety failure (COVID-funding / investigator-death / post-marketing-temporal).
    # Relabel to EXCLUDE_NONDRUG_STOP so they drop from every task cohort.
    # Provenance: data/sources/safety_label_corrections_jun7.csv,
    # notes/safety_label_audit_jun7.md.
    _corr_files = sorted((ROOT / "data/sources").glob("*_label_corrections_*.csv"))
    if _corr_files:
        _fix = {}
        for _f in _corr_files:
            _c = pd.read_csv(_f)
            _fix.update(dict(zip(_c["NCT_ID"], _c["new_outcome"])))
        _n = 0
        for _nct, _new in _fix.items():
            _m = df["NCT_ID"] == _nct
            _n += int(_m.sum())
            df.loc[_m, "Corrected_Outcome"] = _new
        _newvals = pd.Series(list(_fix.values())).value_counts().to_dict()
        print(f"[label-corr] relabeled {_n} audited rows -> {_newvals} "
              f"(from {[f.name for f in _corr_files]})")
    else:
        print("[label-corr] WARNING no *_label_corrections_*.csv files — none applied")

    df.to_csv(OUT, index=False)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    prov = {
        "built_by": "scripts/build_v8_dataset.py",
        "git_sha": sha,
        "source": str(SRC), "source_sha256": sha256(SRC),
        "output": str(OUT), "output_sha256": sha256(OUT),
        "rows_in": n0, "rows_out": len(df), "cols_in": c0, "cols_out": len(df.columns),
        "changes": {
            "L1c_anti_pathogen": {"before": ap_before, "after": ap_after},
            "L3a_rows_dropped": n_drop, "L3a_rows_listed": n_listed,
            "causal_plausibility_blind": {
                "efficacy_coverage": float(cov),
                "availability_auc": float(a), "baserate_spearman": float(r),
            },
        },
    }
    Path(str(OUT) + ".provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {OUT}  ({len(df)} rows, {len(df.columns)} cols)")
    print(f"wrote {OUT}.provenance.json")


if __name__ == "__main__":
    main()
