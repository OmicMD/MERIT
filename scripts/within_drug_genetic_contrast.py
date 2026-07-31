#!/usr/bin/env python3
"""Task #2: within-drug / across-indication genetic-causality contrasts (Jun 6 2026).

The power-independent exhibit for the L1 (target-disease) layer: hold the MOLECULE
constant and ask whether the indication with more genetic support (OpenTargets
genetic_association, leak-free) passes, while the indication with no genetic
support fails. Because the molecule is fixed, the contrast is immune to the
survivorship / molecular-property confounds that plague cross-drug comparisons.

Reuses build() from genetic_novelty_decompression.py (arm dataset + leak-free
genetic score + target novelty). Caches the merged frame to parquet.

Outputs:
  data/cache/within_drug_merged.parquet              (the merged frame, cached)
  results/phase1/within_drug_genetic_contrasts.csv   (per-drug indication table)
  prints the paired within-molecule statistic + named exemplars
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
CACHE = ROOT / "data/cache/within_drug_merged.parquet"
OUT = ROOT / "results/phase1/within_drug_genetic_contrasts.csv"


def get_merged(rebuild=False):
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    from genetic_novelty_decompression import build
    df = build()
    keep = [c for c in ["NCT_ID", "Arm_Label", "Investigational_Drugs", "Disease",
                        "Corrected_Outcome", "ot_genetic_score", "target_n_approved",
                        "disease_is_oncology", "is_anti_pathogen", "is_endogenous"]
            if c in df.columns]
    df = df[keep].copy()
    df.to_parquet(CACHE)
    return df


def main():
    rebuild = "--rebuild" in sys.argv
    df = get_merged(rebuild)
    print(f"merged arms={len(df)}  cols={df.columns.tolist()}")

    # efficacy frame, drop anti-pathogen/endogenous (no efficacy signal by design)
    excl = pd.Series(False, index=df.index)
    for c in ("is_anti_pathogen", "is_endogenous"):
        if c in df:
            excl |= df[c] == 1
    e = df[~excl & df.Corrected_Outcome.isin(["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    e["fail"] = e.Corrected_Outcome.isin(["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)
    # one row per (drug, disease): genetic score is target×disease so constant within;
    # collapse arms, fail if ANY arm of that drug×disease failed efficacy
    dd = (e.groupby(["Investigational_Drugs", "Disease"])
            .agg(fail=("fail", "max"),
                 genetic=("ot_genetic_score", "max"),
                 n_arms=("fail", "size"),
                 target_n_approved=("target_n_approved", "max"),
                 onco=("disease_is_oncology", "max"),
                 nct=("NCT_ID", "first"))
            .reset_index())

    # within-drug contrast pool: drugs with >=2 diseases, both a PASS and a FAIL,
    # and genetic coverage on BOTH sides (so the score difference is measurable)
    print(f"\n(drug,disease) cells: {len(dd)}  with genetic coverage: {dd.genetic.notna().sum()}")
    rows = []
    for drug, sub in dd.groupby("Investigational_Drugs"):
        sub = sub[sub.genetic.notna()]
        if sub.Disease.nunique() < 2:
            continue
        passes = sub[sub.fail == 0]
        fails = sub[sub.fail == 1]
        if len(passes) == 0 or len(fails) == 0:
            continue
        rows.append(dict(
            drug=drug,
            n_pass=len(passes), n_fail=len(fails),
            pass_genetic_mean=passes.genetic.mean(),
            fail_genetic_mean=fails.genetic.mean(),
            delta=passes.genetic.mean() - fails.genetic.mean(),
            pass_inds="; ".join(f"{r.Disease}={r.genetic:.2f}" for r in passes.itertuples()),
            fail_inds="; ".join(f"{r.Disease}={r.genetic:.2f}" for r in fails.itertuples()),
            onco=int(sub.onco.max()),
            min_target_n_approved=int(sub.target_n_approved.min()) if sub.target_n_approved.notna().any() else -1,
        ))
    con = pd.DataFrame(rows).sort_values("delta", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con.to_csv(OUT, index=False)
    print(f"\n=== {len(con)} within-drug contrast drugs (genetic-covered PASS and FAIL indications) ===")

    # POWER-INDEPENDENT PAIRED STATISTIC: within each molecule, is genetic(pass) > genetic(fail)?
    d = con.delta.values
    nz = d[d != 0]
    pos = (nz > 0).sum(); neg = (nz < 0).sum()
    print(f"\nWithin-molecule paired delta (genetic_pass_mean - genetic_fail_mean):")
    print(f"  drugs with measurable delta: {len(nz)}   pass>fail: {pos}   fail>pass: {neg}   ties(0): {(d==0).sum()}")
    print(f"  mean delta: {d.mean():+.3f}   median: {np.median(d):+.3f}")
    # sign test p-value (two-sided) via binomial
    from scipy.stats import binomtest, wilcoxon
    if len(nz) > 0:
        bt = binomtest(pos, len(nz), 0.5, alternative="greater")
        print(f"  sign test (pass>fail) p={bt.pvalue:.4f}")
        try:
            w = wilcoxon(nz)
            print(f"  Wilcoxon signed-rank p={w.pvalue:.4f}")
        except Exception as ex:
            print(f"  Wilcoxon n/a: {ex}")

    # non-onco only (the decompression zone)
    co = con[con.onco == 0]
    dco = co.delta.values; nzc = dco[dco != 0]
    if len(nzc) > 0:
        from scipy.stats import binomtest as bt2
        posc = (nzc > 0).sum()
        print(f"\n  NON-ONCO only: drugs={len(co)} measurable={len(nzc)} pass>fail={posc} "
              f"mean delta={dco.mean():+.3f} sign-p={bt2(posc,len(nzc),0.5,alternative='greater').pvalue:.4f}")

    print("\n=== TOP within-drug contrasts (PASS indication more genetically supported) ===")
    show = con[con.delta > 0].head(20)
    for r in show.itertuples():
        print(f"\n  {r.drug}  (Δ={r.delta:+.2f}, onco={r.onco})")
        print(f"    PASS: {r.pass_inds}")
        print(f"    FAIL: {r.fail_inds}")

    print("\n=== Named exemplars present? ===")
    for name in ["mavacamten", "evacetrapib", "EMA401", "olodanrigan", "PF-07038124",
                 "ezetimibe", "ladarixin", "ataluren", "varespladib"]:
        m = dd[dd.Investigational_Drugs.str.contains(name, case=False, na=False)]
        if len(m):
            print(f"  {name}:")
            print(m[["Disease", "fail", "genetic", "onco", "target_n_approved"]].to_string(index=False))


if __name__ == "__main__":
    main()
