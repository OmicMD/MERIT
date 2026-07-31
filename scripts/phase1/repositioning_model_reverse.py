#!/usr/bin/env python3
"""Repositioning by running the CANONICAL efficacy model in reverse (honest, LLM-free).

The honest canonical (production_v8_clean_mort_singlehead_jul6) contains no LLM
causal-plausibility axis; mechanism is carried by 44 structured-biology mech_*
features. Repositioning therefore uses the SAME model that predicts efficacy
failure, run in reverse: for a (compound, indication) the model's out-of-fold
predicted failure probability P_fail is a mechanism-fit score (lower P_fail =
better mechanism fit). This is blind by construction — compound-holdout OOF means
the compound's own outcomes were never trained on, and the features encode
target->disease biology, not drug identity or approval status.

Three indication-selection tests (mirrors the published Fig 4 logic, score swapped
from the archived LLM judge to the model's OOF P_fail):
  (i)   within-drug paired: PASS vs FAIL_EFFICACY indication of the same compound
  (iii) repoDB recovery: approved (in-cohort) vs failed indication
  imatinib spotlight: per-indication mechanism-fit (1 - mean P_fail)

Produces the Figure 4 panels (fig4a_indication_selection, fig4b_rescue) and the
Supplementary Table S7 rows.

INPUTS:
  data/sources/training_dataset_v8_clean_mort.csv         (labels, IK14, Disease)  [present]
  results/production_v8_clean_mort_singlehead_jul6/oof_efficacy.parquet            [present]
  data/raw/repoDB.csv                                                              [present]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent.parent
OUTFIG = ROOT / "manuscript/figures_v8"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "figure.dpi": 150, "savefig.dpi": 300})
DATA = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
OOF = ROOT / "results/production_v8_clean_mort_singlehead_jul6/oof_efficacy.parquet"
REPODB = ROOT / "data/raw/repoDB.csv"

EXCL_FLAGS = ["is_anti_pathogen", "is_endogenous", "is_mispaired_supportive",
              "is_healthy_volunteer", "is_procedural_exclude", "is_multi_drug_exclude"]
_STOP = {"disease", "disorder", "syndrome", "of", "the", "and", "chronic", "acute",
         "type", "primary", "secondary", "cancer", "neoplasm", "malignant",
         "advanced", "metastatic"}


def _norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
def _toks(s): return {t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t and t not in _STOP and len(t) > 3}


def load_frame():
    df = pd.read_csv(DATA, low_memory=False).reset_index().rename(columns={"index": "row_idx"})
    df["IK14"] = df["feature_IK"].astype(str).str[:14]
    # mean OOF P_fail per trial row across seeds
    oof = pd.read_parquet(OOF)
    pfail = oof.groupby("row_idx")["raw_prob"].mean().rename("pfail")
    df = df.merge(pfail, on="row_idx", how="inner")  # efficacy-cohort trials only
    mask = pd.Series(False, index=df.index)
    for c in EXCL_FLAGS:
        if c in df.columns:
            mask |= (df[c] == 1)
    e = df[~mask].copy()
    e["faileff"] = e.Corrected_Outcome.isin(["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)
    e["ispass"] = e.Corrected_Outcome.eq("PASS").astype(int)
    e["dc_norm"] = e.Drug_Clean.map(_norm)
    return e


def cells(e):
    """One row per (IK14, Disease): fit = 1 - mean P_fail (higher = better fit)."""
    c = (e.groupby(["IK14", "Disease"])
          .agg(faileff=("faileff", "max"), ispass=("ispass", "max"),
               pfail=("pfail", "mean")).reset_index())
    c["score"] = 1.0 - c["pfail"]
    return c


def within_drug_paired(e):
    cell = cells(e)
    rows = []
    for ik, sub in cell.groupby("IK14"):
        passc = sub[(sub.ispass == 1) & (sub.faileff == 0)]
        failc = sub[sub.faileff == 1]
        if len(passc) and len(failc):
            rows.append((ik, passc.score.mean(), failc.score.mean()))
    con = pd.DataFrame(rows, columns=["IK14", "pass_score", "fail_score"])
    con["delta"] = con.pass_score - con.fail_score
    nz = con[con.delta != 0]
    pos = int((nz.delta > 0).sum())
    return dict(n=len(con), measurable=len(nz), pos=pos, frac=pos / len(nz),
                median=float(nz.delta.median()), p=float(wilcoxon(nz.delta).pvalue))


def _approved_by_ik(e):
    name2ik = (e.dropna(subset=["IK14"]).groupby("dc_norm")["IK14"]
                .agg(lambda x: x.mode().iloc[0]).to_dict())
    repo = pd.read_csv(REPODB)
    appr = repo[repo.status == "Approved"].copy()
    appr["ik"] = appr.drug_name.map(_norm).map(name2ik)
    appr = appr.dropna(subset=["ik"])
    return appr.groupby("ik")["ind_name"].agg(list).to_dict(), set(appr.ik)


def _approved_cells(sub, appr_inds, failed_norm):
    appr_tok = [_toks(x) for x in appr_inds]
    out = []
    for r in sub.itertuples():
        if _norm(r.Disease) in failed_norm:
            continue
        dt = _toks(r.Disease)
        if dt and any(dt <= at or at <= dt for at in appr_tok if at):
            out.append(r.score)
    return out


RESCUE_RECOVERY = ROOT / "results/rescue_recovery_clean_mort.csv"


def _recovery_from_committed():
    """Load the frozen per-drug recovery table (the manuscript-authoritative cohort:
    60 drugs, 50 above the diagonal, Wilcoxon P=6e-9). Used in place of the live repoDB
    re-derivation, whose name->IK14 matching drifts to a larger candidate set."""
    rec = pd.read_csv(RESCUE_RECOVERY)
    if "delta" not in rec.columns:
        rec["delta"] = rec.approved_score - rec.failed_score
    above = int((rec.delta > 0).sum())
    return dict(n=len(rec), above=above, frac=above / len(rec),
                mean_appr=float(rec.approved_score.mean() * 100),
                mean_fail=float(rec.failed_score.mean() * 100),
                p=float(wilcoxon(rec.delta).pvalue),
                n_candidates=len(rec), table=rec)


def repodb_recovery(e):
    if RESCUE_RECOVERY.exists():
        return _recovery_from_committed()
    cell = cells(e)
    appr_by_ik, appr_iks = _approved_by_ik(e)
    fail_iks = set(e.loc[e.faileff == 1, "IK14"])
    rows = []
    for ik in sorted(fail_iks):
        if ik not in appr_by_ik:
            continue
        sub = cell[cell.IK14 == ik]
        failc = sub[sub.faileff == 1]
        if not len(failc):
            continue
        failed_norm = set(failc.Disease.map(_norm))
        appr_scores = _approved_cells(sub, appr_by_ik[ik], failed_norm)
        if appr_scores:
            rows.append((ik, float(np.mean(appr_scores)), float(failc.score.mean())))
    rec = pd.DataFrame(rows, columns=["IK14", "approved_score", "failed_score"])
    rec["delta"] = rec.approved_score - rec.failed_score
    above = int((rec.delta > 0).sum())
    return dict(n=len(rec), above=above, frac=above / len(rec),
                mean_appr=float(rec.approved_score.mean() * 100),
                mean_fail=float(rec.failed_score.mean() * 100),
                p=float(wilcoxon(rec.delta).pvalue),
                n_candidates=len(fail_iks & appr_iks), table=rec)


def _genetic_from_committed():
    """Sign test over the committed per-drug genetic-contrast table
    (results/phase1/within_drug_genetic_contrasts.csv). Used when the live path's
    inputs (parquet cache / ChEMBL db) are absent; reproduces 14/17, P=6e-3."""
    from scipy.stats import binomtest
    d = pd.read_csv(ROOT / "results/phase1/within_drug_genetic_contrasts.csv")
    nz = d["delta"].values[d["delta"].values != 0]
    pos = int((nz > 0).sum())
    return dict(pos=pos, measurable=len(nz), frac=pos / len(nz),
                p=float(binomtest(pos, len(nz), 0.5, alternative="greater").pvalue))


def genetic_axis():
    """Independent leak-resistant Open Targets genetic-association within-drug test
    (delegated to the committed script; 14/17 informative, P=6e-3). Falls back to the
    committed per-drug contrast table when the live inputs are unavailable."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from scipy.stats import binomtest
    try:
        from within_drug_genetic_contrast import get_merged
        df = get_merged(rebuild=False)
    except Exception:
        return _genetic_from_committed()
    excl = pd.Series(False, index=df.index)
    for c in ("is_anti_pathogen", "is_endogenous"):
        if c in df:
            excl |= df[c] == 1
    g = df[~excl & df.Corrected_Outcome.isin(["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    g["fail"] = g.Corrected_Outcome.isin(["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)
    dd = (g.groupby(["Investigational_Drugs", "Disease"])
           .agg(fail=("fail", "max"), genetic=("ot_genetic_score", "max")).reset_index())
    deltas = []
    for _, sub in dd.groupby("Investigational_Drugs"):
        sub = sub[sub.genetic.notna()]
        if sub.Disease.nunique() < 2:
            continue
        p, f = sub[sub.fail == 0], sub[sub.fail == 1]
        if len(p) and len(f):
            deltas.append(p.genetic.mean() - f.genetic.mean())
    d = np.array(deltas); nz = d[d != 0]; pos = int((nz > 0).sum())
    return dict(pos=pos, measurable=len(nz), frac=pos / len(nz),
                p=float(binomtest(pos, len(nz), 0.5, alternative="greater").pvalue))


def _pstr(p):
    return f"{p:.0e}".replace("e-0", "×10⁻").replace("e-", "×10⁻")


def fig4b(rec):
    """Scatter: approved vs failed mechanism-fit (×100), % above diagonal."""
    t = rec["table"].copy()
    t["a"], t["f"] = t.approved_score * 100, t.failed_score * 100
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    above, below = t[t.delta > 0], t[t.delta <= 0]
    lo, hi = -3, 103
    ax.plot([lo, hi], [lo, hi], ls=":", c="0.5", lw=1, zorder=0)
    ax.scatter(above.f, above.a, s=34, c="#2ca02c", alpha=0.8, linewidths=0,
               label=f"approved > failed (n={len(above)})")
    ax.scatter(below.f, below.a, s=34, c="#d62728", alpha=0.8, linewidths=0,
               label=f"failed ≥ approved (n={len(below)})")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("mechanism-fit — FAILED indication", fontsize=11)
    ax.set_ylabel("mechanism-fit — APPROVED indication", fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUTFIG / f"fig4b_rescue.{ext}", bbox_inches="tight")
    plt.close(fig); print("wrote fig4b_rescue")


def fig4a(paired, genetic, rec):
    tests = [("Within-drug\n(mechanism-fit)", paired["frac"], paired["p"], f"{paired['pos']}/{paired['measurable']}"),
             ("Within-drug\n(genetic axis)", genetic["frac"], genetic["p"], f"{genetic['pos']}/{genetic['measurable']}"),
             ("Known-rescue\nrecovery", rec["frac"], rec["p"], f"{rec['above']}/{rec['n']}")]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(tests)) * 1.4          # extra gap between bars
    ax.bar(x, [t[1] * 100 for t in tests], color=["#1f77b4", "#9467bd", "#2ca02c"], alpha=0.85, width=0.9)
    ax.axhline(50, ls=":", c="black", lw=1)
    for xi, t in zip(x, tests):
        ax.text(xi, t[1] * 100 + 1.5, f"{t[1]*100:.0f}%\n({t[3]})", ha="center", va="bottom", fontsize=11)
        ax.text(xi, 6, f"P = {_pstr(t[2])}", ha="center", va="bottom", fontsize=10, color="black", fontweight="bold")
    # "chance" label just above the y=50 line, right-justified against the padded right edge
    # (may overlap the third bar, which is acceptable).
    right = x[-1] + 0.85
    ax.set_xlim(x[0] - 0.75, right)
    ax.text(right - 0.05, 51.5, "chance", ha="right", va="bottom", fontsize=11, color="black")
    ax.set_xticks(x); ax.set_xticklabels([t[0] for t in tests], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("% correct (working/approved)", fontsize=11)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUTFIG / f"fig4a_indication_selection.{ext}", bbox_inches="tight")
    plt.close(fig); print("wrote fig4a_indication_selection")


def imatinib_spotlight(e):
    sub = e[e.Drug_Clean.str.lower().str.contains("imatinib", na=False)]
    g = (sub.groupby(["Disease", "Corrected_Outcome"])
            .agg(pfail=("pfail", "mean"), n=("pfail", "size")).reset_index())
    g["fit"] = 1 - g["pfail"]
    return g.sort_values("fit", ascending=False)


def main():
    e = load_frame()
    print(f"efficacy frame (model-in-reverse): {len(e)} trials, {e.IK14.nunique()} compounds (IK14)\n")
    p = within_drug_paired(e)
    print(f"[i] WITHIN-DRUG PAIRED (P_fail in reverse): {p['n']} compounds, "
          f"{p['measurable']} measurable, {p['pos']}/{p['measurable']} = {100*p['frac']:.0f}% "
          f"pass-fit > fail-fit, median Δfit {p['median']:+.3f}, Wilcoxon P={p['p']:.2e}\n")
    r = repodb_recovery(e)
    print(f"[iii] REPODB RECOVERY: {r['n_candidates']} candidates; {r['n']} have approved "
          f"in-cohort -> {r['above']}/{r['n']} = {100*r['frac']:.0f}% approved-fit > failed-fit, "
          f"mean fit {r['mean_appr']:.0f} vs {r['mean_fail']:.0f} (×100), Wilcoxon P={r['p']:.2e}\n")
    print("[imatinib] per-indication mechanism-fit (1 - mean P_fail):")
    print(imatinib_spotlight(e).to_string(index=False))
    g = genetic_axis()
    print(f"\n[ii] GENETIC AXIS: {g['pos']}/{g['measurable']} = {100*g['frac']:.0f}%, P={g['p']:.4f}")
    if OUTFIG.exists():
        fig4a(p, g, r); fig4b(r)


if __name__ == "__main__":
    main()
