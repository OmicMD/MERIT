#!/usr/bin/env python3
"""Figure 3 (mechanism instrument) on the v8 honest cohort.
Panels: a) single-feature fingerprints of the three confident misses, b) disease-pathway
overlap by outcome. Writes to manuscript/figures_v8/; composited by compose_figures.py
into fig3_composite. Also prints case z-scores to reconcile legend.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase1"))
from make_mechanism_figures import OUTCOME_COLORS

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "figure.dpi": 150, "savefig.dpi": 300})
OUT = ROOT / "manuscript/figures_v8"
OUT.mkdir(parents=True, exist_ok=True)
SCORES = ROOT / "results/phase1/analysis_explanatory/systemic_framework_v8/trial_pca_scores_v8.csv"


PATHWAY_FEAT = "net_n_disease_genes_in_pathways"


def fig3_pathway(df):
    """Pathway-overlap feature by outcome (anchors the efficacy/safety separation,
    not PC6). df is the full v8 dataset (has the raw feature). Panel (b)."""
    from scipy.stats import gaussian_kde
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    col = df[PATHWAY_FEAT].astype(float)
    xs = np.linspace(np.nanmin(col), np.nanpercentile(col, 99), 400)
    for oc in ["PASS", "FAIL_EFFICACY", "FAIL_SAFETY"]:
        s = df.loc[df.Corrected_Outcome == oc, PATHWAY_FEAT].dropna().astype(float)
        if len(s) < 30:
            continue
        ys = gaussian_kde(s.values, bw_method=0.4)(xs)
        ax.fill_between(xs, ys, color=OUTCOME_COLORS[oc], alpha=0.35, label=f"{oc} (n={len(s)})")
        ax.plot(xs, ys, color=OUTCOME_COLORS[oc], lw=1.2)
    # Cliff's d for the title: use the bootstrap point estimates (pathway_alignment_bootstrap_v8.csv)
    # so this panel matches Supplementary Note S3 and the Fig. 3b legend exactly. The raw full-sample
    # estimate and the bootstrap point differ at the second decimal (e.g. raw safety -0.2555 rounds to
    # -0.26 while the bootstrap point is -0.247 -> -0.25); reporting one source everywhere avoids a
    # spurious 0.01 mismatch between the figure and its own legend.
    boot = pd.read_csv(SCORES.parent / "pathway_alignment_bootstrap_v8.csv")
    def _bd(contrast):
        r = boot[(boot.feature == PATHWAY_FEAT) & (boot.contrast == contrast)]
        return float(r.d.iloc[0])
    de, ds = _bd("efficacy"), _bd("safety")
    ax.set_xlabel("disease genes in drug's enriched pathways")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig3b_pathway_by_outcome.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"wrote fig3b (eff d={de:+.2f} saf d={ds:+.2f})")


# Confident FAIL-scored-PASS cases with a clear molecular fingerprint (Results case audit).
# Each: (label, raw feature, drug substring, descriptive feature name for the figure).
CASES = [
    ("Nilotinib", "tox_cardiac_burden", "nilotinib",
     "cardiac toxicity-target burden"),
    ("Panobinostat", "net_mech_epigenetic_frac", "panobinostat",
     "epigenetic-mechanism fraction"),
    ("Metoprolol", "drumap_kpbrain", "metoprolol",
     "brain partition coefficient"),
]


def fig3_case(df):
    """Raw-feature fingerprints for the three confident misses with clear molecular
    signatures (matches the Results case audit). Bar = drug mean z-score vs cohort. Panel (a)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels, zs = [], []
    print("\ncase-study raw-feature z-scores:")
    for lab, feat, sub, descr in CASES:
        col = df[feat].astype(float); mu, sd = col.mean(), col.std()
        s = df[df.Drug_Clean.str.lower().str.contains(sub, na=False)]
        z = (s[feat].astype(float).mean() - mu) / sd
        labels.append(f"{lab}\n{descr}"); zs.append(z)
        print(f"  {sub:12s} {feat:26s} z={z:+.1f}")
    y = np.arange(len(labels))
    ax.barh(y, zs, color=["#d62728", "#8c564b", "#17becf"], alpha=0.85)
    for yi, zi in zip(y, zs):
        ax.text(zi + 0.15, yi, f"z = {zi:+.1f}", va="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("feature z-score vs trial cohort")
    ax.set_xlim(0, max(zs) * 1.28)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig3a_case_features.{ext}", bbox_inches="tight")
    plt.close(fig); print("wrote fig3a")


def main():
    # A drug-class PCA panel was retired Jul 2026: classes did not separate in PC space
    # beyond a label-shuffled null (silhouette -0.286 vs -0.285, p=0.56).
    df = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
    fig3_case(df); fig3_pathway(df)
    print(f"\nwrote Figure 3 panels to {OUT}")


if __name__ == "__main__":
    main()
