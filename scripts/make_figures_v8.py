#!/usr/bin/env python3
"""Regenerate the data-driven manuscript figures on the v8 main model.

Outputs to manuscript/figures_v8/: trial-level ROC curves, signal decomposition,
per-fold AUC distributions. These are the Fig. 2 panels consumed by compose_figures.py.
Schematic diagrams (funnel, architecture) are conceptual and not regenerated here.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "results/production_v8_clean_mort_singlehead_jul6"  # canonical (fixed mechanism + is_cytotoxic, Jun 26)
OUT = ROOT / "manuscript/figures_v8"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})
C = {"overall": "#2c3e50", "safety": "#c0392b", "efficacy": "#2980b9"}
# Canonical mean-of-folds AUCs (the numbers cited throughout the manuscript). The
# plotted ROC is necessarily pooled out-of-fold; we label each curve with the
# mean-of-folds AUC so the figure and text agree.
AUC_TRIAL = {"overall": 0.770, "safety": 0.784, "efficacy": 0.765}  # canonical singlehead_jul6 metrics.json (0.7699/0.7838/0.7649); matches Fig 2c full-model + Table 1 + abstract


def pooled(d):
    return d.groupby("row_idx").agg(y=("y", "first"), p=("raw_prob", "mean")).reset_index()


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig); print("wrote", name, flush=True)


# --- Fig 2a: ROC curves (trial-level) ---
fig, ax = plt.subplots(figsize=(4.2, 4.2))
for t in ["overall", "safety", "efficacy"]:
    g = pooled(pd.read_parquet(TRIAL / f"oof_{t}.parquet"))
    fpr, tpr, _ = roc_curve(g.y, g.p)
    ax.plot(fpr, tpr, color=C[t], lw=1.8, label=f"{t} (AUC = {AUC_TRIAL[t]:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax.set(xlabel="False positive rate", ylabel="True positive rate")
ax.legend(loc="lower right", fontsize=8, title="mean-of-folds AUC", title_fontsize=7)
ax.text(0.98, 0.02, "curve: pooled out-of-fold", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6, color="0.5")
save(fig, "fig2a_roc_trial")

# --- Fig 2c: signal decomposition (overall + safety + efficacy), ending at the full model ---
# The first three groups are additive single-GBM blocks (Supplementary Table S1); the fourth is the full
# integrated model, so the final bars EQUAL Fig. 2a / the headline (0.770 / 0.784 / 0.765) and the
# decomposition no longer appears to fall short of the reported AUC.
fig, ax = plt.subplots(figsize=(5.6, 3.6))
steps = ["flags", "+disease\ndifficulty", "+molecular\nmechanism", "full\nmodel"]
ov = [0.615, 0.689, 0.740, 0.770]; sa = [0.667, 0.630, 0.719, 0.784]; ef = [0.641, 0.667, 0.717, 0.765]
x = np.arange(len(steps)); w = 0.27
ax.bar(x - w, ov, w, color=C["overall"], label="overall")
ax.bar(x,     sa, w, color=C["safety"], label="safety")
ax.bar(x + w, ef, w, color=C["efficacy"], label="efficacy")
for xi, (o, s, e) in enumerate(zip(ov, sa, ef)):
    ax.text(xi - w, o + .005, f"{o:.3f}", ha="center", fontsize=6.5)
    ax.text(xi,     s + .005, f"{s:.3f}", ha="center", fontsize=6.5)
    ax.text(xi + w, e + .005, f"{e:.3f}", ha="center", fontsize=6.5)
# separate the additive blocks from the full-model reference group
ax.axvline(2.5, color="0.6", lw=0.8, ls=":")
ax.set(xticks=x, xticklabels=steps, ylabel="AUC", ylim=(0.55, 0.85))
ax.legend(fontsize=8); save(fig, "fig2c_decomposition")

# --- Fig 2b: per-fold AUC distributions (trial-level) ---
fig, ax = plt.subplots(figsize=(4.6, 3.6))
data = []
for t in ["overall", "safety", "efficacy"]:
    d = pd.read_parquet(TRIAL / f"oof_{t}.parquet")
    aucs = [roc_auc_score(x.y, x.raw_prob) for _, x in d.groupby(["seed", "fold"]) if x.y.nunique() == 2]
    data.append(aucs)
bp = ax.boxplot(data, labels=["overall", "safety", "efficacy"], patch_artist=True)
for patch, t in zip(bp["boxes"], ["overall", "safety", "efficacy"]):
    patch.set_facecolor(C[t]); patch.set_alpha(0.6)
ax.set(ylabel="AUC"); save(fig, "fig2b_fold_distributions")


print("\nAll v8 data figures written to", OUT)
