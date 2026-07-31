#!/usr/bin/env python3
"""Supplementary Figure S2 panels: why the field's reported numbers are higher -- a
looser evaluation that lets a drug's other trials sit in training gives a spuriously
higher score by letting the model recognize the drug.
Shows only the split effect (honest vs looser), on a single de-leaked feature set per
context; the separate enrollment-feature leak is discussed in the text, not mixed in here.

Writes the two panels letterless (figS2a_trialbench, figS2b_phase3); panel letters and
layout come from compose_figures.py -> figS2_composite, as for the main figures.

Reads:
  results/benchmark/trialbench_split_compare.csv         (feature-set arm, TrialBench)
  results/benchmark/star_phase3_split_compare_clean.csv  (our model, matched Phase III)
  results/benchmark/hint_holdout_clean_full.csv          (HINT honest test, 2,787 NCTs)
  results/benchmark/hint_blind_full.csv                  (HINT looser test, same NCTs)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "results/benchmark"
OUT = ROOT / "manuscript/figures_v8"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "figure.dpi": 150, "savefig.dpi": 300})
HONEST, LOOSE = "#2c7fb8", "#d95f0e"  # honest test vs looser (leaky) test


def load():
    tb = pd.read_csv(B / "trialbench_split_compare.csv").set_index("feature_set")
    star = pd.read_csv(B / "star_phase3_split_compare_clean.csv")
    sh = star[star.arm == "holdout"].roc_auc.mean()
    sb = star[star.arm == "blind"].roc_auc.mean()
    hh = pd.read_csv(B / "hint_holdout_clean_full.csv").roc_auc.mean()
    hb = pd.read_csv(B / "hint_blind_full.csv").roc_auc.mean()
    return tb, sh, sb, hh, hb


def _split_panel(stem, labels, honest, loose):
    """One panel -> its own letterless PNG. Panel letters are overlaid by
    compose_figures.py, matching the main figures, so nothing is baked in here.

    The honest/looser key is drawn BELOW the axes in every panel: an in-axes legend has
    nowhere to sit without covering a bar or its value label.
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    y = np.arange(len(labels)); h = 0.36
    ax.barh(y + h/2, honest, h, color=HONEST, label="honest test (drug held out)")
    ax.barh(y - h/2, loose, h, color=LOOSE, label="looser test (drug recognizable)")
    for i, (a, b) in enumerate(zip(honest, loose)):
        ax.text(a + 0.004, y[i] + h/2, f"{a:.3f}", va="center", fontsize=7.5, color=HONEST, fontweight="bold")
        ax.text(b + 0.004, y[i] - h/2, f"{b:.3f}", va="center", fontsize=7.5, color=LOOSE)
        ax.text(0.975, y[i], f"+{b-a:.3f}\nspurious", va="center", ha="right",
                fontsize=7, color="black")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    for t, lbl in zip(ax.get_yticklabels(), labels):
        if "ours" in lbl.lower():
            t.set_fontweight("bold")
    ax.set_xlim(0.5, 0.98); ax.set_xlabel("ROC-AUC")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              fontsize=7.5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}")


def fig_inflation(tb, sh, sb, hh, hb):
    """Supplementary Figure S2 panels: the spurious gain from a looser evaluation.

    Ours is the top bar in BOTH panels (consistent placement). The explanatory caption
    lives in the Supplementary Fig. S2 legend, not in the figure.
    """
    # "their legit design" is the leak-free set: enrollment leak AND establishment metadata
    # removed. This panel isolates EVALUATION inflation (blind vs holdout split), so both
    # arms must start from the same leak-free scope; feature-scope leakage is Table 1's job.
    _split_panel(
        "figS2a_trialbench",
        ["Our model\n(ours)", "TrialBench design features\n(leak-free set)"],
        [tb.loc["our biology", "holdout_auc"],
         tb.loc["their legit design", "holdout_auc"]],
        [tb.loc["our biology", "blind_auc"],
         tb.loc["their legit design", "blind_auc"]])
    _split_panel(
        "figS2b_phase3", ["Our model\n(ours)", "HINT\n(Phase III)"],
        [sh, hh], [sb, hb])
    print(f"  (ours Phase III {sh:.3f}->{sb:.3f}; HINT {hh:.3f}->{hb:.3f})")


def main():
    tb, sh, sb, hh, hb = load()
    fig_inflation(tb, sh, sb, hh, hb)


if __name__ == "__main__":
    main()
