#!/usr/bin/env python3
"""Supplementary Figure S3: inClinico's 0.88 is evaluation inflation + a reverse-causation feature.

Single-panel exhibit (the flat-vs-collapse panel was removed Jul 2026: the apparent
precedent-independence was carried by the n_phase2_trials leak, not by biology).
Shows the descent of the Phase 2->3 transition AUC as the leak is removed:
  reported 0.88 -> reproduced under inClinico's random-CV eval (gCV) -> honest structure-holdout
  with n_phase2_trials -> honest with the leak removed (chance).

Numbers from results/benchmark/benchmark_final_v2.json + aact_scale_proof_ladder.json.
Explanatory text belongs in the Supplementary Fig. S3 legend, not in the figure.
Output: manuscript/figures_v8/figS3_proof_ladder.{png,svg}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "results/benchmark"
OUT = ROOT / "manuscript/figures_v8"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "figure.dpi": 150, "savefig.dpi": 300})


def main():
    ic = json.loads((B / "benchmark_final_v2.json").read_text())["inclinico"]
    steps = ["inClinico\nreported",
             "honest split\n(all features)",
             "trial-design +\nsponsor only",
             "compound +\nprecedent only"]
    vals = [ic["reported_inclinico"], ic["honest_full"],
            ic["honest_trial_sponsor_only"], ic["honest_compound_precedent_only"]]
    colors = ["#95a5a6", "#2c7fb8", "#e08214", "#c0392b"]

    # The first value is inClinico's own reported AUC, quoted at the 3 dp their publication
    # gives it to. Our three computed AUCs are shown at 3 dp, matching the Supplementary
    # Fig. S2 legend.
    fmt = ["{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}"]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = range(len(vals))
    ax.bar(x, vals, color=colors, width=0.55)
    for xi, v, f in zip(x, vals, fmt):
        ax.text(xi, v + 0.008, f.format(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="black", lw=0.9, ls=":")
    # Blank margin to the right of the last bar so the label sits clear of the bars
    # rather than on top of them (and inside the axes box, so it is not clipped).
    ax.set_xlim(-0.6, len(vals) - 0.1)
    ax.text(len(vals) - 0.5, 0.5, "chance", fontsize=8, color="black",
            ha="left", va="bottom")
    ax.set_xticks(list(x)); ax.set_xticklabels(steps, fontsize=8.3)
    ax.set_ylim(0.45, 0.95); ax.set_ylabel("Phase 2→3 transition ROC-AUC")
    # Title and explanatory caption both live in the Supplementary Fig. S3 legend, not in the figure.
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"figS3_proof_ladder.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'figS3_proof_ladder.png'}  (ladder {vals})")


if __name__ == "__main__":
    main()
