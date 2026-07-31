#!/usr/bin/env python3
"""Figure 1 (dataset + architecture) on the v8 honest cohort.
Figure 1 has two panels: a) trial-selection funnel, b) computational-architecture
schematic (nine modules). Also writes figS1_module_auc, the per-module standalone
AUC bar, which is a standalone Supplementary Fig. S3, not a Figure 1 panel.
Counts read from the v8 dataset + module_ablation_v8.csv so they cannot drift.
Out: manuscript/figures_v8/
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "manuscript/figures_v8"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "figure.dpi": 150, "savefig.dpi": 300})

df = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
mod = pd.read_csv(ROOT / "results/phase1/analysis_explanatory/systemic_framework_v8/module_ablation_v8.csv")

# ---- panel a: trial-selection funnel ----
# A funnel-shaped cascade (raw records -> classified outcomes -> profiled compounds ->
# blind-audit modeling cohort) that folds in EVERY exclusion category and its reason
# (right-hand annotations) plus the final-cohort composition (bar beneath). Fig. 1a is the
# only place the dataset composition is reported. Counts derive from the v8 cohort so they
# cannot drift; only the two upstream AACT counts are constants.
import numpy as np
from matplotlib.patches import Polygon, Rectangle

vc = df.Corrected_Outcome.value_counts()
_n_profiled = len(df)                        # complete-profile rows (incl. audited-out)
_n_profiled_compounds = df.SMILES.nunique()
_model = df[df.Corrected_Outcome.isin(["PASS", "FAIL_EFFICACY", "FAIL_SAFETY", "FAIL_BOTH"])]
_n_trials, _n_compounds = len(_model), _model.SMILES.nunique()
n_nondrug = int(vc.get("EXCLUDE_NONDRUG_STOP", 0))
n_misscoped = int(vc.get("EXCLUDE_MISSCOPED", 0))
n_noresults = int(vc.get("EXCLUDE_NO_RESULTS", 0))
_n_excluded = n_nondrug + n_misscoped + n_noresults

N_AACT = 428377      # AACT drug-intervention records (documented upstream constant)
N_CLEAR = 12420      # trials with a clear PASS / FAIL outcome (documented upstream constant)

stages = [("AACT drug-intervention records", N_AACT),
          ("Clear PASS / FAIL outcome", N_CLEAR),
          (f"Complete pre-trial profile\n({_n_profiled_compounds} compounds)", _n_profiled),
          (f"Modeling cohort\n({_n_compounds} compounds)", _n_trials)]
# what each narrowing removes, aligned to the neck between consecutive stages
removed = [
    (N_AACT - N_CLEAR, ["non-terminated / unclassifiable", "outcomes removed"]),
    (N_CLEAR - _n_profiled, ["biologics, combinations, and compounds", "without a complete pre-trial profile"]),
    (_n_excluded, ["non-drug / mis-scoped stops (blind audit):",
                   f"{n_nondrug} non-drug (business / enrollment / COVID / admin.)",
                   f"{n_misscoped} mis-scoped  ·  {n_noresults} no posted results"]),
]

CX, BAND_H, Y_TOP = 3.1, 1.7, 10.0
# Schematic even-narrowing silhouette. A faithful width-proportional-to-n mapping cannot
# read as a funnel here: the cohort collapses 428k->12k in a single step, then barely
# changes (12,420 -> 3,320 -> 3,133). So the half-widths give the classic converging-funnel
# shape while the true magnitudes are carried by the bold n labels (standard for a
# CONSORT-style selection funnel).
# Boundary half-widths (one per band edge, len = stages + 1) so EVERY band tapers and the
# funnel narrows continuously to a rounded bottom (no rectangular terminal block). The neck
# cannot taper below ~1.35 half-width: the bold "n = 3,133" label must fit inside the
# narrowest band without crossing the sloping edges.
HWID = [3.30, 2.65, 2.10, 1.65, 1.35]
ys = [Y_TOP - i * BAND_H for i in range(len(stages) + 1)]
blues = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd"]

# Panel a is drawn on a wide, short canvas whose aspect ratio (~1.5) matches the panel-b
# schematic. compose_figures.py scales each panel to a common ROW HEIGHT, so a tall panel a
# would be shrunk far harder than the wide panel b and its type would come out illegibly
# small next to it. Matching the aspect ratios makes both panels shrink by the same factor,
# so these font sizes are also the sizes seen in fig1_composite.
fig, ax = plt.subplots(figsize=(9.4, 6.2))
for i, (lab, n) in enumerate(stages):
    top_hw, bot_hw = HWID[i], HWID[i + 1]
    yt, yb = ys[i], ys[i + 1]
    ax.add_patch(Polygon([(CX - top_hw, yt), (CX + top_hw, yt),
                          (CX + bot_hw, yb), (CX - bot_hw, yb)],
                         closed=True, fc=blues[i], ec="white", lw=1.4))
    yc = (yt + yb) / 2
    ax.text(CX, yc + 0.30, lab, ha="center", va="center",
            fontsize=(14.0 if top_hw > 1.6 else 12.5), linespacing=1.15)
    ax.text(CX, yc - 0.42, f"n = {n:,}", ha="center", va="center", fontsize=16.5, weight="bold")
    if i < len(removed):                              # exclusion note at the neck
        rn, rlines = removed[i]
        ax.annotate("", xy=(CX + bot_hw + 0.55, yb), xytext=(CX + bot_hw + 0.02, yb),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
        head = f"−{rn:,}  {rlines[0]}"
        body = "" if len(rlines) == 1 else "\n" + "\n".join(rlines[1:])
        ax.text(CX + bot_hw + 0.72, yb, head + body, ha="left", va="center",
                fontsize=12.0, color="black", style="italic", linespacing=1.35)

# final-cohort composition bar, beneath the funnel neck
comp = [("PASS", int(vc.get("PASS", 0)), "#2ca02c"),
        ("FAIL_EFFICACY", int(vc.get("FAIL_EFFICACY", 0)), "#ff7f0e"),
        ("FAIL_SAFETY", int(vc.get("FAIL_SAFETY", 0)), "#d62728"),
        ("FAIL_BOTH", int(vc.get("FAIL_BOTH", 0)), "#7f4f24")]
bar_w, bar_h = 2 * HWID[-1], 0.42
bar_x0, bar_y = CX - HWID[-1], ys[-1] - 1.75
# Arrow drops from the funnel neck and stops short; the title sits in the gap below the
# arrow head and directly above the bar it labels.
ax.annotate("", xy=(CX, bar_y + bar_h + 0.42), xytext=(CX, ys[-1] - 0.04),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
ax.text(CX, bar_y + bar_h + 0.06, f"Cohort composition  (n = {_n_trials:,})",
        ha="center", va="bottom", fontsize=12.5, weight="bold")
x = bar_x0
for _, n, c in comp:
    w = bar_w * n / _n_trials
    ax.add_patch(Rectangle((x, bar_y), w, bar_h, fc=c, ec="white", lw=0.8))
    x += w
# Single-row colour-keyed legend with counts (tiny FAIL classes can't be labelled in-bar).
# Uses ax.legend (not hand-placed swatches) so entry spacing follows the real rendered text
# widths: the labels differ in length, and a hand-rolled pitch collides swatches with text.
handles = [Rectangle((0, 0), 1, 1, fc=c) for _, _, c in comp]
labels = [f"{lab.replace('_', ' ')}: {n:,}" for lab, n, _ in comp]
ax.legend(handles, labels, ncol=len(comp), fontsize=12.0, frameon=False,
          loc="upper center", bbox_to_anchor=(CX, bar_y - 0.10), bbox_transform=ax.transData,
          handlelength=0.9, handleheight=0.9, handletextpad=0.5, columnspacing=1.4)

ax.set_xlim(-0.4, 10.6); ax.set_ylim(1.1, 10.4); ax.axis("off")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(OUT / f"fig1a_funnel.{ext}", bbox_inches="tight")
plt.close(fig)
print(f"wrote fig1a_funnel (profiled={_n_profiled}, cohort={_n_trials}, compounds={_n_compounds}, "
      f"excluded={_n_excluded} [{n_nondrug}/{n_misscoped}/{n_noresults}])")

# ---- panel b: computational architecture schematic ----
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(9.4, 6.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

C_IN = "#dce6f1"; C_STR = "#d7e8d2"; C_PRIOR = "#fce4c4"; C_CTX = "#e8dcec"
C_INT = "#cfd8dc"; C_HEAD = "#f4cccc"; C_OUT = "#eeeeee"; C_GATE = "#fdf2d0"
EDGE = "#5a5a5a"          # box outlines stay grey
BUS = ARROW = "black"     # connector rails and flow arrows are black, as in the funnel panel

def box(x, y, w, h, text, fc, fs=11.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=1.6",
                 fc=fc, ec=EDGE, lw=0.9, mutation_aspect=h / w))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, linespacing=1.25)

def arrow(x1, y1, x2, y2, lw=1.0, color=ARROW):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=11, lw=lw, color=color,
                 shrinkA=0, shrinkB=0))

def stub(x1, x2, y):  # short horizontal connector into a bus rail
    ax.add_line(Line2D([x1, x2], [y, y], color=BUS, lw=0.8, zorder=0))

# --- column geometry ---
IN_X, IN_W = 2.0, 18.0
MOD_X, MOD_W = 28.0, 27.0
INT_X, INT_W = 62.0, 17.0
HEAD_X, HEAD_W = 81.0, 13.0
IN_BUS, MOD_BUS = 21.5, 56.5
CY = 54.0  # vertical centre of the data spine

# Column headers (clear above the boxes)
for hx, ht in [(IN_X + IN_W / 2, "Pre-trial inputs"),
               (MOD_X + MOD_W / 2, "Pre-trial modules"),
               (INT_X + INT_W / 2, "Integration"),
               (HEAD_X + HEAD_W / 2, "Predictions")]:
    ax.text(hx, 94, ht, ha="center", va="center", fontsize=12.5, style="italic",
            color="black")

# Inputs (col 1) — centred on the spine
in_centres = [74.0, 54.0, 34.0]
inp = ["Compound structure", "Target / mechanism\nannotations",
       "Disease indication"]
for t, cy in zip(inp, in_centres):
    box(IN_X, cy - 7.5, IN_W, 15, t, C_IN, fs=11)
    stub(IN_X + IN_W, IN_BUS, cy)
ax.add_line(Line2D([IN_BUS, IN_BUS], [in_centres[-1], in_centres[0]],
                   color=BUS, lw=1.0, zorder=0))
arrow(IN_BUS, CY, MOD_X, CY)

# Modules (col 2): eight core modules (1-8), evenly stacked, plus the gated
# ninth (module 9; endpoint/population/trial-design leverage) drawn distinctly
# (dashed border) to mark it as gated and added during cross-validation.
mod_specs = (
    [(t, C_STR) for t in ["Tissue-specific binding",
                          "Pathway engagement",
                          "Binding specificity", "Safety pharmacology",
                          "Pharmacokinetics"]]
    + [(t, C_PRIOR) for t in ["Target–disease\nmechanism",
                              "Genetics"]]
    + [("Disease complexity\n+ trial context", C_CTX)]
)
mod_centres = np.linspace(86.0, 20.0, len(mod_specs) + 1)  # +1 slot for module 9
MOD_H = 7.0
for (t, fc), cy in zip(mod_specs, mod_centres):
    box(MOD_X, cy - MOD_H / 2, MOD_W, MOD_H, t, fc, fs=10)
    stub(MOD_X + MOD_W, MOD_BUS, cy)
g_cy = mod_centres[-1]
ax.add_patch(FancyBboxPatch((MOD_X, g_cy - MOD_H / 2), MOD_W, MOD_H,
             boxstyle="round,pad=0,rounding_size=1.6",
             fc=C_GATE, ec=EDGE, lw=0.9, ls="--", mutation_aspect=MOD_H / MOD_W))
ax.text(MOD_X + MOD_W / 2, g_cy, "Trial design &\nendpoint context",
        ha="center", va="center", fontsize=10, linespacing=1.25)
stub(MOD_X + MOD_W, MOD_BUS, g_cy)
ax.add_line(Line2D([MOD_BUS, MOD_BUS], [g_cy, mod_centres[0]],
                   color=BUS, lw=1.0, zorder=0))
arrow(MOD_BUS, CY, INT_X, CY)

# Integration (col 3)
box(INT_X, CY - 12, INT_W, 24,
    "285\nfeatures", C_INT, fs=15, weight="bold")

# Task heads (col 4)
eff_cy, saf_cy = 64.0, 41.0
box(HEAD_X, eff_cy - 7, HEAD_W, 14,
    "Efficacy / overall\n\ncalibrated\nclassifier", C_HEAD, fs=10)
box(HEAD_X, saf_cy - 8, HEAD_W, 16,
    "Safety\n\nsingle classifier\nover mechanism\ngroups", C_HEAD, fs=10)
arrow(INT_X + INT_W, CY + 2, HEAD_X, eff_cy)
arrow(INT_X + INT_W, CY - 2, HEAD_X, saf_cy)

# Outcome node (bottom) + non-crossing routing from both heads
OY = 8.0
box(68.0, OY - 4.5, 30.0, 9.0,
    "Trial outcome\nPASS · FAIL_EFFICACY · FAIL_SAFETY", C_OUT, fs=9.5)
arrow(HEAD_X + 5, saf_cy - 8, HEAD_X + 5, OY + 4.7)          # safety straight down
ax.add_patch(FancyArrowPatch((HEAD_X + HEAD_W, eff_cy), (HEAD_X + HEAD_W + 3, OY + 4.7),
             arrowstyle="-|>", mutation_scale=11, lw=1.0, color=ARROW,
             connectionstyle="angle,angleA=0,angleB=90", shrinkA=0, shrinkB=0))

# module-type legend: sits fully below the module column's lowest (gated) box,
# whose bottom edge is at y = g_cy - MOD_H/2 = 16.5, so the long legend labels can
# extend right into the module x-range without colliding with any box.
for i, (sw, lab) in enumerate([(C_STR, "structure-computed mechanism"),
                                (C_PRIOR, "prior target–disease evidence"),
                                (C_CTX, "disease / trial context"),
                                (C_GATE, "trial-design context (gated, module 9)")]):
    ly = 12.5 - i * 4.0
    ax.add_patch(FancyBboxPatch((IN_X, ly), 3, 3,
                 boxstyle="round,pad=0,rounding_size=0.8", fc=sw, ec=EDGE, lw=0.7))
    ax.text(IN_X + 4.5, ly + 1.5, lab, fontsize=10.5, va="center", ha="left")

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(OUT / f"fig1b_architecture.{ext}", bbox_inches="tight")
plt.close(fig); print("wrote fig1b_architecture")

# ---- Supplementary Fig. S3: per-module standalone AUC ----
mod = mod.sort_values("auc")
# Names must match the module boxes in panel a (same modules, one label each).
labelmap = {"network": "Pathway engagement", "tissue": "Tissue-specific binding",
            "pharmacokinetics": "Pharmacokinetics", "safety_pharmacology": "Safety pharmacology",
            "binding_specificity": "Binding specificity"}
fig, ax = plt.subplots(figsize=(6.2, 3.4))
y = range(len(mod))
ax.barh(list(y), mod.auc, color="#6a994e", alpha=0.85)
for yi, a in zip(y, mod.auc):
    ax.text(a + 0.004, yi, f"{a:.3f}", va="center", ha="left", fontsize=10)
ax.set_yticks(list(y)); ax.set_yticklabels([labelmap.get(m, m) for m in mod.module], fontsize=11)
ax.set_xlim(0.50, 1.00); ax.set_xticks(np.arange(0.50, 1.001, 0.10))
ax.axvline(0.5, color="k", lw=0.5)
ax.set_xlabel("standalone AUC", fontsize=11)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(OUT / f"figS1_module_auc.{ext}", bbox_inches="tight")
plt.close(fig); print("wrote figS1_module_auc")
