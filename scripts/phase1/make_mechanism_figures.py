"""Three publication figures for the manuscript v6 "Mechanism of failure" subsection.

1. PC1 × PC2 scatter colored by drug class (9 classes)
2. PC6 (disease-network alignment) distribution split by trial outcome
3. Case-study PC profile radar: nilotinib, panobinostat, metoprolol, ivacaftor, olaparib

Inputs:
- results/phase1/analysis_explanatory/systemic_framework/trial_pca_scores.csv
- data/sources/training_dataset_v5_unified.csv  (for drug-class keyword matching)

Outputs (PNG + SVG): results/phase1/analysis_explanatory/figures/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase1/analysis_explanatory/figures"
OUT.mkdir(parents=True, exist_ok=True)

DRUG_CLASSES = {
    "TKI":            ["sorafenib","sunitinib","lenvatinib","neratinib","imatinib","nilotinib","dasatinib","erlotinib","gefitinib","lapatinib","pazopanib","regorafenib","cabozantinib","vandetanib","axitinib","ibrutinib","crizotinib","afatinib","ceritinib","alectinib","ruxolitinib","palbociclib","ribociclib","trametinib","cobimetinib"],
    "PARP":           ["olaparib","talazoparib","niraparib","rucaparib","veliparib","pamiparib"],
    "HDAC":           ["panobinostat","vorinostat","romidepsin","belinostat","entinostat"],
    "Statin":         ["atorvastatin","rosuvastatin","simvastatin","pravastatin","lovastatin","pitavastatin","fluvastatin"],
    "NSAID":          ["celecoxib","diclofenac","ibuprofen","naproxen","ketorolac","aspirin","etoricoxib","indomethacin","meloxicam","piroxicam","rofecoxib"],
    "β-blocker":      ["metoprolol","propranolol","atenolol","carvedilol","bisoprolol","nebivolol","labetalol","esmolol","sotalol"],
    "SSRI/SNRI":      ["sertraline","fluoxetine","paroxetine","citalopram","escitalopram","duloxetine","venlafaxine","fluvoxamine","desvenlafaxine","milnacipran"],
    "Corticosteroid": ["prednisone","prednisolone","dexamethasone","methylprednisolone","hydrocortisone","budesonide","fluticasone","triamcinolone","betamethasone","mometasone"],
    "Cytotoxic chemo": ["cyclophosphamide","doxorubicin","cisplatin","carboplatin","oxaliplatin","paclitaxel","docetaxel","gemcitabine","fluorouracil","capecitabine","irinotecan","etoposide","methotrexate","vincristine","vinblastine","mitomycin","bleomycin"],
}

CLASS_COLORS = {
    "TKI": "#d62728",
    "PARP": "#9467bd",
    "HDAC": "#8c564b",
    "Statin": "#1f77b4",
    "NSAID": "#2ca02c",
    "β-blocker": "#17becf",
    "SSRI/SNRI": "#bcbd22",
    "Corticosteroid": "#e377c2",
    "Cytotoxic chemo": "#7f7f7f",
    "Other": "#cccccc",
}

OUTCOME_COLORS = {
    "PASS":          "#2ca02c",
    "FAIL_EFFICACY": "#ff7f0e",
    "FAIL_SAFETY":   "#d62728",
    "FAIL_BOTH":     "#7f4f24",
}


def assign_class(drug: str) -> str:
    if not isinstance(drug, str):
        return "Other"
    d = drug.lower().strip()
    for klass, members in DRUG_CLASSES.items():
        if any(m in d for m in members):
            return klass
    return "Other"


def load_pca():
    pca = pd.read_csv(ROOT / "results/phase1/analysis_explanatory/systemic_framework/trial_pca_scores.csv")
    pca["drug_class"] = pca["Drug_Clean"].map(assign_class)
    return pca


# ---------------------------------------------------------------------------
# Figure 1: PC1 × PC2 by drug class
# ---------------------------------------------------------------------------
def fig_pc1_pc2(pca: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    # plot "Other" first (background)
    other = pca[pca["drug_class"] == "Other"]
    ax.scatter(other["PC1"], other["PC2"], s=4, c=CLASS_COLORS["Other"],
               alpha=0.15, linewidths=0, label=None)
    # plot labelled classes
    for klass in DRUG_CLASSES.keys():
        sub = pca[pca["drug_class"] == klass]
        if len(sub) == 0:
            continue
        ax.scatter(sub["PC1"], sub["PC2"], s=14, c=CLASS_COLORS[klass],
                   alpha=0.7, linewidths=0, label=f"{klass} (n={len(sub)})")

    ax.axhline(0, color="black", lw=0.4, alpha=0.3)
    ax.axvline(0, color="black", lw=0.4, alpha=0.3)
    ax.set_xlabel("PC1 — tissue engagement breadth (23.3%)")
    ax.set_ylabel("PC2 — binding intensity (18.2%)")
    ax.set_title("Drug classes occupy distinct PC1×PC2 regions consistent with pharmacology")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig5a_pc1_pc2_by_class.png", bbox_inches="tight")
    fig.savefig(OUT / "fig5a_pc1_pc2_by_class.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig5a_pc1_pc2_by_class.{{png,svg}}")


# ---------------------------------------------------------------------------
# Figure 2: PC6 distribution by outcome
# ---------------------------------------------------------------------------
def fig_pc6_outcome(pca: pd.DataFrame):
    from scipy.stats import gaussian_kde
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    outcomes = ["PASS", "FAIL_EFFICACY", "FAIL_SAFETY"]
    xs = np.linspace(-6, 9, 400)
    for o in outcomes:
        sub = pca[pca["Corrected_Outcome"] == o]
        if len(sub) < 30:
            continue
        kde = gaussian_kde(sub["PC6"].values, bw_method=0.35)
        ys = kde(xs)
        ax.fill_between(xs, ys, color=OUTCOME_COLORS[o], alpha=0.35,
                        label=f"{o} (n={len(sub)})")
        ax.plot(xs, ys, color=OUTCOME_COLORS[o], lw=1.2)
    ax.set_xlim(-6, 9)

    # Cliff's d annotation
    pass_pc6 = pca.loc[pca["Corrected_Outcome"] == "PASS", "PC6"].values
    eff_pc6  = pca.loc[pca["Corrected_Outcome"] == "FAIL_EFFICACY", "PC6"].values
    saf_pc6  = pca.loc[pca["Corrected_Outcome"] == "FAIL_SAFETY", "PC6"].values

    def cliffs_d(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return (np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a)*len(b))

    d_eff = cliffs_d(eff_pc6, pass_pc6)
    d_saf = cliffs_d(saf_pc6, pass_pc6)

    ax.set_xlabel("PC6 — disease-network alignment (3.1% var)")
    ax.set_ylabel("Density")
    ax.set_title(f"PC6 separates efficacy failures from PASS  (Cliff's d = {d_eff:+.2f})")
    ax.text(0.02, 0.97,
            f"Efficacy fail vs PASS: d = {d_eff:+.2f}\nSafety fail vs PASS: d = {d_saf:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round,pad=0.3"))
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig5b_pc6_by_outcome.png", bbox_inches="tight")
    fig.savefig(OUT / "fig5b_pc6_by_outcome.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig5b_pc6_by_outcome.{{png,svg}}  (eff d={d_eff:+.2f}, saf d={d_saf:+.2f})")


# ---------------------------------------------------------------------------
# Figure 3: case-study PC radar
# ---------------------------------------------------------------------------
CASE_DRUGS = ["Nilotinib", "Panobinostat", "Metoprolol", "Ivacaftor", "Olaparib"]
CASE_COLORS = ["#d62728", "#8c564b", "#17becf", "#2ca02c", "#9467bd"]


def fig_case_radar(pca: pd.DataFrame):
    # Z-score PC scores against the full distribution for comparability
    z = pca[["PC1","PC2","PC3","PC4","PC5","PC6"]].copy()
    z = (z - z.mean()) / z.std()
    z["Drug_Clean"] = pca["Drug_Clean"]

    fig, ax = plt.subplots(figsize=(6.0, 6.0), subplot_kw=dict(polar=True))
    axes_labels = [
        "PC1\ntissue breadth",
        "PC2\nbinding intensity",
        "PC3\norgan peak",
        "PC4\ntranscript count",
        "PC5\nantitarget tox",
        "PC6\ndisease align",
    ]
    n = len(axes_labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist() + [0]

    for drug, color in zip(CASE_DRUGS, CASE_COLORS):
        rows = z[z["Drug_Clean"].str.lower() == drug.lower()]
        if len(rows) == 0:
            print(f"  WARN: {drug} not found in PCA scores")
            continue
        vals = rows[["PC1","PC2","PC3","PC4","PC5","PC6"]].mean().tolist()
        vals.append(vals[0])
        ax.plot(angles, vals, color=color, lw=1.8, label=drug)
        ax.fill(angles, vals, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=8)
    ax.set_rgrids([-2,-1,0,1,2,3], angle=90, fontsize=7, color="gray")
    ax.set_ylim(-3, 4)
    ax.set_title("Case-study PC fingerprints (z-scored across trial cohort)",
                 y=1.10, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig5c_case_pc_radar.png", bbox_inches="tight")
    fig.savefig(OUT / "fig5c_case_pc_radar.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig5c_case_pc_radar.{png,svg}")


def main():
    pca = load_pca()
    print(f"Loaded PCA scores: {len(pca)} trials")
    print(pca["drug_class"].value_counts().to_string())
    fig_pc1_pc2(pca)
    fig_pc6_outcome(pca)
    fig_case_radar(pca)
    print(f"\nAll figures written to {OUT}")


if __name__ == "__main__":
    main()
