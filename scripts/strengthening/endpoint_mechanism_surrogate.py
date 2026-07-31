#!/usr/bin/env python3
"""Endpoint-mechanism surrogate decision-layer (Jun 17) — COMMITTED, reproducible.

Reads WHAT each trial measured (ctgov primary-outcome text) against WHAT the drug does (ATC L1 +
target-gene organ), and flags trials whose primary endpoint is an ON-MECHANISM pharmacodynamic /
organ-function surrogate the drug directly moves. These pass ~95% of the time regardless of disease
difficulty, so the model (which judges disease) over-flags them. A surgical decision-layer cap on
just these cases removes 13 confident false positives at the cost of 3 false negatives (net -10),
without retraining (retraining redistributes mass globally and nets ~0; see notes).

Outcome-blind: category uses only endpoint text + drug ATC/target. Cap = a-priori operating point.

Outputs:
  data/sources/endpoint_mechanism_v1.csv   (per-trial category, outcome-blind)
  prints the confident-miss delta on the canonical efficacy OOF.

Builds on the per-case investigation (notes/investigation_repurposing_bets_jun17.md and the
endpoint×mechanism thread). Sibling of the risk-tolerance (Supplementary Table S12) and
effect-size-uncertainty (built but not adopted in the manuscript) decision layers.
"""
import json, re, sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent

READOUT = [
    ("CLINICAL_EVENT", r"mortalit|death|survival|\bmace\b|hospitali|stroke|infarction|recurrenc|relapse|composite|progression-free|event-free|disease-free|new ischemic|new lesion|exacerbation|incidence of|treatment failure|global rank"),
    ("ORGAN_FUNCTION", r"ejection fraction|ventricular|strain|egfr|gfr|creatinine clearance|fev1|fvc|walk distance|6-?minute|flow.mediated|systolic function|forced expiratory|spirometr|polysomnograph|sleep time|renal plasma flow|saturation"),
    ("BIOMARKER", r"hba1c|a1c|glycosylated|glucose|insulin resistance|homa|\bldl\b|cholesterol|triglycerid|body weight|\bbmi\b|blood pressure|viral load|platelet reactiv|level|concentration|count|slope|biomarker|histolog|fibrosis (stage|score)|remission|spleen"),
    ("SYMPTOM_SCALE", r"scale|score|\bdays\b|severity|symptom|madrs|panss|ham-?d|rating|questionnaire|\bpain\b|ipss"),
]
ORGAN = [
    ("cardiac", r"cardiac|heart|ventric|ejection|myocard|\bmace\b|coronary|\blv\b|systolic|heart failure"),
    ("renal", r"renal|kidney|egfr|gfr|creatinine|nephro|albumin|plasma flow|dialys"),
    ("metabolic", r"hba1c|a1c|glucose|glycem|insulin|\bbmi\b|body weight|lipid|cholesterol|triglycerid|metabolic|diabet|neonatal"),
    ("respiratory", r"fev1|fvc|copd|asthma|exacerbation|respiratory|pulmonary|oxygen|saturation|spirometr|airway|walk distance"),
    ("cns", r"cogniti|depress|madrs|panss|seizure|sleep|headache|migraine|delirium|\bpain\b|neuro|stroke|epilep"),
    ("hematologic", r"hemoglobin|platelet|remission|thrombo|leukemi|lymphoma|myelo|anemia|blast|spleen"),
    ("hepatic", r"liver|hepatic|fibrosis|nash|steatohepat|histolog|bilirubin"),
    ("oncology", r"tumou?r|response rate|\borr\b|progression-free|overall survival|\bos\b|\bpfs\b|cancer|carcinoma|lesion"),
    ("infection", r"infection|viral|sars|covid|influenza|bacteri|antimicrob"),
]
ATC_ORGAN = {
    "CARDIOVASCULAR SYSTEM": "cardiac", "ALIMENTARY TRACT AND METABOLISM": "metabolic",
    "NERVOUS SYSTEM": "cns", "RESPIRATORY SYSTEM": "respiratory",
    "BLOOD AND BLOOD FORMING ORGANS": "hematologic", "ANTINEOPLASTIC AND IMMUNOMODULATING AGENTS": "oncology",
    "ANTIINFECTIVES FOR SYSTEMIC USE": "infection", "MUSCULO-SKELETAL SYSTEM": "msk",
    "GENITO URINARY SYSTEM AND SEX HORMONES": "endocrine",
    "SYSTEMIC HORMONAL PREPARATIONS, EXCL. SEX HORMONES AND INSULINS": "endocrine",
    "DERMATOLOGICALS": "derm", "SENSORY ORGANS": "sensory", "VARIOUS": "various",
    "ANTIPARASITIC PRODUCTS, INSECTICIDES AND REPELLENTS": "infection",
}
CAP = 0.15  # a-priori operating point for on-mechanism-surrogate trials


def classify(txt, table, default="other"):
    t = txt.lower()
    for label, pat in table:
        if re.search(pat, t):
            return label
    return default


def build_categories():
    ep = json.load(open(ROOT / "data/cache/ctgov_primary_outcomes_protocol.json"))
    def etext(nct):
        v = ep.get(nct)
        if not v: return ""
        o = v[0] if isinstance(v, list) else v
        return f"{o.get('title','')} {o.get('desc','')}"
    atc = pd.read_csv(ROOT / "data/sources/cohort_atc_v1.csv")
    ik2atc = atc.dropna(subset=["level1_description"]).groupby("IK14").level1_description.first().map(ATC_ORGAN).to_dict()
    def drug_organ(ik):
        return ik2atc.get(ik, "unk")

    df = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
    df["IK14"] = df.feature_IK.astype(str).str[:14]
    rows = df[["NCT_ID", "Drug_Clean", "Disease", "IK14"]].drop_duplicates("NCT_ID").copy()
    rows["etxt"] = rows.NCT_ID.map(etext)
    rows = rows[rows.etxt.str.len() > 3].copy()
    rows["endpoint_readout"] = rows.etxt.map(lambda t: classify(t, READOUT))
    rows["endpoint_organ"] = rows.etxt.map(lambda t: classify(t, ORGAN))
    rows["drug_organ"] = rows.IK14.map(drug_organ).fillna("unk")
    on = (rows.endpoint_organ == rows.drug_organ) & (rows.drug_organ != "unk")
    def cat(r):
        if r._on and r.endpoint_readout in ("ORGAN_FUNCTION", "BIOMARKER"): return "on_mech_surrogate"
        if r.endpoint_readout == "CLINICAL_EVENT": return "clinical_on_mech" if r._on else "clinical_off_or_other"
        if not r._on and r.endpoint_organ != "other": return "off_mech"
        return "ambiguous"
    rows["_on"] = on
    rows["endpoint_mech_category"] = rows.apply(cat, axis=1)
    rows["is_on_mech_surrogate"] = (rows.endpoint_mech_category == "on_mech_surrogate").astype(int)
    # surgical-pass = reliably-movable PD surrogate: on-mechanism surrogate OR intrinsically-easy
    # metabolic biomarker (weight/HbA1c/glucose move for most interventions; 3.8% fail, within-phase).
    rows["is_surrogate_pass"] = (rows.is_on_mech_surrogate.astype(bool) |
        ((rows.endpoint_organ == "metabolic") & (rows.endpoint_readout == "BIOMARKER"))).astype(int)
    out = rows[["NCT_ID", "Drug_Clean", "Disease", "endpoint_readout", "endpoint_organ",
                "drug_organ", "endpoint_mech_category", "is_on_mech_surrogate", "is_surrogate_pass"]]
    out.to_csv(ROOT / "data/sources/endpoint_mechanism_v1.csv", index=False)
    print(f"wrote data/sources/endpoint_mechanism_v1.csv ({len(out)} trials; "
          f"on_mech_surrogate={int(out.is_on_mech_surrogate.sum())})")
    return out


def apply_decision_layer(cats):
    oof = pd.read_parquet(ROOT / "results/production_v8_clean_mort_gapBD_jun28/oof_efficacy.parquet")
    df = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
    nct = df[["SMILES", "Disease", "NCT_ID"]].drop_duplicates(["SMILES", "Disease"])
    agg = oof.groupby(["SMILES", "Disease"]).agg(y=("y", "first"), p=("raw_prob", "mean")).reset_index()
    m = agg.merge(nct, on=["SMILES", "Disease"], how="left").merge(
        cats[["NCT_ID", "is_surrogate_pass"]], on="NCT_ID", how="left")
    m["is_surrogate_pass"] = m.is_surrogate_pass.fillna(0)
    oms = m.is_surrogate_pass == 1
    m["p_adj"] = m.p.where(~oms, np.minimum(m.p, CAP))
    def conf(p): return int(((m.y == 1) & (p < 0.30)).sum()), int(((m.y == 0) & (p > 0.60)).sum())
    fnb, fpb = conf(m.p); fna, fpa = conf(m.p_adj)
    print(f"\nDecision-layer surrogate cap (P_fail<={CAP} for surrogate-pass, n={int(oms.sum())}):")
    print(f"  confident FN {fnb} -> {fna} ({fna-fnb:+d}) | FP {fpb} -> {fpa} ({fpa-fpb:+d}) | "
          f"total {fnb+fpb} -> {fna+fpa} ({fna+fpa-fnb-fpb:+d})")
    print(f"  on_mech_surrogate cohort Brier {(((m[oms].p-m[oms].y)**2).mean()):.3f} -> "
          f"{(((m[oms].p_adj-m[oms].y)**2).mean()):.3f}")
    m[["NCT_ID", "Disease", "y", "p", "p_adj", "is_surrogate_pass"]].to_csv(
        ROOT / "results/production_v8_clean_mort_gapBD_jun28/oof_efficacy_surrogate_adjusted.csv", index=False)
    print("  wrote oof_efficacy_surrogate_adjusted.csv")


if __name__ == "__main__":
    cats = build_categories()
    apply_decision_layer(cats)
