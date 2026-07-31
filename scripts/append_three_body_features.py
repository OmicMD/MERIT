#!/usr/bin/env python3
"""Three-body (pathogen x host x drug-at-site) axis features for anti-pathogen trials.

These describe the BIOLOGY OF THE INFECTION, not the drug molecule — the axis the human-target binding
pipeline is structurally blind to, and the reason anti-pathogen trials were excluded from the efficacy
model. Scored A-PRIORI from textbook pathogen/host biology (fixed rule tables) + MEASURED regimen depth
(blind to outcome). Validated in scripts/strengthening/three_body_axis_test.py (a-priori composite raw
AUC 0.809, shuffle 0.497; mono-vs-combo 3x within combination-required classes).

add_three_body_columns(df) adds tb_* columns for is_anti_pathogen rows (neutral 0 elsewhere). They are
PREFIX-GUARDED out of the main model (get_features skips 'tb_'); a dedicated anti-pathogen efficacy head
consumes them, so the canonical small-molecule model is unchanged. See notes/
three_body_characterizations_antipathogen.md.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REGIMEN_CSV = ROOT / "data/sources/ap_regimen_depth_jun16.csv"

IMMUNE_CLEARS = {  # can competent host immunity clear/control WITHOUT the drug? 3=self-limiting .. 0=never
    "ACUTE_RESP_VIRUS": 3, "ACUTE_VIRUS_OTHER": 3, "BACTERIA": 2,
    "PROTOZOA_PLASMODIUM": 1, "PROTOZOA_LEISHMANIA": 1, "FUNGAL": 1,
    "MYCOBACTERIA_TB": 1, "MYCOBACTERIA_NTM": 1, "HERPESVIRUS_CMV": 1, "HERPESVIRUS_HSV": 1,
    "CHRONIC_VIRUS_HCV": 0, "CHRONIC_VIRUS_HBV": 0, "CHRONIC_VIRUS_HIV": 0, "OTHER": 1, "NONE": 1,
}
CURATIVE = {  # does a durable-cure regimen exist for this pathogen as deployed?
    "ACUTE_RESP_VIRUS": 0, "ACUTE_VIRUS_OTHER": 0, "BACTERIA": 1, "FUNGAL": 1,
    "CHRONIC_VIRUS_HCV": 1, "PROTOZOA_PLASMODIUM": 1, "CHRONIC_VIRUS_HIV": 0, "CHRONIC_VIRUS_HBV": 0,
    "MYCOBACTERIA_TB": 1, "MYCOBACTERIA_NTM": 0, "PROTOZOA_LEISHMANIA": 1,
    "HERPESVIRUS_CMV": 1, "HERPESVIRUS_HSV": 1, "OTHER": 1, "NONE": 1,
}
COMBO_REQUIRED = {"CHRONIC_VIRUS_HIV", "CHRONIC_VIRUS_HBV", "MYCOBACTERIA_TB", "MYCOBACTERIA_NTM"}


def _has(s, *t):
    return s.apply(lambda x: any(k in x for k in t))


def classify_pathogen(disease, drug):
    """Per-trial pathogen class from indication + drug intent (a-priori, blind to outcome)."""
    pc = pd.Series("NONE", index=disease.index)
    D, DR = disease.fillna("").str.lower(), drug.fillna("").str.lower()
    pc[_has(D, "covid", "sars", "corona", "influenza", "flu", "respiratory infection")] = "ACUTE_RESP_VIRUS"
    pc[_has(D, "dengue")] = "ACUTE_VIRUS_OTHER"
    pc[_has(D, "hepatitis c", "hcv") | (_has(DR, "sofosbuvir","daclatasvir","asunaprevir","dasabuvir","boceprevir","ribavirin","lcq908") & _has(D,"hepatitis c","hcv","cirrhosis"))] = "CHRONIC_VIRUS_HCV"
    pc[_has(D, "hepatitis b", "hbv") | _has(DR, "telbivudine")] = "CHRONIC_VIRUS_HBV"
    pc[_has(D, "hiv", "immunodeficiency virus")] = "CHRONIC_VIRUS_HIV"
    pc[_has(D, "cmv", "cytomegalovirus")] = "HERPESVIRUS_CMV"
    pc[_has(D, "herpes", "hsv", "zoster", "varicella")] = "HERPESVIRUS_HSV"
    pc[_has(D, "mac ", "nontuberculous", "atypical", "mycobacterium infections")] = "MYCOBACTERIA_NTM"
    pc[_has(D, "tuberculosis", "tuberculoses")] = "MYCOBACTERIA_TB"
    pc[_has(D, "malaria", "plasmodium", "falciparum", "vivax", "knowlesi")] = "PROTOZOA_PLASMODIUM"
    pc[_has(D, "leishmania")] = "PROTOZOA_LEISHMANIA"
    pc[_has(D, "candid", "fungal", "crypto", "aspergill")] = "FUNGAL"
    pc[_has(D, "pneumonia","urinary","uti","bacteremia","skin","tissue","intra-abdominal","cellulitis",
            "gonorrhea","clostridium","pylori","bacterial","infection; surg","pyelonephritis","bsi",
            "appendicitis","osteomyelitis","hospital")] = "BACTERIA"
    abx = _has(DR, "cef","penicillin","amoxicillin","meropenem","ertapenem","vancomycin","daptomycin","linezolid",
               "tedizolid","cipro","moxiflox","delaflox","nitrofurantoin","plazomicin","eravacycline","omadacycline",
               "dalbavancin","metronidazole","sulbactam","ceftaroline","cefiderocol")
    pc[(pc == "NONE") & abx] = "BACTERIA"
    return pc


def add_three_body_columns(df, verbose=True):
    """Add tb_* three-body axis columns. Informative for is_anti_pathogen rows, neutral (0) elsewhere."""
    out = df.copy()
    is_ap = out["is_anti_pathogen"].fillna(0).astype(int) == 1 if "is_anti_pathogen" in out else pd.Series(False, index=out.index)

    pc = classify_pathogen(out["Disease"], out["Drug_Clean"])
    pc[~is_ap] = "NONE"
    out["tb_pathogen_class"] = pc

    D = out["Disease"].fillna("").str.lower()
    DR = out["Drug_Clean"].fillna("").str.lower()
    out["tb_immunity_can_clear"] = pc.map(IMMUNE_CLEARS).astype(float)
    out["tb_curative_class_exists"] = pc.map(CURATIVE).astype(float)
    # drug-pathogen match (a-priori pharmacology): default match; azole vs protozoan = mismatch
    match = pd.Series(1, index=out.index)
    match[_has(DR, "fluconazole", "azole") & (pc == "PROTOZOA_LEISHMANIA")] = 0
    out["tb_drug_pathogen_match"] = match.where(is_ap, 1).astype(float)
    # pathogen is the disease driver, vs bystander/secondary (antimicrobial aimed at a host-organ disease)
    host_organ = _has(D, "alcoholic hepatitis","alcoholic liver","pulmonary fibrosis","copd",
                      "chronic obstructive","respiratory failure","steatohep","nash")
    out["tb_pathogen_is_driver"] = (~(host_organ & is_ap)).astype(float)
    out["tb_site_sanctuary"] = (_has(D, "mening","encephal","cns","osteomyelitis","bone","mac ",
                                     "nontuberculous","endocard","prosth","abscess") & is_ap).astype(float)
    out["tb_combination_required"] = (pc.isin(COMBO_REQUIRED)).astype(float)

    # measured regimen depth (blind, from ct.gov protocols) — committed CSV
    if REGIMEN_CSV.exists():
        rd = pd.read_csv(REGIMEN_CSV)[["NCT_ID", "regimen_depth", "is_monotherapy"]]
        out = out.merge(rd, on="NCT_ID", how="left")
        out["tb_regimen_depth"] = out["regimen_depth"]
        out["tb_mono_when_combo_required"] = (((out["is_monotherapy"] == 1) &
                                               (out["tb_combination_required"] == 1)).astype(float))
        out = out.drop(columns=["regimen_depth", "is_monotherapy"])
    else:
        out["tb_regimen_depth"] = np.nan
        out["tb_mono_when_combo_required"] = 0.0

    # a-priori composite mechanistic FAIL-risk (fixed weights; primary four axes + combo modifier)
    risk = (
        2.0 * (1 - out["tb_drug_pathogen_match"])
      + 2.0 * (1 - out["tb_pathogen_is_driver"])
      + 0.7 * out["tb_immunity_can_clear"] * (1 - out["tb_curative_class_exists"])
      + 0.8 * out["tb_site_sanctuary"]
      + 0.6 * (1 - out["tb_curative_class_exists"]) * (out["tb_immunity_can_clear"] <= 1).astype(float)
    )
    out["tb_mech_fail_risk"] = risk.where(is_ap, 0.0)
    out["tb_mech_fail_risk_combo"] = (risk + 1.5 * out["tb_mono_when_combo_required"]).where(is_ap, 0.0)

    if verbose:
        n = int(is_ap.sum())
        print(f"[three-body] tb_* columns added for {n} anti-pathogen rows (neutral elsewhere); "
              f"pathogen classes: {pc[is_ap].value_counts().to_dict()}")
    return out


if __name__ == "__main__":
    p = ROOT / "data/sources/training_dataset_v8_honest_exposure.csv"
    d = add_three_body_columns(pd.read_csv(p, low_memory=False))
    tb = [c for c in d.columns if c.startswith("tb_")]
    print("tb_ columns:", tb)
