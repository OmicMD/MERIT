#!/usr/bin/env python3
"""
Path C, step 4: tighten the locked set from "novel to the training cohort" to
"genuinely investigational (drug, indication)" + correct the model's known
over-flag, so the prospective predictions are defensible at readout.

Two outcome-blind, prediction-blind passes plus a flagged residual:

  PASS 1 - ChEMBL approval exclusion (deterministic).
    Exclude any (drug, indication) pair whose drug holds ChEMBL
    drug_indication.max_phase_for_ind == 4 (approved) for that indication.
    Matched drug via InChIKey-14 -> molregno; indication via token-subset on
    mesh_heading/efo_term. Label-blind: applied regardless of predicted label
    (e.g. metformin->Type 2 Diabetes is dropped even though predicted PASS).
    This is the bulk, reproducible removal.

  PASS 2 - decision-layer cytotoxic-monotherapy cap (the manuscript's own
    over-flag correction, scripts/strengthening/efficacy_decision_layer.py).
    For oncology + cytotoxic-target (TYMS/DNMT1/TUBB/TOP/...) + MONOTHERAPY
    trials, cap P_fail at 0.15. Monotherapy is derived from the ongoing trial's
    ct.gov experimental arm (NOT the imputed lock value): a single DRUG-type
    intervention in the experimental arm. This corrects broad antiproliferatives
    the mechanism-fit model over-flags (gemcitabine/capecitabine->NPC etc.).

  PASS 3 - human verdict application (committed determination).
    Pairs still predicted FAIL after 1+2 are the confident-FAIL residual. Their
    approval/SOC status is adjudicated by a committed, web-verified determination
    (data/sources/prereg_C_residual_verdicts_v1.csv) because the structured sources
    under-report exactly these: ChEMBL records venetoclax->AML and ruxolitinib->
    myelofibrosis at phase 3 despite FDA approval, and off-label guideline SOC is in
    no local table. Pairs marked EXCLUDE (established/approved/SOC) or KEEP-AMBIG
    (indication too vague to score, e.g. "Cancer") are dropped from the FINAL set.
    Editing the verdicts CSV changes the final set deterministically.

Inputs : results/benchmark/prereg_C/prereg_C_locked_predictions_clean.csv
         results/benchmark/prereg_C_trials.csv  (IK14 per drug)
         data/cache/chembl_36/chembl_36_sqlite/chembl_36.db
         data/cache/ctgov_ongoing_p3.json       (experimental-arm interventions)
         data/sources/ik14_moa_targets_combined_v1.csv
         data/sources/training_dataset_v8_clean_mort.csv  (disease_is_oncology)
         data/sources/prereg_C_residual_verdicts_v1.csv   (committed human verdicts)
Outputs: results/benchmark/prereg_C/prereg_C_locked_predictions_final.csv (+ .sha256)
         results/benchmark/prereg_C/prereg_C_residual_for_review.csv  (audit + verdicts)
"""
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/benchmark/prereg_C"
CLEAN = OUT / "prereg_C_locked_predictions_clean.csv"
TRIALS = ROOT / "results/benchmark/prereg_C_trials.csv"
DB = ROOT / "data/cache/chembl_36/chembl_36_sqlite/chembl_36.db"
CTGOV = ROOT / "data/cache/ctgov_ongoing_p3.json"
MOA = ROOT / "data/sources/ik14_moa_targets_combined_v1.csv"
COHORT = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
# Committed human determination on the confident-FAIL residual (web-verified approval/SOC
# status). Applied deterministically here; edit this CSV to change the final set.
VERDICTS = ROOT / "data/sources/prereg_C_residual_verdicts_v1.csv"
# Gap C (Jun 28): committed outcome-blindness + junior-partner exclusions (web-verified, each row
# evidenced). Unlike VERDICTS (confident-FAIL residual only), these drop a bet at ANY label because
# the trial is not a valid prospective, single-agent test of OUR drug: it read out before the lock,
# our drug is the SOC control / already-approved in the indication, or it is the junior partner of a
# novel lead. See scripts/benchmark/prereg_C_combination_attribution.py (auto-detector + audit).
BLINDNESS_EXCL = ROOT / "data/sources/prereg_C_blindness_exclusions_v1.csv"
CAP = 0.15
CYTOTOXIC_TARGETS = {"TUBB", "TUBB1", "TUBB4B", "TUBA1A", "TUBA1B", "TUBA4A", "TUBA3C", "TYMS",
                     "RRM1", "RRM2", "RRM2B", "DHFR", "GART", "TOP1", "TOP2A", "TOP2B", "DNMT1",
                     "POLA1", "TYMP"}
_STOP = {"disease", "disorder", "syndrome", "of", "the", "and", "chronic", "acute", "type",
         "primary", "secondary", "cancer", "neoplasm", "malignant", "advanced", "metastatic",
         "diseases"}


def toks(s):
    return {t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t and t not in _STOP and len(t) > 3}


def chembl_max_phase(clean):
    """Per (drug-IK14, indication): the highest ChEMBL development phase, by
    token-subset match on mesh_heading/efo_term. None if no indication match."""
    iks = sorted(clean.IK14.dropna().unique())
    con = sqlite3.connect(DB)
    cs = pd.read_sql("SELECT substr(standard_inchi_key,1,14) ik14, molregno FROM compound_structures", con)
    ik2mol = cs[cs.ik14.isin(iks)].groupby("ik14").molregno.apply(list).to_dict()
    allmol = sorted({m for ms in ik2mol.values() for m in ms})
    di = pd.read_sql(
        f"SELECT molregno,max_phase_for_ind,mesh_heading,efo_term FROM drug_indication "
        f"WHERE molregno IN ({','.join(map(str, allmol))})", con)
    con.close()
    mol_rows = {m: di[di.molregno == m] for m in allmol}

    def phase(ik, ind):
        it = toks(ind)
        if not it:
            return None
        best = None
        for m in ik2mol.get(ik, []):
            for r in mol_rows[m].itertuples():
                for term in (r.mesh_heading, r.efo_term):
                    tt = toks(term)
                    if tt and (it <= tt or tt <= it):
                        best = max(best or 0.0, float(r.max_phase_for_ind))
        return best

    return [phase(ik, ind) for ik, ind in zip(clean.IK14, clean.novel_indication)]


def trial_is_combination(clean):
    """Derive monotherapy/combination from the ongoing trial's ct.gov EXPERIMENTAL
    arm: >1 distinct DRUG-type intervention named in any experimental arm = combination.
    (The lock imputed is_combination; this recovers the real status.)"""
    cache = json.load(open(CTGOV))
    nct_exp_drugs = {}
    for _drug, studies in cache.items():
        for s in studies or []:
            p = s.get("protocolSection", {})
            nct = p.get("identificationModule", {}).get("nctId")
            if not nct:
                continue
            am = p.get("armsInterventionsModule", {})
            for a in (am.get("armGroups", []) or []):
                if a.get("type") == "EXPERIMENTAL":
                    drugs = {n.split(":", 1)[-1].strip().lower()
                             for n in (a.get("interventionNames", []) or [])
                             if n.lower().startswith("drug:")}
                    nct_exp_drugs.setdefault(nct, set()).update(drugs)
    out = []
    for nct in clean.NCT_ID:
        d = nct_exp_drugs.get(nct)
        out.append(np.nan if d is None else int(len(d) > 1))
    return out


def main():
    clean = pd.read_csv(CLEAN)
    trials = pd.read_csv(TRIALS)
    drug2ik = (trials.drop_duplicates("Drug_Clean")
               .assign(IK14=lambda x: x.IK14.astype(str).str[:14])
               .set_index("Drug_Clean").IK14.to_dict())
    clean["IK14"] = clean.drug.map(drug2ik)

    coh = pd.read_csv(COHORT, low_memory=False)
    dis_onc = coh.drop_duplicates("Disease").set_index("Disease")["disease_is_oncology"].to_dict()
    moa = pd.read_csv(MOA)
    ik2tg = moa.groupby("ik14").target_gene.apply(lambda s: set(str(x) for x in s.dropna())).to_dict()

    # ---- PASS 1: ChEMBL approval exclusion (label-blind) ----
    clean["chembl_max_phase"] = chembl_max_phase(clean)
    clean["chembl_approved"] = clean.chembl_max_phase >= 4

    # ---- PASS 2: decision-layer cytotoxic-monotherapy cap ----
    clean["is_combination_ctgov"] = trial_is_combination(clean)
    clean["is_oncology"] = clean.novel_indication.map(lambda x: dis_onc.get(x, 0) == 1)
    clean["cyto_target"] = clean.IK14.map(lambda ik: bool(ik2tg.get(ik, set()) & CYTOTOXIC_TARGETS)
                                          if pd.notna(ik) else False)
    # monotherapy = explicitly not-combination (NaN combination status -> NOT capped, conservative)
    clean["cyto_mono"] = (clean.is_oncology & clean.cyto_target & (clean.is_combination_ctgov == 0))
    clean["P_fail_adj"] = np.where(clean.cyto_mono,
                                   np.minimum(clean.P_fail_overall_calibrated, CAP),
                                   clean.P_fail_overall_calibrated)
    clean["label_adj"] = np.where(clean.P_fail_adj >= 0.5, "FAIL", "PASS")

    # ---- candidate surviving set (drop ChEMBL-approved) ----
    cand = clean[~clean.chembl_approved].copy()

    # ---- merge the committed human verdicts on the confident-FAIL residual ----
    key = lambda s: s.str.lower().str.strip()
    cand["_k"] = key(cand.drug) + "||" + key(cand.novel_indication)
    if VERDICTS.exists():
        v = pd.read_csv(VERDICTS)
        v["_k"] = key(v.drug) + "||" + key(v.indication)
        vmap = v.drop_duplicates("_k").set_index("_k")
        cand = cand.merge(vmap[["verdict", "confidence", "rationale", "source"]],
                          left_on="_k", right_index=True, how="left")
    else:
        cand["verdict"] = np.nan
    cand["verdict"] = cand["verdict"].fillna("KEEP")

    # residual = confident-FAIL pairs (the rows the verdicts adjudicate)
    resid = cand[cand.label_adj == "FAIL"].copy()
    resid["ctgov_url"] = "https://clinicaltrials.gov/study/" + resid.NCT_ID
    resid["EXCLUDE_human"] = resid.verdict.map(
        {"EXCLUDE": "Y", "KEEP-AMBIG": "?"}).fillna("N")

    # FINAL locked set: drop pairs the human determination marks EXCLUDE (established/
    # approved/SOC) or KEEP-AMBIG (indication too vague to score, e.g. "Cancer").
    drop = cand.verdict.isin(["EXCLUDE", "KEEP-AMBIG"])
    final = cand[~drop].copy()

    # ---- Gap C: drop outcome-blindness / junior-partner / control-arm / approved-indication
    # exclusions (committed, web-verified). Matched on NCT_ID + drug so a drug excluded in one
    # trial/indication is unaffected elsewhere. ----
    if BLINDNESS_EXCL.exists():
        bx = pd.read_csv(BLINDNESS_EXCL)
        bx_keys = set(zip(bx.NCT_ID.astype(str), bx.drug.str.lower().str.strip()))
        bmask = [(str(n), str(d).lower().strip()) in bx_keys
                 for n, d in zip(final.NCT_ID, final.drug)]
        n_bx = int(sum(bmask))
        if n_bx:
            reasons = bx.groupby("reason").size().to_dict()
            print(f"Gap-C outcome-blindness/attribution exclusions dropped: {n_bx} "
                  f"(committed {len(bx)}; reasons {reasons})")
            for n, d, lab in zip(final.NCT_ID[bmask], final.drug[bmask],
                                 final.label_adj[bmask] if "label_adj" in final else [""] * n_bx):
                print(f"    - {d} / {n} ({lab})")
        final = final[[not m for m in bmask]].copy()

    # LOUD feature-completeness flag in the deposited sheet: a pair missing its target->disease
    # mechanism coverage has that whole biology block absent (not just imputed) and must be
    # re-run through the pipeline before it can be trusted (all forward pairs additionally impute
    # 4 disease-specific direct-target features).
    mcov = (pd.read_csv(ROOT / "data/sources/mechanism_dataderived_prereg_C.csv")
            [["IK14", "Disease", "coverage_disease", "coverage_drug", "direct_target_max"]]
            .drop_duplicates(["IK14", "Disease"]))
    final = final.merge(mcov, left_on=["IK14", "novel_indication"],
                        right_on=["IK14", "Disease"], how="left")
    # All biological sub-blocks are recomputed per pair: direct-target + OT/topology/KEGG mechanism
    # (prereg_C_build_mech.py) AND Mendelian/DepMap/mechanism-impact genetics (prereg_C_build_genetics.py).
    # Honest 3-way completeness (prereg_C_completeness): COMPLETE / N/A (drug acts without a protein
    # target — cytotoxic-DNA or metabolic MOA, mechanism-fit not applicable) / INCOMPLETE (resolvable
    # data gap: disease module not in OT, or drug targets not yet OT-mapped). "Targetless" is never asserted.
    from prereg_C_completeness import label as _completeness, summary as _csum
    final["feature_completeness"] = _completeness(final, ik_col="IK14")
    print(f"deposited set feature completeness: {_csum(final['feature_completeness'])}")

    extra = [c for c in ["P_fail_efficacy_calibrated", "P_fail_safety_calibrated", "model_prediction"]
             if c in cand.columns]
    final_cols = (["NCT_ID", "drug", "novel_indication", "indication_ctgov", "feature_completeness",
                   "overall_status", "P_fail_overall_raw", "P_fail_overall_calibrated"] + extra +
                  ["P_fail_adj", "label_adj", "chembl_max_phase", "cyto_mono", "lock_date"])
    resid_cols = (["NCT_ID", "drug", "novel_indication", "P_fail_overall_calibrated"] + extra +
                  ["verdict", "EXCLUDE_human", "confidence", "rationale", "chembl_max_phase",
                   "is_combination_ctgov", "cyto_mono", "ctgov_url", "source"])
    final_out = OUT / "prereg_C_locked_predictions_final.csv"
    resid_out = OUT / "prereg_C_residual_for_review.csv"
    final[final_cols].to_csv(final_out, index=False)
    sha = hashlib.sha256(final_out.read_bytes()).hexdigest()
    (OUT / "prereg_C_locked_predictions_final.sha256").write_text(
        sha + "  prereg_C_locked_predictions_final.csv\n")
    resid[[c for c in resid_cols if c in resid.columns]].sort_values(
        "P_fail_overall_calibrated", ascending=False).to_csv(resid_out, index=False)

    # ---- report ----
    print(f"START (attribution-clean lock): {len(clean)} preds / {clean.NCT_ID.nunique()} trials / "
          f"{clean.drug.nunique()} drugs ({int((clean.predicted_label=='FAIL').sum())} FAIL)")
    print(f"PASS 1 ChEMBL approved (phase 4) -> EXCLUDE {int(clean.chembl_approved.sum())} pairs "
          f"({int((clean.chembl_approved & (clean.predicted_label=='FAIL')).sum())} FAIL)")
    print(f"PASS 2 cytotoxic-mono cap: {int(clean.cyto_mono.sum())} pairs "
          f"(real ct.gov combo status; NaN not capped: {int(clean.is_combination_ctgov.isna().sum())})")
    nE = int((resid.verdict == 'EXCLUDE').sum()); nA = int((resid.verdict == 'KEEP-AMBIG').sum())
    print(f"PASS 3 human verdicts on {len(resid)} confident-FAIL residual: "
          f"EXCLUDE {nE}, KEEP-AMBIG {nA}, KEEP {len(resid)-nE-nA}")
    print(f"\nFINAL locked set: {len(final)} preds / {final.NCT_ID.nunique()} trials / "
          f"{final.drug.nunique()} drugs ({int((final.label_adj=='FAIL').sum())} FAIL)")
    print(f"SHA-256 {sha}")
    print(f"  -> {final_out.name} + residual audit {resid_out.name}")
    if (resid.verdict == "").any() or resid.verdict.isna().any():
        print("  WARNING: some residual rows have no committed verdict.")


if __name__ == "__main__":
    main()
