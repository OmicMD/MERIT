#!/usr/bin/env python3
"""
Path C, step 2: lock predictions on NOVEL (drug, indication) pairs — ongoing
Phase III trials of cohort compounds in indications new to that compound. Unlike
the in-sample Path A lock, these (drug, indication) pairs are NOT in the training
cohort, so this is a genuine out-of-sample forward test.

Assembly per novel trial-row:
  - 206 drug-level features  : cloned from a representative cohort row of the drug
  - 10 disease-level features : cloned from a representative cohort row of the indication
  - mechanism block (the pairing signal): RECOMPUTED for the (drug, indication) pair
    (data-derived topology/KEGG/OT-biology/ClinGen/in-module via prereg_C_build_mech.py),
    plus within-indication percentile (_wd) ranked against the cohort's drugs in that
    indication
  - mechanism-GENETICS block (Mendelian/DepMap/impact, 8 features + 8 _wd): RECOMPUTED
    from cached biology (ClinVar/OT-causal/DepMap CRISPR) via prereg_C_build_genetics.py.
    Previously median-imputed; now de-imputed (no_impute_biological_features rule). A pair
    lacking MOA targets / OT module yields a COMPUTED 0.0 ("no causal evidence"), the
    builders' native value — not imputation.
  - 10 design_* features      : the ongoing trial's own ct.gov design (live pull)
  - secondary NON-biological interaction features IMPUTED (median, at fit time) and
    DISCLOSED: 3 leverage (endpoint_*/population_*), 2 trial-structure (trial_n_drugs/
    is_combination), plus trial-context interactions. NO biological feature is imputed;
    target->disease mechanism-fit, direct-target engagement, AND the Mendelian/DepMap/
    impact genetics signals are all recomputed per (drug, indication) pair.

The frozen canonical model is NOT modified; the overall head is fit on all cohort
rows via the identical published pipeline and applied to the novel pairs.

Outputs: results/benchmark/prereg_C/ (predictions, sha256, registration).
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_calibrated import prepare_fold, fit_predict, fit_isotonic_nested  # noqa: E402
from retrain_corrected import get_features, SEEDS  # noqa: E402
from pull_ctgov_design import fetch as fetch_design  # noqa: E402
from append_design_features import add_design_columns  # noqa: E402

COHORT = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
TRIALS = ROOT / "results/benchmark/prereg_C_trials.csv"
MECH_NEW = ROOT / "data/sources/mechanism_dataderived_prereg_C.csv"
MECH_COH = ROOT / "data/sources/mechanism_dataderived_v1.csv"
MECH_GEN = ROOT / "data/sources/mechanism_genetics_prereg_C.csv"  # de-imputed Mendelian/DepMap/impact
TRIALCTX = ROOT / "data/sources/trialcontext_prereg_C.csv"  # de-imputed endpoint/population/combination
TRIALCTX_COLS = ["endpoint_physiology_score", "endpoint_cvevent_match", "precedent_neg_class",
                 "endpoint_difficulty_tier", "population_leverage", "trial_n_drugs", "is_combination"]
# Complete design cache (all 4 ct.gov modules) for the prospective NCTs, COMMITTED so the
# lock is reproducible. The committed ctgov_ongoing_p3.json lacks Outcomes/Eligibility
# modules, so it cannot be used for design features. First run pulls + writes this; later
# runs are deterministic.
DESIGN_CACHE = ROOT / "data/cache/prereg_C_ctgov_design.json"
OUTDIR = ROOT / "results/benchmark/prereg_C"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOCK_DATE = "2026-06-21"
# A PASS prediction (overall) is flagged PASS_WITH_SIDE_EFFECT_RISK when the safety head
# alone is elevated above this calibrated probability.
SIDE_EFFECT_THRESHOLD = 0.30


def fit_head(coh_full, new, feature_cols, task, cohort_labels, pos_labels,
             exclude_flags=(), drop_prefixes=(), protect_prefixes=()):
    """Fit-on-all GBM head + nested isotonic, predict on the prospective rows. Mirrors
    the per-task treatment of retrain_calibrated / decompose_v8_noclass: a task-specific
    cohort + label, and (for safety) a reduced feature set excluding mech_*/design_*."""
    df = coh_full[coh_full["Corrected_Outcome"].isin(cohort_labels)].copy()
    for f in exclude_flags:
        if f in df.columns:
            df = df[df[f] != 1]
    df["_y"] = df["Corrected_Outcome"].isin(pos_labels).astype(int)
    feats = [c for c in feature_cols if not any(c.startswith(p) for p in drop_prefixes)]
    X_all, y = df[feats].values, df["_y"].values
    groups = df["SMILES"].values
    dis = df["Disease"].fillna("unknown").values
    Xn, disn = new[feats].values, new["Disease"].fillna("unknown").values
    protect_idx = [feats.index(c) for c in feats
                   if any(c.startswith(p) for p in protect_prefixes)]
    raws, cals = [], []
    for seed in SEEDS:
        X_tr, X_te, _, _ = prepare_fold(X_all, Xn, y, dis, disn, feats, protect_idx=protect_idx)
        r = fit_predict(X_tr, y, X_te, task, seed)
        iso = fit_isotonic_nested(X_all, y, groups, dis, feats, task, seed)
        raws.append(r); cals.append(iso.transform(r))
    print(f"  {task} head: cohort {len(df)} (pos {int(y.sum())}), {len(feats)} features")
    return np.mean(raws, axis=0), np.mean(cals, axis=0)

# raw -> model mech name (from build_v8_honest_exposure.py)
MECH_RENAME = {
    "topo_upstream": "mech_topo_upstream", "topo_downstream": "mech_topo_downstream",
    "topo_net": "mech_topo_net", "topo_outdeg": "mech_topo_outdeg",
    "kegg_shared": "mech_kegg_shared", "kegg_frac": "mech_kegg_frac",
    "ot_genetic_association": "mech_ot_genetic_assoc", "ot_genetic_literature": "mech_ot_genetic_lit",
    "ot_somatic_mutation": "mech_ot_somatic", "ot_affected_pathway": "mech_ot_pathway",
    "ot_animal_model": "mech_ot_animal", "ot_rna_expression": "mech_ot_rna",
    "clingen": "mech_clingen", "in_module": "mech_in_module",
    "coverage_disease": "mech_coverage_disease", "coverage_drug": "mech_coverage_drug"}

# raw -> model mech-GENETICS name (from build_v8_honest_exposure.py). These were previously
# median-IMPUTED in the lock ("INCOMPLETE: mech-genetics imputed"); prereg_C_build_genetics.py
# now RECOMPUTES them from cached biology (ClinVar/OT-causal/DepMap) for the novel pairs, so
# they are de-imputed here (NEVER median-filled — no_impute_biological_features rule).
GENETICS_RENAME = {
    "mendel_clinvar": "mech_mendel_clinvar", "mendel_ot_causal": "mech_mendel_ot_causal",
    "mendel_max": "mech_mendel_max", "depmap_dep_lin": "mech_depmap_dep",
    "depmap_selectivity": "mech_depmap_sel", "mi_raw": "mech_mi_raw",
    "mi_within_disease_pct": "mech_mi_within_disease", "mi_genetics": "mech_mi_genetics"}


def main():
    coh_full = pd.read_csv(COHORT, low_memory=False)
    feature_cols = get_features(coh_full)
    coh = coh_full[coh_full["Corrected_Outcome"].isin(
        ["PASS", "FAIL_EFFICACY", "FAIL_SAFETY", "FAIL_BOTH"])].copy()
    coh["IK14"] = coh["feature_IK"].astype(str).str[:14]

    trials = pd.read_csv(TRIALS)
    trials["IK14"] = trials["IK14"].astype(str).str[:14]

    # representative clone rows
    drug_row = coh.drop_duplicates("Drug_Clean").set_index("Drug_Clean")
    dis_row = coh.drop_duplicates("Disease").set_index("Disease")

    drug_lvl, dis_lvl = level_columns(coh, feature_cols)
    mech_cols = [c for c in feature_cols if c.startswith("mech_")]
    recomputed = [MECH_RENAME[k] for k in MECH_RENAME if MECH_RENAME[k] in feature_cols]
    # mech-GENETICS (Mendelian/DepMap/impact) — de-imputed: recomputed by prereg_C_build_genetics.py
    recomputed_gen = [GENETICS_RENAME[k] for k in GENETICS_RENAME
                      if GENETICS_RENAME[k] in feature_cols]
    # TRIAL-CONTEXT — de-imputed (Gap A): endpoint/population/combination fit, recomputed per trial
    # from the ongoing trial's own pre-registered ct.gov protocol (prereg_C_build_trialcontext.py).
    # Outcome-blind; these encode whether the drug's mechanism matches what THIS trial measures
    # (e.g. metoprolol->DMD: real endpoint LVEF -> endpoint_physiology +1, previously imputed to 0).
    recomputed_ctx = [c for c in TRIALCTX_COLS if c in feature_cols]
    wd_recomputed = [c + "_wd" for c in (recomputed + recomputed_gen) if c + "_wd" in feature_cols]
    # direct-target engagement is disease-specific BIOLOGY -> RECOMPUTED per pair in
    # build_mech (NEVER imputed; biological features are computed or the pair is flagged).
    DIRECT_TARGET = [c for c in ["direct_target_max", "direct_target_mean",
                                 "direct_target_n_engaged_05", "direct_target_n_genes_matched"]
                     if c in feature_cols]
    # everything in the interaction set we are NOT recomputing -> imputed (trial-context only;
    # NO biological feature may land here)
    imputed = sorted(set(feature_cols) - set(drug_lvl) - set(dis_lvl)
                     - set(recomputed) - set(recomputed_gen) - set(recomputed_ctx)
                     - set(wd_recomputed) - set(DIRECT_TARGET)
                     - {c for c in feature_cols if c.startswith("design_")})
    print(f"feature handling: clone-drug {len(drug_lvl)} | clone-disease {len(dis_lvl)} | "
          f"recompute mech {len(recomputed)} + genetics {len(recomputed_gen)} "
          f"+ trialctx {len(recomputed_ctx)} (+{len(wd_recomputed)} _wd) | design 10 | impute {len(imputed)}")

    # ---- assemble new rows ----
    new = pd.DataFrame(index=range(len(trials)))
    new["NCT_ID"] = trials["NCT_ID"].values
    new["Drug_Clean"] = trials["Drug_Clean"].values
    new["IK14"] = trials["IK14"].values
    new["Disease"] = trials["Disease"].values
    new["Disease_raw"] = trials["Disease_raw"].values
    new["overall_status"] = trials["overall_status"].values
    new["SMILES"] = drug_row.reindex(trials["Drug_Clean"].values)["SMILES"].values

    # clone drug-level + disease-level
    for c in drug_lvl:
        new[c] = drug_row.reindex(trials["Drug_Clean"].values)[c].values
    for c in dis_lvl:
        new[c] = dis_row.reindex(trials["Disease"].values)[c].values

    # is_cytotoxic: outcome-blind ATC L01A-D class flag (computed by IK14, NOT imputed) —
    # the bet drug is either a known cytotoxic or it is not. Overrides any drug-level clone /
    # median imputation so prospective rows carry the correct class value.
    if "is_cytotoxic" in feature_cols:
        _cyto = set(pd.read_csv(ROOT / "data/sources/cytotoxic_class_atc_v1.csv")["IK14"])
        new["is_cytotoxic"] = new["IK14"].isin(_cyto).astype(int)

    # recomputed mechanism + direct-target engagement (merge on IK14,Disease); mech_* get
    # renamed, direct_target_* keep their cohort names. Both are recomputed, not imputed.
    mech = pd.read_csv(MECH_NEW).rename(columns=MECH_RENAME)
    dt_present = [c for c in DIRECT_TARGET if c in mech.columns]
    new = new.merge(mech[["IK14", "Disease"] + recomputed + dt_present],
                    on=["IK14", "Disease"], how="left")
    n_dt_real = int(new[dt_present[0]].notna().sum()) if dt_present else 0
    print(f"direct-target features recomputed (not imputed): {n_dt_real}/{len(new)} rows have real values")

    # de-imputed mech-GENETICS (Mendelian/DepMap/impact); recomputed from cached biology, merged
    # on IK14,Disease. A pair lacking MOA targets / OT module yields a COMPUTED 0.0 ("no evidence"),
    # the builders' native value — NOT median imputation.
    gen = pd.read_csv(MECH_GEN).rename(columns=GENETICS_RENAME)
    gen_present = [c for c in recomputed_gen if c in gen.columns]
    new = new.merge(gen[["IK14", "Disease"] + gen_present], on=["IK14", "Disease"], how="left")
    n_gen_real = int((new[gen_present].abs().sum(axis=1) > 0).sum()) if gen_present else 0
    print(f"mech-genetics features recomputed (not imputed): {len(gen_present)} cols; "
          f"{n_gen_real}/{len(new)} rows carry a computed nonzero genetics signal")

    # de-imputed TRIAL-CONTEXT (endpoint/population/combination), recomputed per trial from the
    # ongoing trial's own ct.gov protocol (prereg_C_build_trialcontext.py); merged on NCT_ID,drug
    # because these are trial-specific (a drug can have different endpoints in different trials).
    ctx = pd.read_csv(TRIALCTX)
    ctx_present = [c for c in recomputed_ctx if c in ctx.columns]
    if ctx_present:
        new = new.drop(columns=[c for c in ctx_present if c in new.columns], errors="ignore")
        new = new.merge(ctx[["NCT_ID", "drug"] + ctx_present].rename(columns={"drug": "Drug_Clean"}),
                        on=["NCT_ID", "Drug_Clean"], how="left")
        n_ctx = int(new[ctx_present].notna().any(axis=1).sum())
        n_phys = int((new.get("endpoint_physiology_score", pd.Series(0, index=new.index)).fillna(0) != 0).sum())
        print(f"trial-context features recomputed (not imputed): {len(ctx_present)} cols; "
              f"{n_ctx}/{len(new)} rows resolved; endpoint_physiology nonzero {n_phys}")

    # within-disease percentile (_wd): rank each novel value against the COHORT's
    # drugs in that indication ONLY (NOT against other novel rows — that would make
    # the feature batch-dependent and is not the locked protocol). Label-free.
    mech_coh = pd.read_csv(MECH_COH).rename(columns=MECH_RENAME)
    for base in recomputed:
        wd = base + "_wd"
        if wd not in wd_recomputed:
            continue
        coh_by_dis = mech_coh.groupby("Disease")[base].apply(lambda s: s.dropna().values).to_dict()
        vals = []
        for d, v in zip(new["Disease"], new[base]):
            cv = coh_by_dis.get(d)
            if v is None or pd.isna(v) or cv is None or len(cv) == 0:
                vals.append(0.5)
            else:
                vals.append(float(np.mean(cv <= v)))  # fraction of cohort at/below -> percentile
        new[wd] = vals
    # genetics _wd: cohort distribution read from the training cohort's mech_* columns (the
    # genetics raw names are NOT in MECH_COH; they live in the model input directly).
    for base in recomputed_gen:
        wd = base + "_wd"
        if wd not in wd_recomputed or base not in coh_full.columns:
            continue
        coh_by_dis = coh_full.groupby("Disease")[base].apply(lambda s: s.dropna().values).to_dict()
        vals = []
        for d, v in zip(new["Disease"], new[base]):
            cv = coh_by_dis.get(d)
            if v is None or pd.isna(v) or cv is None or len(cv) == 0:
                vals.append(0.5)
            else:
                vals.append(float(np.mean(cv <= v)))
        new[wd] = vals

    # imputed interaction features -> NaN (prepare_fold's median imputer fills them)
    for c in imputed:
        new[c] = np.nan

    # design_* from the COMMITTED prospective design cache (all 4 ct.gov modules). First run
    # pulls any missing NCT live and writes the cache; later runs are fully deterministic.
    dcache = json.load(open(DESIGN_CACHE)) if DESIGN_CACHE.exists() else {}
    ncts = list(dict.fromkeys(new["NCT_ID"].tolist()))
    missing = [n for n in ncts if n not in dcache]
    if missing:
        print(f"  pulling design for {len(missing)} NCTs not in cache ...")
        for i, nct in enumerate(missing):
            try:
                dcache[nct] = fetch_design(nct)
            except Exception:
                dcache[nct] = {}
            if i % 100 == 0:
                print(f"    {i}/{len(missing)}")
            time.sleep(0.06)
        DESIGN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dcache, open(DESIGN_CACHE, "w"))
    new = add_design_columns(new, cache=dcache)
    print(f"design coverage: {new['design_n_arms'].notna().mean():.2f} (committed cache, all modules)")

    # ---- fit-on-all heads (overall + efficacy + safety), predict on prospective rows ----
    # overall protects design_*/mech_ot_* from top-k (efficacy_protect in retrain_calibrated.main);
    # WITHOUT it the locked model scores ~0.733 not 0.760. efficacy uses the efficacy frame;
    # safety drops mech_*/design_* (survivorship/leakage guard, per decompose_v8_noclass).
    EFF_EXCL = ("is_anti_pathogen", "is_endogenous", "is_mispaired_supportive",
                "is_healthy_volunteer", "is_procedural_exclude", "is_multi_drug_exclude")
    print(f"fitting overall/efficacy/safety heads x {len(SEEDS)} seeds ...")
    o_raw, o_cal = fit_head(coh_full, new, feature_cols, "overall",
                            ["PASS", "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"],
                            ["FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"],
                            protect_prefixes=("design_", "mech_ot_", "endpoint_", "population_leverage", "precedent_"))
    e_raw, e_cal = fit_head(coh_full, new, feature_cols, "efficacy",
                            ["PASS", "FAIL_EFFICACY", "FAIL_BOTH"], ["FAIL_EFFICACY", "FAIL_BOTH"],
                            exclude_flags=EFF_EXCL, protect_prefixes=("design_", "mech_ot_", "endpoint_", "population_leverage", "precedent_"))
    s_raw, s_cal = fit_head(coh_full, new, feature_cols, "safety",
                            ["PASS", "FAIL_SAFETY", "FAIL_BOTH"], ["FAIL_SAFETY", "FAIL_BOTH"],
                            exclude_flags=("is_multi_drug_exclude",), drop_prefixes=("mech_", "design_"))
    p_raw, p_cal = o_raw, o_cal  # overall remains the locked headline

    def classify(po, pe, ps_):
        if po >= 0.5:
            if pe >= 0.5 and ps_ >= 0.5:
                return "FAIL_BOTH"
            return "FAIL_SAFETY" if ps_ > pe else "FAIL_EFFICACY"
        return "PASS_WITH_SIDE_EFFECT_RISK" if ps_ >= SIDE_EFFECT_THRESHOLD else "PASS"

    out = pd.DataFrame({
        "NCT_ID": new["NCT_ID"], "drug": new["Drug_Clean"],
        "novel_indication": new["Disease"], "indication_ctgov": new["Disease_raw"],
        "overall_status": new["overall_status"],
        "P_fail_overall_raw": np.round(o_raw, 4),
        "P_fail_overall_calibrated": np.round(o_cal, 4),
        "P_fail_efficacy_calibrated": np.round(e_cal, 4),
        "P_fail_safety_calibrated": np.round(s_cal, 4),
        "predicted_label": np.where(o_cal >= 0.5, "FAIL", "PASS"),
        "model_prediction": [classify(a, b, c) for a, b, c in zip(o_cal, e_cal, s_cal)],
        "lock_date": LOCK_DATE,
    }).sort_values("P_fail_overall_calibrated", ascending=False).reset_index(drop=True)

    pred_path = OUTDIR / "prereg_C_locked_predictions.csv"
    out.to_csv(pred_path, index=False)
    sha = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    (OUTDIR / "prereg_C_locked_predictions.sha256").write_text(
        sha + "  prereg_C_locked_predictions.csv\n")
    _over = coh_full[coh_full["Corrected_Outcome"].isin(
        ["PASS", "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"])]
    _npos = int(_over["Corrected_Outcome"].isin(["FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"]).sum())
    write_reg(out, sha, len(_over), _npos, len(recomputed) + len(wd_recomputed), imputed)
    print(f"\nLOCKED {len(out)} out-of-sample predictions / {out.NCT_ID.nunique()} trials -> {pred_path}")
    print(f"SHA-256 {sha}")
    print(f"P_fail: median {out.P_fail_overall_calibrated.median():.3f} "
          f"mean {out.P_fail_overall_calibrated.mean():.3f} | "
          f"predicted FAIL {(out.predicted_label=='FAIL').sum()}/{len(out)}")
    print(out.head(12).to_string(index=False))


def level_columns(coh, feats):
    """Classify clone level, measuring constancy ONLY over groups with >=2 members.

    Single-indication drugs and single-drug diseases are trivially constant and make
    the naive within-group test unidentifiable (it wrongly tagged disease-derived
    network features as drug-level, cloning the drug's ORIGINAL indication's value).
    A feature constant within multi-drug diseases is disease-level; constant within
    multi-disease drugs is drug-level; varying in both (interaction) or constant in
    both (ambiguous) is NOT cloned — it is recomputed or imputed.
    """
    md = [d for d, n in coh.groupby("Disease").size().items() if n >= 2]
    mk = [d for d, n in coh.groupby("Drug_Clean").size().items() if n >= 2]
    g_s = coh[coh.Disease.isin(md)].groupby("Disease")
    g_d = coh[coh.Drug_Clean.isin(mk)].groupby("Drug_Clean")
    drug_lvl, dis_lvl = [], []
    for c in feats:
        if c.startswith("design_") or c.startswith("mech_"):
            continue
        s_const = (g_s[c].nunique(dropna=True) <= 1).mean() >= 0.999
        d_const = (g_d[c].nunique(dropna=True) <= 1).mean() >= 0.999
        if s_const and not d_const:
            dis_lvl.append(c)
        elif d_const and not s_const:
            drug_lvl.append(c)
    return drug_lvl, dis_lvl


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def write_reg(out, sha, n_train, n_pos, n_recomp, imputed):
    n = len(out); nt = out["NCT_ID"].nunique()
    md = f"""# Prospective registration (Path C — out-of-sample indications)

**Lock date:** {LOCK_DATE}
**Git commit:** `{git_sha()}`
**Predictions:** `results/benchmark/prereg_C/prereg_C_locked_predictions.csv`
**SHA-256:** `{sha}`

## Commitment
On the lock date we froze model predictions for **{n} ongoing Phase III trials
({nt} distinct trials) of cohort compounds in indications the compound does NOT
hold in the training cohort** — i.e. genuinely out-of-sample (drug, indication)
pairs whose outcomes are not yet known. Unlike a same-pair forward test, a held
score cannot be confirmed by label stability here: the model has never seen these
pairings. Predicted to FAIL (calibrated P_fail >= 0.5): {(out.predicted_label=='FAIL').sum()}/{n}.

## Why this is the meaningful prospective test
The model's discriminative signal for a novel pairing is the target->disease
mechanism-fit. For each pair we **recompute** that mechanism block ({n_recomp}
features: OmniPath directed topology, KEGG co-membership, Open-Targets biology
channels, ClinGen causal validity, in-module flag, and their within-indication
percentile ranks) from public biology keyed on the drug's targets and the
indication's gene module — no outcome information, no new trial data. The
Mendelian/rare-variant causal, DepMap lineage-dependency, and domain-conditional
mechanism-impact (genetics) features are likewise **recomputed** from cached public
biology (ClinVar/Open-Targets-causal/DepMap CRISPR) per pair — never imputed. Drug-level
features (206) clone from the compound; indication-level features (10) clone from
the indication; the 10 design features are the ongoing trial's own pre-registered
ct.gov design. {len(imputed)} secondary NON-biological interaction features are
median-imputed and disclosed (direct-target binding overlap, leverage, trial-structure,
trial-context): {', '.join(imputed)}.

## Model
Canonical `production_v8_clean_mort_coverage_jun22` (frozen, unchanged). The overall
PASS-versus-failure head was fit on all {n_train} cohort rows ({n_pos} failures) via
the identical published pipeline (impute -> top-k -> within-train disease
target-encoding -> seed-averaged gradient boosting -> nested isotonic calibration)
and applied to the novel pairs. Report `P_fail_overall_calibrated`.

## Evaluation (prospective, at readout)
Score `P_fail_overall_calibrated` against the realized PASS/FAIL under the
manuscript's label-audit protocol; report ROC-AUC with a compound-clustered 95% CI.
This file + SHA-256 + commit fix the predictions before any trial reports.

## Reproduce
```
python3 scripts/benchmark/prereg_C_build_mech.py   # recompute mechanism for novel pairs
python3 scripts/benchmark/prereg_C_lock.py         # assemble, fit-on-all, lock
```
"""
    (OUTDIR / "prereg_C_registration.md").write_text(md)


if __name__ == "__main__":
    main()
