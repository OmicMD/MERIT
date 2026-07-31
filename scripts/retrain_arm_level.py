#!/usr/bin/env python3
"""Arm-level cohort builder + superseded baseline run.

KEEP: this module is the shared arm-level LIBRARY. retrain_arm_level_v18_production.py,
retrain_arm_level_v12_stratified.py, compute_opentargets_assoc.py and
genetic_novelty_decompression.py all import DATA / prepare_arm_dataset / build_task_subsets
from here. Do not delete it.

DO NOT RUN AS A SCRIPT (superseded). Its main() writes results/arm_level_v8/ and gives
overall 0.757 / safety 0.682 / efficacy 0.790 -- numbers that CONTRADICT the published
Supplementary Table S14 (0.710 / 0.688 / 0.720). The published arm-level run is produced by
retrain_arm_level_v18_production.py -> results/arm_level_v18/ (shared safety head +
dili_dose_x_logp). Import from this file; run that one.

Inputs:  data/sources/training_dataset_arm_level.csv (8,653 arms × 260 cols)
Outputs (superseded): results/arm_level_v8/
    metrics.json, fold_metrics.csv
    oof_safety.parquet, oof_efficacy.parquet, oof_overall.parquet

Differences vs production_v2 (trial-level):
- Filters to test arms (n_investigational ≥ 1) and drops methodology / dosing-
  schedule arms (is_methodology_study = 0).
- Group column = first investigational SMILES (StratifiedGroupKFold prevents
  drug leakage across folds, same as before).
- No --crosstask / no calibration for baseline run.

v4 vs v3: ACTIVE_COMPARATOR FAIL→PASS relabeling (29 arms corrected). When a
trial is stopped for futility the active comparator drug did not fail — the
experimental drug failed to beat it. Labeling comparators as FAIL_EFFICACY
trains the wrong signal. All 29 ACTIVE_COMPARATOR FAIL arms relabeled to PASS
in build_arm_level_dataset.py.

v5: 0.760 overall / 0.750 safety / 0.756 efficacy (3,589 arms) — EXPERIMENTAL-only, AC relabeled
v6: 0.662 safety (dedup hurt small safety class) — discarded
v7: fix empty-disease methodology overfilter (+184 arms); exclude imputed-feature arms
v8 (May 28 2026): CT.gov-mislabel recovery in build_arm_level_dataset.py
  (_recover_mislabeled_test_arms) retypes 350 fallback-active (drug-vs-placebo)
  + 5 single-arm-FAIL arms ACTIVE_COMPARATOR/OTHER → EXPERIMENTAL, so they flow
  into this mask and escape the active-comparator FAIL→PASS relabel. Baseline
  2,964 → 3,233 arms (+240 PASS, +25 FE, +4 FS). Sensitivity (WITH/WITHOUT,
  toggle on is_recovered_test_arm) is performance-neutral: overall 0.767→0.763,
  safety 0.757→0.767, efficacy 0.767→0.760, all within fold SD; survivorship
  check clean (recovered-indicator→FAIL AUC 0.480). Justification is label
  correctness. See scripts/retrain_recovery_sensitivity.py.
trial-level prod_v2: 0.837 overall / 0.772 safety / 0.828 efficacy
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Reuse the production pipeline's training helpers (model fitting, nested
# feature selection, disease encoding, fold reporting).
from retrain_calibrated import (run_task_cv, mechanism_groups,  # noqa: E402
                                noisy_or_safety_oof)
from retrain_corrected import get_features  # noqa: E402

DATA = ROOT / "data" / "sources" / "training_dataset_arm_level.csv"
OUT = ROOT / "results" / "arm_level_v8"   # superseded; created in main(), not on import, so
                                          # importing this library does not spawn an empty dir


EXCLUSION_FLAGS = [
    "inv_biologic_only", "inv_large_peptide", "inv_approved_biologic_coinv",
    "inv_investigational_biologic_coinv", "is_business_stop",
    "is_narrow_population", "is_wrong_drug_assignment",
    "is_treatment_duration_study",
]

# --- Training-population toggles (May 29 2026) ---------------------------------
# These admit two arm classes the user asked to represent faithfully. Both are
# kept behind toggles so the with/without ablation can be run and flipped off if
# they degrade experimental-only AUC or show a survivorship proxy signal.
#   INCLUDE_COMPARATOR_ARMS: distinct active-comparator treatment arms (e.g.
#     mitoxantrone/prednisone vs cabozantinib). 99% PASS (established SOC) → risks
#     a "recognized SOC = PASS" survivorship signal; evaluate before trusting.
#   INCLUDE_BIO_COMBO_ARMS: small-molecule + biologic combination arms (e.g.
#     rucaparib+nivolumab), trained via the small-molecule anchor; outcome is
#     partly driven by the (unfeaturized) biologic.
INCLUDE_COMPARATOR_ARMS = True
INCLUDE_BIO_COMBO_ARMS = True


def training_mask(df: pd.DataFrame) -> pd.Series:
    """THE single definition of a trainable arm. Imported by the review-xlsx
    builder so its 'Training Ready' tab equals exactly what the model trains on
    — do not duplicate this logic elsewhere. See prepare_arm_dataset() for the
    rationale behind each exclusion.

    Dosing/phase-duplicate arms (is_dosing_arm_duplicate) are EXCLUDED: they share
    the same drug, features and outcome as their primary arm, so they only inflate
    the dataset with identical rows. The primary arm per (NCT_ID, anchor) is kept.
    """
    pipeline_feat_cols = [c for c in df.columns
                          if c.startswith(("net_", "binding_", "tissue_", "tox_"))]
    has_pipeline_features = df[pipeline_feat_cols].notna().any(axis=1)

    # Allowed arm types: EXPERIMENTAL always; ACTIVE_COMPARATOR only when it is a
    # distinct active-treatment arm (is_active_comparator_treatment) and the
    # comparator toggle is on.
    allowed_type = (df["Arm_Type"] == "EXPERIMENTAL")
    if INCLUDE_COMPARATOR_ARMS and "is_active_comparator_treatment" in df.columns:
        allowed_type = allowed_type | (
            (df["Arm_Type"] == "ACTIVE_COMPARATOR")
            & df["is_active_comparator_treatment"].fillna(False)
        )

    mask = (df["n_investigational"] >= 1) & \
           (df.get("is_methodology_study", 0) == 0) & \
           df["Corrected_Outcome"].isin(["PASS", "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"]) & \
           df["All_SMILES"].notna() & \
           has_pipeline_features & \
           allowed_type & \
           ~df.get("is_dosing_arm_duplicate", pd.Series(False, index=df.index)).fillna(False)

    exclusions = list(EXCLUSION_FLAGS)
    if INCLUDE_BIO_COMBO_ARMS and "inv_investigational_biologic_coinv" in exclusions:
        # Keep small-molecule+biologic combination arms (trained via the
        # small-molecule anchor) instead of dropping them.
        exclusions.remove("inv_investigational_biologic_coinv")
    for f in exclusions:
        if f in df.columns:
            mask &= ~df[f].fillna(False)
    return mask


def prepare_arm_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)
    print(f"Loaded arm-level dataset: {df.shape}", flush=True)

    # Filter to trainable test arms. Exclusions:
    #   inv_biologic_only          — all investigational drugs are biologics (no IK14)
    #   inv_large_peptide          — anchor MW >900 Da (Binding not designed for these)
    #   inv_approved_biologic_coinv — approved biologic co-investigational with small
    #                                 molecule; outcome driven by the biologic, not the
    #                                 small-molecule anchor whose features we use
    #   is_business_stop           — trial stopped for sponsor/portfolio decision, not
    #                                 scientific efficacy failure; label is unreliable
    #   is_narrow_population       — twin pregnancy, compassionate use, expanded access;
    #                                 too niche to generalise from
    #   no pipeline features       — drug has SMILES but no Binding/network/tissue data
    #                                 yet; median imputation creates a generic-average
    #                                 drug rather than real molecular signal. These are
    #                                 in drugs_needing_full_pipeline_run.csv and will be
    #                                 added back as pipeline runs complete.
    # Dosing-schedule trials are flagged but kept — they test real drug outcomes.
    # Pediatric trials are flagged but kept — legitimate scientific outcomes.
    # Dosing-arm duplicates (is_dosing_arm_duplicate) ARE now excluded by
    #   training_mask (May 29 2026) — same investigational drug SET at a different
    #   dose/schedule/phase, identical features+outcome to the primary arm, so they
    #   only inflate the dataset with duplicate rows (~733).
    #
    # Active-comparator treatment arms and small-molecule+biologic combination arms
    # are admitted via the INCLUDE_COMPARATOR_ARMS / INCLUDE_BIO_COMBO_ARMS toggles
    # (see training_mask). Ablation (scripts/ablation_comparator_arms.py, May 29):
    # including comparators IMPROVED EXPERIMENTAL-only AUC 0.729→0.742 (the honest
    # same-population comparison) while the feature→is_comparator proxy was only
    # 0.585 (marginal). NOTE: report EXPERIMENTAL-only AUC as the honest headline —
    # the all-arms AUC (~0.752) is inflated ~0.01 by easy comparator-PASS arms.
    pipeline_feat_cols = [c for c in df.columns
                          if c.startswith(("net_", "binding_", "tissue_", "tox_"))]
    has_pipeline_features = df[pipeline_feat_cols].notna().any(axis=1)

    mask = training_mask(df)  # single shared definition (also used by review xlsx)

    # Report how many arms are held out pending pipeline runs
    base_mask = (df["n_investigational"] >= 1) & \
                (df.get("is_methodology_study", 0) == 0) & \
                df["Corrected_Outcome"].isin(["PASS", "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"]) & \
                df["All_SMILES"].notna() & \
                (df["Arm_Type"] == "EXPERIMENTAL")
    for f in ["inv_biologic_only", "inv_large_peptide", "inv_approved_biologic_coinv",
              "inv_investigational_biologic_coinv", "is_business_stop",
              "is_narrow_population", "is_wrong_drug_assignment",
              "is_treatment_duration_study"]:
        if f in df.columns:
            base_mask &= ~df[f].fillna(False)
    pending = df[base_mask & ~has_pipeline_features]
    pending_vc = pending["Corrected_Outcome"].value_counts()
    print(f"Arms held out (pending pipeline run): {len(pending)} "
          f"[FAIL_SAFETY={pending_vc.get('FAIL_SAFETY',0)+pending_vc.get('FAIL_BOTH',0)}, "
          f"FAIL_EFF={pending_vc.get('FAIL_EFFICACY',0)}, "
          f"PASS={pending_vc.get('PASS',0)}]", flush=True)

    df = df[mask].copy()

    # Group column = first INVESTIGATIONAL drug's SMILES (not first All_SMILES,
    # because All_SMILES is ordered by intervention list and can put a backbone
    # drug first — Cisplatin in 'Pembrolizumab + Cisplatin + 5-FU + Carbo' for
    # example. Grouping by backbone would let backbone drugs leak across folds.)
    #
    # Strategy: take the first name in Investigational_Drugs, look it up in the
    # chembl_smiles_lookup, use that SMILES as the group key. Fall back to first
    # All_SMILES if the inv drug isn't resolvable (mostly biologics).
    chembl = pd.read_csv(ROOT / "data" / "sources" / "chembl_smiles_lookup.csv")
    name_to_smi = {}
    for _, r in chembl.iterrows():
        for k in ("Drug_Clean", "chembl_pref_name"):
            n = (r.get(k) or "").strip().lower() if isinstance(r.get(k), str) else ""
            if n and pd.notna(r.get("chembl_smiles")) and n not in name_to_smi:
                name_to_smi[n] = r["chembl_smiles"]

    def first_inv_smiles(row):
        inv = row.get("Investigational_Drugs")
        all_smi = row.get("All_SMILES")
        if isinstance(inv, str):
            first_inv = inv.split(";")[0].strip()
            if first_inv:
                # Look up by exact name then by first token (matches P10 canon)
                key = first_inv.lower()
                if key in name_to_smi: return name_to_smi[key]
                tok = key.split()[0]
                if tok in name_to_smi: return name_to_smi[tok]
        # Fallback: first SMILES in All_SMILES
        if isinstance(all_smi, str):
            parts = [p.strip() for p in all_smi.split(";") if p.strip()]
            if parts: return parts[0]
        return None

    df["SMILES"] = df.apply(first_inv_smiles, axis=1)
    df = df[df["SMILES"].notna() & (df["SMILES"].astype(str).str.len() > 0)].copy()

    print(f"After filtering: {len(df)} arms, "
          f"{df['SMILES'].nunique()} unique drug groups, "
          f"{df['NCT_ID'].nunique()} unique NCTs", flush=True)
    print(f"Outcome distribution:\n{df['Corrected_Outcome'].value_counts().to_string()}",
          flush=True)
    return df


def build_task_subsets(df: pd.DataFrame):
    """Mirror retrain_calibrated.main()'s per-task cohort construction."""
    safety_mask = df["Corrected_Outcome"].isin(["PASS", "FAIL_SAFETY", "FAIL_BOTH"])
    df_safety = df[safety_mask].copy()
    df_safety["_y"] = df_safety["Corrected_Outcome"].isin(
        ["FAIL_SAFETY", "FAIL_BOTH"]).astype(int)

    # Efficacy exclusion flags carried over from the trial-level pipeline.
    eff_excl = pd.Series(False, index=df.index)
    for c in ("is_anti_pathogen", "is_endogenous"):
        if c in df.columns:
            eff_excl |= df[c] == 1
    df_eff = df[~eff_excl & df["Corrected_Outcome"].isin(
        ["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    df_eff["_y"] = df_eff["Corrected_Outcome"].isin(
        ["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)

    df_over = df[df["Corrected_Outcome"].isin(
        ["PASS", "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    df_over["_y"] = df_over["Corrected_Outcome"].isin(
        ["FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)

    return df_safety, df_eff, df_over


def run_safety_noisy_or(df_safety, feature_cols):
    """Safety head: per-mechanism detectors combined by noisy-OR.

    Thin wrapper over retrain_calibrated.noisy_or_safety_oof (the canonical
    implementation, shared with the trial-level head) so arm and trial use
    identical logic. Returns (oof, fold_metrics, detail). No exposure features
    exist at arm level, so protect_cols is omitted.
    """
    return noisy_or_safety_oof(df_safety, feature_cols, return_detail=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = prepare_arm_dataset()
    # Block arm-specific columns that aren't molecular/disease features. The
    # is_methodology_study / is_dosing_schedule_trial flags are constant in
    # the filtered training set (all 0), so they add no value. n_drugs and
    # n_investigational duplicate trial_n_drugs / is_combination semantically.
    # Noise / leakage features excluded from arm-level training.
    # net_mech_* features score AUC 0.503-0.524 (effectively 0.50) and carry
    # zero RF importance — they are noise at this dataset size.
    # n_high_protein_in_* features also carry zero RF importance.
    # net_n/frac_disease_in_ppi: 92.6% zero across training arms — dead feature.
    # max_*_interaction tissue features: 87-90% of arms score exactly 1.0 (ceiling
    #   saturation from a single high-binding transcript) — no discriminative power.
    #   min_* and mean_* variants have real variance; min_sensory_interaction is the
    #   best single tissue feature (safety AUC 0.639, efficacy 0.589).
    ARM_LEVEL_BLOCKLIST = {
        "n_drugs", "n_investigational",
        "is_methodology_study", "is_dosing_schedule_trial",
        "net_mech_apoptosis_n", "net_mech_apoptosis_frac",
        "net_mech_cell_cycle_n", "net_mech_cell_cycle_frac",
        "net_mech_dna_damage_n", "net_mech_dna_damage_frac",
        "net_mech_epigenetic_n", "net_mech_epigenetic_frac",
        "net_mech_immune_n", "net_mech_immune_frac",
        "net_n_disease_in_ppi", "net_frac_disease_in_ppi",
    }
    ARM_LEVEL_BLOCKLIST_PREFIXES = ("n_high_protein_in_",)
    feature_cols = [c for c in get_features(df)
                    if c not in ARM_LEVEL_BLOCKLIST
                    and not any(c.startswith(p) for p in ARM_LEVEL_BLOCKLIST_PREFIXES)
                    and not (c.startswith("max_") and c.endswith("_interaction"))]
    print(f"Feature columns retained: {len(feature_cols)}", flush=True)

    df_safety, df_eff, df_over = build_task_subsets(df)
    print(f"\nsafety  n={len(df_safety)} pos={df_safety._y.sum()}", flush=True)
    print(f"efficacy n={len(df_eff)} pos={df_eff._y.sum()}", flush=True)
    print(f"overall n={len(df_over)} pos={df_over._y.sum()}", flush=True)

    all_fold_metrics = []

    print("\n=== SAFETY (noisy-OR mechanism detectors) ===", flush=True)
    # The safety head is a parameter-free noisy-OR over per-mechanism GBM
    # detectors (promiscuity / hepatic-DILI / cardiac / network / tissue /
    # context), NOT a single GBM. A single class-balanced GBM lets the dominant
    # promiscuity signal wash out the minority-mechanism detectors; noisy-OR
    # ("fail if ANY mechanism trips") matches the biology and lifts arm-level
    # safety AUC +0.033 (5/5 seeds, p=0.012). calibrate=False throughout: nested
    # isotonic hurts safety (too few positives for inner 3-fold). Class balancing
    # is applied via compute_sample_weight('balanced') inside each detector's GBM.
    oof_s, fm_s, detail_s = run_safety_noisy_or(df_safety, feature_cols)
    oof_s.to_parquet(OUT / "oof_safety.parquet", index=False)
    detail_s.to_parquet(OUT / "safety_mechanism_detail.parquet", index=False)
    all_fold_metrics.append(fm_s)

    # Single-GBM safety baseline kept as an audit sidecar (NOT the headline).
    print("\n--- safety single-GBM baseline (audit sidecar) ---", flush=True)
    oof_s_single, _ = run_task_cv(df_safety, feature_cols, "safety", calibrate=False)
    oof_s_single.to_parquet(OUT / "oof_safety_single_gbm.parquet", index=False)
    single_fold = [roc_auc_score(g["y"], g["raw_prob"])
                   for _, g in oof_s_single.groupby(["seed", "fold"])
                   if g["y"].nunique() == 2]
    nor_fold = [roc_auc_score(g["y"], g["raw_prob"])
                for _, g in oof_s.groupby(["seed", "fold"])
                if g["y"].nunique() == 2]
    print(f"  safety single-GBM AUC {np.mean(single_fold):.4f} | "
          f"noisy-OR AUC {np.mean(nor_fold):.4f} | "
          f"Δ{np.mean(nor_fold) - np.mean(single_fold):+.4f}", flush=True)

    print("\n=== EFFICACY ===", flush=True)
    oof_e, fm_e = run_task_cv(df_eff, feature_cols, "efficacy", calibrate=False)
    oof_e.to_parquet(OUT / "oof_efficacy.parquet", index=False)
    all_fold_metrics.append(fm_e)

    print("\n=== OVERALL ===", flush=True)
    oof_o, fm_o = run_task_cv(df_over, feature_cols, "overall", calibrate=False)
    oof_o.to_parquet(OUT / "oof_overall.parquet", index=False)
    all_fold_metrics.append(fm_o)

    fold_df = pd.concat(all_fold_metrics, ignore_index=True)
    fold_df.to_csv(OUT / "fold_metrics.csv", index=False)

    # Headline metrics
    metrics = {}
    for task, oof in (("safety", oof_s), ("efficacy", oof_e), ("overall", oof_o)):
        per_fold = []
        for (seed, fold), grp in oof.groupby(["seed", "fold"]):
            if grp["y"].nunique() < 2: continue
            per_fold.append(roc_auc_score(grp["y"], grp["raw_prob"]))
        metrics[f"{task}_auc_mean"] = float(np.mean(per_fold))
        metrics[f"{task}_auc_std"] = float(np.std(per_fold))
        metrics[f"{task}_n_folds"] = len(per_fold)
        metrics[f"{task}_n_arms"] = int(len(oof))
        metrics[f"{task}_n_pos"] = int(oof["y"].sum())

    with open(DATA, "rb") as f:
        metrics["data_sha256"] = hashlib.sha256(f.read()).hexdigest()
    metrics["data_path"] = str(DATA)
    metrics["n_features"] = len(feature_cols)

    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {OUT/'metrics.json'}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
