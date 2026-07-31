#!/usr/bin/env python3
"""Build the final honest+exposure trial-level dataset: the attribution-fixed honest dataset
(training_dataset_v8_honest.csv) plus the validated exposure axis as STANDING features:
  - max_daily_dose_mg + logdose (max recommended total daily systemic dose, FDA labels; 99% cov)
  - off-target × dose interactions (realized off-target toxicity = binding potential x exposure).

get_features() is a denylist over numeric cols, so these become standing features automatically
and per-fold median-imputation handles them. Validated (Jun 7): non-onc safety base 0.52 ->
+dose+offtarget*dose 0.62 (+0.101, 5/5 seeds), leak-clean (dose availability AUC 0.51).

Out: data/sources/training_dataset_v8_honest_exposure.csv (+ .provenance.json)
"""
from __future__ import annotations
import json, hashlib, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/sources/training_dataset_v8_honest.csv"
DOSE = ROOT / "data/sources/max_daily_dose_v1.csv"
OUT = ROOT / "data/sources/training_dataset_v8_honest_exposure.csv"

# off-target / promiscuity / anti-target binding features to interact with dose
OT = ["binding_drug_n_bound", "binding_n_above_80", "tox_antitarget_burden",
      "tox_cardiac_burden", "tox_hepatic_burden", "tox_renal_burden",
      "binding_score_max", "tox_vs_overall_ratio"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    df = pd.read_csv(SRC, low_memory=False)

    # Jun 12 2026 — DISEASE BACKFILL (applied HERE, after attribution, so the upstream attribution
    # determination — which is keyed on the original NaN Disease — still matches; doing it in
    # build_v8_dataset broke that merge and silently dropped 201 rows). 208 trials kept Disease=NaN
    # because notebook 01 (where the backfill was first wired) is not re-run from a committed
    # v5_unified — this caused the persistent elagolix/endometriosis FP cluster (NaN disease -> no
    # mechanism-score join -> imputed median, and zeroed disease_is_*). Backfill from ct.gov + re-derive
    # disease_is_* so the mechanism block and disease context populate. notes/error_explainability_map_jun12.md.
    _rep = ROOT / "data/sources/disease_repopulation_jun12.csv"
    if _rep.exists():
        _rmap = dict(pd.read_csv(_rep)[["NCT_ID", "Disease"]].values)
        _m = df["Disease"].isna() & df["NCT_ID"].isin(_rmap)
        df.loc[_m, "Disease"] = df.loc[_m, "NCT_ID"].map(_rmap)
        _dt = df["Disease"].fillna("").str.lower()
        _pats = {
            "disease_is_oncology": "cancer|carcinoma|lymphoma|leukemia|melanoma|sarcoma|myeloma|glioma|glioblastoma|neoplasm|tumor|nsclc|sclc|neuroblastoma|mesothelioma|oncology|metastat|malignant",
            "disease_is_infectious": "hiv|hepatitis|hcv|hbv|influenza|covid|sars|tuberculosis|malaria|bacterial|fungal|infection|pneumonia|sepsis|cmv|herpes|rsv|dengue",
            # Jun 13: added "depressive" (Major Depressive Disorder matched "depressive" not the old
            # "depression" token — 50 MDD trials were silently mis-zoned to non-CNS) + mood/psychiatric terms.
            "disease_is_cns": "alzheimer|parkinson|epilepsy|seizure|schizophreni|depress|bipolar|anxiety|dementia|multiple sclerosis|neuropath|migraine|stroke|brain|psychiatr|mood disorder|obsessive|ptsd|adhd|autism|huntington",
            "disease_is_cardiac": "heart failure|atrial fibrillation|hypertension|coronary|myocardial|arrhythmia|angina|cardiac|cardiovascular|atherosclerosis",
            # Jun 13: added atopic dermatitis / eczema / other immune-mediated dermatoses (16 atopic
            # dermatitis trials were mis-zoned to non-autoimmune; baricitinib/JAK is an immune-zone drug there).
            "disease_is_autoimmune": "rheumatoid|lupus|psoriasis|crohn|colitis|autoimmune|immunolog|ankylosing|scleroderma|vasculitis|pemphigus|atopic|eczema|dermatitis|hidradenitis|alopecia areata|vitiligo",
            "disease_is_metastatic": "metastat|advanced|stage iv|stage 4|unresectable|refractory",
            "disease_is_transplant": "transplant|graft|rejection",
            "disease_is_severe": "severe|critical|intensive care|icu|acute respiratory distress|ards|life.threatening|end.stage|terminal",
        }
        for _c, _p in _pats.items():
            if _c in df.columns:
                df[_c] = _dt.str.contains(_p, regex=True).astype(int)
        print(f"disease backfill: {int(_m.sum())} NaN-disease rows filled from ct.gov + disease_is_* re-derived")

    dose = pd.read_csv(DOSE)[["drug", "max_daily_dose_mg"]].rename(columns={"drug": "Drug_Clean"})
    df = df.merge(dose, on="Drug_Clean", how="left")
    cov = df["max_daily_dose_mg"].notna().mean()
    df["logdose"] = np.log10(df["max_daily_dose_mg"] + 1)

    # leak guard: dose-availability must not predict outcome
    for task, pos in [("safety", ["FAIL_SAFETY", "FAIL_BOTH"]), ("efficacy", ["FAIL_EFFICACY", "FAIL_BOTH"])]:
        s = df[df["Corrected_Outcome"].isin(["PASS"] + pos)].copy()
        y = s["Corrected_Outcome"].isin(pos).astype(int)
        a = roc_auc_score(y, s["max_daily_dose_mg"].notna().astype(int))
        print(f"  {task} dose-availability proxy AUC {max(a, 1 - a):.3f} (<0.58 required)")
        assert max(a, 1 - a) < 0.58, "dose availability is a leak"

    made = []
    for c in OT:
        if c in df.columns:
            df[c + "_xdose"] = df[c].fillna(df[c].median()) * df["logdose"]
            made.append(c + "_xdose")
    print(f"honest n={len(df)}  dose coverage {cov:.0%}  added: logdose + {len(made)} off-target*dose feats")

    # Trial-design context primitives (leak-safe; notes/efficacy_signal_is_trial_design_jun11.md):
    # the efficacy ceiling is substantially trial-CONTEXT, not mechanism (17% of efficacy fails are
    # same drug+disease as a pass). De-confounded set adds +0.016 efficacy / +0.022 overall.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from append_design_features import add_design_columns
        df = add_design_columns(df, with_placebo=False)
        print(f"appended design context primitives (coverage {df['design_n_arms'].notna().mean():.0%})")
    except FileNotFoundError:
        print("WARN: data/cache/ctgov_design_v2.json absent — run scripts/pull_ctgov_design.py; skipping design feats")

    # iDILI Axis A — dose x lipophilicity (Chen rule-of-two), leak-safe + measured.
    # Validated jun13: the endothelin-antagonist natural experiment (ambrisentan/macitentan 10mg safe
    # vs bosentan/sitaxentan high-dose hepatotoxic) + cohort 7.2% vs 2.1% fail; +0.017 safety mof.
    # It is an "is-a-hepatotoxin" prior, NOT the fatal-vs-managed discriminator (that = HLA/idiosyncratic,
    # axes B/C tested+rejected jun13/14). Routed to the noisy-OR hepatic detector + dose-protected in
    # retrain_calibrated.py via the dili_ prefix. notes/idiosyncratic_dili_mechanism_stash_jun13.md.
    from build_dili_dose_lipophilicity import add_dili_columns
    df = add_dili_columns(df)
    print(f"appended iDILI dose x lipophilicity feats (logP cov {df['dili_logp'].notna().mean():.0%})")

    # Jun 16 2026 — THREE-BODY anti-pathogen axes (pathogen x host x drug-at-site). These describe the
    # biology of the INFECTION, not the molecule — the axis the human-target binding pipeline is blind to,
    # and the reason anti-pathogen trials were excluded from efficacy. Scored a-priori from pathogen/host
    # biology + measured regimen depth (blind to outcome); validated a-priori composite raw AUC 0.809
    # (shuffle 0.497). tb_* are PREFIX-GUARDED out of the canonical model (get_features skips 'tb_') and
    # consumed only by the anti-pathogen efficacy head (scripts/strengthening/antipathogen_efficacy_head.py),
    # so the canonical small-molecule model is unchanged. notes/three_body_characterizations_antipathogen.md.
    from append_three_body_features import add_three_body_columns
    df = add_three_body_columns(df)

    # Jun 14 2026 — TARGET -> AE-CLASS pharmacology priors: TESTED & NOT BANKED (characterized
    # honest negative; notes/target_ae_class_result_jun14.md). Drug MoA target(s) -> union of the
    # targets' known on-target organ-tox classes (aeclass_*), scored BLIND to drug/outcome from pure
    # target physiology (leak-safe; availability-proxy AUC 0.513). The flags correctly hit the right
    # organ per-case (cinacalcet->metabolic/endocrine, dexmedetomidine->cardiac/vascular,
    # perampanel->cns) BUT flip 0/32 confident-miss safety FN (max move +0.02, well short of a 0.3
    # threshold) and add 9 new PASS false alarms; pooled safety mof -0.0028 (noise). The on-target
    # organ-tox signal is already saturated by disease + ATC-class encoding (the predicted ATC
    # redundancy), and most flagged drugs PASS (managed risk). Not wired. Build the standalone
    # artifacts with scripts/build_target_aeclass_features.py; re-test by uncommenting below +
    # the aeclass_ guard in retrain_calibrated.py. Same shape as line-of-therapy / tdc_dili negatives.
    # from build_target_aeclass_features import add_aeclass_columns
    # df = add_aeclass_columns(df)

    # Jun 14 2026 — REACTIVE-METABOLITE / HAPTENATION axis (iDILI drug-side): TESTED & NOT BANKED
    # (notes/forward_paths_both_negative_jun14.md). Leak-safe structural-alert bioactivation score
    # (RDKit SMARTS, Kalgutkar). The alerts fire CORRECTLY per-case on the iDILI FN (sitaxentan->
    # thiophene+methylenedioxy, duloxetine->thiophene, fasiglifam->acyl-glucuronide) but are
    # NON-DISCRIMINATIVE (within confident-PASS region AUC 0.516; reactive groups are common — thiophene
    # in 130 drugs, carboxylic_acid in 412, mostly safe) -> 0/32 FN flipped, safety mof -0.0026. The
    # structure flags the liability; only the patient's risk-HLA converts it to injury -> irreducible
    # patient-level, confirms the PROSPECTIVE-enrichment-tool framing. Build standalone with
    # scripts/build_reactive_metabolite_features.py; re-test by uncommenting + the rmet_ guards in
    # retrain_calibrated.py.
    # from build_reactive_metabolite_features import add_reactmetab_columns
    # df = add_reactmetab_columns(df)

    # Jun 12 2026 — MECHANISM-FITNESS block replaces the inverted blind-plausibility feature.
    # The old causal_plausibility_blind asked "is the target RELATED to the disease" (soft) and was
    # demonstrably INVERTED on the diagnosed cases (scored genuine fails higher than drugs that worked).
    # The replacement asks the SHARP question — "is the target the CAUSAL, RATE-LIMITING driver of THIS
    # disease in THIS population" — scored blind-to-outcome by an anchored LLM judge for every drug-disease
    # pair (committed determination: data/sources/causal_centrality_jun12.csv). Validated: right-mechanism
    # drugs fail 7% vs wrong-mechanism 27%, holding WITHIN every disease area; +0.21 AUC over disease flags;
    # leak guards hold (within-disease consistency; scores failed-but-sound drugs high). 4 components carry
    # distinct signal (population_fit is strongest, AUC 0.78). notes/causal_centrality_two_bar_jun12.md.
    # Jun 16 2026 — ARCHIVED the LLM mechanism judge. The causal_centrality_jun12.csv scores were produced
    # by an anchored LLM judge that sees ONLY the drug + disease names, so "reason from biology, not recalled
    # outcomes" is unenforceable — it can memorise known trial results. Decomposition (Jun-14/15) showed only
    # ~63% of its edge is reconstructable from clean structured biology within-disease; the remainder is
    # consistent with memorisation, and a target-shuffle does not rule that out (a memorised feature genuinely
    # carries outcome info). REPLACED with the biology-derived mechanism block (Open Targets target->disease
    # evidence channels, network topology, KEGG pathway overlap, ClinGen, DepMap dependency, Mendelian/ClinVar
    # causal evidence) — all computed from databases keyed by target gene + disease, NO LLM. Prefixed mech_*
    # so they (a) replace the LLM features and (b) inherit the existing "mechanism stays out of the safety
    # head" exclusion in retrain_calibrated.py. Archived: data/archive/causal_centrality_jun12.csv +
    # score_mechanism_fitness.py. See notes/mech_llm_archival_jun16.md.
    df["_ik14"] = df["feature_IK"].astype(str).str[:14]
    _mech_sources = {
        "data/sources/mechanism_dataderived_v1.csv": {
            "topo_upstream": "mech_topo_upstream", "topo_downstream": "mech_topo_downstream",
            "topo_net": "mech_topo_net", "topo_outdeg": "mech_topo_outdeg",
            "kegg_shared": "mech_kegg_shared", "kegg_frac": "mech_kegg_frac",
            "ot_genetic_association": "mech_ot_genetic_assoc", "ot_genetic_literature": "mech_ot_genetic_lit",
            "ot_somatic_mutation": "mech_ot_somatic", "ot_affected_pathway": "mech_ot_pathway",
            "ot_animal_model": "mech_ot_animal", "ot_rna_expression": "mech_ot_rna",
            "clingen": "mech_clingen", "in_module": "mech_in_module",
            "coverage_disease": "mech_coverage_disease", "coverage_drug": "mech_coverage_drug"},
        "data/sources/mechanism_impact_v1.csv": {
            "mi_raw": "mech_mi_raw", "mi_within_disease_pct": "mech_mi_within_disease",
            "mi_genetics": "mech_mi_genetics", "depmap_dep_lin": "mech_depmap_dep",
            "depmap_selectivity": "mech_depmap_sel"},
        "data/sources/mendelian_causal_v1.csv": {
            "mendel_clinvar": "mech_mendel_clinvar", "mendel_ot_causal": "mech_mendel_ot_causal",
            "mendel_max": "mech_mendel_max"},
    }
    _all_mech = []
    for _src, _ren in _mech_sources.items():
        _p = ROOT / _src
        if not _p.exists():
            print(f"WARN: {_src} absent — mechanism biology block incomplete"); continue
        _m = pd.read_csv(_p).rename(columns=_ren)
        _keep = [c for c in _ren.values() if c in _m.columns]
        _m = _m[["IK14", "Disease"] + _keep].drop_duplicates(["IK14", "Disease"])
        df = df.merge(_m, left_on=["_ik14", "Disease"], right_on=["IK14", "Disease"], how="left")
        if "IK14" in df.columns:
            df = df.drop(columns=["IK14"])
        _all_mech += _keep
    _cov = df[_all_mech].notna().any(axis=1).mean() if _all_mech else 0.0
    # coverage flags: out-of-file pairs are UNCOVERED -> 0 (NOT median, which would falsely mark them covered)
    _cov_cols = [c for c in _all_mech if c.startswith("mech_coverage_")]
    for _c in _all_mech:
        df[_c] = df[_c].fillna(0.0 if _c in _cov_cols else df[_c].median())
    # WITHIN-DISEASE normalization: the mechanism concepts are relative ("is THIS the right mechanism FOR
    # this disease"), and the raw OT/DepMap/topology signals are sparse + domain-specific. Percentile-rank
    # each within disease (uses FEATURE values only, NOT outcomes -> no outcome leak; shuffle-clean 0.513).
    # Lifts mechanism-only efficacy 0.622 -> 0.641 (notes/mech_llm_archival_jun16.md).
    for _c in list(_all_mech):
        if _c in _cov_cols:   # binary coverage flags: a within-disease percentile is meaningless
            continue
        df[_c + "_wd"] = df.groupby("Disease")[_c].rank(pct=True).fillna(0.5)
        _all_mech.append(_c + "_wd")
    df = df.drop(columns=["_ik14"])
    if "causal_plausibility_blind" in df.columns:
        df = df.drop(columns=["causal_plausibility_blind"])
    print(f"mechanism BIOLOGY block merged: {len(_all_mech)} features (LLM-free), coverage {_cov:.0%}; "
          f"dropped causal_plausibility_blind + archived LLM causal_centrality")

    # Jun 13 2026 — TESTED & NOT BANKED: a zone-scoped leak-free mechanism feature
    # (mech_ot_immunecns = ot_hybrid × CNS/immune zone). Grounded in read-verified confident misses
    # (notes/research_scratchpad.md ZONE-W) where the model under-weights mechanism on sound CNS/immune
    # winners. Per-case flip test FAILED (0/12 even with corrected zones): the leak-free OT
    # disease→target signal is structurally too weak in psychiatry/dermatology (CNS ~0.6, atopic
    # dermatitis ~0.4 — OT genetics is sparse there) to override the model. The signal that WOULD flip
    # them is the LEAKY LLM population_fit (within-zone AUC 0.83–0.93), which is excluded by policy.
    # Conclusion: the CNS/immune miss zone is EXPLAINED but not leak-free-fixable with current data;
    # needs a stronger leak-free CNS/immune mechanism representation. Feature reverted (AUC-neutral, no
    # flip). The zone-coverage bug fix above (depress→MDD, atopic→immune) is kept (genuine correctness).

    # Jun 12 2026 — drop has_black_box. It is (a) INVERTED for safety (univariate AUC 0.398: black-box
    # drugs are 45% of safe vs 25% of safety-fails — survivorship, BBW only exists for approved/established
    # drugs whose risks are managed) and (b) temporally leaky (the warning is usually added POST-marketing,
    # after the trial). It contributes to the managed-risk safety false positives (metformin x44, paclitaxel
    # x31). Removal is safety-AUC-neutral. notes/error_explainability_map_jun12.md.
    if "has_black_box" in df.columns:
        df = df.drop(columns=["has_black_box"])
        print("dropped has_black_box (inverted/leaky survivorship marker)")

    df.to_csv(OUT, index=False)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    prov = {
        "built_by": "scripts/build_v8_honest_exposure.py", "git_sha": sha,
        "source": str(SRC), "source_sha256": sha256(SRC),
        "dose_table": str(DOSE), "output": str(OUT), "output_sha256": sha256(OUT),
        "dose_coverage": float(cov), "added_features": ["max_daily_dose_mg", "logdose"] + made,
    }
    Path(str(OUT) + ".provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"wrote {OUT} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
