#!/usr/bin/env python3
"""Add the trial-DESIGN modality (inClinico's 2nd input) to the registry-only AACT-scale model
and re-test the inClinico replication. All design fields come from the AACT dump (registered at
trial start -> pre-readout). Goal: does precedent + establishment + design reach ~0.88 under the
structure-blind temporal split, with NO molecular features?

Design features (per the pair's earliest Phase 2 NCT):
  enrollment (log), number_of_arms, number_of_groups, randomized, single_group, treatment_purpose,
  n_masked (0-4 blinding depth), has_us_facility, has_single_facility, actual_duration.

Reuses the clean cohort (pairs_clean.csv) for labels + precedent features; reconstructs the
clean name->IK14 map (small-molecule, in-cohort, non-therapeutic-excluded) to attach the earliest
Phase 2 NCT per (IK14, condition) pair.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/benchmark"
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
import aact_scale_lib as L  # noqa: E402
TRAIN_MAX, TEST_LO, TEST_HI = 2017, 2018, 2021


def load_design():
    d = pd.read_csv(ROOT / "data/raw/aact_designs.txt", sep="|", low_memory=False,
                    usecols=["nct_id", "allocation", "intervention_model", "primary_purpose",
                             "subject_masked", "caregiver_masked", "investigator_masked", "outcomes_assessor_masked"])
    s = pd.read_csv(ROOT / "data/raw/aact_studies_full.txt", sep="|", low_memory=False,
                    usecols=["nct_id", "enrollment", "number_of_arms", "number_of_groups"])
    cv = pd.read_csv(ROOT / "data/raw/aact_calculated_values.txt", sep="|", low_memory=False,
                     usecols=["nct_id", "actual_duration", "has_us_facility", "has_single_facility"])
    d = d.merge(s, on="nct_id", how="outer").merge(cv, on="nct_id", how="outer")
    f = pd.DataFrame({"nct_id": d.nct_id})
    f["d_log_enrollment"] = np.log1p(pd.to_numeric(d.enrollment, errors="coerce"))
    f["d_n_arms"] = pd.to_numeric(d.number_of_arms, errors="coerce")
    f["d_n_groups"] = pd.to_numeric(d.number_of_groups, errors="coerce")
    f["d_randomized"] = (d.allocation == "RANDOMIZED").astype(float)
    f["d_single_group"] = (d.intervention_model == "SINGLE_GROUP").astype(float)
    f["d_treatment"] = (d.primary_purpose == "TREATMENT").astype(float)
    f["d_n_masked"] = sum((d[c] == "t").astype(float) for c in
                          ["subject_masked", "caregiver_masked", "investigator_masked", "outcomes_assessor_masked"])
    f["d_has_us"] = (d.has_us_facility == "t").astype(float)
    f["d_single_facility"] = (d.has_single_facility == "t").astype(float)
    f["d_duration"] = pd.to_numeric(d.actual_duration, errors="coerce")
    return f.set_index("nct_id")


def main():
    pairs = pd.read_csv(OUT / "aact_scale_transition_pairs_clean.csv", low_memory=False)
    cohort_ik = set(pairs.ik14.unique())
    # reconstruct clean name -> ik14 (small-mol, in-cohort, non-therapeutic excluded)
    res = pd.read_csv(ROOT / "data/sources/aact_drug_chembl_resolved.csv", low_memory=False)
    res = res[res.molecule_type.eq("Small molecule") & res.ik14.isin(cohort_ik)]
    res = res[~res.aact_name.map(lambda n: L.is_non_therapeutic(n)[0])]
    name2ik = dict(zip(res.aact_name, res.ik14))

    tr = pd.read_csv(ROOT / "data/raw/aact_drug_trials.csv",
                     usecols=["nct_id", "intervention_name", "phase", "start_date"], low_memory=False)
    tr["ik14"] = tr.intervention_name.map(name2ik)
    tr = tr.dropna(subset=["ik14"])
    tr["year"] = pd.to_datetime(tr.start_date, errors="coerce").dt.year
    tr = tr.dropna(subset=["year"]); tr["year"] = tr.year.astype(int)
    tr["p2"] = tr.phase.astype(str).str.contains("PHASE2")
    cond = pd.read_csv(ROOT / "data/raw/aact_conditions_full.txt", sep="|",
                       usecols=["nct_id", "downcase_name"], low_memory=False).dropna()
    nct2cond = cond.groupby("nct_id").downcase_name.apply(lambda s: sorted(set(s))).to_dict()
    tr = tr.assign(condition=tr.nct_id.map(nct2cond)).dropna(subset=["condition"]).explode("condition")

    # earliest Phase 2 NCT per (ik14, condition)
    p2 = tr[tr.p2].sort_values("year")
    earliest = p2.drop_duplicates(["ik14", "condition"], keep="first")[["ik14", "condition", "nct_id"]]
    earliest = earliest.rename(columns={"nct_id": "p2_nct"})

    design = load_design()
    pairs = pairs.merge(earliest, on=["ik14", "condition"], how="left")
    pairs = pairs.merge(design, left_on="p2_nct", right_index=True, how="left")
    dcols = [c for c in pairs.columns if c.startswith("d_")]
    print(f"design attached to {pairs[dcols].notna().any(axis=1).mean():.0%} of pairs ({len(dcols)} design features)")

    prec = [c for c in pairs.columns if c.startswith(("tprec_", "ind_", "analog_", "gnomad_"))] + ["n_phase2_trials"]
    y = pairs.transitioned.values; ik = pairs.ik14.values; yr = pairs.earliest_p2_year.values
    trn = yr <= TRAIN_MAX; tst = (yr >= TEST_LO) & (yr <= TEST_HI)

    def blind(cols):
        m = HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.05,
                                            max_leaf_nodes=31, l2_regularization=1.0).fit(pairs[cols].values[trn], y[trn])
        return roc_auc_score(y[tst], m.predict_proba(pairs[cols].values[tst])[:, 1])

    def holdout(cols):
        keep = trn & ~np.isin(ik, list(set(ik[tst])))
        m = HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.05,
                                            max_leaf_nodes=31, l2_regularization=1.0).fit(pairs[cols].values[keep], y[keep])
        return roc_auc_score(y[tst], m.predict_proba(pairs[cols].values[tst])[:, 1])

    print(f"\ntemporal split: train<= {TRAIN_MAX} n={trn.sum()} ({y[trn].sum()} pos); "
          f"test {TEST_LO}-{TEST_HI} n={tst.sum()} ({y[tst].sum()} pos)   vs inClinico 0.88")
    rows = {}
    for name, cols in [("registry only (precedent+establishment)", prec),
                       ("design only", dcols),
                       ("registry + design (full inClinico-style)", prec + dcols)]:
        b, h = blind(cols), holdout(cols)
        rows[name] = (b, h)
        print(f"  {name:42s} structure-BLIND {b:.4f}  | structure-HOLDOUT {h:.4f}  | inflation {b-h:+.4f}")
    summary = {"n_pairs": int(len(pairs)), "n_design_features": len(dcols),
               "test_n": int(tst.sum()), "test_pos": int(y[tst].sum()),
               "results": {k: {"structure_blind": round(v[0], 4), "structure_holdout": round(v[1], 4)} for k, v in rows.items()},
               "inclinico_published": 0.88}
    (OUT / "aact_scale_replicate_with_design.json").write_text(json.dumps(summary, indent=2))
    top = rows["registry + design (full inClinico-style)"][0]
    print(f"\nHEADLINE: registry + design, structure-blind temporal (inClinico's protocol, NO molecular) "
          f"= {top:.4f}  vs their 0.88.")
    print(f"wrote {OUT/'aact_scale_replicate_with_design.json'}")


if __name__ == "__main__":
    main()
