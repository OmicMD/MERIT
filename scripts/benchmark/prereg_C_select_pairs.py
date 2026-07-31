#!/usr/bin/env python3
"""
Path C, step 0: select the novel (drug, indication) pairs to lock.

A pair qualifies when a cohort compound (complete molecular profile) appears in an
ongoing Phase III trial in an indication that compound does NOT hold in the
training cohort, AND that indication is itself present in the cohort (so its
disease-side data is cached). Pairs whose (IK14, Disease) collide with a training
row are EXCLUDED (CLAUDE.md #11: stereoisomer/name variants share the IK14
connectivity layer, so the mechanism block would be identical to a trained pair —
not out-of-sample).

Emits:
  results/benchmark/prereg_C_novel_pairs.csv   distinct (IK14, Disease) for mechanism recompute
  results/benchmark/prereg_C_trials.csv        trial-level rows for scoring
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRIALS = ROOT / "results/benchmark/prereg_ongoing_p3_trials.csv"
COHORT = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"


def norm(s):
    return str(s).lower().strip().rstrip("s") if pd.notna(s) else ""


# Master/platform/basket protocols enroll across many indications under one NCT.
# A histology-agnostic basket (e.g. the DETERMINE BRAF trial, NCT05768178) lists a
# dozen ct.gov conditions, and without this guard the pipeline fans each one out into
# a separate spurious bet (vemurafenib -> multiple myeloma, etc.). A valid prospective
# bet must be a compound tested for ONE clear indication.
MASTER_RE = re.compile(
    r"master protocol|platform trial|basket trial|umbrella|screening protocol|sub-?study",
    re.I)


def drop_basket_trials(trials):
    """Keep only single-condition trials; drop named platform/master/basket protocols.

    n_conditions is counted over the FULL ct.gov condition list per NCT (across all
    of the trial's interventions), so a multi-indication basket is removed even if the
    cohort compound appears in only one of its condition rows."""
    ncond = trials.groupby("NCT_ID")["Disease_raw"].transform("nunique")
    title = trials.get("brief_title", pd.Series("", index=trials.index)).fillna("")
    drop = (ncond > 1) | title.str.contains(MASTER_RE)
    n_multi = trials.loc[ncond > 1, "NCT_ID"].nunique()
    n_master = trials.loc[title.str.contains(MASTER_RE), "NCT_ID"].nunique()
    print(f"basket/multi-indication filter: dropped {trials.loc[drop, 'NCT_ID'].nunique()} "
          f"trials ({n_multi} multi-condition, {n_master} master/platform-titled); "
          f"kept {trials.loc[~drop, 'NCT_ID'].nunique()} single-indication trials")
    return trials[~drop].copy()


# Procedural anesthesia / sedation / neuromuscular / PK-booster agents: these are
# operative tools or pharmacokinetic enhancers, not disease-modifying therapies, so a
# (drug, disease) efficacy bet on them is a role mis-attribution (e.g. ritonavir as a
# CYP3A4 booster scored against COVID-19, dexmedetomidine as a sedative scored against
# breast cancer). The binding pipeline cannot represent their actual role.
PURE_ADJUNCT = {
    "propofol", "sevoflurane", "isoflurane", "desflurane", "dexmedetomidine",
    "midazolam", "remimazolam", "etomidate", "remifentanil", "rocuronium",
    "cisatracurium", "neostigmine", "sugammadex", "cobicistat",
}


def drop_scope_excluded(nov, cohd):
    """Apply the cohort's efficacy-scope exclusions to the novel pairs (Supplementary Table S13).

    The training/eval cohort excludes anti-pathogen and endogenous-ligand compounds from
    efficacy (the binding pipeline models human-protein, not drug-pathogen or endogenous
    physiology); a prospective efficacy bet must respect the same scope. anti-pathogen and
    endogenous are drug-level flags read from the cohort; PURE_ADJUNCT covers procedural/
    booster agents that have no disease-therapy role."""
    cd = cohd.assign(drn=cohd.Drug_Clean.astype(str).str.lower().str.strip())
    anti = cd.groupby("drn").is_anti_pathogen.max()
    endo = cd.groupby("drn").is_endogenous.max()
    drn = nov.Drug_Clean.astype(str).str.lower().str.strip()
    is_anti = drn.map(anti).fillna(0).astype(bool)
    is_endo = drn.map(endo).fillna(0).astype(bool)
    is_adj = drn.isin(PURE_ADJUNCT)
    drop = is_anti | is_endo | is_adj
    print(f"efficacy-scope filter: dropped {nov.loc[drop, 'NCT_ID'].nunique()} trials "
          f"({int(is_anti.sum())} anti-pathogen rows, {int(is_endo.sum())} endogenous, "
          f"{int(is_adj.sum())} procedural/booster adjunct)")
    return nov[~drop].copy()


def main():
    trials = pd.read_csv(TRIALS)
    trials = drop_basket_trials(trials)
    coh = pd.read_csv(COHORT, low_memory=False)
    cohd = coh[coh.Corrected_Outcome.isin(
        ["PASS", "FAIL_EFFICACY", "FAIL_SAFETY", "FAIL_BOTH"])].copy()
    cohd["dn"] = cohd.Disease.map(norm)
    cohd["IK14"] = cohd["feature_IK"].astype(str).str[:14]

    ik_by_drug = cohd.dropna(subset=["feature_IK"]).groupby("Drug_Clean")["IK14"].first().to_dict()
    drug_dis = cohd.groupby("Drug_Clean")["dn"].apply(set).to_dict()
    canon = cohd.groupby("dn")["Disease"].agg(lambda s: s.value_counts().index[0]).to_dict()
    cohort_ik_disease = set(zip(cohd["IK14"], cohd["dn"]))  # for IK14-collision exclusion

    trials["dn"] = trials.Disease_raw.map(norm)

    def status(d, dn):
        if d not in drug_dis:
            return None
        if dn in drug_dis[d]:
            return None                 # in-cohort pair (not novel)
        if dn in canon:
            return "novel_cached"        # novel pair, disease cached
        return None

    trials["ok"] = [status(d, dn) for d, dn in zip(trials.Drug_Clean, trials.dn)]
    nov = trials[trials.ok == "novel_cached"].copy()
    nov["IK14"] = nov.Drug_Clean.map(ik_by_drug)
    nov["Disease"] = nov.dn.map(canon)
    nov = nov.dropna(subset=["IK14"])

    # exclude IK14-level collisions with the cohort (not truly out-of-sample)
    before = len(nov)
    collide = np.array([(ik, dn) in cohort_ik_disease
                        for ik, dn in zip(nov["IK14"], nov["dn"])])
    nov = nov[~collide]
    print(f"excluded {int(collide.sum())} trial-rows on (IK14, Disease) cohort collision")

    nov = drop_scope_excluded(nov, cohd)

    cols = ["NCT_ID", "Drug_Clean", "IK14", "Disease", "Disease_raw",
            "overall_status", "n_arms", "primary_completion_date"]
    tr = nov[cols].drop_duplicates(["NCT_ID", "IK14", "Disease"])
    pairs = nov[["IK14", "Disease"]].drop_duplicates().reset_index(drop=True)
    tr.to_csv(ROOT / "results/benchmark/prereg_C_trials.csv", index=False)
    pairs.to_csv(ROOT / "results/benchmark/prereg_C_novel_pairs.csv", index=False)
    print(f"novel pairs: {len(pairs)} distinct (IK14,Disease) | trials {tr.NCT_ID.nunique()} | "
          f"trial-rows {len(tr)} | compounds {tr.IK14.nunique()} | diseases {pairs.Disease.nunique()}")


if __name__ == "__main__":
    main()
