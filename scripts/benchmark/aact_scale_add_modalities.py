#!/usr/bin/env python3
"""Add inClinico's remaining registry modalities to the AACT-scale transition replication —
OT "target choice" biology channels + eligibility-criteria features — with NO molecular (STAR)
features. Re-test under the structure-blind vs structure-holdout temporal split (their protocol).

Builds on aact_scale_replicate_inclinico.py (registry precedent+establishment, 0.788/0.710) and
aact_scale_add_design.py (+ trial design, 0.811/0.738). This adds:

  (3) OT target-choice  — drug target genes (ChEMBL molregno->gene) x disease (EFO) Open Targets
      association channels, max over the drug's targets. Canonical OT_SAFE set only:
      genetic_association, genetic_literature, somatic_mutation, affected_pathway, animal_model,
      rna_expression. EXCLUDES `clinical` (encodes known-drug clinical advancement = near-circular
      with the transition label) and `literature` (snapshot temporal-leak risk). OT is a current
      snapshot (not date-filtered) — that bias hits structure-blind and structure-holdout equally,
      so the inflation delta stays clean; absolute number carries a mild forward-looking caveat.
  (4) eligibility       — age bounds, gender, healthy-volunteer, criteria length, biomarker/
      refractory/line-of-therapy keyword flags, on the pair's earliest Phase 2 NCT. Registered at
      trial start -> pre-readout, structure-blind-legitimate.

Reports each cumulative stack as structure-BLIND | structure-HOLDOUT | inflation, vs inClinico 0.88,
on the full cohort AND on the OT-covered subset, plus a leak audit (availability->outcome AUC, #8).
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
import aact_scale_add_design as AD  # noqa: E402  (reuse load_design)

TRAIN_MAX, TEST_LO, TEST_HI = 2017, 2018, 2021
OT_SAFE = ["genetic_association", "genetic_literature", "somatic_mutation",
           "affected_pathway", "animal_model", "rna_expression"]


def clf(seed=0):
    return HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, l2_regularization=1.0)


def build_ot_features(pairs):
    """OT target-choice: max channel score over the drug's target genes, for the pair's disease."""
    res = pd.read_csv(ROOT / "data/sources/aact_drug_chembl_resolved.csv", usecols=["molregno", "ik14"])
    mt = pd.read_csv(ROOT / "data/sources/chembl_molregno_targets.csv")
    m = res.merge(mt, on="molregno").dropna(subset=["gene"])
    ik2genes = m.groupby("ik14").gene.apply(lambda s: set(s)).to_dict()

    dt = json.load(open(ROOT / "data/cache/disease_targets_cache.json"))
    name2id = {k[7:]: v for k, v in dt.items() if k.startswith("search:") and isinstance(v, str)}
    # broadened AACT condition->EFO map (pull_ot_aact_conditions.py) takes precedence where present
    emap_path = ROOT / "data/cache/aact_condition_efo_map.json"
    if emap_path.exists():
        emap = json.load(open(emap_path))
        name2id = {**name2id, **{k: v for k, v in emap.items() if v}}
    # long-tail normalization recoveries (aact_scale_efo_normalize.py; zero new pulls)
    norm_path = ROOT / "data/cache/aact_condition_efo_normalized.json"
    if norm_path.exists():
        name2id = {**name2id, **{k: v for k, v in json.load(open(norm_path)).items() if v}}
    otc = json.load(open(ROOT / "data/cache/ot_all_channels.json"))

    recs = []
    for ik, cond in zip(pairs.ik14, pairs.condition.astype(str).str.lower()):
        genes = ik2genes.get(ik, set())
        efo = name2id.get(cond)
        dmap = otc.get(efo, {}) if efo else {}
        rec = {}
        covered = bool(genes) and bool(dmap)
        for ch in OT_SAFE:
            vals = [dmap.get(g, {}).get(ch, 0.0) or 0.0 for g in genes if isinstance(dmap.get(g), dict)]
            rec[f"ot_{ch}"] = max(vals) if vals else np.nan
        rec["ot_max_any"] = (np.nanmax([rec[f"ot_{ch}"] for ch in OT_SAFE])
                             if covered and any(not np.isnan(rec[f"ot_{ch}"]) for ch in OT_SAFE) else np.nan)
        rec["_ot_covered"] = covered
        recs.append(rec)
    f = pd.DataFrame(recs, index=pairs.index)
    return f


def attach_design_and_elig(pairs):
    """Attach trial-design + eligibility features on each pair's earliest Phase 2 NCT."""
    cohort_ik = set(pairs.ik14.unique())
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

    p2 = tr[tr.p2].sort_values("year")
    earliest = p2.drop_duplicates(["ik14", "condition"], keep="first")[["ik14", "condition", "nct_id"]]
    earliest = earliest.rename(columns={"nct_id": "p2_nct"})

    design = AD.load_design()
    elig = pd.read_csv(ROOT / "data/sources/aact_elig_features_compact.csv", low_memory=False)
    # drop the dead line-count columns (newlines collapsed during parse)
    elig = elig.drop(columns=[c for c in ["elig_crit_n_lines", "elig_n_inclusion", "elig_n_exclusion"]
                              if c in elig.columns]).set_index("NCT_ID")
    spn = pd.read_csv(ROOT / "data/sources/aact_sponsor_features_compact.csv", low_memory=False)
    spn_name = spn.set_index("nct_id").spn_lead_name           # kept for as-of-date track record
    spn = spn.drop(columns=[c for c in ["spn_lead_class", "spn_lead_name"] if c in spn.columns]).set_index("nct_id")
    fac = pd.read_csv(ROOT / "data/sources/aact_facility_features_compact.csv", low_memory=False).set_index("nct_id")

    p = pairs.merge(earliest, on=["ik14", "condition"], how="left")
    p = p.merge(design, left_on="p2_nct", right_index=True, how="left")
    p = p.merge(elig, left_on="p2_nct", right_index=True, how="left")
    p = p.merge(spn, left_on="p2_nct", right_index=True, how="left")
    p = p.merge(fac, left_on="p2_nct", right_index=True, how="left")

    # as-of-date sponsor track record: # prior Phase-3 starts by this pair's lead sponsor, before
    # the pair's earliest-P2 year, self-excluded (leak-safe establishment prior, sponsor level).
    p["spn_lead_name"] = p.p2_nct.map(spn_name)
    trc = tr.copy()
    trc["spn"] = trc.nct_id.map(spn_name)
    p3 = trc[trc.phase.astype(str).str.contains("PHASE3")].dropna(subset=["spn", "year"])
    p3u = p3.drop_duplicates(["spn", "nct_id"])[["spn", "year"]]
    from collections import defaultdict
    spn_years = defaultdict(list)
    for s, yv in zip(p3u.spn, p3u.year):
        spn_years[s].append(int(yv))
    for s in spn_years:
        spn_years[s].sort()
    import bisect
    def prior_p3(row):
        ys = spn_years.get(row.spn_lead_name)
        if not ys:
            return 0.0
        return float(bisect.bisect_left(ys, int(row.earliest_p2_year)))
    p["spn_prior_p3_starts"] = p.apply(lambda r: prior_p3(r) if pd.notna(r.spn_lead_name) else 0.0, axis=1)
    p["spn_prior_p3_log"] = np.log1p(p["spn_prior_p3_starts"])
    p = p.drop(columns=["spn_lead_name"])
    return p


def leak_audit(pairs, cols, y):
    """CLAUDE.md #8: availability(notnull)->outcome AUC must be <0.58; report value AUC too."""
    rows = []
    for c in cols:
        avail = pairs[c].notna().astype(int).values
        a_auc = roc_auc_score(y, avail) if 0 < avail.sum() < len(avail) else np.nan
        v = pairs[c].values
        ok = ~np.isnan(v) if v.dtype.kind == "f" else np.ones(len(v), bool)
        v_auc = roc_auc_score(y[ok], v[ok]) if ok.sum() > 10 and len(set(v[ok])) > 1 else np.nan
        v_auc = max(v_auc, 1 - v_auc) if not np.isnan(v_auc) else np.nan
        rows.append({"feature": c, "avail_auc": round(a_auc, 3) if not np.isnan(a_auc) else None,
                     "value_auc": round(v_auc, 3) if not np.isnan(v_auc) else None,
                     "coverage": round(float(avail.mean()), 3),
                     "FLAG_avail>0.58": (not np.isnan(a_auc)) and a_auc > 0.58})
    return pd.DataFrame(rows)


def main():
    pairs = pd.read_csv(OUT / "aact_scale_transition_pairs_clean.csv", low_memory=False)
    reg = [c for c in pairs.columns if c.startswith(("tprec_", "ind_", "analog_", "gnomad_"))] + ["n_phase2_trials"]

    ot = build_ot_features(pairs)
    ot_cols = [c for c in ot.columns if c.startswith("ot_")]
    ot_covered = ot["_ot_covered"].values
    pairs = pd.concat([pairs, ot[ot_cols]], axis=1)

    pairs = attach_design_and_elig(pairs)
    d_cols = [c for c in pairs.columns if c.startswith("d_")]
    e_cols = [c for c in pairs.columns if c.startswith("elig_")]
    s_cols = [c for c in pairs.columns if c.startswith("spn_")]
    f_cols = [c for c in pairs.columns if c.startswith("fac_")]

    y = pairs.transitioned.values
    ik = pairs.ik14.values
    yr = pairs.earliest_p2_year.values
    trn = yr <= TRAIN_MAX
    tst = (yr >= TEST_LO) & (yr <= TEST_HI)

    print(f"{len(pairs)} pairs | reg {len(reg)} | OT {len(ot_cols)} (covered {ot_covered.mean():.1%}) "
          f"| design {len(d_cols)} | elig {len(e_cols)}")
    print(f"temporal: train<= {TRAIN_MAX} n={trn.sum()} ({y[trn].sum()} pos); "
          f"test {TEST_LO}-{TEST_HI} n={tst.sum()} ({y[tst].sum()} pos)   vs inClinico 0.88\n")

    def evaluate(cols, mask=None):
        idx = np.ones(len(pairs), bool) if mask is None else mask
        tr_i, te_i = trn & idx, tst & idx
        if te_i.sum() < 20 or y[te_i].sum() < 5:
            return None, None
        X = pairs[cols].values
        mb = clf().fit(X[tr_i], y[tr_i])
        blind = roc_auc_score(y[te_i], mb.predict_proba(X[te_i])[:, 1])
        keep = tr_i & ~np.isin(ik, list(set(ik[te_i])))
        mh = clf().fit(X[keep], y[keep])
        hold = roc_auc_score(y[te_i], mh.predict_proba(X[te_i])[:, 1])
        return blind, hold

    stacks = [
        ("registry (precedent+establishment)", reg),
        ("+ design", reg + d_cols),
        ("+ OT target-choice", reg + d_cols + ot_cols),
        ("+ eligibility", reg + d_cols + ot_cols + e_cols),
        ("+ funding/sponsor", reg + d_cols + ot_cols + e_cols + s_cols),
        ("+ facility/geography", reg + d_cols + ot_cols + e_cols + s_cols + f_cols),
        ("ALL modalities", reg + d_cols + ot_cols + e_cols + s_cols + f_cols),
        ("funding/sponsor only (on registry)", reg + s_cols),
    ]
    results = {}
    print(f"{'stack':42s} {'BLIND':>7} {'HOLD':>7} {'infl':>7}   (full cohort)")
    for name, cols in stacks:
        b, h = evaluate(cols)
        results[name] = {"full": {"blind": round(b, 4), "holdout": round(h, 4), "inflation": round(b - h, 4)}}
        print(f"{name:42s} {b:7.4f} {h:7.4f} {b-h:+7.4f}")

    print(f"\nOT-covered subset (n={ot_covered.sum()}, drug-genes AND disease-in-OT):")
    print(f"{'stack':42s} {'BLIND':>7} {'HOLD':>7} {'infl':>7}")
    for name, cols in [("registry only", reg), ("registry + OT", reg + ot_cols),
                       ("registry + design + OT + elig", reg + d_cols + ot_cols + e_cols)]:
        b, h = evaluate(cols, mask=ot_covered)
        if b is None:
            print(f"{name:42s}  (subset too small)")
            continue
        results.setdefault("ot_subset", {})[name] = {"blind": round(b, 4), "holdout": round(h, 4),
                                                      "inflation": round(b - h, 4)}
        print(f"{name:42s} {b:7.4f} {h:7.4f} {b-h:+7.4f}")

    print("\nLEAK AUDIT (new features; availability->outcome AUC must be <0.58):")
    audit = leak_audit(pairs, ot_cols + e_cols + s_cols + f_cols, y)
    print(audit.to_string(index=False))
    flagged = audit[audit["FLAG_avail>0.58"]]
    if len(flagged):
        print(f"\n!! {len(flagged)} features flagged (avail>0.58): {list(flagged.feature)}")

    summary = {"n_pairs": int(len(pairs)), "ot_covered": int(ot_covered.sum()),
               "test_n": int(tst.sum()), "test_pos": int(y[tst].sum()),
               "results": results, "inclinico_published": 0.88,
               "leak_audit": audit.to_dict(orient="records")}
    (OUT / "aact_scale_add_modalities.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'aact_scale_add_modalities.json'}")


if __name__ == "__main__":
    main()
