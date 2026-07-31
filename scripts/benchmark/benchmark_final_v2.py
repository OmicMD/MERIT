#!/usr/bin/env python3
"""Consolidated, leak-audited benchmark numbers (Jul 2026 principled regen).

Locks every Table-1 / ED-figure benchmark number under ONE rule: strip leaky / reverse-
causation / establishment features from BOTH sides, and report both the FAIR comparison and
the "on their turf" (their leaky features/eval) comparison. Writes results/benchmark/
benchmark_final_v2.json. See notes: the earlier numbers silently included trial-design
(our side) and establishment proxies (their side); inClinico's honest signal was almost
entirely the future-peeking n_phase2_trials proxy.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
B = ROOT / "results/benchmark"
sys.path.insert(0, str(ROOT / "scripts/benchmark"))

# feature scopes ------------------------------------------------------------------
FL = pd.read_csv(ROOT / "data/sources/feature_list_v8.csv")
MOD9 = set(FL[FL.module_family == "9_endpoint_population_design_leverage"].feature)
# our trial-context / non-molecule features to exclude for a "biology, no trial-design" set
OUR_TRIAL_CTX = MOD9 | {"trial_n_drugs", "is_combination", "is_healthy_volunteer",
                        "is_multi_drug_exclude", "is_procedural_exclude",
                        "is_mispaired_supportive", "attr_misindexed"}


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return dict(mean_a=round(float(a.mean()), 4), mean_b=round(float(b.mean()), 4),
                delta=round(float(a.mean() - b.mean()), 4), wins=int((a > b).sum()), n=len(a),
                ttest_p=float(stats.ttest_rel(a, b).pvalue),
                wilcoxon_p=float(stats.wilcoxon(a, b).pvalue))


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_pertrial(star_csv, hint_csv):
    """Per-sample DeLong test (Sun & Xu 2014) comparing our vs HINT AUC on the
    aligned per-trial predictions (seed 0, one row per NCT). A canonical
    two-correlated-AUC test that complements the per-fold paired t-test; it is
    computed on the pooled single-seed AUC, so its AUC values run below the
    mean-of-folds headline and it is reported only as confirmation."""
    s = pd.read_csv(star_csv).groupby("nct").agg(label=("label", "first"), m=("proba", "mean")).reset_index()
    h = pd.read_csv(hint_csv).groupby("nctid").agg(h=("pred", "mean")).reset_index().rename(columns={"nctid": "nct"})
    d = s.merge(h, on="nct")
    y = d.label.values.astype(float); preds = np.vstack([d.m.values, d.h.values])
    order = (-y).argsort(); y = y[order]; preds = preds[:, order]
    mm = int(y.sum()); nn = len(y) - mm
    tx = np.array([_midrank(preds[r, :mm]) for r in range(2)])
    ty = np.array([_midrank(preds[r, mm:]) for r in range(2)])
    tz = np.array([_midrank(preds[r]) for r in range(2)])
    aucs = tz[:, :mm].sum(1) / mm / nn - (mm + 1) / 2 / nn
    cov = np.cov((tz[:, :mm] - tx) / nn) / mm + np.cov(1 - (tz[:, mm:] - ty) / mm) / nn
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return dict(n=int(len(d)), auc_ours=round(float(aucs[0]), 4), auc_hint=round(float(aucs[1]), 4),
                delta=round(float(aucs[0] - aucs[1]), 4), z=round(float(z), 3),
                delong_p=float(2 * stats.norm.sf(abs(z))))


def trialbench():
    import trialbench_split_compare as T
    import importlib; importlib.reload(T)
    tb, tb_feats = T.load_trialbench(); ours, star = T.load_ours()
    j = ours.merge(tb, on="nct", how="inner").drop_duplicates("nct")
    y = j.outcome.astype(int).values; g = j.ik14.values
    bio = [c for c in star if c not in OUR_TRIAL_CTX]
    enroll = [c for c in tb_feats if "enroll" in c.lower()]
    estab = [c for c in tb_feats if any(k in c.lower() for k in
             ["enroll", "sponsor", "agency", "oversight", "dmc", "fda_regulated",
              "sharing_ipd", "collaborator", "responsible", "funded"])]
    sets = {"our_biology": bio,
            "their_all": list(tb_feats),
            "their_minus_enrollment": [c for c in tb_feats if c not in enroll],
            "their_legit_design": [c for c in tb_feats if c not in estab],
            "combined_ours_plus_theirs": bio + [c for c in tb_feats if c not in enroll]}
    perfold = {}
    for name, feats in sets.items():
        X = j[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        aucs = []
        for s in T.SEEDS:
            for tr, te in StratifiedGroupKFold(T.NSPLIT, shuffle=True, random_state=s).split(X, y, g):
                if len(np.unique(y[te])) < 2:
                    continue
                aucs.append(T.fit_auc(X, y, tr, te, s))
        perfold[name] = np.array(aucs)
    out = {"n_trials": int(len(j)), "n_compounds": int(j.ik14.nunique()),
           "approval_rate": round(float(y.mean()), 3),
           "auc": {k: round(float(v.mean()), 4) for k, v in perfold.items()},
           "n_biology_feats": len(bio), "n_legit_design_feats": len(sets["their_legit_design"]),
           "fair_biology_vs_legit_design": paired(perfold["our_biology"], perfold["their_legit_design"])}
    print("TrialBench:", {k: round(float(v.mean()), 3) for k, v in perfold.items()})
    return out


def hint():
    import matched_star_phase3_split_compare_clean as M
    import importlib; importlib.reload(M)
    d, feats = M.prep()
    ho, _ = M.fold_arm(d, feats, M.HOLDOUT_FOLDS, "holdout")
    bl, _ = M.fold_arm(d, feats, M.BLIND_FOLDS, "blind")
    star_ho = pd.DataFrame(ho)[["seed", "fold", "roc_auc"]].rename(columns={"roc_auc": "m"})
    star_bl = pd.DataFrame(bl)[["seed", "fold", "roc_auc"]].rename(columns={"roc_auc": "m"})
    hint_ho = pd.read_csv(B / "hint_holdout_clean_full.csv")[["seed", "fold", "roc_auc"]].rename(columns={"roc_auc": "h"})
    hint_bl = pd.read_csv(B / "hint_blind_full.csv")[["seed", "fold", "roc_auc"]].rename(columns={"roc_auc": "h"})
    ph = star_ho.merge(hint_ho, on=["seed", "fold"]); pb = star_bl.merge(hint_bl, on=["seed", "fold"])
    out = {"n_feats": len(feats),
           "fair_holdout": paired(ph.m, ph.h), "their_turf_blind": paired(pb.m, pb.h),
           "delong_pertrial_seed0": delong_pertrial(B / "star_pertrial_seed0.csv",
                                                     B / "hint_pertrial_seed0.csv")}
    print(f"HINT fair: {ph.m.mean():.3f} vs {ph.h.mean():.3f} | blind: {pb.m.mean():.3f} vs {pb.h.mean():.3f}")
    return out


def inclinico():
    """Registry-level demonstration that the 'honest' number is the n_phase2_trials leak.
    (Full OT+mech proof ladder is in aact_scale_proof_ladder.py; committed gCV reproduces 0.882.)"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    import aact_scale_add_modalities as MOD, aact_scale_add_mechanism as MECH
    p = pd.read_csv(B / "aact_scale_transition_pairs_clean.csv", low_memory=False)
    reg = [c for c in p.columns if c.startswith(("tprec_", "ind_", "analog_", "gnomad_"))] + ["n_phase2_trials"]
    # full modality set exactly as the proof ladder (registry + OT + design + eligibility +
    # sponsor + facility + gene->disease mechanism), so the honest number is not understated.
    ot = MOD.build_ot_features(p); otc = [c for c in ot.columns if c.startswith("ot_")]
    p = pd.concat([p, ot[otc]], axis=1); p = MOD.attach_design_and_elig(p)
    d = [c for c in p if c.startswith("d_")]; e = [c for c in p if c.startswith("elig_")]
    sp = [c for c in p if c.startswith("spn_")]; fc = [c for c in p if c.startswith("fac_")]
    ik2g = MECH.load_drug_targets(); n2i = MECH.load_efo_map()
    mech = MECH.build_mech_features(p, ik2g, n2i); mc = [c for c in mech if c.startswith("mech_")]
    p = pd.concat([p, mech[mc]], axis=1)
    allm = reg + d + otc + e + sp + fc + mc
    prec = [c for c in reg if c != "n_phase2_trials"]                 # target/indication precedent
    trial_sponsor = d + e + sp + fc + ["n_phase2_trials"]            # NOT pre-trial compound properties
    compound_precedent = prec + otc + mc                            # knowable as a compound property (+ precedent)
    bio = otc + mc  # compound biology only
    y = p.transitioned.values; ik = p.ik14.values; yr = p.earliest_p2_year.values
    trn = yr <= 2015; tst = (yr >= 2016) & (yr <= 2021); keep = trn & ~np.isin(ik, list(set(ik[tst])))

    def clf(s):
        return HistGradientBoostingClassifier(random_state=s, max_iter=300, learning_rate=0.05,
                                              max_leaf_nodes=31, l2_regularization=1.0)

    def hold(cols):  # mean over 3 seeds
        X = p[cols].values
        return round(float(np.mean([roc_auc_score(y[tst], clf(s).fit(X[keep], y[keep]).predict_proba(X[tst])[:, 1])
                                    for s in range(3)])), 3)
    gcv = json.loads((B / "aact_scale_proof_ladder.json").read_text())
    out = {"reported_inclinico": 0.882,
           "reproduced_gcv_full": gcv["all_modalities_plus_mech"]["gcv"],
           "honest_full": hold(allm),
           "honest_trial_sponsor_only": hold(trial_sponsor),
           "honest_compound_precedent_only": hold(compound_precedent),
           "honest_biology_only": hold(bio),
           "nphase2_univariate_auc": round(float(roc_auc_score(y, p.n_phase2_trials)), 3)}
    print("inClinico:", out)
    return out


def main():
    res = {"trialbench": trialbench(), "hint": hint(), "inclinico": inclinico()}
    (B / "benchmark_final_v2.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {B/'benchmark_final_v2.json'}")


if __name__ == "__main__":
    main()
