#!/usr/bin/env python3
"""Add the leak-safe gene->disease MECHANISM block (the honest analogue of inClinico's
knowledge-graph "target choice" modality) to the AACT-scale Phase 2->3 transition replication,
on top of ALL existing registry modalities. NO molecular (STAR) features.

The transition harness (aact_scale_add_modalities.py) already ports ONE slice of the canonical
mechanism representation -- the OT datatypeScores channels, max-over-targets (ot_*). This script
ports the FOUR channels the canonical build_dataderived_mechanism.py computes that the transition
cohort still lacks, per (drug-target gene set x disease gene module):

  mech_topo_*   OmniPath directed-signalling topology: reachability from the drug's target(s) INTO
                the disease gene module (upstream-ness = disease-modifying) vs downstream/peripheral;
                net up-down; mean target out-degree.    [data/cache/omnipath_directed.tsv]
  mech_kegg_*   KEGG pathway co-membership: target shares a curated pathway with the disease genes
                (shared count, frac of target pathways).  [kegg_hsa_genes.tsv, kegg_gene_pathway.tsv]
  mech_clingen  ClinGen curated causal-validity strength of the drug's target as a disease gene.
                [clingen_gene_validity.csv]
  mech_in_module  is the drug's target itself a disease-associated gene (in the OT module).

Disease module (Gabe's call, Jun 20): genes in ot_all_channels.json with any channel score >=
MOD_THR (default 0.1) -- full coverage across all 3,404 broadened-pull diseases, broader than the
canonical builder's disease_targets_cache top-30.

LEAK DISCIPLINE: these are EXTERNAL biology snapshots, not registry-derived -- the index pair's own
trials never enter, so no self-exclusion is needed. The forward-looking snapshot bias hits the
structure-BLIND and structure-HOLDOUT splits equally, so the inflation delta stays clean (same
caveat the harness already states for OT). Every new feature is leak-audited (availability->outcome
AUC must be < 0.58, CLAUDE.md #8). Per the inflation thesis, real gene->disease biology should lift
the DISCIPLINED (holdout) number more than the inflated blind one.

Reports registry -> +ALL-modalities -> +mech as BLIND | HOLDOUT | inflation, on the full cohort and
the mech-covered subset, plus a drug-grouped shuffle control. Match drugs by IK14, never raw SMILES.
Run: python3 scripts/benchmark/aact_scale_add_mechanism.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/benchmark"
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
import aact_scale_add_modalities as M  # noqa: E402  (reuse build_ot_features + attach_design_and_elig)

TRAIN_MAX, TEST_LO, TEST_HI = 2017, 2018, 2021
MOD_THR = 0.1   # disease-module inclusion: max channel score >= MOD_THR


def clf(seed=0):
    return HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, l2_regularization=1.0)


# ---------- biology resources (mirrors build_dataderived_mechanism.py) ----------

def load_drug_targets():
    res = pd.read_csv(ROOT / "data/sources/aact_drug_chembl_resolved.csv", usecols=["molregno", "ik14"])
    mt = pd.read_csv(ROOT / "data/sources/chembl_molregno_targets.csv")
    m = res.merge(mt, on="molregno").dropna(subset=["gene"])
    return m.groupby("ik14").gene.apply(lambda s: set(map(str, s))).to_dict()


def load_efo_map():
    dt = json.load(open(ROOT / "data/cache/disease_targets_cache.json"))
    name2id = {k[7:]: v for k, v in dt.items() if k.startswith("search:") and isinstance(v, str)}
    emap = json.load(open(ROOT / "data/cache/aact_condition_efo_map.json"))
    name2id = {**name2id, **{k: v for k, v in emap.items() if v}}
    # long-tail normalization recoveries (aact_scale_efo_normalize.py; zero new pulls)
    norm_path = ROOT / "data/cache/aact_condition_efo_normalized.json"
    if norm_path.exists():
        name2id = {**name2id, **{k: v for k, v in json.load(open(norm_path)).items() if v}}
    return name2id


def load_omnipath():
    op = pd.read_csv(ROOT / "data/cache/omnipath_directed.tsv", sep="\t")
    fwd, rev = defaultdict(set), defaultdict(set)
    for s, t in zip(op.source_genesymbol, op.target_genesymbol):
        if isinstance(s, str) and isinstance(t, str):
            fwd[s].add(t)
            rev[t].add(s)
    return fwd, rev


def reach(starts, adj, hops=3):
    seen, frontier = set(starts), set(starts)
    for _ in range(hops):
        nxt = set().union(*[adj.get(n, set()) for n in frontier]) if frontier else set()
        nxt -= seen
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return seen - set(starts)


def load_kegg():
    sym2ez = {}
    for ln in open(ROOT / "data/cache/kegg_hsa_genes.tsv"):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 4:
            ez = p[0].split(":")[1]
            for s in p[3].split(";")[0].split(","):
                sym2ez.setdefault(s.strip(), ez)
    ez2path = defaultdict(set)
    for ln in open(ROOT / "data/cache/kegg_gene_pathway.tsv"):
        g, pth = ln.rstrip("\n").split("\t")
        ez2path[g.split(":")[1]].add(pth.split(":")[1])

    def kpaths(sym):
        ez = sym2ez.get(sym)
        return ez2path.get(ez, set()) if ez else set()
    return kpaths


def load_clingen():
    rows = list(csv.reader(open(ROOT / "data/cache/clingen_gene_validity.csv")))
    hi = [i for i, r in enumerate(rows) if r and r[0] == "GENE SYMBOL"][0]
    cg = pd.DataFrame([r[:8] for r in rows[hi + 1:] if len(r) >= 8 and r[0] and r[0] != "+++++++++++"],
                      columns=rows[hi][:8])
    sstr = {"Definitive": 5, "Strong": 4, "Moderate": 3, "Limited": 2, "Disputed": 1,
            "No Known Disease Relationship": 0, "Refuted": 0}
    cg["s"] = cg["CLASSIFICATION"].map(sstr).fillna(0)
    return cg.groupby("GENE SYMBOL")["s"].max().to_dict()


def build_mech_features(pairs, ik2genes, name2id):
    otc = json.load(open(ROOT / "data/cache/ot_all_channels.json"))
    # disease module: genes whose max channel score >= MOD_THR (full broadened-pull coverage)
    efo2mod = {}
    for efo, gmap in otc.items():
        if isinstance(gmap, dict):
            efo2mod[efo] = {g for g, ch in gmap.items()
                            if isinstance(ch, dict) and ch and max(ch.values()) >= MOD_THR}

    fwd, rev = load_omnipath()
    kpaths = load_kegg()
    clingen = load_clingen()

    # precompute per-target reach + kegg pathways (cohort target universe only)
    alltg = set().union(*[v for v in ik2genes.values()]) if ik2genes else set()
    alltg = {g for g in alltg if g and g != "nan"}
    tdown = {g: reach([g], fwd) for g in alltg}
    tup = {g: reach([g], rev) for g in alltg}
    tout = {g: len(fwd.get(g, ())) for g in alltg}
    tkeg = {g: kpaths(g) for g in alltg}
    tcg = {g: clingen.get(g, 0) for g in alltg}
    dmod_kegg = {}  # cache disease-module kegg pathway union per efo

    recs = []
    for ik, cond in zip(pairs.ik14, pairs.condition.astype(str).str.lower()):
        tg = [g for g in ik2genes.get(ik, set()) if g and g != "nan"]
        efo = name2id.get(cond)
        dgenes = efo2mod.get(efo, set()) if efo else set()
        covered = bool(tg) and bool(dgenes)
        if not covered:
            recs.append({"mech_topo_upstream": np.nan, "mech_topo_downstream": np.nan,
                         "mech_topo_net": np.nan, "mech_topo_outdeg": np.nan,
                         "mech_kegg_shared": np.nan, "mech_kegg_frac": np.nan,
                         "mech_clingen": np.nan, "mech_in_module": np.nan, "_mech_covered": False})
            continue
        down = set().union(*[tdown.get(g, set()) for g in tg])
        up = set().union(*[tup.get(g, set()) for g in tg])
        nd = max(len(dgenes), 1)
        tp = set().union(*[tkeg.get(g, set()) for g in tg])
        if efo not in dmod_kegg:
            dmod_kegg[efo] = set().union(*[kpaths(g) for g in dgenes]) if dgenes else set()
        dp = dmod_kegg[efo]
        recs.append({
            "mech_topo_upstream": len(dgenes & down) / nd,
            "mech_topo_downstream": len(dgenes & up) / nd,
            "mech_topo_net": (len(dgenes & down) - len(dgenes & up)) / nd,
            "mech_topo_outdeg": float(np.mean([tout.get(g, 0) for g in tg])),
            "mech_kegg_shared": float(len(tp & dp)),
            "mech_kegg_frac": len(tp & dp) / max(len(tp), 1),
            "mech_clingen": float(max([tcg.get(g, 0) for g in tg], default=0)),
            "mech_in_module": int(any(g in dgenes for g in tg)),
            "_mech_covered": True,
        })
    return pd.DataFrame(recs, index=pairs.index)


def evaluate(pairs, cols, trn, tst, ik, y, mask=None):
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


def main():
    pairs = pd.read_csv(OUT / "aact_scale_transition_pairs_clean.csv", low_memory=False)
    reg = [c for c in pairs.columns if c.startswith(("tprec_", "ind_", "analog_", "gnomad_"))] + ["n_phase2_trials"]

    # OT channels + design/elig/sponsor/facility (reuse the existing modality harness verbatim)
    ot = M.build_ot_features(pairs)
    ot_cols = [c for c in ot.columns if c.startswith("ot_")]
    pairs = pd.concat([pairs, ot[ot_cols]], axis=1)
    pairs = M.attach_design_and_elig(pairs)
    d_cols = [c for c in pairs.columns if c.startswith("d_")]
    e_cols = [c for c in pairs.columns if c.startswith("elig_")]
    s_cols = [c for c in pairs.columns if c.startswith("spn_")]
    f_cols = [c for c in pairs.columns if c.startswith("fac_")]
    all_mod = reg + d_cols + ot_cols + e_cols + s_cols + f_cols

    # NEW mechanism block
    ik2genes = load_drug_targets()
    name2id = load_efo_map()
    mech = build_mech_features(pairs, ik2genes, name2id)
    m_cols = [c for c in mech.columns if c.startswith("mech_")]
    mech_covered = mech["_mech_covered"].values
    pairs = pd.concat([pairs, mech[m_cols]], axis=1)

    y = pairs.transitioned.values
    ik = pairs.ik14.values
    yr = pairs.earliest_p2_year.values
    trn = yr <= TRAIN_MAX
    tst = (yr >= TEST_LO) & (yr <= TEST_HI)

    print(f"{len(pairs)} pairs | reg {len(reg)} | OT {len(ot_cols)} | design {len(d_cols)} | "
          f"elig {len(e_cols)} | spn {len(s_cols)} | fac {len(f_cols)} | MECH {len(m_cols)} "
          f"(covered {mech_covered.mean():.1%}, {mech_covered.sum()})")
    print(f"temporal: train<= {TRAIN_MAX} n={trn.sum()} ({y[trn].sum()} pos); "
          f"test {TEST_LO}-{TEST_HI} n={tst.sum()} ({y[tst].sum()} pos)   vs inClinico 0.88\n")

    def ev(cols, mask=None):
        return evaluate(pairs, cols, trn, tst, ik, y, mask)

    stacks = [
        ("registry (precedent+establishment)", reg),
        ("ALL modalities (current best)", all_mod),
        ("+ mechanism (topo/kegg/clingen)", all_mod + m_cols),
        ("registry + mechanism only", reg + m_cols),
    ]
    results = {}
    print(f"{'stack':42s} {'BLIND':>7} {'HOLD':>7} {'infl':>7}   (full cohort)")
    for name, cols in stacks:
        b, h = ev(cols)
        results[name] = {"full": {"blind": round(b, 4), "holdout": round(h, 4), "inflation": round(b - h, 4)}}
        print(f"{name:42s} {b:7.4f} {h:7.4f} {b-h:+7.4f}")

    print(f"\nMECH-covered subset (n={mech_covered.sum()}, drug-targets AND disease-module):")
    print(f"{'stack':42s} {'BLIND':>7} {'HOLD':>7} {'infl':>7}")
    for name, cols in [("ALL modalities", all_mod),
                       ("ALL + mechanism", all_mod + m_cols),
                       ("registry only", reg),
                       ("registry + mechanism", reg + m_cols)]:
        b, h = ev(cols, mask=mech_covered)
        if b is None:
            print(f"{name:42s}  (subset too small)")
            continue
        results.setdefault("mech_subset", {})[name] = {"blind": round(b, 4), "holdout": round(h, 4),
                                                       "inflation": round(b - h, 4)}
        print(f"{name:42s} {b:7.4f} {h:7.4f} {b-h:+7.4f}")

    # leak audit (CLAUDE.md #8)
    print("\nLEAK AUDIT (mechanism features; availability->outcome AUC must be <0.58):")
    audit = M.leak_audit(pairs, m_cols, y)
    print(audit.to_string(index=False))
    flagged = audit[audit["FLAG_avail>0.58"]]
    if len(flagged):
        print(f"\n!! {len(flagged)} features flagged (avail>0.58): {list(flagged.feature)}")

    # drug-grouped shuffle control on the mech-covered subset (mech features only)
    rng = np.random.RandomState(0)
    sub = pairs[mech_covered]
    lab = sub.groupby("ik14").transitioned.first()
    perm = pd.Series(rng.permutation(lab.values), index=lab.index)
    yshuf = sub.ik14.map(perm).values
    real_b, real_h = ev(reg + m_cols, mask=mech_covered)
    # quick shuffled holdout: fit on shuffled labels temporally
    Xs = sub[reg + m_cols].values
    ys = sub.transitioned.values
    syr = sub.earliest_p2_year.values
    strn, stst = syr <= TRAIN_MAX, (syr >= TEST_LO) & (syr <= TEST_HI)
    if stst.sum() > 20 and yshuf[stst].sum() >= 5:
        ms = clf().fit(Xs[strn], yshuf[strn])
        shuf_auc = roc_auc_score(ys[stst], ms.predict_proba(Xs[stst])[:, 1])
    else:
        shuf_auc = float("nan")
    print(f"\nshuffle control (reg+mech, mech-covered): real holdout {real_h:.3f} | "
          f"label-shuffled blind {shuf_auc:.3f} (should be ~0.50)")

    summary = {"n_pairs": int(len(pairs)), "mech_covered": int(mech_covered.sum()),
               "mod_thr": MOD_THR, "test_n": int(tst.sum()), "test_pos": int(y[tst].sum()),
               "results": results, "inclinico_published": 0.88,
               "shuffle_holdout_real": round(float(real_h), 4), "shuffle_blind_shuffled": round(float(shuf_auc), 4),
               "leak_audit": audit.to_dict(orient="records")}
    (OUT / "aact_scale_add_mechanism.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'aact_scale_add_mechanism.json'}")


if __name__ == "__main__":
    main()
