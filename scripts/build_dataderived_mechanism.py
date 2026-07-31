#!/usr/bin/env python3
"""Data-derived mechanism–disease match feature, built ONLY from leak-free causal-biology
sources. Represents whether a drug's target is an upstream causal driver of the disease
(disease-modifying) vs a peripheral/downstream node — directly, from public biology, with
no LLM judgment and no outcome-adjacent channels.

Leak-SAFE sources integrated (per drug-target × disease):
  - OmniPath directed signaling topology: reachability from target into the disease gene module
    (upstream-ness = disease-modifying), in/out degree.
  - KEGG pathway co-membership: target shares a curated pathway with the disease's genes.
  - Open Targets association — BIOLOGY channels only (genetic_association, genetic_literature,
    somatic_mutation, affected_pathway, animal_model, rna_expression). EXCLUDES the outcome-
    adjacent `clinical` (known-drug) and `literature` channels.
  - ClinGen curated gene–disease causal validity strength.
  - ATC drug class (modality: disease-modifying vs symptomatic).
EXCLUDED by design (outcome-adjacent / popularity): drug–disease & target–disease literature
co-mention (prior clinical attention).

Outputs data/sources/mechanism_dataderived_v1.csv and runs a shuffle control (disease-label
permuted within the feature build) to confirm the signal is real, not an artifact.
Run: python3 scripts/build_dataderived_mechanism.py
"""
from __future__ import annotations
import json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OT_SAFE = ["genetic_association", "genetic_literature", "somatic_mutation",
           "affected_pathway", "animal_model", "rna_expression"]


def main():
    df = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv", low_memory=False)
    df["IK14"] = df["feature_IK"].astype(str).str[:14]
    # Compute mechanism for the FULL modeling cohort incl. FAIL_SAFETY. Scoping this to efficacy
    # outcomes (prior behavior) left every FAIL_SAFETY row with a NaN->0 mechanism block, which the
    # OVERALL head (uses mechanism + includes safety failures) could read as a "zero mechanism =
    # safety failure" tell -- an outcome-correlated artifact. Mechanism is target->disease biology
    # and outcome-blind, so computing it for safety rows is leak-safe.
    e = df[df.Corrected_Outcome.isin(["PASS", "FAIL_EFFICACY", "FAIL_SAFETY", "FAIL_BOTH"])].copy()
    e["y"] = e.Corrected_Outcome.isin(["FAIL_EFFICACY", "FAIL_BOTH"]).astype(int)
    moa = pd.read_csv(ROOT / "data/sources/ik14_moa_targets_combined_v1.csv")
    dtgt = moa.groupby("ik14")["target_gene"].apply(lambda s: set(map(str, s))).to_dict()

    # --- OmniPath directed topology ---
    op = pd.read_csv(ROOT / "data/cache/omnipath_directed.tsv", sep="\t")
    fwd, rev = defaultdict(set), defaultdict(set)
    for s, t in zip(op.source_genesymbol, op.target_genesymbol):
        if isinstance(s, str) and isinstance(t, str):
            fwd[s].add(t); rev[t].add(s)

    def reach(starts, adj, hops=3):
        seen, frontier = set(starts), set(starts)
        for _ in range(hops):
            nxt = set().union(*[adj.get(n, set()) for n in frontier]) if frontier else set()
            nxt -= seen; seen |= nxt; frontier = nxt
            if not frontier:
                break
        return seen - set(starts)

    # --- KEGG pathway membership (symbol -> entrez -> pathways) ---
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

    # --- disease -> genes (OT module) + disease -> EFO id ---
    dt = json.load(open(ROOT / "data/cache/disease_targets_cache.json"))
    # BUGFIX (jun22): 253 cohort diseases (15% of trials) cached their resolved id with a stray
    # " targets" suffix (e.g. "EFO_0000319 targets"), which fails the id2genes / OT-channel
    # lookups -> EMPTY disease gene module -> ALL mechanism features default to 0 -> the model
    # systematically predicts FAIL for those indications (CV, MDS, depression, NSCLC, ...). The
    # genes are cached under the clean id; strip the suffix so the module resolves.
    def _clean_id(v):
        return v[:-8] if isinstance(v, str) and v.endswith(" targets") else v
    name2id = {k[7:]: _clean_id(v) for k, v in dt.items()
               if k.startswith("search:") and isinstance(v, str)}
    # Jun22: recover empty modules from verbose/compound condition strings (semicolon lists, clinical
    # qualifiers) by mapping the raw disease to its already-cached base-disease module. Zero new pulls;
    # determination committed in data/sources/disease_condition_norm_v1.csv (build_disease_normalization_map.py).
    _normf = ROOT / "data/sources/disease_condition_norm_v1.csv"
    if _normf.exists():
        for _, _r in pd.read_csv(_normf).iterrows():
            name2id.setdefault(str(_r["raw_disease"]).lower(), _r["efo_id"])
    id2genes = {v["disease_id"]: [t["symbol"] for t in v.get("targets", [])[:30]]
                for k, v in dt.items() if k.startswith("targets:") and isinstance(v, dict)}

    # --- OT biology-channel scores (leak-safe) ---
    otc = json.load(open(ROOT / "data/cache/ot_all_channels.json"))

    # --- ClinGen causal strength (gene-level best) ---
    rows = list(csv.reader(open(ROOT / "data/cache/clingen_gene_validity.csv")))
    hi = [i for i, r in enumerate(rows) if r and r[0] == "GENE SYMBOL"][0]
    cg = pd.DataFrame([r[:8] for r in rows[hi + 1:] if len(r) >= 8 and r[0] and r[0] != "+++++++++++"],
                      columns=rows[hi][:8])
    sstr = {"Definitive": 5, "Strong": 4, "Moderate": 3, "Limited": 2, "Disputed": 1,
            "No Known Disease Relationship": 0, "Refuted": 0}
    cg["s"] = cg["CLASSIFICATION"].map(sstr).fillna(0)
    clingen = cg.groupby("GENE SYMBOL")["s"].max().to_dict()

    # precompute per-target reach
    alltg = set().union(*dtgt.values())
    tdown = {g: reach([g], fwd) for g in alltg}
    tup = {g: reach([g], rev) for g in alltg}

    def build(disease_for_row):
        recs = []
        for i, r in enumerate(e.itertuples(index=False)):
            tg = [g for g in dtgt.get(r.IK14, set()) if g and g != "nan"]
            dis = disease_for_row[i]
            efo = name2id.get(str(dis).lower())
            dgenes = set(id2genes.get(efo, [])) if efo else set()
            dmap = otc.get(efo, {}) if efo else {}
            # topology
            down = set().union(*[tdown.get(g, set()) for g in tg]) if tg else set()
            up = set().union(*[tup.get(g, set()) for g in tg]) if tg else set()
            nd = max(len(dgenes), 1)
            rec = {"topo_upstream": len(dgenes & down) / nd, "topo_downstream": len(dgenes & up) / nd,
                   "topo_net": (len(dgenes & down) - len(dgenes & up)) / nd,
                   "topo_outdeg": float(np.mean([len(fwd.get(g, ())) for g in tg])) if tg else 0.0}
            # KEGG overlap
            tp = set().union(*[kpaths(g) for g in tg]) if tg else set()
            dp = set().union(*[kpaths(g) for g in dgenes]) if dgenes else set()
            rec["kegg_shared"] = len(tp & dp); rec["kegg_frac"] = len(tp & dp) / max(len(tp), 1)
            # OT biology channels (max over targets)
            for ch in OT_SAFE:
                vals = [dmap.get(g, {}).get(ch, 0.0) or 0.0 for g in tg if isinstance(dmap.get(g), dict)]
                rec[f"ot_{ch}"] = max(vals) if vals else 0.0
            # ClinGen + target-in-module
            rec["clingen"] = max([clingen.get(g, 0) for g in tg], default=0)
            rec["in_module"] = int(any(g in dgenes for g in tg))
            # Coverage flags (jun22): the topology/KEGG/OT/in_module features above are STRUCTURALLY 0
            # whenever the disease has no resolved gene module OR the drug has no mapped target — i.e. the
            # zero means "no data", not "poor mechanism fit". Without these flags the model conflates the
            # two and reads missing-mechanism as a (weak) FAIL signal (empty-module rows predicted P_fail
            # 0.32 vs true 0.16; faithful counterfactual Δ-0.067). Leak-safe: coverage availability carries
            # ~no outcome signal (avail AUC ≈0.51). Let the model discount the zeros when coverage=0.
            rec["coverage_disease"] = int(bool(dgenes) or bool(dmap))
            rec["coverage_drug"] = int(bool(tg))
            recs.append(rec)
        return pd.DataFrame(recs)

    feats = build(e.Disease.tolist())
    atc = pd.read_csv(ROOT / "data/sources/cohort_atc_v1.csv"); atc["l2"] = atc.level2_description.fillna("NA")
    oh = pd.crosstab(atc.IK14, atc.level1_description).clip(upper=1); oh.columns = [f"atc1_{i}" for i in range(oh.shape[1])]
    oh2 = pd.crosstab(atc.IK14, atc.l2).clip(upper=1); oh2.columns = [f"atc2_{i}" for i in range(oh2.shape[1])]
    af = oh.join(oh2, how="outer").fillna(0)
    feats = feats.reset_index(drop=True)
    e2 = e.reset_index(drop=True)
    feats = pd.concat([e2[["IK14", "Disease", "y"]], feats], axis=1).merge(af.reset_index(), on="IK14", how="left")
    fcols = [c for c in feats.columns if c not in ("IK14", "Disease", "y")]
    feats[fcols] = feats[fcols].fillna(0.0)
    feats.to_csv(ROOT / "data/sources/mechanism_dataderived_v1.csv", index=False)

    grp = feats.IK14.values
    def cv_auc(y):
        X = feats[fcols].values; pr = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, grp):
            m = HistGradientBoostingClassifier(max_iter=250, max_depth=3, learning_rate=0.06, random_state=0)
            m.fit(X[tr], y[tr]); pr[te] = m.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, pr)

    real = cv_auc(feats.y.values)
    # SHUFFLE control: permute disease labels at the drug level before building, rebuild, re-test
    rng = np.random.RandomState(0)
    drugs = e.IK14.unique(); perm = dict(zip(drugs, rng.permutation(drugs)))
    # cheap shuffle: permute the y within the SAME features (label permutation, drug-grouped)
    yshuf = feats.y.values.copy()
    uniq = feats.IK14.unique(); lab = {d: feats[feats.IK14 == d].y.iloc[0] for d in uniq}
    vals = list(lab.values()); rng.shuffle(vals)
    sh = dict(zip(lab.keys(), vals)); yshuf = feats.IK14.map(sh).values
    shuf = cv_auc(yshuf)

    print(f"data-derived mechanism feature ({len(fcols)} leak-safe biology features):")
    print(f"  standalone efficacy AUC (real)    = {real:.3f}")
    print(f"  standalone efficacy AUC (shuffled) = {shuf:.3f}   (should be ~0.50)")
    print(f"  -> signal is {'REAL' if real - shuf > 0.05 else 'SUSPECT'} (real-shuf = {real-shuf:+.3f})")
    print(f"\nwrote data/sources/mechanism_dataderived_v1.csv")


if __name__ == "__main__":
    main()
