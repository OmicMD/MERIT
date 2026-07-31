#!/usr/bin/env python3
"""
Path C, step 1: recompute the data-derived mechanism block (topology / KEGG /
Open-Targets biology channels / ClinGen / in-module) for NOVEL (drug, indication)
pairs — ongoing Phase III trials of cohort compounds in indications the compound
does NOT have in the training cohort. This is the genuine out-of-sample pairing
signal; everything else is drug- or disease-level and clones.

Reuses the exact cached-biology logic of scripts/build_dataderived_mechanism.py
(no external pulls): MOA drug-targets x {OmniPath directed topology, KEGG
co-membership, OT biology-only channels, ClinGen causal validity, OT disease
gene module}. Output columns match mechanism_dataderived_v1.csv's raw names so
the downstream rename map applies unchanged.

Input : results/benchmark/prereg_C_novel_pairs.csv   (IK14, Disease columns)
Output: data/sources/mechanism_dataderived_prereg_C.csv
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OT_SAFE = ["genetic_association", "genetic_literature", "somatic_mutation",
           "affected_pathway", "animal_model", "rna_expression"]
PAIRS = ROOT / "results/benchmark/prereg_C_novel_pairs.csv"
OUT = ROOT / "data/sources/mechanism_dataderived_prereg_C.csv"


def main():
    pairs = pd.read_csv(PAIRS)
    pairs["IK14"] = pairs["IK14"].astype(str).str[:14]

    moa = pd.read_csv(ROOT / "data/sources/ik14_moa_targets_combined_v1.csv")
    dtgt = moa.groupby("ik14")["target_gene"].apply(lambda s: set(map(str, s))).to_dict()

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

    dt = json.load(open(ROOT / "data/cache/disease_targets_cache.json"))
    # BUGFIX (jun22): strip stray " targets" suffix on cached resolved ids (else 253 diseases
    # get an EMPTY gene module -> all-zero mechanism -> default FAIL). See build_dataderived_mechanism.py.
    name2id = {k[7:]: (v[:-8] if isinstance(v, str) and v.endswith(" targets") else v)
               for k, v in dt.items() if k.startswith("search:") and isinstance(v, str)}
    # Jun22: recover verbose/compound condition strings to their base-disease module (zero new pulls).
    # Same determination as the canonical builder: data/sources/disease_condition_norm_v1.csv.
    _normf = ROOT / "data/sources/disease_condition_norm_v1.csv"
    if _normf.exists():
        for _, _r in pd.read_csv(_normf).iterrows():
            name2id.setdefault(str(_r["raw_disease"]).lower(), _r["efo_id"])
    id2genes = {v["disease_id"]: [t["symbol"] for t in v.get("targets", [])[:30]]
                for k, v in dt.items() if k.startswith("targets:") and isinstance(v, dict)}
    otc = json.load(open(ROOT / "data/cache/ot_all_channels.json"))

    rows = list(csv.reader(open(ROOT / "data/cache/clingen_gene_validity.csv")))
    hi = [i for i, r in enumerate(rows) if r and r[0] == "GENE SYMBOL"][0]
    cg = pd.DataFrame([r[:8] for r in rows[hi + 1:] if len(r) >= 8 and r[0] and r[0] != "+++++++++++"],
                      columns=rows[hi][:8])
    sstr = {"Definitive": 5, "Strong": 4, "Moderate": 3, "Limited": 2, "Disputed": 1,
            "No Known Disease Relationship": 0, "Refuted": 0}
    cg["s"] = cg["CLASSIFICATION"].map(sstr).fillna(0)
    clingen = cg.groupby("GENE SYMBOL")["s"].max().to_dict()

    alltg = set().union(*dtgt.values()) if dtgt else set()
    tdown = {g: reach([g], fwd) for g in alltg}
    tup = {g: reach([g], rev) for g in alltg}

    # --- Direct disease-target engagement (RECOMPUTED per pair, NEVER imputed; ports
    # notebooks/01 cell 41): the drug's cached Binding scores intersected with the top-10
    # OpenTargets disease genes. No GPU — pure lookup from already-computed binding scores.
    g2enst = json.load(open(ROOT / "data/cache/gene_to_enst_cache.json"))
    ik2file = {}
    for f in (ROOT / "data/raw/pipeline_all/binding").glob("*_drug_scores.tsv"):
        ik2file.setdefault(f.name[:14], f)
    _scache = {}

    def drug_scores(ik14):
        if ik14 in _scache:
            return _scache[ik14]
        f = ik2file.get(ik14)
        if f is None:
            _scache[ik14] = None; return None
        d = pd.read_csv(f, sep="\t")
        txc = next(c for c in d.columns if c.lower() in ("transcript", "transcipt"))
        scc = next(c for c in d.columns if c.lower() == "score")
        _scache[ik14] = dict(zip(d[txc], d[scc]))
        return _scache[ik14]

    NA_DT = {"direct_target_max": np.nan, "direct_target_mean": np.nan,
             "direct_target_n_engaged_05": np.nan, "direct_target_n_genes_matched": 0}

    def direct_target(ik14, dgenes_top10):
        scores = drug_scores(ik14)
        if scores is None or not dgenes_top10:
            return dict(NA_DT)
        per_gene = []
        for gene in dgenes_top10:
            ensts = g2enst.get(gene, [])
            if ensts:
                per_gene.append(max((scores.get(e, 0) for e in ensts), default=0))
        if not per_gene:
            return dict(NA_DT)
        return {"direct_target_max": float(max(per_gene)),
                "direct_target_mean": float(np.mean(per_gene)),
                "direct_target_n_engaged_05": int(sum(1 for x in per_gene if x >= 0.5)),
                "direct_target_n_genes_matched": len(per_gene)}

    recs = []
    n_dt = 0
    n_efo, n_tg = 0, 0
    for r in pairs.itertuples(index=False):
        tg = [g for g in dtgt.get(r.IK14, set()) if g and g != "nan"]
        efo = name2id.get(str(r.Disease).lower())
        dgenes = set(id2genes.get(efo, [])) if efo else set()
        dmap = otc.get(efo, {}) if efo else {}
        n_efo += int(bool(efo)); n_tg += int(bool(tg))
        down = set().union(*[tdown.get(g, set()) for g in tg]) if tg else set()
        up = set().union(*[tup.get(g, set()) for g in tg]) if tg else set()
        nd = max(len(dgenes), 1)
        rec = {"IK14": r.IK14, "Disease": r.Disease,
               "topo_upstream": len(dgenes & down) / nd, "topo_downstream": len(dgenes & up) / nd,
               "topo_net": (len(dgenes & down) - len(dgenes & up)) / nd,
               "topo_outdeg": float(np.mean([len(fwd.get(g, ())) for g in tg])) if tg else 0.0}
        tp = set().union(*[kpaths(g) for g in tg]) if tg else set()
        dp = set().union(*[kpaths(g) for g in dgenes]) if dgenes else set()
        rec["kegg_shared"] = len(tp & dp); rec["kegg_frac"] = len(tp & dp) / max(len(tp), 1)
        for ch in OT_SAFE:
            vals = [dmap.get(g, {}).get(ch, 0.0) or 0.0 for g in tg if isinstance(dmap.get(g), dict)]
            rec[f"ot_{ch}"] = max(vals) if vals else 0.0
        rec["clingen"] = max([clingen.get(g, 0) for g in tg], default=0)
        rec["in_module"] = int(any(g in dgenes for g in tg))
        # coverage flags (jun22) — see build_dataderived_mechanism.py: distinguishes structural-zero
        # (no disease module / no drug target) from genuine poor mechanism fit, so the model + the
        # prereg OOD-abstention gate can discount missing-mechanism instead of reading it as FAIL.
        rec["coverage_disease"] = int(bool(dgenes) or bool(dmap))
        rec["coverage_drug"] = int(bool(tg))
        # recomputed direct-target engagement (top-10 OT disease genes, same as training)
        rec.update(direct_target(r.IK14, list(id2genes.get(efo, []))[:10]))
        n_dt += int(pd.notna(rec["direct_target_max"]))
        recs.append(rec)

    out = pd.DataFrame(recs)
    out.to_csv(OUT, index=False)
    print(f"recomputed mechanism for {len(out)} novel pairs -> {OUT}")
    print(f"  EFO-mapped disease: {n_efo}/{len(out)} | drug has MOA targets: {n_tg}/{len(out)} "
          f"| direct_target computed: {n_dt}/{len(out)} (rest = no disease module, flagged INCOMPLETE)")
    print(f"  nonzero in_module: {int(out['in_module'].sum())} | "
          f"mean ot_genetic_association: {out['ot_genetic_association'].mean():.3f}")


if __name__ == "__main__":
    main()
