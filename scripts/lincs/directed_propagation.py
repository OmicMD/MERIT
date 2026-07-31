#!/usr/bin/env python3
"""Angle (b) variant: DIRECTED downstream propagation over the OmniPath curated
signaling network (signed, directed), instead of undirected STRING.

Rationale: STRING-based RWR failed with a pure hubness confound (matched==
mismatched, spec_pct=0.50). A directed signaling graph propagates flow from drug
targets DOWNSTREAM to effectors along regulatory edges, which should align better
with the measured transcriptional response and be less hub-dominated.

Same validation as the STRING run: matched vs mismatched, specificity percentile,
degree baseline, and the generic-core vs drug-specific GT split.
Gene-symbol space throughout (OmniPath provides genesymbols).
"""
import json, os, glob, sys
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.metrics import roc_auc_score
from collections import Counter

TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RESTART = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
SEED, K_NULL = 17, 25
L = "data/sources/lincs"

# ---- directed signaling graph (gene symbols) ----
op = pd.read_csv(f"{L}/omnipath_signaling.tsv", sep="\t", low_memory=False)
op = op[op.is_directed == True][["source_genesymbol", "target_genesymbol"]].dropna()
op = op[op.source_genesymbol != op.target_genesymbol].drop_duplicates()
genes = sorted(set(op.source_genesymbol) | set(op.target_genesymbol))
gidx = {g: i for i, g in enumerate(genes)}
n = len(genes)
src = op.source_genesymbol.map(gidx).to_numpy()
tgt = op.target_genesymbol.map(gidx).to_numpy()
# downstream transition: M[v,u] = 1/outdeg(u) for edge u->v
outdeg = np.bincount(src, minlength=n).astype(float)
w = 1.0 / outdeg[src]
M = sparse.csr_matrix((w, (tgt, src)), shape=(n, n))
dangling = (outdeg == 0)
degree_total = np.bincount(src, minlength=n) + np.bincount(tgt, minlength=n)
print(f"OmniPath directed graph: {n} nodes, {len(src)} edges", file=sys.stderr)

def rwr(seed_idx, seed_w):
    p0 = np.zeros(n)
    sw = seed_w if seed_w.sum() > 0 else np.ones_like(seed_w)
    p0[seed_idx] = sw / sw.sum()
    p = p0.copy()
    for _ in range(100):
        dm = p[dangling].sum()                     # dangling mass -> restart dist
        pn = (1 - RESTART) * (M @ p + dm * p0) + RESTART * p0
        if np.abs(pn - p).sum() < 1e-9:
            return pn
        p = pn
    return p

# ---- seeds / GT / bridge (gene symbols directly) ----
g2e = json.load(open("data/cache/gene_to_enst_mapping.json")); e2g = {v: k for k, v in g2e.items()}
drh = pd.read_csv(f"{L}/drh_samples.txt", sep="\t", skiprows=9, low_memory=False)[["pert_iname", "InChIKey"]].dropna()
name2ik = {r.pert_iname.strip().lower(): r.InChIKey[:14] for r in drh.itertuples()}
gt = {}
with open(f"{L}/lincs_chem_consensus.txt") as f:
    for line in f:
        p = line.rstrip("\n").split("\t"); h = p[0]
        for suf in (" Up", " Down"):
            if h.endswith(suf): gt.setdefault(h[:-len(suf)].strip().lower(), set()).update(g for g in p[1:] if g); break
binding = {os.path.basename(p).split("_")[0]: p for p in glob.glob("data/raw/pipeline_all/binding/*_drug_scores.tsv")}
cohort = pd.read_csv("data/sources/training_dataset_v8_honest_exposure.csv", low_memory=False)
ck = set(cohort["feature_IK"].dropna().str[:14])
drugs, seen = [], set()
for name, gg in gt.items():
    ik = name2ik.get(name)
    if ik and ik in ck and ik in binding and ik not in seen and len(gg) >= 5:
        seen.add(ik); drugs.append((name, ik))

freq = Counter()
for name, ik in drugs: freq.update(gt[name])
N = len(drugs); core = {g for g, c in freq.items() if c > 0.25 * N}

prof, seedset, gtn_all, gtn_core, gtn_spec, seedcov = {}, {}, {}, {}, {}, []
for name, ik in drugs:
    df = pd.read_csv(binding[ik], sep="\t"); df["gene"] = df["Transcript"].map(e2g)
    g = df.dropna(subset=["gene"]).groupby("gene")["Score"].max().sort_values(ascending=False)
    pairs = [(gidx[gene], float(s)) for gene, s in g.head(TOPN).items() if gene in gidx]
    if not pairs: continue
    idx = np.array([p[0] for p in pairs]); ws = np.array([p[1] for p in pairs])
    seedcov.append(len(idx) / min(TOPN, len(g)))
    p = rwr(idx, ws)
    if not np.isfinite(p).all(): continue
    prof[ik] = p; seedset[ik] = set(idx.tolist())
    alln = {gidx[x] for x in gt[name] if x in gidx}
    gtn_all[ik] = alln
    gtn_core[ik] = {gidx[x] for x in gt[name] if x in core and x in gidx}
    gtn_spec[ik] = {gidx[x] for x in gt[name] if x not in core and x in gidx}

iks = [ik for ik in prof if len(gtn_all[ik]) >= 5]
print(f"evaluable drugs: {len(iks)}; mean seed coverage in graph: {np.mean(seedcov):.2f}", file=sys.stderr)
print(f"mean GT genes in graph: all={np.mean([len(gtn_all[k]) for k in iks]):.0f} "
      f"core={np.mean([len(gtn_core[k]) for k in iks]):.0f} spec={np.mean([len(gtn_spec[k]) for k in iks]):.0f}", file=sys.stderr)

def auc(score, pos, exclude):
    mask = np.ones(n, bool); mask[list(exclude)] = False
    pos = [x for x in pos if mask[x]]
    if len(pos) < 5: return np.nan
    y = np.zeros(n, int); y[pos] = 1
    return roc_auc_score(y[mask], score[mask])

rng = np.random.default_rng(SEED)
rows = []
for i, ik in enumerate(iks):
    p, sd = prof[ik], seedset[ik]
    o = iks[(i + 1) % len(iks)]
    others = rng.choice([x for x in iks if x != ik], size=min(K_NULL, len(iks) - 1), replace=False)
    md = [auc(p, gtn_all[x], sd | seedset[x]) for x in others]; md = [v for v in md if np.isfinite(v)]
    rand_idx = rng.choice(n, size=len(sd), replace=False)
    ps = rwr(rand_idx, np.ones(len(rand_idx)))
    rows.append(dict(ik14=ik, n_gt=len(gtn_all[ik]),
        auc_matched=auc(p, gtn_all[ik], sd),
        auc_degree=auc(degree_total.astype(float), gtn_all[ik], sd),
        auc_seedshuf=auc(ps, gtn_all[ik], set(rand_idx.tolist())),
        auc_mismatched_mean=np.mean(md) if md else np.nan,
        spec_percentile=np.mean([auc(p, gtn_all[ik], sd) > v for v in md]) if md else np.nan,
        core_matched=auc(p, gtn_core[ik], sd),
        core_mism=auc(p, gtn_core[o], sd | seedset[o]),
        spec_matched=auc(p, gtn_spec[ik], sd),
        spec_mism=auc(p, gtn_spec[o], sd | seedset[o])))
r = pd.DataFrame(rows)
r.to_csv(f"{L}/directed_propagation_top{TOPN}_r{int(RESTART*100)}.csv", index=False)

def med(c): return r[c].dropna().median()
print(f"\n==== OmniPath DIRECTED propagation (TOPN={TOPN}, restart={RESTART}) ====")
print(f"n={len(r)}")
print(f"matched   {med('auc_matched'):.3f} | mismatched {med('auc_mismatched_mean'):.3f} | "
      f"degree {med('auc_degree'):.3f} | seedshuf {med('auc_seedshuf'):.3f} | spec_pct {r.spec_percentile.mean():.2f}")
def paired(a, b, tag):
    d = r.dropna(subset=[a, b])
    if len(d) < 10: print(f"  {tag}: n<10"); return
    print(f"  {tag}: matched={d[a].median():.3f} vs {d[b].median():.3f}  "
          f"delta={ (d[a]-d[b]).median():+.3f}  p={stats.wilcoxon(d[a],d[b]).pvalue:.1e} (n={len(d)})")
paired("auc_matched", "auc_mismatched_mean", "matched vs mismatched")
paired("auc_matched", "auc_seedshuf", "matched vs seedshuf  ")
paired("core_matched", "core_mism", "GENERIC-CORE specificity")
paired("spec_matched", "spec_mism", "DRUG-SPECIFIC specificity")
print("\nspec_pct~0.5 & drug-specific delta~0 => still no drug-specificity (directed network does not rescue).")
