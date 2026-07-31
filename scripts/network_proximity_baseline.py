#!/usr/bin/env python3
"""Network-proximity (Guney & Barabasi 2016) efficacy baseline on the R08 v8 cohort.

Feeds the negative-control AUC reported in Results and Methods (0.470, 95% CI 0.406-0.538;
results/benchmark/network_proximity_summary.json).

NOT CURRENTLY RUNNABLE (Jul 2026): three inputs are absent from the repo --
data/cache/chembl_mechanisms.json, data/sources/lincs/string_graph_400_nodes.json and
data/sources/lincs/string_graph_400.npz (plus the chembl_36 sqlite it opens). They are read
directly, not fetched on demand, and nothing regenerates them. The committed
network_proximity_summary.json is intact and is what the manuscript reports.

Computes the closest network distance d_c between each drug's target set S and the
disease gene module T over the STRING v12 PPI graph, z-scored against degree-preserving
random reference gene sets. Reports the STANDALONE efficacy AUC of the proximity z-score
on the subset of efficacy trials where both a drug-target annotation and a disease gene
module are available, for the benchmarking section (alongside TrialBench / HINT).

Drug targets : ChEMBL mechanism_of_action genes (resolved via chembl_36 sqlite) UNION the
               proprietary-pipeline target list, matched to the cohort by name / InChIKey-14.
Disease genes: Open Targets associated targets (top-K by association score) -> gene module.
               (Topology only is used for prediction; association scores are NOT a feature,
               so this is the classical network-medicine baseline, not the genetic axis.)

Guney d_c(S,T) = (1/|T|) * sum_{t in T} min_{s in S} d(s,t)   (closest measure)
z = (d_c - mean(d_c over R degree-matched random S',T')) / std

Predictor: higher z (drug targets FARTHER from disease module) => more likely FAIL_EFFICACY.

Outputs:
  results/benchmark/network_proximity_efficacy.csv   (per trial: z, d_c, label)
  results/benchmark/network_proximity_summary.json   (AUC + bootstrap CI + coverage)
"""
import json, sqlite3, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
ROOT = Path(".")
RNG = np.random.default_rng(20260610)
R_RAND = 50          # degree-matched random reference draws per (S,T) pair
TOPK_DISEASE = 100   # cap disease module to top-K OT-associated genes (focused module)

# ---------------------------------------------------------------- STRING graph
nd = json.load(open(ROOT / "data/sources/lincs/string_graph_400_nodes.json"))
ensp_order = nd["nodes"]                       # idx -> ENSP
ensp2sym = nd["ensp2sym"]
sym2idx = {}                                   # gene symbol -> representative graph node idx
for i, ensp in enumerate(ensp_order):
    s = ensp2sym.get(ensp, ensp)
    sym2idx.setdefault(s, i)                    # first ENSP per symbol
A = sparse.load_npz(ROOT / "data/sources/lincs/string_graph_400.npz")
A = (A + A.T)                                   # ensure symmetric for undirected BFS
A.data[:] = 1.0                                 # unweighted (hop distance)
n_nodes = A.shape[0]
deg = np.asarray((A > 0).sum(1)).ravel()
print(f"STRING graph: {n_nodes} nodes, {int((A>0).nnz/2)} undirected edges", flush=True)

# degree bins (log-spaced) for degree-preserving randomization, only over annotated nodes
nz = np.where(deg > 0)[0]
log_deg = np.log10(deg[nz] + 1)
bin_edges = np.quantile(log_deg, np.linspace(0, 1, 21))         # 20 bins
bin_edges[-1] += 1e-6
node_bin = np.full(n_nodes, -1)
node_bin[nz] = np.clip(np.digitize(log_deg, bin_edges) - 1, 0, 19)
bin_members = {b: nz[node_bin[nz] == b] for b in range(20)}

def degree_matched(nodes):
    out = []
    for v in nodes:
        b = node_bin[v]
        pool = bin_members.get(b, nz)
        out.append(pool[RNG.integers(len(pool))])
    return np.array(out, dtype=int)

# ---------------------------------------------------------------- drug targets
def load_drug_targets():
    from compute_opentargets_assoc import load_ot, drug_target_genes
    search, tgt = load_ot()
    string_syms = set(sym2idx)
    # source A: ChEMBL mechanism genes by drug name
    lk = pd.read_csv(ROOT / "data/sources/chembl_smiles_lookup.csv")
    n2c = {}
    for _, r in lk.iterrows():
        for k in ("Drug_Clean", "chembl_pref_name"):
            n = str(r.get(k) or "").strip().lower()
            if n and pd.notna(r.get("chembl_id")) and n not in n2c:
                n2c[n] = r["chembl_id"]
    mech = json.load(open(ROOT / "data/cache/chembl_mechanisms.json"))
    conn = sqlite3.connect(ROOT / "data/cache/chembl_36/chembl_36_sqlite/chembl_36.db")
    resolve = drug_target_genes(conn, mech, n2c)
    # source B: pipeline target list by InChIKey-14
    pt = pd.read_csv(ROOT / "data/sources/pipeline_drugs_with_targets.csv")
    ik2 = {}
    for _, r in pt.iterrows():
        if isinstance(r["drug_targets_symbols"], str) and r["drug_targets_symbols"]:
            ik2.setdefault(str(r["inchikey"])[:14], set()).update(r["drug_targets_symbols"].split(";"))
    return search, tgt, resolve, ik2, string_syms, conn

# ---------------------------------------------------------------- proximity
_bfs_cache = {}
def dist_from_set(S):
    """Multi-source hop distance: dist[v] = min_{s in S} d(s,v)."""
    key = frozenset(S)
    if key not in _bfs_cache:
        d = shortest_path(A, method="D", unweighted=True, indices=list(S))
        _bfs_cache[key] = d.min(0)
    return _bfs_cache[key]

def closest_dc(S, T, dist=None):
    if dist is None:
        dist = dist_from_set(S)
    dt = dist[list(T)]
    dt = dt[np.isfinite(dt)]
    return float(dt.mean()) if len(dt) else np.nan

def proximity_z(S, T):
    S, T = list(S), list(T)
    if not S or not T:
        return np.nan, np.nan
    d_obs = closest_dc(S, T)
    if not np.isfinite(d_obs):
        return np.nan, d_obs
    rand = np.empty(R_RAND)
    for r in range(R_RAND):
        rand[r] = closest_dc(degree_matched(S), degree_matched(T))
    rand = rand[np.isfinite(rand)]
    mu, sd = rand.mean(), rand.std()
    z = (d_obs - mu) / sd if sd > 1e-9 else 0.0
    return float(z), d_obs

# ---------------------------------------------------------------- main
def main():
    search, tgt, resolve, ik2, string_syms, conn = load_drug_targets()
    from rdkit import Chem

    df = pd.read_csv(ROOT / "data/sources/training_dataset_v8_clean_mort.csv")  # migrated to canonical (Jun 14)
    eff = df[df["Corrected_Outcome"].isin(["PASS", "FAIL_EFFICACY", "FAIL_BOTH"])].copy()
    eff["y"] = (eff["Corrected_Outcome"] != "PASS").astype(int)

    def did(dis):
        if not isinstance(dis, str):
            return None
        for k in (dis.strip().lower(), dis.strip().lower().split(";")[0].strip()):
            v = search.get(k)
            if isinstance(v, str) and v in tgt:
                return v
        return None

    def ik14(smi):
        try:
            m = Chem.MolFromSmiles(smi)
            return Chem.MolToInchiKey(m)[:14] if m else None
        except Exception:
            return None

    # drug target node set per Drug_Clean (union ChEMBL-resolver + pipeline-by-IK14)
    smi2ik = {s: ik14(s) for s in eff["SMILES"].dropna().unique()}
    dt_cache = {}
    def drug_nodes(name, smi):
        key = (name, smi)
        if key in dt_cache:
            return dt_cache[key]
        genes = set(resolve(name)) if isinstance(name, str) else set()
        ik = smi2ik.get(smi)
        if ik and ik in ik2:
            genes |= ik2[ik]
        nodes = {sym2idx[g] for g in genes if g in sym2idx}
        dt_cache[key] = nodes
        return nodes

    # disease module node set per EFO (top-K OT genes by score)
    dm_cache = {}
    def disease_nodes(efo):
        if efo in dm_cache:
            return dm_cache[efo]
        gmap = tgt.get(efo, {})
        top = sorted(gmap.items(), key=lambda kv: -kv[1])[:TOPK_DISEASE]
        nodes = {sym2idx[g] for g, _ in top if g in sym2idx}
        dm_cache[efo] = nodes
        return nodes

    eff["efo"] = eff["Disease"].map(did)
    rows = []
    pair_cache = {}
    for _, r in eff.iterrows():
        S = drug_nodes(r["Drug_Clean"], r["SMILES"])
        T = disease_nodes(r["efo"]) if r["efo"] else set()
        if S and T:
            pk = (frozenset(S), frozenset(T))
            if pk not in pair_cache:
                pair_cache[pk] = proximity_z(S, T)
            z, dc = pair_cache[pk]
        else:
            z, dc = np.nan, np.nan
        rows.append({"NCT_ID": r.get("NCT_ID"), "Drug_Clean": r["Drug_Clean"],
                     "Disease": r["Disease"], "SMILES": r["SMILES"],
                     "y": r["y"], "prox_z": z, "d_c": dc, "n_S": len(S), "n_T": len(T)})
        if len(rows) % 400 == 0:
            print(f"  {len(rows)}/{len(eff)} trials, {len(pair_cache)} unique pairs, "
                  f"{len(_bfs_cache)} BFS cached", flush=True)
    conn.close()
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/benchmark/network_proximity_efficacy.csv", index=False)

    comp = out[out["prox_z"].notna()].copy()
    y, score = comp["y"].values, comp["prox_z"].values   # higher z (farther) -> FAIL
    auc = roc_auc_score(y, score)
    auc_dc = roc_auc_score(y, comp["d_c"].values)
    # drug-grouped bootstrap CI
    drugs = comp["Drug_Clean"].values
    uniq = comp["Drug_Clean"].unique()
    boots = []
    for _ in range(2000):
        samp = RNG.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(drugs == d)[0] for d in samp])
        yb, sb = y[idx], score[idx]
        if 0 < yb.sum() < len(yb):
            boots.append(roc_auc_score(yb, sb))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    summary = {
        "task": "efficacy (PASS vs FAIL_EFFICACY/FAIL_BOTH)",
        "method": "Guney-Barabasi closest network proximity z-score over STRING v12",
        "auc_prox_z": round(auc, 4),
        "auc_raw_dc": round(auc_dc, 4),
        "ci95_prox_z": [round(lo, 4), round(hi, 4)],
        "n_trials_computable": int(len(comp)),
        "n_trials_efficacy_total": int(len(out)),
        "coverage_frac": round(len(comp) / len(out), 4),
        "n_positives_computable": int(comp["y"].sum()),
        "n_unique_pairs": int(len(pair_cache)),
        "n_drugs": int(comp["Drug_Clean"].nunique()),
        "R_rand": R_RAND, "topk_disease": TOPK_DISEASE,
    }
    json.dump(summary, open(ROOT / "results/benchmark/network_proximity_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
