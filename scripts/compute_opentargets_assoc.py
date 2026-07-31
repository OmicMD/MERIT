#!/usr/bin/env python3
"""Efficacy lever: OpenTargets target-disease genetic-association score, arm-level.

Found by reading the efficacy failures: most are drug-disease MISMATCHES /
repurposing failures (Rosuvastatin->sepsis, Ondansetron->OCD, Hydroxyurea->MS) —
the drug's target isn't causally linked to the disease. OpenTargets association
score (genetic/literature/expression evidence) directly measures that link.
Distinct from net_* (STRING/KEGG pathway overlap) — this is direct target-disease
evidence (the Nelson 2015 'genetically supported targets succeed 2x' axis).

Chain: Disease -> OpenTargets id (disease_targets_cache search:) -> scored targets.
Drug -> on-target gene(s) (ChEMBL drug_mechanism, full DB via synonyms; fallback
to chembl_mechanisms.json). Feature = max OT score of the drug's targets for the
disease (0 = target absent from disease's association list = mismatch).

Output: data/models/opentargets_assoc_feature.csv
"""
from __future__ import annotations
import json, sqlite3, sys, warnings
from pathlib import Path
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_arm_level import prepare_arm_dataset
CHEMBL = ROOT / "data/cache/chembl_36/chembl_36_sqlite/chembl_36.db"


def load_ot():
    d = json.load(open(ROOT / "data/cache/disease_targets_cache.json"))
    search = {k[len("search:"):]: v for k, v in d.items() if k.startswith("search:")}
    tgt = {}
    for k, v in d.items():
        if k.startswith("targets:") and isinstance(v, dict):
            tgt[v["disease_id"]] = {t["symbol"]: t["score"] for t in v.get("targets", [])}
    return search, tgt


def drug_target_genes(conn, mech, n2c):
    """Return resolver: drug name -> set of on-target gene symbols."""
    def resolve(inv):
        genes = set()
        if not isinstance(inv, str):
            return genes
        for nm in inv.split(";"):
            k = nm.strip().lower()
            # via chembl_mechanisms.json (fast path)
            c = n2c.get(k) or n2c.get(k.split()[0] if k else "")
            if c:
                for mm in mech.get(c, {}).get("mechanisms", []):
                    t = mm.get("target_chembl_id")
                    if t:
                        for (g,) in conn.execute(
                            "SELECT cs.component_synonym FROM target_dictionary td "
                            "JOIN target_components tc ON td.tid=tc.tid "
                            "JOIN component_synonyms cs ON tc.component_id=cs.component_id "
                            "WHERE cs.syn_type='GENE_SYMBOL' AND td.chembl_id=?", (t,)):
                            genes.add(g)
            # via full ChEMBL drug_mechanism by synonym name
            if not genes:
                for (g,) in conn.execute(
                    "SELECT DISTINCT cs.component_synonym FROM molecule_synonyms ms "
                    "JOIN drug_mechanism dm ON ms.molregno=dm.molregno "
                    "JOIN target_dictionary td ON dm.tid=td.tid "
                    "JOIN target_components tc ON td.tid=tc.tid "
                    "JOIN component_synonyms cs ON tc.component_id=cs.component_id "
                    "WHERE cs.syn_type='GENE_SYMBOL' AND UPPER(ms.synonyms)=UPPER(?)",
                    (nm.strip(),)):
                    genes.add(g)
        return genes
    return resolve


def main():
    search, tgt = load_ot()
    print(f"OT: {len(search)} disease-name searches, {len(tgt)} diseases with scored targets")
    lk = pd.read_csv(ROOT / "data/sources/chembl_smiles_lookup.csv")
    n2c = {}
    for _, r in lk.iterrows():
        for k in ("Drug_Clean", "chembl_pref_name"):
            n = str(r.get(k) or "").strip().lower()
            if n and pd.notna(r.get("chembl_id")) and n not in n2c:
                n2c[n] = r["chembl_id"]
    mech = json.load(open(ROOT / "data/cache/chembl_mechanisms.json"))
    conn = sqlite3.connect(CHEMBL)
    resolve = drug_target_genes(conn, mech, n2c)

    df = prepare_arm_dataset()

    def disease_targets(dis):
        if not isinstance(dis, str):
            return None
        key = dis.strip().lower()
        did = search.get(key)
        if isinstance(did, str) and did in tgt:
            return tgt[did]
        # try first clause before ';'
        k2 = key.split(";")[0].strip()
        did = search.get(k2)
        return tgt.get(did) if isinstance(did, str) else None

    rows = []
    tgt_cache = {}
    for _, r in df.iterrows():
        dt = disease_targets(r["Disease"])
        inv = r["Investigational_Drugs"]
        if inv not in tgt_cache:
            tgt_cache[inv] = resolve(inv)
        genes = tgt_cache[inv]
        score = np.nan
        has_dis = dt is not None
        if dt is not None and genes:
            vals = [dt[g] for g in genes if g in dt]
            score = max(vals) if vals else 0.0   # 0 = target not associated (mismatch)
        rows.append({"NCT_ID": r["NCT_ID"], "Arm_Label": r["Arm_Label"],
                     "ot_assoc_score": score, "ot_disease_covered": int(has_dis),
                     "ot_has_target": int(bool(genes))})
    out = pd.DataFrame(rows)
    conn.close()
    cov = out["ot_assoc_score"].notna().mean()
    print(f"arms with OT assoc score (disease+target resolved): {out['ot_assoc_score'].notna().sum()} ({cov:.1%})")
    out.to_csv(ROOT / "data/models/opentargets_assoc_feature.csv", index=False)
    print("Saved data/models/opentargets_assoc_feature.csv")


if __name__ == "__main__":
    main()
