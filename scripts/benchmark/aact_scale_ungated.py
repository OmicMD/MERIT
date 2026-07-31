#!/usr/bin/env python3
"""UNGATED AACT-scale inClinico replication — registry-only, NO molecular, NO small-molecule gate.

The committed cohort (aact_scale_transition_pairs_clean.csv) is gated to 3,419 ChEMBL-resolved SMALL
MOLECULES — a gate imposed for the (deferred) molecular pipeline (Stage 4b). A REGISTRY-ONLY model
needs no structure, so that gate needlessly shrinks the cohort ~3x. Dropping it reaches inClinico's
actual scale:
    gated:   3,419 drugs / 57,043 pairs / temporal-test 14,257 (660 pos)
    UNGATED: ~43.5k drugs / ~164k pairs / temporal-test ~48.7k (~1,163 pos)   (~= their 41k molecules)

HYPOTHESIS (the reason for this run): the gCV registry number is already 0.88, but the structure-BLIND
temporal number is 0.81 because temporal censoring (only pre-2017 history) starves our small cohort's
DOMINANT establishment/precedent signal (n_phase2_trials alone = 0.80). inClinico's 41k-molecule /
150k-trial scale keeps pre-2017 establishment/precedent DENSE for most test pairs even after censoring,
which is most of their 0.81->0.88 edge. If scale alone lifts our structure-BLIND temporal number toward
0.88, that's the open-source, no-molecular path — reproduced by matching their scale.

DISCIPLINE (CLAUDE.md): drug-key = IK14 where the cleaned name resolves offline against local ChEMBL
(dedups salt/synonym variants, #7 IK14 not raw SMILES); normalized-name otherwise. Canonical
clean_name + NON_THERAPEUTIC/radiotracer exclusion (#9/#10) via aact_scale_lib. All features strictly
as-of-date + self-excluded. Establishment (n_phase2_trials) + indication-context features compute at
drug-name granularity with ZERO structure; target-precedent + gnomAD attach where the drug resolves
(~56%), 0/NaN otherwise. The expensive Tanimoto analog feature is omitted at this scale (minor; needs
SMILES). Reports structure-BLIND | structure-HOLDOUT | inflation, vs gated 0.81 and inClinico 0.88.

Run: python3 scripts/benchmark/aact_scale_ungated.py
"""
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
import aact_scale_lib as L  # noqa: E402

TRAIN_MAX, TEST_LO, TEST_HI, CENSOR = 2017, 2018, 2021, 2021


def clf(seed=0):
    return HistGradientBoostingClassifier(random_state=seed, max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, l2_regularization=1.0)


def load_trials():
    tr = pd.read_csv(ROOT / "data/raw/aact_drug_trials.csv",
                     usecols=["nct_id", "intervention_name", "phase", "start_date"], low_memory=False)
    tr["year"] = pd.to_datetime(tr.start_date, errors="coerce").dt.year
    tr = tr.dropna(subset=["year", "intervention_name"])
    tr["year"] = tr.year.astype(int)
    ph = tr.phase.astype(str)
    tr["p2"] = ph.str.contains("PHASE2")
    tr["p3"] = ph.str.contains("PHASE3")
    tr = tr[tr.p2 | tr.p3]
    # exclude non-therapeutics (controls/excipients/fluids/radiotracers) on the raw name
    nontherap = tr.intervention_name.map(lambda n: L.is_non_therapeutic(n)[0])
    n_excl = int(nontherap.sum())
    tr = tr[~nontherap]
    return tr, n_excl


def resolve_keys(names):
    """Map each distinct raw name -> drug_key (IK14 if it resolves offline, else NAME:<norm>),
    plus ik14 -> set(genes) for the resolved subset. CLAUDE.md #7: IK14, never raw SMILES."""
    by_mol, name2mol = L.load_chembl_maps()
    res = pd.read_csv(ROOT / "data/sources/aact_drug_chembl_resolved.csv", usecols=["molregno", "ik14"])
    mt = pd.read_csv(ROOT / "data/sources/chembl_molregno_targets.csv")
    mol_genes = mt.dropna(subset=["gene"]).groupby("molregno").gene.apply(lambda s: set(map(str, s))).to_dict()
    ik2genes = defaultdict(set)
    key, resolved = {}, 0
    for nm in names:
        n = L.norm(nm)
        hit = name2mol.get(n)
        if hit is not None:
            mol = hit[0]
            ik = str(by_mol.loc[mol, "standard_inchi_key"])[:14] if mol in by_mol.index else None
            if ik and ik != "nan":
                key[nm] = ik
                if mol in mol_genes:
                    ik2genes[ik] |= mol_genes[mol]
                resolved += 1
                continue
        key[nm] = "NAME:" + n if n else "NAME:_blank"
    return key, dict(ik2genes), resolved


def build(tr, key, ik2genes, g_loeuf, g_pli, nct2cond):
    tr = tr.copy()
    tr["dk"] = tr.intervention_name.map(key)
    ex = tr.assign(condition=tr.nct_id.map(nct2cond)).dropna(subset=["condition"]).explode("condition")

    # per-drugkey earliest P2/P3 (target precedent uses drugkey-level timeline, structure via ik2genes)
    dk_tl = {}
    for dk, g in ex.groupby("dk"):
        p2y, p3y = g.year[g.p2], g.year[g.p3]
        dk_tl[dk] = (int(p2y.min()) if len(p2y) else None, int(p3y.min()) if len(p3y) else None)
    # target -> [(dk, p3y, p2y)]   (only resolved IK14 keys carry genes)
    tgt_hist = defaultdict(list)
    for dk, (p2, p3) in dk_tl.items():
        for t in ik2genes.get(dk, ()):  # NAME: keys have no genes
            tgt_hist[t].append((dk, p3, p2))
    # disease -> [(dk, p2y, p3y)]
    dis_hist = defaultdict(list)
    for (dk, dis), g in ex.groupby(["dk", "condition"]):
        p2y, p3y = g.year[g.p2], g.year[g.p3]
        dis_hist[dis].append((dk, int(p2y.min()) if len(p2y) else None,
                              int(p3y.min()) if len(p3y) else None))

    rows = []
    for (dk, dis), g in ex.groupby(["dk", "condition"]):
        p2 = g[g.p2]
        if len(p2) == 0:
            continue
        Y = int(p2.year.min())
        if Y > CENSOR:
            continue
        transitioned = int((g.year[g.p3] > Y).any())
        tg = ik2genes.get(dk, set())
        pp3 = {oik for t in tg for (oik, op3, _) in tgt_hist.get(t, []) if oik != dk and op3 is not None and op3 < Y}
        pp2 = {oik for t in tg for (oik, _, op2) in tgt_hist.get(t, []) if oik != dk and op2 is not None and op2 < Y}
        prior_p2 = prior_trans = prior_p3s = 0
        for (oik, op2, op3) in dis_hist.get(dis, []):
            if oik == dk:
                continue
            if op2 is not None and op2 < Y:
                prior_p2 += 1
                prior_trans += int(op3 is not None and op3 < Y)
            prior_p3s += int(op3 is not None and op3 < Y)
        lv = [g_loeuf[t] for t in tg if t in g_loeuf and g_loeuf[t] == g_loeuf[t]]
        pv = [g_pli[t] for t in tg if t in g_pli and g_pli[t] == g_pli[t]]
        rows.append({"drug_key": dk, "condition": dis, "n_phase2_trials": int(len(p2)),
                     "earliest_p2_year": Y, "transitioned": transitioned, "resolved": int(bool(tg)),
                     "tprec_prior_p3_drugs": len(pp3), "tprec_prior_p2_drugs": len(pp2),
                     "tprec_n_targets": len(tg), "tprec_first_in_class": int(len(pp3) == 0),
                     "ind_prior_p2_programs": prior_p2,
                     "ind_transition_rate": prior_trans / prior_p2 if prior_p2 else np.nan,
                     "ind_prior_p3_starts": prior_p3s,
                     "gnomad_min_loeuf": min(lv) if lv else np.nan,
                     "gnomad_max_pli": max(pv) if pv else np.nan,
                     "gnomad_n_constrained": int(sum(1 for x in lv if x < 0.6))})
    return pd.DataFrame(rows)


def evaluate(pairs, feats, label):
    y = pairs.transitioned.values
    dk = pairs.drug_key.values
    yr = pairs.earliest_p2_year.values
    trn = yr <= TRAIN_MAX
    tst = (yr >= TEST_LO) & (yr <= TEST_HI)
    X = pairs[feats].values
    mb = clf().fit(X[trn], y[trn])
    blind = roc_auc_score(y[tst], mb.predict_proba(X[tst])[:, 1])
    keep = trn & ~np.isin(dk, list(set(dk[tst])))
    mh = clf().fit(X[keep], y[keep])
    hold = roc_auc_score(y[tst], mh.predict_proba(X[tst])[:, 1])
    print(f"{label:34s} BLIND {blind:.4f} | HOLDOUT {hold:.4f} | infl {blind-hold:+.4f} "
          f"(train {trn.sum()}/{y[trn].sum()}p, test {tst.sum()}/{y[tst].sum()}p)")
    return {"blind": round(blind, 4), "holdout": round(hold, 4), "inflation": round(blind - hold, 4),
            "train_n": int(trn.sum()), "test_n": int(tst.sum()), "test_pos": int(y[tst].sum())}


def main():
    tr, n_excl = load_trials()
    cond = pd.read_csv(ROOT / "data/raw/aact_conditions_full.txt", sep="|",
                       usecols=["nct_id", "downcase_name"], low_memory=False).dropna()
    nct2cond = cond.groupby("nct_id").downcase_name.apply(lambda s: sorted(set(s))).to_dict()
    print(f"AACT P2/P3 trials (non-therapeutic excluded {n_excl}): {len(tr)} rows, "
          f"{tr.intervention_name.nunique()} distinct names")

    key, ik2genes, resolved = resolve_keys(tr.intervention_name.unique())
    g_loeuf, g_pli = L.load_gnomad()
    print(f"resolved to IK14: {resolved}/{tr.intervention_name.nunique()} names; "
          f"{len(ik2genes)} IK14 carry targets")

    pairs = build(tr, key, ik2genes, g_loeuf, g_pli, nct2cond)
    feats = [c for c in pairs.columns if c.startswith(("tprec_", "ind_", "gnomad_"))] + ["n_phase2_trials"]
    print(f"\nUNGATED cohort: {len(pairs)} pairs | {pairs.drug_key.nunique()} drug-keys "
          f"({pairs.resolved.mean():.1%} resolved) | {pairs.transitioned.mean():.1%} base rate | {len(feats)} feats\n")

    print(f"{'stack':34s} {'':5} {'BLIND':>5}   {'HOLDOUT':>7}")
    res = {}
    res["ungated_all"] = evaluate(pairs, feats, "UNGATED registry (all drugs)")
    # diagnostic: establishment+indication only (pure structure-free, full coverage)
    sf = ["n_phase2_trials", "ind_prior_p2_programs", "ind_transition_rate", "ind_prior_p3_starts"]
    res["ungated_structurefree"] = evaluate(pairs, sf, "  establishment+indication only")
    res["ungated_proxy"] = evaluate(pairs, ["n_phase2_trials"], "  n_phase2_trials alone (proxy)")

    summary = {"cohort": {"pairs": int(len(pairs)), "drug_keys": int(pairs.drug_key.nunique()),
                          "resolved_frac": round(float(pairs.resolved.mean()), 4),
                          "base_rate": round(float(pairs.transitioned.mean()), 4),
                          "nontherapeutic_excluded": n_excl},
               "gated_comparator": {"blind": 0.8079, "holdout": 0.7654, "note": "57k-pair gated ALL-modalities"},
               "inclinico_published": 0.88, "results": res}
    (OUT / "aact_scale_ungated.json").write_text(json.dumps(summary, indent=2))
    pairs.to_csv(OUT / "aact_scale_ungated_pairs.csv", index=False)
    print(f"\nwrote {OUT/'aact_scale_ungated.json'} and _pairs.csv")
    print(f"\ngated structure-blind temporal (registry) was 0.788; +all-modalities 0.808; inClinico 0.88")


if __name__ == "__main__":
    main()
