#!/usr/bin/env python3
"""Library for the AACT-scale Phase 2->3 transition build, applying the SAME data-cleanup
discipline as the canonical provenance notebook (notebooks/01_data_provenance_rebuild_executed)
plus the extensions the full-registry scale requires. Imported by notebooks/05_aact_scale_
transition.ipynb so every transformation is shown with intermediate outputs (CLAUDE.md rule #4).

Cleanup rules honoured:
  #1  zero silent data loss   -> every dropped/unresolved name is returned in a categorized log.
  #7  match by IK14            -> compound identity = first 14 chars of InChIKey.
  #9  exclude non-drugs        -> canonical NON_THERAPEUTIC (faithful) + documented scale extension
                                 (electrolytes/excipients/imaging) + radiotracer regex, partial match.
  #10 clean drug names         -> strip (R)/(TM), trailing ';', whitespace (canonical clean).
  #11 stereoisomers != same    -> IK14s mapping to >1 full InChIKey are flagged (not silently merged).
Leak-safety (transition task): all precedent features are as-of-date + self-excluded; see the
builder. n_phase2_trials is the establishment baseline, never the leak-safe headline.
"""
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHEMBL = ROOT / "data/cache/chembl_36/chembl_36_sqlite/chembl_36.db"

# ---- CLAUDE.md #9: non-therapeutic exclusion ----
# (a) the canonical set, verbatim from notebooks/01_data_provenance_rebuild_executed.ipynb:
CANONICAL_NON_THERAPEUTIC = {
    "0.9% Sodium chloride", "Sodium Chloride", "Broncho-Vaxom",
    "NaCl 0.9%", "Sodium Lactate", "11C-MC1", "11C-PS13",
}
# (b) scale extension: control fluids, electrolytes, excipients, sugars, gases, and imaging
#     agents that only appear once the full registry (not the curated cohort) is in scope.
#     Partial, case-insensitive match (mirrors the canonical str.contains application).
NON_THERAPEUTIC_EXT = {
    "saline", "sodium chloride", "nacl", "dextrose", "glucose", "sucrose", "fructose",
    "lactated ringer", "ringer", "water for injection", "sterile water", "normal saline",
    "mannitol", "sorbitol", "oxygen", "nitrogen", "carbon dioxide", "medical air",
    "ethanol", "alcohol", "potassium chloride", "magnesium sulfate", "calcium chloride",
    "sodium bicarbonate", "sodium lactate", "placebo", "vehicle", "excipient", "diluent",
    "contrast", "gadolinium", "iodine contrast", "barium",
}
# imaging / PET-SPECT radiotracers: isotope prefixes and FDG (CLAUDE.md #9 lists PET tracers)
RADIOTRACER_RE = re.compile(
    r"(^|[^a-z0-9])(11c|13n|15o|18f|64cu|68ga|89zr|99mtc|111in|123i|124i|125i|131i|177lu|"
    r"153sm|223ra|225ac|212pb|211at)[-\s\[\]]|fdg|fludeoxyglucose|technetium|gallium ?68|"
    r"fluorodopa|florbetapir|flortaucipir|pittsburgh compound", re.I)


def clean_name(s):
    """Canonical drug-name clean (CLAUDE.md #10): strip (R)/(TM), trailing ';', whitespace."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[®™]", "", s).rstrip(";").strip()


def is_non_therapeutic(name):
    """True if the (cleaned) name is a control/excipient/fluid/imaging agent. Returns (bool, reason)."""
    n = clean_name(name).lower()
    if not n:
        return True, "empty"
    if RADIOTRACER_RE.search(n):
        return True, "radiotracer/imaging"
    for nd in CANONICAL_NON_THERAPEUTIC:
        if nd.lower() in n:
            return True, f"canonical:{nd}"
    for nd in NON_THERAPEUTIC_EXT:
        if nd in n:
            return True, f"ext:{nd}"
    return False, ""


# ---- name normalization for offline ChEMBL resolution (#7 resolve offline, no PubChem) ----
DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|ug|g|kg|ml|l|%|units?|iu|mmol|mol|nm|um)\b", re.I)
FORM = re.compile(r"\b(tablet|tablets|capsule|capsules|injection|injectable|solution|oral|"
                  r"intravenous|iv|infusion|suspension|cream|gel|ointment|patch|inhaled|"
                  r"inhalation|spray|drops?|powder|sachet|syrup|film|coated|extended|release|"
                  r"sustained|modified|hydrochloride|hcl|sulfate|sulphate|sodium|potassium|"
                  r"mesylate|maleate|citrate|acetate|phosphate|besylate|fumarate|tartrate|"
                  r"dihydrate|monohydrate|hydrate|salt|free base)\b", re.I)


def norm(s):
    s = clean_name(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = DOSE.sub(" ", s)
    s = FORM.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" -")


def load_chembl_maps():
    con = sqlite3.connect(CHEMBL)
    struct = pd.read_sql(
        "SELECT cs.molregno, cs.canonical_smiles, cs.standard_inchi_key, "
        "md.chembl_id, md.pref_name, md.molecule_type "
        "FROM compound_structures cs JOIN molecule_dictionary md ON cs.molregno=md.molregno "
        "WHERE cs.canonical_smiles IS NOT NULL", con)
    syn = pd.read_sql("SELECT molregno, synonyms FROM molecule_synonyms WHERE synonyms IS NOT NULL", con)
    con.close()
    struct = struct.dropna(subset=["molregno"]).drop_duplicates("molregno")
    by_mol = struct.set_index("molregno")
    have = set(by_mol.index)
    name2mol = {}
    for molregno, nm in zip(syn.molregno, syn.synonyms):
        if molregno in have:
            n = norm(nm)
            if n and n not in name2mol:
                name2mol[n] = (molregno, "synonym")
    for molregno, nm in zip(struct.molregno, struct.pref_name):
        n = norm(nm)
        if n and n not in name2mol:
            name2mol[n] = (molregno, "pref_name")
    return by_mol, name2mol


def fail_reason(name):
    """Categorize WHY an unresolved name didn't resolve (CLAUDE.md #1: no silent loss)."""
    n = clean_name(name).lower()
    if " + " in n or "/" in n or " and " in n or "+" in n:
        return "combination"
    if any(k in n for k in ("mab", "cept", "antibody", "vaccine", "cells", "car-t", "car t",
                            "interferon", "insulin", "albumin", "immunoglobulin", "toxin")):
        return "likely_biologic"
    if any(k in n for k in ("extract", "herbal", "traditional", "plant", "oil")):
        return "botanical/other"
    return "no_chembl_match"


# ---- transition cohort build (leak-safe, as-of-date, self-excluded) ----
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, DataStructs  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

RDLogger.DisableLog("rdApp.*")
CENSOR_AFTER, NSPLIT, NSEED, SIM_THRESH = 2021, 5, 3, 0.5


def _fp(smiles):
    m = Chem.MolFromSmiles(str(smiles)) if isinstance(smiles, str) else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def load_gnomad():
    g = pd.read_csv(ROOT / "data/cache/gnomad_loeuf_by_gene.txt", sep="\t", low_memory=False).dropna(subset=["gene"])
    return (pd.to_numeric(g.set_index("gene")["oe_lof_upper"], errors="coerce").to_dict(),
            pd.to_numeric(g.set_index("gene")["pLI"], errors="coerce").to_dict())


def build_pairs(trials, nct2cond, ik2genes, ik2smiles, g_loeuf, g_pli):
    """trials: rows with columns ik14, phase(p2/p3 bool), year. Already cleaned + small-molecule +
    non-therapeutic-excluded + stereo-unambiguous. Returns the (ik14, condition) pair table with
    as-of-date, self-excluded precedent/indication/analog/gnomAD features."""
    ik_tl = {}
    for ik, g in trials.groupby("ik14"):
        p2y, p3y = g[g.p2].year, g[g.p3].year
        ik_tl[ik] = {"p2": int(p2y.min()) if len(p2y) else None,
                     "p3": int(p3y.min()) if len(p3y) else None}
    tgt_hist = {}
    for ik, info in ik_tl.items():
        for t in ik2genes.get(ik, set()):
            tgt_hist.setdefault(t, []).append((ik, info["p3"], info["p2"]))
    fps = {ik: _fp(ik2smiles.get(ik)) for ik in ik_tl}
    p3_list = [(ik, info["p3"]) for ik, info in ik_tl.items() if info["p3"] is not None]

    ex = trials.assign(condition=trials.nct_id.map(nct2cond)).dropna(subset=["condition"]).explode("condition")
    dis_hist = {}
    for (ik, dis), g in ex.groupby(["ik14", "condition"]):
        p2y, p3y = g[g.p2].year, g[g.p3].year
        dis_hist.setdefault(dis, []).append((ik, int(p2y.min()) if len(p2y) else None,
                                             int(p3y.min()) if len(p3y) else None))
    rows = []
    for (ik, dis), g in ex.groupby(["ik14", "condition"]):
        p2 = g[g.p2]; p2y = p2.year
        if len(p2y) == 0:
            continue
        Y = int(p2y.min())
        if Y > CENSOR_AFTER:
            continue
        transitioned = int((g[g.p3].year > Y).any())
        tg = ik2genes.get(ik, set())
        pp3 = {oik for t in tg for (oik, op3, _) in tgt_hist.get(t, []) if oik != ik and op3 is not None and op3 < Y}
        pp2 = {oik for t in tg for (oik, _, op2) in tgt_hist.get(t, []) if oik != ik and op2 is not None and op2 < Y}
        prior_p2 = prior_trans = prior_p3s = 0
        for (oik, op2, op3) in dis_hist.get(dis, []):
            if oik == ik:
                continue
            if op2 is not None and op2 < Y:
                prior_p2 += 1
                prior_trans += int(op3 is not None and op3 < Y)
            prior_p3s += int(op3 is not None and op3 < Y)
        f = fps.get(ik); max_sim = 0.0; n_an = 0
        if f is not None:
            for (oik, op3) in p3_list:
                if oik == ik or op3 is None or op3 >= Y:
                    continue
                of = fps.get(oik)
                if of is None:
                    continue
                s = DataStructs.TanimotoSimilarity(f, of)
                max_sim = max(max_sim, s); n_an += int(s >= SIM_THRESH)
        lv = [g_loeuf[t] for t in tg if g_loeuf.get(t) == g_loeuf.get(t) and t in g_loeuf]
        pv = [g_pli[t] for t in tg if g_pli.get(t) == g_pli.get(t) and t in g_pli]
        rows.append({"ik14": ik, "condition": dis, "n_phase2_trials": int(len(p2)),
                     "earliest_p2_year": Y, "transitioned": transitioned,
                     "tprec_prior_p3_drugs": len(pp3), "tprec_prior_p2_drugs": len(pp2),
                     "tprec_n_targets": len(tg), "tprec_first_in_class": int(len(pp3) == 0),
                     "ind_prior_p2_programs": prior_p2,
                     "ind_transition_rate": prior_trans / prior_p2 if prior_p2 else np.nan,
                     "ind_prior_p3_starts": prior_p3s,
                     "analog_max_sim_priorp3": max_sim, "analog_n_priorp3": n_an,
                     "gnomad_min_loeuf": min(lv) if lv else np.nan,
                     "gnomad_max_pli": max(pv) if pv else np.nan,
                     "gnomad_n_constrained": int(sum(1 for x in lv if x < 0.6))})
    return pd.DataFrame(rows)


def _gauc(X, y, groups):
    aucs = []
    for s in range(NSEED):
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedGroupKFold(NSPLIT, shuffle=True, random_state=s).split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            clf = HistGradientBoostingClassifier(random_state=s, max_iter=300, learning_rate=0.05,
                                                 max_leaf_nodes=31, l2_regularization=1.0)
            clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        m = ~np.isnan(oof); aucs.append(roc_auc_score(y[m], oof[m]))
    return float(np.mean(aucs)), float(np.std(aucs))


def evaluate(pairs):
    y = pairs.transitioned.values
    groups = pairs.ik14.values
    np2 = pairs.n_phase2_trials.values.astype(float)
    feats = [c for c in pairs.columns if c.startswith(("tprec_", "ind_", "analog_", "gnomad_"))]
    full = _gauc(pairs[feats].values, y, groups)
    proxy = _gauc(np2.reshape(-1, 1), y, groups)
    s1 = np2 == 1
    strat = _gauc(pairs[feats].values[s1], y[s1], groups[s1])
    return {"n_features": len(feats), "full_grouped_auc": round(full[0], 4),
            "proxy_only_auc": round(proxy[0], 4), "within_single_p2_n": int(s1.sum()),
            "within_single_p2_pos": int(y[s1].sum()), "within_single_p2_auc": round(strat[0], 4),
            "within_single_p2_sd": round(strat[1], 4)}
