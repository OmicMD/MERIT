#!/usr/bin/env python3
"""Pull the Open-Targets disease gene modules that the prereg novel-pair set needed but
the cohort caches lacked, and record the parent-disease mappings for niche conditions
that Open Targets does not catalogue as their own entity.

This is the one-time, auditable data step behind completing two confident bets whose
disease module was missing (NOT a model change):

  - "neurogenic detrusor overactivity" -> overactive bladder (MONDO_0006624). NDO is the
    neurogenic subtype of overactive bladder; OT has no NDO entity, and vibegron (a beta-3
    agonist, ADRB3) treats both — ADRB3 sits at score ~0.61 in the OAB module, so the
    mechanism-fit signal is recovered honestly via the parent disease.
  - "preeclampsia" -> MONDO_0005081 (direct entity; was simply unpulled).

Writes into the committed caches (resumable, idempotent — skips ids already present):
  data/cache/disease_targets_cache.json   ("search:<name>" -> id ; "targets:<id>" -> {targets})
  data/cache/ot_all_channels.json         (id -> {gene: {datatype: score}})

Re-run after a cache wipe needs online access to the OT GraphQL API; the committed caches
already contain these modules, so the prereg chain reproduces offline.
"""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DT = ROOT / "data/cache/disease_targets_cache.json"
OTC = ROOT / "data/cache/ot_all_channels.json"
API = "https://api.platform.opentargets.org/api/v4/graphql"
Q = """query($efo:String!){disease(efoId:$efo){name associatedTargets(page:{index:0,size:500}){
rows{score target{approvedSymbol} datatypeScores{id score}}}}}"""

# (search-name written into the cache, resolved OT/MONDO id). Niche conditions map to the
# closest catalogued parent disease (documented above); direct entities map to themselves.
JOBS = [("neurogenic detrusor overactivity", "MONDO_0006624"),
        ("preeclampsia", "MONDO_0005081")]


def fetch(efo):
    body = json.dumps({"query": Q, "variables": {"efo": efo}}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "dt-research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    dis = (d.get("data") or {}).get("disease")
    if not dis:
        return None, None, None
    targets, channels = [], {}
    for row in dis["associatedTargets"]["rows"]:
        sym = row["target"]["approvedSymbol"]
        targets.append({"symbol": sym, "score": row["score"]})
        sc = {ds["id"]: ds["score"] for ds in row["datatypeScores"]}
        if sc:
            channels[sym] = sc
    return dis["name"], targets, channels


def main():
    dt = json.load(open(DT))
    otc = json.load(open(OTC))
    for name_q, efo in JOBS:
        if dt.get("search:" + name_q) == efo and efo in otc:
            print(f"{name_q} -> {efo}: already cached, skipping")
            continue
        name, targets, channels = fetch(efo)
        if not targets:
            print(f"{name_q} ({efo}): NO targets returned — skipped")
            continue
        dt["search:" + name_q] = efo
        dt["targets:" + efo] = {"disease_name": name, "disease_id": efo, "targets": targets}
        otc[efo] = channels
        print(f"{name_q} -> {efo} ({name}): {len(targets)} targets cached")
    json.dump(dt, open(DT, "w"))
    json.dump(otc, open(OTC, "w"))
    print("caches updated.")


if __name__ == "__main__":
    main()
