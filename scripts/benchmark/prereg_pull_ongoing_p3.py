#!/usr/bin/env python3
"""
Prospective-registration step 1: pull currently-ONGOING Phase 3 trials of the
755 compounds already in our modeling cohort (complete molecular profile), so we
can lock model predictions on trials whose outcomes are not yet known.

Ongoing = overall status in {RECRUITING, ACTIVE_NOT_RECRUITING,
ENROLLING_BY_INVITATION, NOT_YET_RECRUITING}. Restricting to cohort compounds
means molecular features are reused (no new featurization).

Per-compound caching makes this resumable. Output:
  data/cache/ctgov_ongoing_p3.json   {drug_clean: [study dicts]}
  results/benchmark/prereg_ongoing_p3_trials.csv   one row per (drug, NCT, condition)
"""
import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
CACHE = ROOT / "data/cache/ctgov_ongoing_p3.json"
OUT = ROOT / "results/benchmark/prereg_ongoing_p3_trials.csv"
API = "https://clinicaltrials.gov/api/v2/studies"
ONGOING = "RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION,NOT_YET_RECRUITING"
# module-level fields (valid v2 names; granular DesignAllocation/... names 400)
FIELDS = ("IdentificationModule,StatusModule,DesignModule,ConditionsModule,"
          "ArmsInterventionsModule")


def query_drug(term):
    studies, token = [], None
    for _ in range(20):  # page cap
        params = {
            "query.intr": term,
            "filter.overallStatus": ONGOING,
            "filter.advanced": "AREA[Phase]PHASE3",
            "fields": FIELDS,
            "pageSize": "100",
        }
        if token:
            params["pageToken"] = token
        url = API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "dt-paper-prereg/1.0"})
        with urllib.request.urlopen(req, timeout=45) as r:
            d = json.load(r)
        studies.extend(d.get("studies", []))
        token = d.get("nextPageToken")
        if not token:
            break
        time.sleep(0.1)
    return studies


def clean_term(name):
    # strip trademark fragments / trailing parenthetical openers
    t = re.sub(r"\(.*$", "", str(name)).strip()
    t = t.replace("®", "").replace("™", "").strip(" ;,")
    return t


def main():
    df = pd.read_csv(SRC, low_memory=False)
    cohort = df[df["Corrected_Outcome"].isin(
        ["PASS", "FAIL_EFFICACY", "FAIL_SAFETY", "FAIL_BOTH"])]
    drugs = sorted(cohort["Drug_Clean"].dropna().unique())
    print(f"{len(drugs)} cohort compounds")

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for i, drug in enumerate(drugs):
        if drug in cache:
            continue
        term = clean_term(drug)
        try:
            cache[drug] = query_drug(term)
        except Exception as e:
            print(f"  ERR {drug!r}: {type(e).__name__} {e}")
            cache[drug] = []
        if i % 25 == 0:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache))
            print(f"  {i}/{len(drugs)} done")
        time.sleep(0.08)
    CACHE.write_text(json.dumps(cache))

    rows = []
    for drug, studies in cache.items():
        term = clean_term(drug).lower()
        for s in studies:
            p = s.get("protocolSection", {})
            interventions = [iv.get("name", "") for iv in
                             p.get("armsInterventionsModule", {}).get("interventions", [])]
            # confirm our compound is genuinely an intervention (substring, case-insensitive)
            if not any(term in iv.lower() for iv in interventions) and term:
                # fall back: term in brief title
                if term not in p.get("identificationModule", {}).get("briefTitle", "").lower():
                    continue
            idm = p.get("identificationModule", {})
            st = p.get("statusModule", {})
            des = p.get("designModule", {})
            conds = p.get("conditionsModule", {}).get("conditions", []) or [None]
            arms = p.get("armsInterventionsModule", {}).get("armGroups", []) or []
            arm_types = [a.get("type", "") for a in arms]
            di = des.get("designInfo", {})
            for cond in conds:
                rows.append({
                    "Drug_Clean": drug,
                    "NCT_ID": idm.get("nctId"),
                    "brief_title": idm.get("briefTitle"),
                    "overall_status": st.get("overallStatus"),
                    "Disease_raw": cond,
                    "start_date": (st.get("startDateStruct") or {}).get("date"),
                    "primary_completion_date":
                        (st.get("primaryCompletionDateStruct") or {}).get("date"),
                    "n_arms": len(arms),
                    "allocation": di.get("allocation"),
                    "intervention_model": di.get("interventionModel"),
                    "primary_purpose": di.get("primaryPurpose"),
                    "has_placebo": int(any("PLACEBO" in t for t in arm_types)),
                    "has_active_comparator": int(any("ACTIVE_COMPARATOR" in t for t in arm_types)),
                    "n_primary_outcomes": len(des.get("primaryOutcomes", []) if isinstance(des, dict) else []),
                })
    out = pd.DataFrame(rows).drop_duplicates(subset=["Drug_Clean", "NCT_ID", "Disease_raw"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\n{len(out)} (drug,NCT,condition) rows | "
          f"{out['NCT_ID'].nunique()} distinct trials | "
          f"{out['Drug_Clean'].nunique()} compounds with >=1 ongoing P3")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
