#!/usr/bin/env python3
"""
Build a blinded second-rater kit for the terminated-trial label audit, so an
independent human annotator can re-classify a stratified sample and we can
report Cohen's kappa against the protocol-establishing (first) rater.

Population: every candidate audit-decision trial carrying a stated termination
reason (Why_Stopped non-null) — this is exactly the population the manuscript's
limitation sentence concerns ("every terminated-trial label entering the failure
set", Methods "Label audit"). Results-only determinations (completion-logic
efficacy failures with no Why_Stopped, EXCLUDE_NO_RESULTS) are a different,
results-reading task and are intentionally out of scope for this kappa; the
scope is stated in the README so the reported statistic is not over-claimed.

First-rater decision = `Corrected_Outcome` in the canonical training set.

Outputs (data/review/second_rater/):
  - audit_sample_BLINDED.csv   <- the rater fills rater2_category / rater2_notes
  - audit_key_HIDDEN.csv       <- first-rater decisions; DO NOT show the rater
  - RUBRIC.md                  <- frozen category definitions
  - README.md                  <- how to rate + how to score
Deterministic (fixed seed); re-running reproduces the same sample.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
OUT = ROOT / "data/review/second_rater"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260621
AUDIT_CATS = [
    "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH",
    "EXCLUDE_NONDRUG_STOP", "EXCLUDE_MISSCOPED", "EXCLUDE_NO_RESULTS",
]
# per-stratum sample target (capped at availability); small strata taken whole
TARGET = {
    "FAIL_SAFETY": 20,
    "FAIL_EFFICACY": 20,
    "FAIL_BOTH": 5,
    "EXCLUDE_NONDRUG_STOP": 20,
    "EXCLUDE_MISSCOPED": 10,
}


def main():
    df = pd.read_csv(SRC, low_memory=False)
    cand = df[df["Corrected_Outcome"].isin(AUDIT_CATS)].copy()
    # terminated population = stated termination reason present
    cand = cand[cand["Why_Stopped"].notna() & (cand["Why_Stopped"].str.strip() != "")]
    # one row per (NCT, disease) audit decision; drop exact dup NCT+disease
    cand = cand.drop_duplicates(subset=["NCT_ID", "Disease"]).reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    picks = []
    for cat, n in TARGET.items():
        pool = cand[cand["Corrected_Outcome"] == cat]
        take = min(n, len(pool))
        if take == 0:
            continue
        idx = rng.choice(pool.index.values, size=take, replace=False)
        picks.append(pool.loc[idx])
    sample = pd.concat(picks).reset_index(drop=True)

    # shuffle so categories are not grouped (blind), assign stable audit_id
    order = rng.permutation(len(sample))
    sample = sample.iloc[order].reset_index(drop=True)
    sample.insert(0, "audit_id", [f"A{ i+1:03d}" for i in range(len(sample))])

    drug_col = "Drug_Clean" if "Drug_Clean" in sample else "Drug"
    sample["ctgov_url"] = "https://clinicaltrials.gov/study/" + sample["NCT_ID"].astype(str)

    blinded = pd.DataFrame({
        "audit_id": sample["audit_id"],
        "NCT_ID": sample["NCT_ID"],
        "drug": sample[drug_col],
        "disease": sample["Disease"],
        "why_stopped": sample["Why_Stopped"],
        "ctgov_url": sample["ctgov_url"],
        "rater2_category": "",   # <- FILL: one of the RUBRIC categories
        "rater2_notes": "",      # <- optional free text
    })
    key = pd.DataFrame({
        "audit_id": sample["audit_id"],
        "NCT_ID": sample["NCT_ID"],
        "rater1_category": sample["Corrected_Outcome"],
    })

    blinded.to_csv(OUT / "audit_sample_BLINDED.csv", index=False)
    key.to_csv(OUT / "audit_key_HIDDEN.csv", index=False)

    write_rubric(OUT / "RUBRIC.md")
    write_readme(OUT / "README.md", key)

    # provenance
    src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    prov = {
        "seed": SEED,
        "source": str(SRC.relative_to(ROOT)),
        "source_sha256": src_sha,
        "population": "audit-decision trials with stated Why_Stopped (terminated)",
        "n_population": int(len(cand)),
        "n_sample": int(len(sample)),
        "per_stratum_population": {c: int((cand["Corrected_Outcome"] == c).sum()) for c in AUDIT_CATS},
        "per_stratum_sample": key.merge(
            sample[["audit_id", "Corrected_Outcome"]], on="audit_id"
        )["Corrected_Outcome"].value_counts().to_dict(),
    }
    (OUT / "audit_sample.provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"wrote {len(sample)} blinded rows -> {OUT/'audit_sample_BLINDED.csv'}")
    print("per-stratum sample:", prov["per_stratum_sample"])


def write_rubric(path):
    path.write_text("""# Second-rater label-audit rubric (FROZEN)

Read each trial's ClinicalTrials.gov record (`ctgov_url`) **and** its stated
termination reason (`why_stopped`). Classify the **investigational compound's
outcome in this trial** into exactly one category. Decide **blind** to the first
rater's decision (not shown). When in genuine doubt, use `UNCERTAIN`.

Core question (the audit decision): *Did this termination reflect a genuine,
drug-attributable safety or efficacy failure of THIS compound, or a non-drug
stop / mis-scoped outcome?*

| Category | Definition |
|---|---|
| `FAIL_SAFETY` | Stopped/failed for a safety or toxicity reason attributable to **our** compound (AE/SAE/DLT, organ toxicity, unfavorable benefit–risk driven by toxicity in our compound's arm). |
| `FAIL_EFFICACY` | Stopped for futility or missed/insufficient efficacy of **our** compound (lack of efficacy, futility analysis, did not meet primary endpoint). |
| `FAIL_BOTH` | Both a genuine safety **and** a genuine efficacy failure of our compound. |
| `EXCLUDE_NONDRUG_STOP` | Termination not attributable to our compound's safety/efficacy: business/strategic/funding/sponsor decision; slow accrual / enrollment-only; COVID-19 / operational / logistical; administrative/regulatory-process; **drug supply**; OR toxicity/efficacy attributable to a **combination partner** (our compound was the backbone/comparator); OR **no in-trial exposure** (terminated before dosing). |
| `EXCLUDE_MISSCOPED` | The stated outcome does not reflect our compound's success/failure for this indication (e.g., outcome belongs to a different arm; comparator-arm event; wrong indication attribution). |
| `UNCERTAIN` | Record/why_stopped insufficient to decide confidently. |

Notes:
- Judge **this compound**, not the trial as a whole. A trial can be terminated
  for a partner drug's toxicity while our compound is fine → `EXCLUDE_NONDRUG_STOP`.
- "Data Safety Monitoring Board" appearing in text is **not** itself a safety
  failure — read why the DSMB acted (futility vs toxicity).
- Strategic/portfolio/"sponsor decision" with no efficacy/safety reason → `EXCLUDE_NONDRUG_STOP`.
""")


def write_readme(path, key):
    n = len(key)
    path.write_text(f"""# Second-rater label audit — instructions

This kit lets an **independent human rater** re-classify a stratified sample of
the terminated-trial label audit so we can report an inter-rater agreement
(Cohen's kappa) against the first (protocol-establishing) rater. The manuscript
currently states the audit is single-rater with no kappa; this closes that.

## What to do
1. Open `audit_sample_BLINDED.csv` ({n} rows). Do **not** open `audit_key_HIDDEN.csv`.
2. For each row, read `why_stopped` and the linked `ctgov_url` record.
3. Put exactly one category from `RUBRIC.md` in `rater2_category`. Optional `rater2_notes`.
4. Save as `audit_sample_RATED.csv` in this folder.

## Scope (so the kappa is not over-claimed)
Population = audit-decision trials **with a stated termination reason**
(`Why_Stopped` present). This matches the manuscript's limitation sentence
("every terminated-trial label entering the failure set"). Completion-logic
efficacy failures (no termination reason; determined from posted results) and
EXCLUDE_NO_RESULTS are a separate results-reading task and are **out of scope**
for this statistic.

## Scoring
After rating:
```
python3 scripts/benchmark/score_second_rater.py
```
Reports: 6-way Cohen's kappa, the headline **binary** kappa
(GENUINE_FAIL = safety/efficacy/both vs EXCLUDE), percent agreement, and a
confusion matrix → `results/benchmark/second_rater_kappa.json`.

The kit is deterministic (seed in `build_second_rater_kit.py`); the same sample
regenerates on re-run.
""")


if __name__ == "__main__":
    main()
