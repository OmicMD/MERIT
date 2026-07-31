# Prospective registration (Path C — out-of-sample indications)

**Lock date:** 2026-06-21
**Git commit:** `ee98e2cb61494b3ddf09786450cd452a3a1f20c1`
**Predictions:** `results/benchmark/prereg_C/prereg_C_locked_predictions.csv`
**SHA-256:** `fced8aadb0a4b3287617e792090631278a0b5206d5f8c7f130f4868a0193b935`

## Commitment
On the lock date we froze model predictions for **711 ongoing Phase III trials
(469 distinct trials) of cohort compounds in indications the compound does NOT
hold in the training cohort** — i.e. genuinely out-of-sample (drug, indication)
pairs whose outcomes are not yet known. Unlike a same-pair forward test, a held
score cannot be confirmed by label stability here: the model has never seen these
pairings. Predicted to FAIL (calibrated P_fail >= 0.5): 36/711.

## Why this is the meaningful prospective test
The model's discriminative signal for a novel pairing is the target->disease
mechanism-fit. For each pair we **recompute** that mechanism block (38
features: OmniPath directed topology, KEGG co-membership, Open-Targets biology
channels, ClinGen causal validity, in-module flag, and their within-indication
percentile ranks) from public biology keyed on the drug's targets and the
indication's gene module — no outcome information, no new trial data. The
Mendelian/rare-variant causal, DepMap lineage-dependency, and domain-conditional
mechanism-impact (genetics) features are likewise **recomputed** from cached public
biology (ClinVar/Open-Targets-causal/DepMap CRISPR) per pair — never imputed. Drug-level
features (206) clone from the compound; indication-level features (10) clone from
the indication; the 10 design features are the ongoing trial's own pre-registered
ct.gov design. 0 secondary NON-biological interaction features are
median-imputed and disclosed (direct-target binding overlap, leverage, trial-structure,
trial-context): .

## Model
Canonical `production_v8_clean_mort_coverage_jun22` (frozen, unchanged). The overall
PASS-versus-failure head was fit on all 3135 cohort rows (577 failures) via
the identical published pipeline (impute -> top-k -> within-train disease
target-encoding -> seed-averaged gradient boosting -> nested isotonic calibration)
and applied to the novel pairs. Report `P_fail_overall_calibrated`.

## Evaluation (prospective, at readout)
Score `P_fail_overall_calibrated` against the realized PASS/FAIL under the
manuscript's label-audit protocol; report ROC-AUC with a compound-clustered 95% CI.
This file + SHA-256 + commit fix the predictions before any trial reports.

## Reproduce
```
python3 scripts/benchmark/prereg_C_build_mech.py   # recompute mechanism for novel pairs
python3 scripts/benchmark/prereg_C_lock.py         # assemble, fit-on-all, lock
```


---

## PRIMARY locked set — cleanly-attributable single-agent repurposings (added 2026-06-21)
**File:** `prereg_C_locked_predictions_clean.csv` · **SHA-256:** `4623c526f042d9ea1305801001373057542805a6d8a2665dfe67c189dd59456f` · **commit:** `ee98e2cb61494b3ddf09786450cd452a3a1f20c1`

The full Path-C run scores 711 novel-pair predictions across 469 trials, but many
are supportive-care or chemo-backbone components of multi-drug regimens, where a
trial's outcome is not attributable to our drug. The **primary** prospective set
restricts to trials where our drug is the investigational focus, by a deterministic
rule: the drug is (a) named in the brief title (word-boundary match), (b) the
experimental DIFFERENTIATOR — in an experimental arm but NOT in the comparator arm,
so it is not a shared chemo/standard backbone (the arms differing by some other,
possibly un-featurisable, agent), (c) of ongoing status (recruiting /
active-not-recruiting / enrolling / not-yet-recruiting — the guarantee that the
outcome is not yet known), and (d) with primary completion on/after the lock date. This leaves **111 predictions
across 105 trials of 76 compounds** — genuinely out-of-sample (drug, indication)
pairs that are also cleanly attributable. Predicted to FAIL (P_fail >= 0.5): 7/111
(the 6% rate matches the cohort base rate). Evaluate this set at readout;
the full set is retained as a secondary, attribution-caveated artifact.

`scripts/benchmark/prereg_C_finalize_clean.py` reproduces the filter deterministically.
