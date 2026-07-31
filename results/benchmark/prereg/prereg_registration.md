# Prospective registration — locked forward predictions

**Lock date:** 2026-06-21
**Git commit:** `e957cb7b36fb72922457377972ecbea04c5c1103`
**Predictions file:** `results/benchmark/prereg/prereg_locked_predictions.csv`
**Predictions SHA-256:** `f1339a163dd570feb7f0b9f058ca4444ca1c5c61a40746cdc5e51b885b3f16c2`

## Commitment
On the lock date above we froze model predictions for **866 (drug, disease)
predictions across 639 currently-ongoing Phase 3 clinical trials** whose
outcomes are **not yet known**. The predictions and their SHA-256 are registered
here as a pre-specified, outcome-blind forward test of the model reported in the
manuscript. No outcome information for these trials exists at lock time (all are
RECRUITING / ACTIVE_NOT_RECRUITING / ENROLLING_BY_INVITATION / NOT_YET_RECRUITING).
Predicted to FAIL (calibrated P_fail >= 0.5): 26/866.

## Model
- Canonical model: `production_v8_clean_mort_leverage_jun17` (frozen; unchanged).
- Scoring: a fit-on-all run of the *identical* published pipeline (median impute
  -> top-k univariate selection -> within-train Bayesian disease target-encoding
  -> seed-averaged gradient-boosting overall head -> nested isotonic calibration),
  fit on all 3135 cohort rows (577 failures) and applied to the ongoing
  trials. Report `P_fail_overall_calibrated` for thresholds.

## Features (all pre-trial / pre-readout)
Every model feature is a drug-level or drug x disease-level property available
before the trial reads out. For each ongoing trial we reuse the cohort
(drug, disease) feature vector and override the 10 trial-design features
(`design_*`) with the ongoing trial's own ClinicalTrials.gov design
(pre-registered fields: allocation, masking, arms, primary/secondary outcome
counts, primary-endpoint timeframe, eligibility). No outcome, enrollment-result,
or post-readout information enters the features.

## Scope (stated so the test is not over-claimed)
Restricted to ongoing Phase 3 trials of the 184 cohort
compounds that already carry a complete molecular profile, and to (drug, disease)
pairs present in the modeling cohort (so features are reused without new
featurization). This is a forward test of the model's (drug, disease)-level
discrimination under real ongoing-trial designs, not of out-of-distribution
compounds.

## Evaluation (prospective, when trials read out)
When these trials report, score `P_fail_overall_calibrated` against the realized
PASS/FAIL using the same outcome-definition and label-audit protocol as the
manuscript (Methods, "Label audit"), and report ROC-AUC with a 95% CI. This file
+ its SHA-256 + the git commit fix the predictions immutably as of the lock date.

## Reproduce
```
python3 scripts/benchmark/prereg_pull_ongoing_p3.py     # pull ongoing P3 of cohort drugs
python3 scripts/benchmark/prereg_lock_predictions.py    # match, override design, fit-all, lock
```

## Provenance
- cohort source: `data/sources/training_dataset_v8_clean_mort.csv` sha256 `648743604a9acc1e...`
- ongoing-trial pull: `results/benchmark/prereg_ongoing_p3_trials.csv` sha256 `d87b36a350efd0d0...`
