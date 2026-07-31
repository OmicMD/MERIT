#!/usr/bin/env python3
"""Endpoint-difficulty tier — a per-trial ordinal the model otherwise lacks (Jun 17).

The model has NO endpoint feature, so for same-drug+disease trials that differ only by endpoint it gives
the same prediction. But endpoint difficulty separates them: a PD/surrogate readout (the drug does it
pharmacologically) passes far more than a hard clinical outcome (the PD effect must translate).

endpoint_difficulty_tier (drug-blind, outcome-blind, pre-specified-endpoint-text only):
  1 = surrogate / PD biomarker (HbA1c, FEV1, blood pressure, hematocrit, viral load, plasma level, pupil)
  2 = functional / symptom scale (pain, depression/psych scales, walk distance, QoL) [DEFAULT]
  3 = clinical outcome / event (mortality, progression, remission, relapse, events, exacerbation, cure)

Fail-rate gradient (clean_mort efficacy): tier1 6% < tier2 17% < tier3 24%. Adds over the model
global 0.7674->0.7697 (coef +0.25 p=0.008) and within Phase 3 0.7538->0.7564 (p=0.003); tier is resolved
for ~100% of trials (no availability confound). notes/leverage_matching_framework_jun17.md.
Mechanism: the PD-biomarker-passes / clinical-outcome-fails translation gap (atropine dilates pupils but
does not stop myopia progression; testosterone raises serum-T but does not improve physical performance).
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_PO = None

T1 = (r'plasma level|serum concentration|pharmacokinet|\bauc\b|pupil|drug level|\btrough\b|c-?peptide|'  # PD/PK
      r'hba1c|glycated|glycaemic|blood pressure|systolic|diastolic|hematocrit|haematocrit|platelet count|'
      r'fev1|forced expiratory|forced vital|\bfvc\b|\bldl\b|cholesterol|triglycer|viral load|hemoglobin|'
      r'haemoglobin|serum testosterone|bone mineral|uric acid|urate|\begfr\b|spleen volume|intraocular|'
      r'urine osmolality|liver fat|proteinuria')
T3 = (r'mortality|survival|progression|hospitali|remission|relapse|recurrence|event-free|disease-free|'
      r'composite|\bstroke\b|exacerbation|\bcure\b|eradicat|complete response|complete remission|'
      r'time to (first |confirmed )?(relapse|progression|event|recurrence)|seizure freedom|vaso-occlusive')


def endpoint_difficulty_tier(title):
    t = (title or '').lower()
    if re.search(T3, t):
        return 3
    if re.search(T1, t):
        return 1
    return 2


def _titles(nct):
    global _PO
    if _PO is None:
        _PO = json.load(open(ROOT / 'data/cache/ctgov_primary_outcomes_protocol.json'))
    o = _PO.get(nct)
    return ' '.join(x.get('title') or '' for x in o if isinstance(x, dict)) if isinstance(o, list) else ''


def add_endpoint_difficulty_columns(df):
    df['endpoint_difficulty_tier'] = [endpoint_difficulty_tier(_titles(n)) for n in df['NCT_ID']]
    return df
