#!/usr/bin/env python3
"""Gap B (Jun 28): endpoint_cvevent_match — mechanism-match for atherothrombotic/cardioembolic
EVENT-RATE & secondary-prevention endpoints (MACE / CV-death / MI / recurrent-stroke / VTE).

This extends the endpoint-physiology mechanism-match idea from narrow STRUCTURAL surrogates to
discrete cardiovascular EVENTS, but as its OWN feature (not folded into endpoint_physiology_score,
because a +1 STRUCT match and a +1 CV-event match carry different absolute risk).

Construct (outcome-blind, set pre-trial from the ct.gov primary-outcome + the drug's MOA target):
  demand   = File A row 'cv_event_*' (endpoint_physiology_v1.csv) fires on an atherothrombotic/
             cardioembolic event endpoint in a cardiovascular/cerebrovascular/thrombotic disease;
             its demanded axis is the process 'atherothrombotic_cardiometabolic'.
  +1 MATCH    : the drug has >=1 MOA target on the atherothrombotic_cardiometabolic axis
                (anticoagulant F2/F10/VKORC1, antiplatelet P2RY12/PDE3A/ITGA2B/ITGB3,
                 LDL-lowering HMGCR/PCSK9/NPC1L1/APOB, SGLT2 SLC5A2, GLP1R) — File B.
  -1 MISMATCH : the drug HAS a curated MOA target but NONE is on that axis (it bets on an
                unvalidated CV pathway: anti-inflammatory colchicine/MTX, CETP, ranolazine, MRA...).
  0  UNKNOWN  : not a cv_event endpoint, OR the drug has no curated MOA target at all.

Why -1 requires a KNOWN-but-off-axis target (not 'no target'): a discrete CV event has a small set
of validated rate-limiting axes; a drug with a known target NOT among them is a genuine mechanism
bet against precedent. A targetless drug is genuinely unknown -> 0 (conservative).

Leak-validated (efficacy cohort, training_dataset_v8_clean_mort): MATCH n=32 fail 0.219 vs
MISMATCH n=13 fail 0.538, shuffle p=0.008, availability AUC 0.508 (<0.58), survives within-Phase-3.
Match does NOT zero residual risk (apixaban-ESUS, ticagrelor-acute-stroke still fail) — it lowers it.
Protected for efficacy/overall via the endpoint_ prefix in retrain_calibrated / prereg_C_lock.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts/strengthening'))
import endpoint_physiology_score as S  # noqa: E402  (provides demand(), tgtmap)

# the demanded axis token (curated in target_physiology_v1.csv)
AXIS = 'atherothrombotic_cardiometabolic'
# genes carrying the axis, read from File B so the curation stays in one place
_B = pd.read_csv(ROOT / 'data/sources/target_physiology_v1.csv')
ATH_GENES = {r.gene for _, r in _B.iterrows() if AXIS in str(r.physiological_process).split(';')}


def _score(nct, disease, ik14):
    ep, _dem = S.demand(nct, disease)
    if not (isinstance(ep, str) and ep.startswith('cv_event')):
        return 0
    ts = set(S.tgtmap.get(ik14, []))
    if not ts:
        return 0                       # targetless -> genuinely unknown
    return 1 if (ts & ATH_GENES) else -1


def add_cvevent_match_columns(df):
    ik14 = df['feature_IK'].astype(str).str[:14]
    df['endpoint_cvevent_match'] = [
        _score(nct, dis, k) for nct, dis, k in zip(df['NCT_ID'], df['Disease'], ik14)]
    return df


def main():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    inp = ROOT / 'data/sources/training_dataset_v8_clean_mort_physio.csv'
    if not inp.exists():
        inp = ROOT / 'data/sources/training_dataset_v8_clean_mort.csv'
    out = ROOT / 'data/sources/training_dataset_v8_clean_mort_cvevent.csv'
    df = pd.read_csv(inp, low_memory=False)
    df = add_cvevent_match_columns(df)
    print(f'axis genes (File B): {sorted(ATH_GENES)}')
    print(f'endpoint_cvevent_match dist: {df.endpoint_cvevent_match.value_counts().to_dict()}')
    # leak battery on the efficacy cohort
    eff = df[df.Corrected_Outcome.isin(['PASS', 'FAIL_EFFICACY', 'FAIL_BOTH'])].drop_duplicates('NCT_ID').copy()
    eff['fail'] = (eff.Corrected_Outcome != 'PASS').astype(int)
    act = eff[eff.endpoint_cvevent_match != 0]
    print('\n[discrimination]')
    print(act.groupby('endpoint_cvevent_match').agg(n=('fail', 'size'), fail=('fail', 'mean')).round(3).to_string())
    rng = np.random.default_rng(0)
    sc = act.endpoint_cvevent_match.values
    real = act[sc == -1].fail.mean() - act[sc == 1].fail.mean()
    null = np.array([act.fail.values[rng.permutation(sc) == -1].mean()
                     - act.fail.values[rng.permutation(sc) == 1].mean() for _ in range(5000)])
    print(f'[shuffle] mismatch-minus-match {real:+.3f} | p={(np.abs(null) >= abs(real)).mean():.4f}')
    print(f'[availability] scored!=0 vs fail AUC: '
          f'{roc_auc_score(eff.fail, (eff.endpoint_cvevent_match != 0).astype(int)):.3f} (want <0.58)')
    df.to_csv(out, index=False)
    print(f'\nwrote {out.name} (candidate; canonical untouched)')


if __name__ == '__main__':
    main()
