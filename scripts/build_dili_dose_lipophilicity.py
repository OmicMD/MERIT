#!/usr/bin/env python3
"""Append leak-safe dose x lipophilicity iDILI features (Chen 'rule of two').

Biology (notes/idiosyncratic_dili_mechanism_stash_jun13.md): idiosyncratic DILI risk is
first-order driven by daily dose x lipophilicity — high hepatic exposure of a lipophilic
molecule -> larger reactive-metabolite/parent burden. Chen et al. Hepatology 2013: daily
dose >=100 mg AND logP >=3 marks high iDILI risk. Validated here by the endothelin-antagonist
natural experiment: ambrisentan/macitentan (10 mg, low) safe; bosentan/sitaxentan (250/100 mg,
lipophilic) hepatotoxic. This is the dose-aware axis the dose-BLIND structural DILI classifier
(tdc_dili) lacked.

Leak-safe + measured: daily dose (protocol) and logP (Crippen from SMILES) are pre-trial molecular
/ design facts. Adds (all routed to the noisy-OR hepatic detector + dose-protected in
retrain_calibrated.py via the `dili_` prefix):
  dili_logp          - Crippen MolLogP of the trial SMILES
  dili_rule_of_two   - 1.0 if max_daily_dose_mg>=100 and logP>=3 else 0.0
  dili_dose_x_logp   - log10(max_daily_dose_mg) * max(logP,0)   (continuous interaction)

Caveat (known, see stash note): the feature is ABSORPTION/ROUTE-blind — it over-flags
gut-restricted (fidaxomicin) or locally-injected (deoxycholic acid) high-logP drugs that are
not systemically exposed. A drumap_papp_caco2 / bioavailability gate is the next refinement.
It is a PRIOR-shift on the high-dose-lipophilic stratum, NOT a fix for the confident-near-zero
idiosyncratic misses (those need the fatal-vs-managed HLA layer).

Usage: python scripts/build_dili_dose_lipophilicity.py <in_csv> [out_csv]
Re-runnable: drops any existing dili_* columns first.
"""
import sys, numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
import warnings; warnings.filterwarnings("ignore")


def _logp(s):
    m = Chem.MolFromSmiles(str(s))
    return Crippen.MolLogP(m) if m else np.nan


def add_dili_columns(df):
    """Append leak-safe dose x lipophilicity iDILI features to a dataframe with SMILES +
    max_daily_dose_mg columns. Re-runnable (drops existing dili_* first). Importable by
    build_v8_honest_exposure.py. NOTE: an absorption (Caco-2) gate was evaluated and rejected
    (jun13) — it does not fix the 2 absorption-blind false-alarms (fidaxomicin NaN, deoxycholic
    acid locally-injected) and would wrongly down-weight low-Caco-2 real hepatotoxins (bosentan)."""
    df = df.drop(columns=[c for c in df.columns if c.startswith("dili_")], errors="ignore")
    uniq = df.drop_duplicates("SMILES")[["SMILES"]].copy()
    uniq["dili_logp"] = uniq["SMILES"].map(_logp)
    df = df.merge(uniq, on="SMILES", how="left")
    dose = df["max_daily_dose_mg"]
    dose_imp = dose.fillna(dose.median())
    df["dili_rule_of_two"] = ((dose_imp >= 100) & (df["dili_logp"] >= 3)).astype(float)
    df["dili_dose_x_logp"] = np.log10(dose_imp.clip(lower=1)) * df["dili_logp"].clip(lower=0)
    return df


if __name__ == "__main__":
    in_csv = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else in_csv
    d = add_dili_columns(pd.read_csv(in_csv, low_memory=False))
    d.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  (+dili_logp, dili_rule_of_two, dili_dose_x_logp; logP cov "
          f"{d['dili_logp'].notna().mean():.2f})")
