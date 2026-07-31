#!/usr/bin/env python3
"""Shared feature-completeness labelling for the Path C prereg deposit.

There is no such thing as a "targetless" drug — but a *target -> disease mechanism-fit*
feature genuinely does not apply to every drug. This assigns one honest label per
(drug, indication) pair from the recomputed biology, distinguishing three cases:

  COMPLETE   — disease gene module present AND the drug's protein target is resolved;
               all biological features (mechanism, direct-target, Mendelian/DepMap/impact)
               are computed for this pair.
  N/A        — the drug acts WITHOUT a druggable protein target (DNA-damaging cytotoxic,
               or a metabolic / cofactor / antioxidant MOA), so target->disease
               mechanism-fit is not applicable. Its recomputed mechanism values are a
               correct 0, not a gap. (e.g. melphalan->DNA, acetylcysteine/citrulline/
               thiamine.) Cytotoxics are additionally carried by the is_cytotoxic axis.
  INCOMPLETE — a genuine, resolvable DATA gap: either the disease has no Open-Targets
               gene module (needs an OT pull), or the drug has known targets that are
               not yet mapped into the OT target universe (needs target resolution).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MOA = ROOT / "data/sources/ik14_moa_targets_combined_v1.csv"

LABEL_COMPLETE = ("COMPLETE: all biology recomputed "
                  "(mechanism + direct-target + Mendelian/DepMap/impact)")
LABEL_NA = ("N/A: drug acts without a protein target "
            "(cytotoxic/DNA or metabolic MOA) — mechanism-fit not applicable")
LABEL_INC_DISEASE = ("INCOMPLETE: disease gene module unavailable in Open Targets "
                     "(needs disease pull)")
LABEL_INC_DRUG = ("INCOMPLETE: drug targets not mapped to Open Targets "
                  "(needs target resolution)")


def moa_ik14s():
    """IK14s with >= 1 curated/known protein target in the combined MOA file."""
    return set(pd.read_csv(MOA)["ik14"].dropna().astype(str))


def label(df, ik_col="IK14", moa_iks=None):
    """Return a feature_completeness label Series for df.

    df must carry coverage_disease, coverage_drug, direct_target_max and the IK14 column.
    """
    if moa_iks is None:
        moa_iks = moa_ik14s()
    cov_dis = df["coverage_disease"].fillna(0).astype(float) > 0
    cov_drug = (df["coverage_drug"].fillna(0).astype(float) > 0) & df["direct_target_max"].notna()
    has_moa = df[ik_col].astype(str).isin(moa_iks)
    no_protein_target = (~cov_drug) & (~has_moa)
    return pd.Series(np.select(
        [cov_dis & cov_drug, no_protein_target, ~cov_dis],
        [LABEL_COMPLETE, LABEL_NA, LABEL_INC_DISEASE],
        default=LABEL_INC_DRUG), index=df.index)


def summary(labels):
    n_complete = int(labels.eq(LABEL_COMPLETE).sum())
    n_na = int(labels.eq(LABEL_NA).sum())
    n_inc = int(labels.str.startswith("INCOMPLETE").sum())
    return (f"{n_complete} COMPLETE / {n_na} N/A (mechanism-fit not applicable) / "
            f"{n_inc} INCOMPLETE (resolvable data gap)")
