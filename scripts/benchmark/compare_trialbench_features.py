#!/usr/bin/env python3
"""Feature head-to-head against the TrialBench approval benchmark, BioBERT-free.

On the NCT-ID intersection of our cohort with TrialBench's trial-approval-forecasting
task, predict TRIALBENCH'S approval label with a single GBM under compound-holdout
(group = first-14-char InChIKey), comparing three feature sets:
  (1) their trial-design tabular features (arm counts, masking, enrollment, sponsor...)
  (2) our pre-trial STAR mechanism features
  (3) both combined
This isolates feature-set value on identical trials / labels / classifier / split, and
sidesteps the benchmark's BioBERT text branch entirely. Note: their design features can
carry mild outcome leakage (e.g. enrollment of terminated trials), so matching them with
mechanism-only features is a conservative comparison in our favor.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
TB = ROOT / "data/external/trialbench/trial-approval-forecasting"
OUR = ROOT / "data/sources/training_dataset_v8_clean_mort.csv"
OUT = ROOT / "results/benchmark"; OUT.mkdir(parents=True, exist_ok=True)

# Their text/id columns to drop when forming the "trial-design tabular" feature set.
TEXT_COLS = {"Unnamed: 0", "brief_summary/textblock", "brief_title", "condition",
             "condition_browse/mesh_term", "eligibility/criteria/textblock",
             "intervention/description", "intervention/intervention_name",
             "intervention_browse/mesh_term", "keyword",
             "location/facility/address/city", "smiless", "icdcode", "phase"}


def load_trialbench():
    """Concatenate all phases (train+test), dedup by NCT. Return label + their tabular X."""
    xs, ys = [], []
    for ph in ["Phase1", "Phase2", "Phase3", "Phase4"]:
        for sp in ["train", "test"]:
            x = pd.read_csv(TB / ph / f"{sp}_x.csv").rename(columns={"Unnamed: 0": "nct"})
            y = pd.read_csv(TB / ph / f"{sp}_y.csv").rename(columns={"Unnamed: 0": "nct"})
            x["phase_tb"] = ph
            xs.append(x); ys.append(y[["nct", "outcome"]])
    X = pd.concat(xs, ignore_index=True).drop_duplicates("nct")
    Y = pd.concat(ys, ignore_index=True).drop_duplicates("nct")
    df = X.merge(Y, on="nct", how="inner")
    # their tabular design features: everything not text/id, coerced numeric (one-hot cats)
    feat = [c for c in X.columns if c not in TEXT_COLS and c not in ("nct", "phase_tb")]
    tab = df[feat].copy()
    num = tab.apply(pd.to_numeric, errors="coerce")
    keep_num = [c for c in num.columns if num[c].notna().mean() > 0.5]
    cat = [c for c in feat if c not in keep_num and tab[c].nunique() <= 12]
    Xtab = pd.concat([num[keep_num], pd.get_dummies(tab[cat].astype(str), dummy_na=True)], axis=1)
    Xtab.columns = [f"tb__{c}" for c in Xtab.columns]
    Xtab = Xtab.loc[:, ~Xtab.columns.duplicated()]   # dummy_na can collide; keep unique
    out = pd.concat([df[["nct", "outcome"]], Xtab], axis=1)
    return out, list(Xtab.columns)


def load_ours():
    d = pd.read_csv(OUR, low_memory=False)
    d = d.rename(columns={"NCT_ID": "nct"})
    # STAR feature block = numeric columns after the metadata head (col index >= 18),
    # excluding obvious non-features. Take numeric dtype columns only.
    meta = {"nct", "ClinicalTrials_URL", "Drug", "Phase", "Start_Year", "Enrollment",
            "Final_Outcome", "Confidence", "matched_approved_name", "Why_Stopped",
            "Trial_Title", "Drug_Clean", "Disease", "SMILES", "Is_Biologic", "Source",
            "Corrected_Outcome", "feature_IK"}
    star = [c for c in d.columns if c not in meta and pd.api.types.is_numeric_dtype(d[c])]
    d["ik14"] = d["feature_IK"].astype(str).str[:14]
    return d, star


def cv_auc(X, y, groups, seeds=(0, 1, 2), nsplit=5):
    aucs = []
    for s in seeds:
        skf = StratifiedGroupKFold(n_splits=nsplit, shuffle=True, random_state=s)
        for tr, te in skf.split(X, y, groups):
            if len(np.unique(y[te])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                               max_iter=300, random_state=s)
            m.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.mean(aucs), np.std(aucs), len(aucs)


def main():
    tb, tb_feats = load_trialbench()
    ours, star_feats = load_ours()
    print(f"TrialBench approval: {len(tb)} NCTs, {len(tb_feats)} design features")
    print(f"Our cohort: {ours.nct.nunique()} NCTs, {len(star_feats)} STAR features")

    j = ours.merge(tb, on="nct", how="inner").drop_duplicates("nct")
    print(f"\nNCT overlap: {len(j)} trials, {j.ik14.nunique()} unique compounds (IK14)")
    print(f"TrialBench label balance on overlap: {dict(j.outcome.value_counts())} "
          f"(approval rate {j.outcome.mean():.3f})")
    # agreement with our own label (sanity)
    our_pass = (j.Corrected_Outcome == "PASS").astype(int)
    print(f"Agreement TB-outcome vs our PASS: {(our_pass == j.outcome).mean():.3f}")

    y = j.outcome.astype(int).values
    g = j.ik14.values
    # Their single most outcome-leaky design feature is enrollment: terminated trials
    # enroll far fewer patients (reverse causation), so its predictive value is partly an
    # artifact of the outcome it predicts. Report their design with and without it.
    enroll_cols = [c for c in tb_feats if "enroll" in c.lower()]
    tb_noenroll = [c for c in tb_feats if c not in enroll_cols]
    print(f"\nenrollment cols dropped for the leak-controlled design set: {enroll_cols}")
    for c in enroll_cols:
        print(f"  median {c}: approved={j[c][y==1].median():.0f}  failed={j[c][y==0].median():.0f}")
    feat_sets = {
        "their design tabular (all)": tb_feats,
        "their design minus enrollment": tb_noenroll,
        "our STAR mechanism": star_feats,
        "STAR + their design minus enrollment": star_feats + tb_noenroll,
        "combined (all)": tb_feats + star_feats,
    }
    print("\nPredicting TrialBench approval label, compound-holdout 3x5 CV (mean-of-folds AUC):")
    rows = []
    for name, feats in feat_sets.items():
        X = j[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        m, sd, k = cv_auc(X, y, g)
        print(f"  {name:24s}  AUC {m:.3f} ± {sd:.3f}  ({len(feats)} feats, {k} folds)")
        rows.append(dict(feature_set=name, auc=m, sd=sd, n_feats=len(feats), n_folds=k))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "trialbench_approval_feature_headtohead.csv", index=False)
    print(f"\nwrote {OUT/'trialbench_approval_feature_headtohead.csv'}")
    print(f"overlap n={len(j)}, compounds={j.ik14.nunique()}, approval_rate={j.outcome.mean():.3f}")


if __name__ == "__main__":
    main()
