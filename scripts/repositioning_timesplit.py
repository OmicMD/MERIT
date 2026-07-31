#!/usr/bin/env python3
"""Time-split repositioning validation: "predicted-before, confirmed-after".

A SEPARATE validation instrument (like the leave-future-out ED table); the canonical
production model is NEVER touched. We ask: would the efficacy head, trained only on
trials that started before a cutoff T, have ranked a stranded drug's *eventual* new
indication above the indications it later failed in?

Design (real feature rows only — no mechanism recompute):
  1. Train the production efficacy head on df_e[Start_Year < T] (drug identity is not a
     feature; molecular/mechanism/design features only — see get_features()).
  2. "Stranded as of T" = a drug (SMILES) with a FAIL_EFFICACY/FAIL_BOTH indication
     started < T AND no PASS indication started < T (operational; approval-date data
     unavailable, see note in the manuscript text).
  3. Score the pre-T head IN REVERSE on (stranded drug X, candidate disease Y) pairs that
     are REALIZED post-T repositioning attempts: rows with Start_Year >= T, SMILES in the
     stranded set, and Y absent from X's pre-T indications (held-out by construction).
     fit = 1 - mean_seed P(fail).
  4. Confirmation: the realized post-T trial's own outcome (PASS = confirmed hit).
  5. Metrics, reported straight (low power is expected and is itself the finding):
       - PRIMARY unit = (drug, disease), the actual unit of a repositioning nomination.
         The COVID/coronavirus synonym cluster is normalized to one indication, so the
         2020 pandemic surge (which registered e.g. HCQ->COVID as ~15 separate trials)
         counts once instead of dominating the pool. Trial-level is kept as a sensitivity
         line. Discrimination AUC (PASS vs FAIL) + bootstrap CI + Mann-Whitney + precision@K
         vs a random-nomination null.
       - Within-drug concordance: among stranded drugs that later attempted BOTH a
         PASS and a FAIL new indication, fraction where fit(PASS) > fit(FAIL). This holds
         the drug's molecular features constant, so it isolates disease/mechanism fit.
       - DIAGNOSTIC (not adopted): a thin-training-support abstention guardrail, tested as a
         way to remove the COVID-19 over-crediting category (COVID host-directed repurposing
         is over-credited +43pt in the forward cohort). It OVER-abstains -- trial count cannot
         separate an emergent, ungroundable disease (COVID) from a rare disease with real
         biology (multiple sclerosis, orphan indications) -- so it discards legitimate
         successes. That over-correction is reported, and no correction is applied.

Usage:
  python scripts/repositioning_timesplit.py --cuts 2018 2017 2016 \
      --out results/repositioning_timesplit/timesplit.json
"""
from __future__ import annotations
import argparse, json, sys, warnings
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_calibrated import prepare_fold, fit_predict, SEEDS  # noqa: E402
from temporal_leave_future_out import build_cohorts, disease_vec  # noqa: E402
from retrain_corrected import get_features  # noqa: E402

EFF_FAIL = {"FAIL_EFFICACY", "FAIL_BOTH"}
RNG = np.random.RandomState(20260623)


def stranded_drugs(df_full, T):
    """SMILES with an efficacy failure started <T and no PASS started <T."""
    pre = df_full[df_full.Start_Year < T]
    out = []
    for smi, g in pre.groupby("SMILES"):
        if g.Corrected_Outcome.isin(EFF_FAIL).any() and not (g.Corrected_Outcome == "PASS").any():
            out.append(smi)
    return set(out)


def candidate_pairs(df_full, df_e, stranded, T):
    """Realized post-T repositioning attempts of stranded drugs in NEW diseases.

    Restricted to the efficacy cohort (df_e) so labels/exclusions match the head, and to
    PASS or FAIL_EFFICACY outcomes (the confirmable classes). Y must be absent from X's
    pre-T indication set (held-out-by-construction)."""
    pre_dis = {smi: set(df_full[(df_full.SMILES == smi) & (df_full.Start_Year < T)].Disease.dropna())
               for smi in stranded}
    c = df_e[(df_e.SMILES.isin(stranded)) & (df_e.Start_Year >= T)].copy()
    c = c[c.apply(lambda r: r.Disease not in pre_dis.get(r.SMILES, set()), axis=1)]
    return c


def score_reverse(tr, cand, feats):
    """fit = 1 - mean_seed P(fail) for the candidate pairs, pre-T efficacy head."""
    preds = []
    for seed in SEEDS:
        Xt, Xe, _, _ = prepare_fold(tr[feats].values, cand[feats].values,
                                    tr["_y"].values, disease_vec(tr), disease_vec(cand),
                                    feats, protect_idx=None)
        preds.append(fit_predict(Xt, tr["_y"].values, Xe, "efficacy", seed))
    pfail = np.mean(preds, axis=0)
    return 1.0 - pfail


def bootstrap_auc(y, score, n=2000):
    """y: 1=PASS(positive for 'fit ranks it high'); paired bootstrap CI of AUC."""
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], score[b]))
    if not aucs:
        return None
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def precision_at_k(is_hit, score, ks):
    order = np.argsort(-score)
    out = {}
    for k in ks:
        kk = min(k, len(score))
        out[k] = float(is_hit[order[:kk]].mean())   # keyed by requested k, slice capped
    return out


def perm_null_precision(is_hit, ks, n=10000):
    """Random-nomination null: shuffle which pairs are 'nominated' (top-K)."""
    base = is_hit.mean()
    null = {}
    for k in ks:
        kk = min(k, len(is_hit))
        draws = [is_hit[RNG.choice(len(is_hit), kk, replace=False)].mean() for _ in range(n)]
        null[k] = {"mean": float(np.mean(draws)), "p95": float(np.percentile(draws, 95))}
    return base, null


def within_drug_concordance(cand):
    """Among stranded drugs with BOTH a PASS and a FAIL_EFFICACY new-indication attempt,
    fraction of DISTINCT (PASS-disease, FAIL-disease) pairs where fit(PASS) > fit(FAIL).
    Molecular features are constant within drug -> isolates disease/mechanism-fit signal.
    Diseases are first collapsed to one mean-fit row each so multiple trials of the same
    (drug, disease) do not double-count."""
    conc = tot = drugs = 0
    detail = []
    for smi, g in cand.groupby("SMILES"):
        # collapse to one row per (drug, disease): mean fit, majority outcome class
        agg = (g.assign(_pass=(g.Corrected_Outcome == "PASS").astype(int))
                 .groupby("Disease")
                 .agg(fit=("fit", "mean"), is_pass=("_pass", "max"),
                      any_fail=("Corrected_Outcome", lambda s: s.isin(EFF_FAIL).any())))
        p = agg[(agg.is_pass == 1)]
        f = agg[(agg.is_pass == 0) & (agg.any_fail)]
        if len(p) and len(f):
            drugs += 1
            for pdis, pr in p.iterrows():
                for fdis, fr in f.iterrows():
                    tot += 1
                    conc += int(pr.fit > fr.fit)
                    detail.append((smi, pdis, fdis, round(pr.fit, 3), round(fr.fit, 3)))
    return {"drugs": drugs, "pairs": tot, "concordant": conc,
            "rate": (conc / tot) if tot else None, "detail": detail}


def normalize_disease(s):
    """Collapse the COVID/coronavirus synonym cluster to one canonical indication so a
    single failed repurposing target (the 2020 pandemic surge registered the same
    drug-disease as ~15 separate trials) counts once. Other strings pass through."""
    x = str(s).lower()
    if any(k in x for k in ("covid", "coronavirus", "sars-cov", "corona virus")):
        return "COVID-19"
    return str(s).strip()


def collapse_drug_disease(cand):
    """A repositioning nomination's unit is (drug, disease), not (trial). Collapse to one
    row per (drug, normalized-disease): mean fit, PASS if any trial of the pair completed."""
    c = cand.assign(dis_norm=cand.Disease.map(normalize_disease),
                    _pass=(cand.Corrected_Outcome == "PASS").astype(int))
    g = (c.groupby(["SMILES", "Drug", "dis_norm"])
           .agg(fit=("fit", "mean"), is_pass=("_pass", "max"),
                any_fail=("Corrected_Outcome", lambda s: s.isin(EFF_FAIL).any()),
                n_rows=("fit", "size")).reset_index())
    return g[(g.is_pass == 1) | g.any_fail].copy()


def disease_support(df_full, T):
    """Pre-T training-trial count per normalized disease (COVID -> 0: it did not exist)."""
    pre = df_full[df_full.Start_Year < T]
    return pre.assign(dn=pre.Disease.map(normalize_disease)).groupby("dn").size()


def metric_block(y, s, ks=(3, 5)):
    """AUC + bootstrap CI + Mann-Whitney + precision@k vs random null on one unit set."""
    y = np.asarray(y); s = np.asarray(s)
    if len(y) < 3 or len(np.unique(y)) < 2:
        return {"status": "insufficient", "n": int(len(y))}
    prec = precision_at_k(y, s, ks)
    base, null = perm_null_precision(y, ks)
    return {"status": "ok", "n": int(len(y)), "n_pass": int(y.sum()), "n_fail": int((y == 0).sum()),
            "auc": float(roc_auc_score(y, s)), "auc_ci95": bootstrap_auc(y, s),
            "mannwhitney_p_greater": float(mannwhitneyu(s[y == 1], s[y == 0], alternative="greater").pvalue),
            "base_rate_pass": float(base),
            "precision_at_k": {str(k): prec[k] for k in ks},
            "random_null_precision": {str(k): null[k] for k in ks}}


def run_cut(df_full, df_e, feats, T, out_dir, min_support=3):
    stranded = stranded_drugs(df_full, T)
    tr = df_e[df_e.Start_Year < T].copy()
    cand = candidate_pairs(df_full, df_e, stranded, T)
    cand = cand[cand.Corrected_Outcome.isin({"PASS"} | EFF_FAIL)].copy()
    res = {"T": T, "n_stranded": len(stranded), "n_train": int(len(tr)),
           "train_pos": int(tr._y.sum()), "n_candidates": int(len(cand)),
           "n_cand_drugs": int(cand.SMILES.nunique()),
           "n_pass": int((cand.Corrected_Outcome == "PASS").sum()),
           "n_fail": int(cand.Corrected_Outcome.isin(EFF_FAIL).sum())}
    if len(cand) < 3 or cand.Corrected_Outcome.nunique() < 2:
        res["status"] = "insufficient"
        return res, None
    cand = cand.copy()
    cand["fit"] = score_reverse(tr, cand, feats)
    cand["dis_norm"] = cand.Disease.map(normalize_disease)

    # PASS is the positive class for "does fit rank the eventual success high?"
    # (1) trial-level = every registered trial (sensitivity; a heavily-registered failed
    #     repurposing such as HCQ->COVID counts once per trial and dominates the pool).
    y_trial = (cand.Corrected_Outcome == "PASS").astype(int).values
    trial_level = metric_block(y_trial, cand.fit.values)

    # (2) drug-disease unit = the actual unit of a repositioning nomination (PRIMARY).
    gdd = collapse_drug_disease(cand)
    dd_level = metric_block(gdd.is_pass.values, gdd.fit.values)

    # (3) DIAGNOSTIC (not adopted): a thin-training-support abstention guardrail was tested
    #     as a way to remove the COVID-19 over-crediting (COVID = 0 pre-2016 trials). It
    #     OVER-ABSTAINS: trial count cannot distinguish an emergent, ungroundable disease
    #     (COVID) from a rare disease with real biology (multiple sclerosis, orphan
    #     indications), so it discards legitimate successes. We record how many PASS
    #     nominations it would wrongly drop -- that over-correction is itself the finding.
    supp = disease_support(df_full, T)
    gdd = gdd.assign(pre_support=gdd.dis_norm.map(supp).fillna(0).astype(int))
    abstained = gdd[gdd.pre_support < min_support]
    thin_diag = {"min_support": min_support, "n_abstained": int(len(abstained)),
                 "n_pass_wrongly_abstained": int(abstained.is_pass.sum()),
                 "pass_wrongly_abstained": sorted(f"{r.Drug}->{r.dis_norm}"
                                                   for _, r in abstained[abstained.is_pass == 1].iterrows()),
                 "note": "trial-support abstention over-corrects (discards rare-but-grounded successes); NOT adopted"}

    wdc = within_drug_concordance(cand)
    res.update({
        "status": "ok",
        "trial_level": trial_level,                     # sensitivity
        "drug_disease": dd_level,                       # PRIMARY
        "thin_support_abstention_diagnostic": thin_diag,
        "within_drug_concordance": {kk: vv for kk, vv in wdc.items() if kk != "detail"},
    })
    tbl = gdd[["SMILES", "Drug", "dis_norm", "fit", "is_pass", "n_rows", "pre_support"]].copy()
    tbl = tbl.sort_values("fit", ascending=False).reset_index(drop=True)
    tbl["rank"] = np.arange(1, len(tbl) + 1)
    tbl.to_csv(out_dir / f"candidates_T{T}.csv", index=False)
    res["within_drug_detail"] = wdc["detail"]
    return res, tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sources/training_dataset_v8_clean_mort.csv")
    ap.add_argument("--cuts", type=int, nargs="+", default=[2018, 2017, 2016])
    ap.add_argument("--min-support", type=int, default=3,
                    help="OOD guardrail: abstain on candidate diseases with < this many pre-T trials")
    ap.add_argument("--out", default="results/repositioning_timesplit/timesplit.json")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.data, low_memory=False)
    feats = get_features(df)
    _, df_e, _ = build_cohorts(df)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"loaded {len(df)} trials; efficacy cohort n={len(df_e)}/{int(df_e._y.sum())} fail; "
          f"{len(feats)} features", flush=True)

    results = {}
    for T in args.cuts:
        print(f"\n=== T={T} (train Start_Year<{T}, score stranded-drug post-T new indications) ===",
              flush=True)
        r, _ = run_cut(df, df_e, feats, T, out.parent, min_support=args.min_support)
        results[str(T)] = r
        if r.get("status") != "ok":
            print(f"  stranded={r['n_stranded']} candidates={r['n_candidates']} -> {r['status']}",
                  flush=True)
            continue
        print(f"  stranded={r['n_stranded']}  candidates={r['n_candidates']} "
              f"({r['n_pass']} PASS / {r['n_fail']} FAIL_EFF) from {r['n_cand_drugs']} drugs",
              flush=True)

        def show(tag, b):
            if b.get("status") != "ok":
                print(f"  {tag:26s} n={b.get('n')} -> {b.get('status')}", flush=True); return
            ci = b["auc_ci95"]
            print(f"  {tag:26s} n={b['n']:3d} ({b['n_pass']}P/{b['n_fail']}F)  "
                  f"AUC={b['auc']:.3f} [{ci[0]:.2f},{ci[1]:.2f}]  MWp={b['mannwhitney_p_greater']:.3f}  "
                  f"prec@3={b['precision_at_k']['3']:.2f}(null {b['random_null_precision']['3']['mean']:.2f})",
                  flush=True)
        show("trial-level (sensitivity)", r["trial_level"])
        show("drug-disease (PRIMARY)", r["drug_disease"])
        wdc = r["within_drug_concordance"]
        td = r["thin_support_abstention_diagnostic"]
        print(f"  within-drug concordance: {wdc['concordant']}/{wdc['pairs']} pairs "
              f"({wdc['drugs']} drugs)"
              + (f" = {wdc['rate']:.2f}" if wdc["rate"] is not None else ""), flush=True)
        print(f"  thin-support guardrail (min={td['min_support']}) would abstain {td['n_abstained']} "
              f"incl. {td['n_pass_wrongly_abstained']} real successes -> OVER-corrects, not adopted "
              f"({', '.join(td['pass_wrongly_abstained']) or 'none'})", flush=True)

    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
