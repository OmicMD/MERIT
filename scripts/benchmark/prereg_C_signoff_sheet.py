#!/usr/bin/env python3
"""
Build ONE current human-signoff sheet for the prospective registration.

Joins the committed human verdicts (prereg_C_residual_verdicts_v1.csv) to the CURRENT
post-confidence-gate state (confident bets + abstained) so the reviewer sees, for each
verdict, whether it still affects a live confident bet — and only spends attention where
it matters. Replaces the stale ad-hoc prereg_C_ALL_verdicts_for_signoff.xlsx.

Out: results/benchmark/prereg_C/prereg_C_signoff_current.xlsx (+ .csv)
     sorted REVIEW-PRIORITY first.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "results/benchmark/prereg_C"


def norm(s):
    return str(s).lower().strip()


def main():
    v = pd.read_csv(ROOT / "data/sources/prereg_C_residual_verdicts_v1.csv")
    bets = pd.read_csv(P / "prereg_C_confident_bets.csv")
    abst = pd.read_csv(P / "prereg_C_abstained.csv")
    bet_keys = {(norm(d), norm(i)) for d, i in zip(bets.drug, bets.novel_indication)}
    abst_keys = {(norm(d), norm(i)) for d, i in zip(abst.drug, abst.novel_indication)}
    fail_keys = {(norm(d), norm(i)) for d, i, b in
                 zip(bets.drug, bets.novel_indication, bets.bet) if b == "FAIL"}

    def status(r):
        k = (norm(r.drug), norm(r.indication))
        if k in fail_keys:
            return "LIVE — confident FAIL bet"
        if k in bet_keys:
            return "LIVE — confident PASS bet"
        if k in abst_keys:
            return "abstained by gate (verdict no longer affects a bet)"
        return "filtered out of funnel (not a current bet)"

    v["current_status"] = v.apply(status, axis=1)
    # review priority: a verdict needs human attention if it's uncertain AND still live,
    # or it's a vague-indication ambiguity, or it's a FAIL bet (the against-base-rate calls).
    def priority(r):
        live = r.current_status.startswith("LIVE")
        if r.current_status == "LIVE — confident FAIL bet":
            return "1-REVIEW (live FAIL bet)"
        if r.verdict == "KEEP-AMBIG":
            return "2-REVIEW (ambiguous indication)"
        if live and r.confidence in ("med", "low"):
            return "2-REVIEW (live, low-confidence verdict)"
        if live:
            return "3-spot-check (live, high-confidence verdict)"
        return "4-no action (not live)"

    v["review_priority"] = v.apply(priority, axis=1)
    v = v.sort_values(["review_priority", "current_status", "drug"]).reset_index(drop=True)
    cols = ["review_priority", "current_status", "drug", "indication",
            "verdict", "confidence", "rationale", "source"]
    v = v[cols]

    out_xlsx = P / "prereg_C_signoff_current.xlsx"
    out_csv = P / "prereg_C_signoff_current.csv"
    v.to_csv(out_csv, index=False)
    try:
        v.to_excel(out_xlsx, index=False, engine="openpyxl")
        print(f"wrote {out_xlsx.relative_to(ROOT)}")
    except Exception as e:
        print(f"(xlsx skipped: {e}); wrote {out_csv.relative_to(ROOT)}")

    print(f"\n{len(v)} committed verdicts. Review burden by priority:")
    print(v.review_priority.value_counts().sort_index().to_string())
    print("\n=== rows that actually need your eyes (priority 1–2) ===")
    need = v[v.review_priority.str.startswith(("1", "2"))]
    for r in need.itertuples():
        print(f"  [{r.review_priority}] {r.drug} -> {r.indication}  "
              f"({r.verdict}/{r.confidence}; {r.current_status})")


if __name__ == "__main__":
    main()
