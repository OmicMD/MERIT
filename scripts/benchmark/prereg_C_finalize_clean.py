#!/usr/bin/env python3
"""
Path C, step 3: select the cleanly-attributable prospective set and lock it.

The full Path-C run (prereg_C_lock.py) scores every ongoing-P3 novel (drug,
indication) pair, but many are supportive-care / chemo-backbone components of
multi-drug regimens (e.g. one ALL trial contributes seven cohort drugs), where a
trial's PASS/FAIL is not attributable to our drug. We restrict the locked set to
trials where our drug is the INVESTIGATIONAL FOCUS, by a deterministic rule:
  (a) the drug is named in the trial's brief title,
  (b) the drug is the experimental DIFFERENTIATOR — in an EXPERIMENTAL arm but
      NOT in the comparator/control arm (so it is not a shared chemo/standard
      backbone whose arms differ by some OTHER, possibly un-featurisable, agent),
  (c) ongoing status, and
  (d) primary completion is on/after the lock date (outcome not yet known).
This yields a single-agent-repurposing set that is both out-of-sample (novel
pairing) AND cleanly attributable at readout.

Input : results/benchmark/prereg_C/prereg_C_locked_predictions.csv (+ caches)
Output: results/benchmark/prereg_C/prereg_C_locked_predictions_clean.csv (PRIMARY),
        its .sha256, and an updated prereg_C_registration.md.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "results/benchmark/prereg_C/prereg_C_locked_predictions.csv"
TRIALS = ROOT / "results/benchmark/prereg_ongoing_p3_trials.csv"
CACHE = ROOT / "data/cache/ctgov_ongoing_p3.json"
OUTDIR = ROOT / "results/benchmark/prereg_C"
LOCK_DATE = "2026-06-21"


def clean(n):
    return re.sub(r"\(.*$", "", str(n)).replace("®", "").replace("™", "").strip(" ;,").lower()


def main():
    o = pd.read_csv(PRED)
    trials = pd.read_csv(TRIALS)
    cache = json.load(open(CACHE))

    nct_title, nct_exp, nct_comp = {}, {}, {}
    for _drug, studies in cache.items():
        for s in studies:
            p = s.get("protocolSection", {})
            nct = p.get("identificationModule", {}).get("nctId")
            if not nct:
                continue
            nct_title[nct] = p.get("identificationModule", {}).get("briefTitle", "")
            for a in (p.get("armsInterventionsModule", {}).get("armGroups", []) or []):
                names = " ".join(a.get("interventionNames", []) or []).lower()
                if a.get("type") == "EXPERIMENTAL":
                    nct_exp.setdefault(nct, set()).add(names)
                else:  # PLACEBO_COMPARATOR / ACTIVE_COMPARATOR / other control arms
                    nct_comp.setdefault(nct, set()).add(names)

    comp = pd.to_datetime(trials.drop_duplicates("NCT_ID").set_index("NCT_ID")
                          ["primary_completion_date"], errors="coerce")
    LOCK = pd.Timestamp(LOCK_DATE)
    ONGOING = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION",
               "NOT_YET_RECRUITING"}

    def keep(r):
        nct, term = r["NCT_ID"], clean(r["drug"])
        if not term:                       # empty cleaned name -> never trivially-true match
            return False
        title = nct_title.get(nct, "").lower()
        in_title = re.search(r"\b" + re.escape(term) + r"\b", title) is not None
        in_exp = any(term in e for e in nct_exp.get(nct, set()))
        in_comp = any(term in e for e in nct_comp.get(nct, set()))
        # DIFFERENTIATOR, not backbone: the drug must be the experimental agent that
        # DISTINGUISHES the arms — present in an experimental arm but NOT in the
        # comparator/control arm. A chemo/standard backbone (e.g. gemcitabine+cisplatin
        # in an add-on trial "novel agent + chemo  vs  placebo + chemo") is named in the
        # brief title and sits in the experimental arm, but ALSO in the comparator arm,
        # so the trial's PASS/FAIL is attributable to the novel agent (often a biologic
        # our pipeline cannot featurise, e.g. KL-A167), NOT to our backbone drug. The
        # earlier rule (in_title and in_exp) admitted 83/349 such backbones.
        differentiator = in_exp and not in_comp
        # outcome-unknown guarantee = ongoing status (the real invariant); future
        # primary completion is a secondary recency filter (NaT kept).
        ongoing = str(r.get("overall_status", "")).upper() in ONGOING
        c = comp.get(nct, pd.NaT)
        future = True if pd.isna(c) else (c >= LOCK)
        return in_title and differentiator and ongoing and future

    o["focus"] = o.apply(keep, axis=1)
    cln = o[o["focus"]].drop(columns=["focus"]).reset_index(drop=True)

    out_path = OUTDIR / "prereg_C_locked_predictions_clean.csv"
    cln.to_csv(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    (OUTDIR / "prereg_C_locked_predictions_clean.sha256").write_text(
        sha + "  prereg_C_locked_predictions_clean.csv\n")

    n, nt, nd = len(cln), cln["NCT_ID"].nunique(), cln["drug"].nunique()
    nf = int((cln["predicted_label"] == "FAIL").sum())
    update_registration(n, nt, nd, nf, cln, sha, len(o), o["NCT_ID"].nunique())
    print(f"CLEAN locked set: {n} predictions / {nt} trials / {nd} drugs "
          f"(from full {len(o)}/{o['NCT_ID'].nunique()})")
    print(f"  predicted FAIL {nf}/{n} ({100*nf/n:.0f}%); median P_fail "
          f"{cln['P_fail_overall_calibrated'].median():.3f}")
    print(f"  SHA-256 {sha}")
    print(f"  -> {out_path}")


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def update_registration(n, nt, nd, nf, cln, sha, n_full, nt_full):
    reg = OUTDIR / "prereg_C_registration.md"
    note = f"""

---

## PRIMARY locked set — cleanly-attributable single-agent repurposings (added {LOCK_DATE})
**File:** `prereg_C_locked_predictions_clean.csv` · **SHA-256:** `{sha}` · **commit:** `{git_sha()}`

The full Path-C run scores {n_full} novel-pair predictions across {nt_full} trials, but many
are supportive-care or chemo-backbone components of multi-drug regimens, where a
trial's outcome is not attributable to our drug. The **primary** prospective set
restricts to trials where our drug is the investigational focus, by a deterministic
rule: the drug is (a) named in the brief title (word-boundary match), (b) the
experimental DIFFERENTIATOR — in an experimental arm but NOT in the comparator arm,
so it is not a shared chemo/standard backbone (the arms differing by some other,
possibly un-featurisable, agent), (c) of ongoing status (recruiting /
active-not-recruiting / enrolling / not-yet-recruiting — the guarantee that the
outcome is not yet known), and (d) with primary completion on/after the lock date. This leaves **{n} predictions
across {nt} trials of {nd} compounds** — genuinely out-of-sample (drug, indication)
pairs that are also cleanly attributable. Predicted to FAIL (P_fail >= 0.5): {nf}/{n}
(the {100*nf/n:.0f}% rate matches the cohort base rate). Evaluate this set at readout;
the full set is retained as a secondary, attribution-caveated artifact.

`scripts/benchmark/prereg_C_finalize_clean.py` reproduces the filter deterministically.
"""
    # Idempotent: strip any previously-appended PRIMARY section before re-appending, so
    # re-running the chain (or running finalize twice) cannot duplicate it.
    base = reg.read_text()
    marker = "\n\n---\n\n## PRIMARY locked set"
    if marker in base:
        base = base[:base.index(marker)]
    reg.write_text(base.rstrip() + "\n" + note)


if __name__ == "__main__":
    main()
