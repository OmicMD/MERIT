#!/usr/bin/env python3
"""Extend is_anti_pathogen to the verified direct-antipathogen drugs missing from
the flag (Jun 6 2026, L1c data hygiene). Classification by anchored LLM judgment
(host-vs-pathogen teaching examples, 3 agents) — NOT keyword matching. Drugs whose
efficacy target is a microbe/virus/parasite/fungus protein (outside the human
proteome) are invisible to the human-target pipeline → exclude from efficacy.
HOST-directed drugs in infectious settings (dexamethasone-COVID, baricitinib,
ARBs, statins, vasopressors) are KEPT — they carry real, model-visible efficacy.

Output: data/sources/anti_pathogen_extension_jun6.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/sources/anti_pathogen_extension_jun6.csv"

# unambiguous DIRECT_ANTIPATHOGEN — flag ALL efficacy rows (antimicrobial wherever it appears)
ALL_INDICATIONS = {
    "abacavir": "HIV RT", "asunaprevir": "HCV NS3/4A protease", "boceprevir": "HCV protease",
    "brincidofovir": "viral DNA pol", "cefazolin": "bacterial PBP", "cefiderocol": "bacterial PBP",
    "cefotaxime": "bacterial PBP", "ceftriaxone": "bacterial PBP", "ciprofloxacin": "bacterial gyrase",
    "cloxacillin": "bacterial PBP", "daclatasvir": "HCV NS5A", "daptomycin": "bacterial membrane",
    "dasabuvir": "HCV NS5B", "delamanid": "Mtb mycolic acid", "D-Mannose": "E. coli FimH adhesin",
    "dolutegravir": "HIV integrase", "efavirenz": "HIV RT", "emtricitabine": "HIV RT",
    "eravacycline": "bacterial 30S", "ertapenem": "bacterial PBP", "ethambutol": "Mtb arabinosyl transferase",
    "famciclovir": "HSV/VZV DNA pol", "fluconazole": "fungal CYP51", "ganciclovir": "CMV DNA pol",
    "isoniazid": "Mtb InhA", "lamivudine": "HIV RT/HBV pol", "lopinavir": "HIV protease",
    "mebendazole": "helminth b-tubulin", "metronidazole": "anaerobe/protozoan DNA",
    "moxifloxacin": "bacterial gyrase", "nitrofurantoin": "bacterial enzymes", "pentamidine": "Pneumocystis",
    "piperacillin": "bacterial PBP", "plazomicin": "bacterial 30S", "pretomanid": "Mtb",
    "primaquine": "Plasmodium", "pyrazinamide": "Mtb", "quinine": "Plasmodium heme",
    "raltegravir": "HIV integrase", "remdesivir": "SARS-CoV-2 RdRp", "rilpivirine": "HIV RT",
    "ritonavir": "HIV protease", "stavudine": "HIV RT", "sulbactam": "bacterial b-lactamase",
    "tafenoquine": "Plasmodium", "tazobactam": "bacterial b-lactamase", "tedizolid phosphate": "bacterial 50S",
    "telbivudine": "HBV pol", "tenofovir alafenamide": "HIV RT", "valacyclovir": "HSV DNA pol",
    "valganciclovir": "CMV DNA pol", "vancomycin": "bacterial cell wall", "zidovudine": "HIV RT",
    # composite-named anti-pathogen rows missed by clean-name matching (found via confident-FN scan):
    "antimicrobial therapy: co-trimoxazole or doxycycline": "bacterial (CleanUP-IPF microbiome hypothesis; IPF)",
}
# indication-specific: (drug, rule) — rule(disease_str)->True means flag THAT row
def _has(s, *terms): return any(t in str(s).lower() for t in terms)
INDICATION_SPECIFIC = {
    # chloroquine: antimalarial=antipathogen; COVID=host (proposed endosomal) -> keep COVID
    "chloroquine": ("Plasmodium heme (malaria only; COVID=host, kept)",
                    lambda d: _has(d, "malaria", "vivax", "knowlesi", "falciparum", "plasmodium") and not _has(d, "covid", "sars", "corona")),
    # oseltamivir: influenza/COVID=viral neuraminidase; NASH=host NEU1 -> keep NASH
    "oseltamivir": ("viral neuraminidase (influenza/COVID; NASH=host NEU1, kept)",
                    lambda d: _has(d, "influenza", "covid", "flu") and not _has(d, "steatohep", "nash", "nafld")),
    # sertraline: cryptococcal meningitis=direct antifungal; other uses=host SERT
    "sertraline": ("Cryptococcus translation (cryptococcal/fungal only)",
                   lambda d: _has(d, "cryptococc", "fungal", "mening")),
}
# Explicitly KEPT (host-directed or contested) — documented, NOT flagged:
KEPT_HOST = ("dexamethasone/hydrocortisone/methylprednisolone/prednis* (host GR), baricitinib/ruxolitinib (host JAK), "
             "ARBs azilsartan/candesartan/olmesartan/valsartan/irbesartan/eprosartan/telmisartan/losartan (host AT1R), "
             "rosuvastatin/pitavastatin (host HMG-CoA), norepinephrine/dobutamine (host adrenergic), colchicine/apremilast/"
             "celecoxib/leflunomide/metformin/imatinib-COVID/fostamatinib (host immune), cilastatin (host DHP-1), "
             "lonafarnib-HDV (host farnesyltransferase), fluorouracil/imiquimod-HPV (host), tamsulosin/tramadol/ketamine/"
             "sevoflurane/vitamin-D3/varenicline/methadone/naltrexone/bupropion (host); CONTESTED kept: hydroxychloroquine-COVID, "
             "niclosamide-COVID, nitazoxanide-COVID/flu (host/antiviral hypothesis, repurposing fails).")

rows = []
for d, tgt in ALL_INDICATIONS.items():
    rows.append((d, "ALL", "DIRECT_ANTIPATHOGEN", tgt))
for d, (tgt, _) in INDICATION_SPECIFIC.items():
    rows.append((d, "INDICATION_SPECIFIC", "DIRECT_ANTIPATHOGEN", tgt))
ext = pd.DataFrame(rows, columns=["Drug_Clean", "scope", "classification", "target_rationale"])
ext["kept_host_note"] = KEPT_HOST
ext.to_csv(OUT, index=False)
print(f"Wrote {len(ext)} drug rules -> {OUT}")
print(f"  {len(ALL_INDICATIONS)} all-indication + {len(INDICATION_SPECIFIC)} indication-specific")
