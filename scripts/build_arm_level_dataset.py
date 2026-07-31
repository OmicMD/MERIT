"""Build arm-level training dataset by aggregating per-drug pipeline features.

For each arm in trial_arms.csv:
- Parse All_InChIKeys → list of drug IK27s
- Look up each drug's pre-computed features (tissue/binding/tox/network/drumap)
- Aggregate row-wise: MAX across drugs per feature (cumulative-effect semantics)
- Add arm metadata + outcome + disease context

This is feature-level aggregation. It approximates raw-level (max per
transcript/target) for most binding features because each binding feature
is already a max/sum over a (organ × transcript × drug) cell.

Output: data/sources/training_dataset_arm_level.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources"
CACHE = ROOT / "data" / "cache"
MODELS = ROOT / "data" / "models"
OUT = SOURCES / "training_dataset_arm_level.csv"

# Feature-IK aliases (per-drug data edge cases, May 31 2026).
# A handful of arm drugs carry a full InChIKey whose stereochemistry/isotope layer
# differs from the pipeline feature row for the SAME connectivity (IK14), so the
# exact full-IK feature match silently misses them. Binding/tissue features are
# unaffected by these specific differences (deuteration; a single defined stereo-
# center vs. the racemate/parent), so we redirect ONLY the feature lookup to the
# equivalent compound while leaving each drug's true identity intact in All_*.
# Maps arm-drug IK27 -> feature-table IK27. This is a closed list of vetted edge
# cases, NOT a general stereo-collapse rule — true diastereomers that are distinct
# drugs (e.g. ephedrine vs. pseudoephedrine) are deliberately NOT listed.
FEATURE_IK_ALIASES = {
    # Donafenib (deuterated sorafenib) -> sorafenib. Deuteration is invisible to
    # structure-based binding prediction; same targets. (6 investigational arms)
    "MLDQJTXFUGDVEO-FIBGUPNXSA-N": "MLDQJTXFUGDVEO-UHFFFAOYSA-N",
    # Arbaclofen (R-baclofen) -> baclofen (undefined-stereo parent row). Same
    # GABA-B target engagement. (4 investigational arms)
    "KPYSYYIEGFHWSV-QMMMGPOBSA-N": "KPYSYYIEGFHWSV-UHFFFAOYSA-N",
    # Folinic acid / leucovorin -> the defined-stereo feature row present in the
    # pipeline. Mostly a supportive backbone; ~3 investigational arms.
    "VVIAGPKUTFNRDU-OLZOCXBDSA-N": "VVIAGPKUTFNRDU-ABLWVSNPSA-N",
}


def load_drug_features():
    """Load all per-drug feature CSVs and inner-join on feature_IK (27-char IK)."""
    from rdkit import Chem
    from rdkit.Chem import inchi as rdkinchi

    def smi_to_ik(smi):
        if not smi or pd.isna(smi): return None
        mol = Chem.MolFromSmiles(str(smi))
        if not mol: return None
        # No salt-stripping: keep consistent with the pipeline feature keys, which
        # use the same un-stripped lookup SMILES (see smiles_to_ik in decompose).
        inch = rdkinchi.MolToInchi(mol)
        return rdkinchi.InchiToInchiKey(inch) if inch else None

    feats = {}
    for name, fname in [
        ("tissue", "tissue_interaction_features_v5.csv"),
        ("binding", "binding_specificity_features_v5.csv"),
        ("tox", "toxicity_binding_features_v5.csv"),
        ("essential", "essential_gene_features_v5.csv"),
        ("network", "network_enrichment_features_v5.csv"),
    ]:
        p = MODELS / fname
        if not p.exists():
            print(f"  WARN: {fname} missing — skipping")
            continue
        df = pd.read_csv(p)
        if "InChIKey" in df.columns:
            df["feature_IK"] = df["InChIKey"].apply(lambda x: str(x)[:27] if pd.notna(x) else None)
        elif "SMILES" in df.columns:
            df["feature_IK"] = df["SMILES"].apply(smi_to_ik)
        if name == "tissue":
            # NULL organ interaction values mean zero binding (no targets above threshold),
            # NOT missing data. Imputing median (~0.25) falsely signals average binding.
            organ_null_zero = [c for c in df.columns if (
                any(c.startswith(p) for p in ("max_", "min_", "mean_")) and "_interaction" in c
            ) or c.startswith("weighted_score_")]
            df[organ_null_zero] = df[organ_null_zero].fillna(0)
        feats[name] = df

    # DruMAP separately (keyed by SMILES)
    drumap_path = SOURCES / "drumap_combined.csv"
    if drumap_path.exists():
        d = pd.read_csv(drumap_path)
        keep = ["smiles", "clint_reg", "fup_rat_reg", "fup_reg", "kpbrain_reg",
                "papp_caco2_reg", "vd_reg", "cyp1a2_prob", "cyp2c9_prob",
                "cyp2d6_prob", "cyp3a4_prob"]
        d = d[[c for c in keep if c in d.columns]].copy()
        d = d.rename(columns={
            "smiles": "SMILES", "clint_reg": "drumap_clint", "fup_rat_reg": "drumap_fup_rat",
            "fup_reg": "drumap_fup", "kpbrain_reg": "drumap_kpbrain",
            "papp_caco2_reg": "drumap_papp_caco2", "vd_reg": "drumap_vd",
            "cyp1a2_prob": "drumap_cyp1a2", "cyp2c9_prob": "drumap_cyp2c9",
            "cyp2d6_prob": "drumap_cyp2d6", "cyp3a4_prob": "drumap_cyp3a4",
        })
        for col in ["drumap_cyp1a2", "drumap_cyp2c9", "drumap_cyp2d6", "drumap_cyp3a4"]:
            if col in d.columns:
                d[col] = d[col].map({"substrate": 1, "non-substrate": 0}).astype(float)
        d["feature_IK"] = d["SMILES"].apply(smi_to_ik)
        feats["drumap"] = d

    # Outer-join all per-drug feature tables
    drug = None
    for name, df in feats.items():
        df = df[df["feature_IK"].notna()]
        cols = [c for c in df.columns if c not in ("SMILES", "InChIKey", "Drug_ID")]
        deduped = df[cols].drop_duplicates("feature_IK")
        if drug is None:
            drug = deduped
        else:
            drug = drug.merge(deduped, on="feature_IK", how="outer", suffixes=("", f"_{name}_dup"))
            drug = drug.drop(columns=[c for c in drug.columns if c.endswith("_dup")])
    print(f"Drug-level feature table: {len(drug)} drugs, {len(drug.columns)} columns")

    # SANITY GUARD (R7): the source v5 feature files contain leaked/template
    # rows where N drugs share identical values (most commonly score_max=1,
    # score_mean=0.298, drug_n_bound=4234). These propagate into training data.
    # When ≥5 drugs share the exact same Binding feature signature, NULL out
    # the corrupted feature group for those drugs so they're treated as
    # missing-features rather than as false signal.
    signature_cols = ["binding_score_max", "binding_score_mean", "binding_drug_n_bound"]
    if all(c in drug.columns for c in signature_cols):
        sig_counts = drug.groupby(signature_cols).size()
        bad_signatures = sig_counts[sig_counts >= 5].index.tolist()
        if bad_signatures:
            # Find all rows matching a bad signature
            n_nulled = 0
            for sig in bad_signatures:
                mask = (drug["binding_score_max"] == sig[0]) & \
                       (drug["binding_score_mean"] == sig[1]) & \
                       (drug["binding_drug_n_bound"] == sig[2])
                # Null the full row's pipeline features (treat as missing)
                # Keep feature_IK so the row still exists; only null the values.
                feat_cols_to_null = [c for c in drug.columns if c != "feature_IK"]
                drug.loc[mask, feat_cols_to_null] = None
                n_nulled += mask.sum()
            print(f"  ⚠ SANITY GUARD: nulled {n_nulled} drug rows with leaked/template features "
                  f"({len(bad_signatures)} distinct corrupted signatures, ≥5 drugs each). "
                  f"Affected drugs will appear as missing-features (correct), not false signal.")

    # Feature-IK aliases: duplicate the source compound's feature row under the
    # alias key so the downstream exact full-IK match finds it. Only adds a row
    # when the alias key is genuinely absent and the source row is present and
    # non-null (not nulled by the sanity guard above).
    present = set(drug["feature_IK"].dropna())
    for alias_ik, source_ik in FEATURE_IK_ALIASES.items():
        if alias_ik in present:
            continue  # arm drug already has its own feature row; no alias needed
        src = drug[drug["feature_IK"] == source_ik]
        if len(src) != 1:
            print(f"  WARN: feature-IK alias source {source_ik} not unique/found "
                  f"({len(src)} rows) — {alias_ik} NOT aliased")
            continue
        if src.drop(columns=["feature_IK"]).isna().all(axis=1).iloc[0]:
            print(f"  WARN: feature-IK alias source {source_ik} is all-null "
                  f"(nulled/empty) — {alias_ik} NOT aliased")
            continue
        new_row = src.copy()
        new_row["feature_IK"] = alias_ik
        drug = pd.concat([drug, new_row], ignore_index=True)
        print(f"  feature-IK alias applied: {alias_ik} ← {source_ik} (edge case)")

    return drug


def aggregate_arm_features(arm_iks_str: str, inv_iks_str: str | None,
                           drug_features: pd.DataFrame,
                           feature_cols: list[str]):
    """Aggregate per-drug pipeline features into a per-arm row.

    Aggregation rule (the systemic rule the project follows): MAX over the
    INVESTIGATIONAL drugs only when the arm has any (Investigational_InChIKeys
    non-empty). Approved comparators / chemo backbones in the same arm are
    excluded from the feature MAX — otherwise their (already-approved) feature
    signal contaminates the inv drug's representation and leaks survivorship.
    When the arm has no investigational drug (control / SOC arm), fall back
    to MAX over all delivered drugs (All_InChIKeys), since there is no inv
    candidate to anchor on.

    Returns (feature_row, reorder_indices, n_with_features, anchor_iks).
        feature_row: per-arm row of MAX-aggregated features, or None when no
            drug from the chosen scope has features.
        reorder_indices: positional ordering for All_* parallel lists so the
            feature-contributing drugs come first.
        n_with_features: count of distinct drugs that contributed.
        anchor_iks: list of IK27s that drove the aggregation (inv subset when
            applicable, else the resolved subset of all drugs).
    """
    def _parse_full(s):
        # Keep empty placeholders so positions stay aligned with All_Drugs/All_SMILES.
        if not isinstance(s, str): return []
        return [ik.strip()[:27] for ik in s.split(";")]

    all_iks_full = _parse_full(arm_iks_str)            # length = len(All_Drugs)
    all_iks = [ik for ik in all_iks_full if ik]         # non-empty, for feature logic
    inv_iks = [ik for ik in _parse_full(inv_iks_str) if ik]
    if not all_iks: return None, None, 0, []

    # Pick the scope to MAX over: investigational subset if present, else all.
    if inv_iks:
        scope_iks = [ik for ik in inv_iks if ik in set(all_iks)] or inv_iks
    else:
        scope_iks = all_iks

    feat_iks_set = set(drug_features["feature_IK"].dropna().unique())
    contributing_iks = [ik for ik in scope_iks if ik in feat_iks_set]
    if not contributing_iks:
        return None, None, 0, []

    # Reorder All_* to put inv-feature-bearing drugs first, then any other
    # feature-bearing all_iks, then no-feature drugs. This makes the first
    # listed IK14 a feature-anchor AND an investigational drug whenever the
    # arm has one.
    # Order over FULL positions (incl. empty placeholders) so new_order aligns
    # with the co-indexed All_Drugs/All_SMILES/All_InChIKeys lists. Empty/no-feature
    # positions sort last, so the first listed IK stays a feature-bearing inv anchor.
    scope_set = set(contributing_iks)
    inv_with_feat_idx = [i for i, ik in enumerate(all_iks_full) if ik and ik in scope_set]
    other_with_feat_idx = [i for i, ik in enumerate(all_iks_full)
                           if ik and ik in feat_iks_set and ik not in scope_set]
    no_feat_idx = [i for i, ik in enumerate(all_iks_full)
                   if not (ik and ik in feat_iks_set)]
    new_order = inv_with_feat_idx + other_with_feat_idx + no_feat_idx

    rows = drug_features[drug_features["feature_IK"].isin(contributing_iks)]
    if len(rows) == 1:
        feats = rows.iloc[0][feature_cols]
    else:
        feats = rows[feature_cols].max(axis=0)
        # Disease-pathway net_* features: anchor drug only, not MAX across all
        # investigational drugs. MAX inflates disease-alignment scores by ~26% for
        # multi-drug arms (picks the best-matching drug's score, not the actual
        # primary drug's signal) while fail rates are unchanged. Binding features
        # (tissue/binding/tox) use MAX correctly — cumulative-exposure semantics.
        net_cols = [c for c in feature_cols if c.startswith("net_")]
        if net_cols:
            anchor_rows = drug_features[drug_features["feature_IK"] == contributing_iks[0]]
            if not anchor_rows.empty:
                feats = feats.copy()
                feats[net_cols] = anchor_rows.iloc[0][net_cols].values
    return feats, new_order, len(set(contributing_iks)), contributing_iks


def _split_top_level(s: str, sep: str = ";", keep_empty: bool = False) -> list[str]:
    """Split on `sep` but ignore separators inside parentheses. Handles drug
    names like "Elotuzumab (BMS-901608; HuLuc63)" where the parenthesized
    synonyms contain a literal semicolon that must not break the parent list.

    keep_empty=True preserves empty slots — required when reordering co-indexed
    parallel lists (All_SMILES/All_InChIKeys carry empty placeholders for
    unresolved drugs, and dropping them would desync positions from All_Drugs)."""
    out, buf, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1; buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1); buf.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    parts = [p.strip() for p in out]
    return parts if keep_empty else [p for p in parts if p]


def _reorder_list(s: str | float, order: list[int]) -> str | float:
    """Reorder a semicolon-separated parallel list using positional indices.
    Returns the original value unchanged if it isn't a string or the part count
    doesn't match the expected length — this avoids silently scrambling lists
    whose component delimiter assumption breaks down."""
    if not isinstance(s, str): return s
    parts = _split_top_level(s, keep_empty=True)
    if len(parts) != len(order):
        return s
    return "; ".join(parts[i] for i in order)


def add_disease_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add disease context flags derived from Disease text."""
    dt = df["Disease"].fillna("").str.lower()
    df["disease_is_oncology"] = dt.str.contains(
        r"cancer|carcinoma|lymphoma|leukemia|melanoma|sarcoma|myeloma|glioma|"
        r"glioblastoma|neoplasm|tumor|nsclc|sclc|neuroblastoma|mesothelioma|"
        r"oncology|metastat|malignant", regex=True).astype(int)
    df["disease_is_infectious"] = dt.str.contains(
        r"hiv|hepatitis|hcv|hbv|influenza|covid|sars|tuberculosis|malaria|"
        r"bacterial|fungal|infection|pneumonia|sepsis|cmv|herpes|rsv|dengue", regex=True).astype(int)
    df["disease_is_cns"] = dt.str.contains(
        r"alzheimer|parkinson|epilepsy|seizure|schizophreni|depression|bipolar|"
        r"anxiety|dementia|multiple sclerosis|neuropath|migraine|stroke|brain", regex=True).astype(int)
    df["disease_is_cardiac"] = dt.str.contains(
        r"heart failure|atrial fibrillation|hypertension|coronary|myocardial|"
        r"arrhythmia|angina|cardiac|cardiovascular|atherosclerosis", regex=True).astype(int)
    df["disease_is_autoimmune"] = dt.str.contains(
        r"rheumatoid|lupus|psoriasis|crohn|colitis|autoimmune|immunolog|"
        r"ankylosing|scleroderma|vasculitis|pemphigus", regex=True).astype(int)
    df["disease_is_metastatic"] = dt.str.contains(
        r"metastat|advanced|stage iv|stage 4|unresectable|refractory", regex=True).astype(int)
    df["disease_is_transplant"] = dt.str.contains(
        r"transplant|graft|rejection", regex=True).astype(int)
    df["disease_is_severe"] = dt.str.contains(
        r"severe|critical|intensive care|icu|acute respiratory distress|ards|"
        r"life.threatening|end.stage|terminal", regex=True).astype(int)
    return df


METHODOLOGY_TITLE_KEYWORDS = (
    "ratio of", "potency of", "pharmacology of", "validation of",
    "comparison of breathing", "comparing inhaler", "anaesthetic",
    "anesthetic potency", "polysomnography", "questionnaire validation",
    "scale validation", "biomarker discovery",
    # R6 additions (May 22): PET imaging / digital behavioral / procedural-pain studies.
    # PET tracer studies measure neurochemistry, not drug efficacy.
    "pet imaging of", "pet imaging study", "pet study", "pet/ct imaging",
    "imaging of cyclooxygenase", "imaging of dopamine",
    "tracer development", "radiotracer", "radiolabeled tracer",
    # Behavioral / digital health interventions are NOT drug efficacy trials.
    "digital intervention", "digital health intervention", "phone app",
    "lock screen", "mobile application", "wearable device",
    "promote physical activity", "behavioral study", "behavioral intervention",
    "fitbit", "smartphone-based",
    # Procedural pain / sedation in newborns using sucrose/glucose (not drugs)
    "intraoral glucose", "oral sucrose for", "sucrose for newborn",
    "glucose for newborn", "glucose effect in newborns",
    "30% glucose", "24% sucrose", "12% sucrose",
    # Endothelial function research probes (Acetylcholine + L-Arginine + SNP)
    # are vascular function tests, not drug efficacy
    "endothelial function", "endothelial dysfunction",
    "vascular reactivity", "flow-mediated dilation",
)
METHODOLOGY_DISEASE_KEYWORDS = (
    "anesthesia", "anaesthesia", "healthy volunteer", "healthy subject",
    # R6 additions: nonsense-disease patterns where classifier mis-combined words
    "heartphone",  # NCT03953326 — disease classifier mis-glued "HeartPhone" + "Cancer"
)
DOSING_SCHEDULE_TITLE_KEYWORDS = (
    "adaptive ", "intermittent vs continuous", "intermittent versus continuous",
    "dose holiday", "dose interruption", "dose-finding", "dose finding",
    "schedule comparison", "extended vs standard dosing", "withdrawal study",
)


def add_special_trial_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Pattern P3/P4 flags: methodology and dosing-schedule trials.

    is_methodology_study=1 when the trial tests a technique, scale, or
    pharmacology comparison rather than drug efficacy. Detected via Disease
    keywords (Anesthesia/Healthy) or title keywords (potency, ratio of, …).

    is_dosing_schedule_trial=1 when the trial tests dosing strategy
    (adaptive vs continuous, intermittent, …) for already-approved drugs.
    Detected via title keywords.

    When either flag is set, the trial should be excluded from drug efficacy
    training (the model would learn the wrong association).
    """
    title = df["Trial_Title"].fillna("").str.lower()
    disease = df["Disease"].fillna("").str.lower()
    is_method = pd.Series(0, index=df.index, dtype=int)
    for kw in METHODOLOGY_TITLE_KEYWORDS:
        is_method |= title.str.contains(kw, regex=False).astype(int)
    for kw in METHODOLOGY_DISEASE_KEYWORDS:
        is_method |= disease.str.contains(kw, regex=False).astype(int)
    df["is_methodology_study"] = is_method.clip(0, 1)

    is_schedule = pd.Series(0, index=df.index, dtype=int)
    for kw in DOSING_SCHEDULE_TITLE_KEYWORDS:
        is_schedule |= title.str.contains(kw, regex=False).astype(int)
    df["is_dosing_schedule_trial"] = is_schedule.clip(0, 1)
    return df


def add_trial_design(df: pd.DataFrame) -> pd.DataFrame:
    """is_combination, trial_n_drugs from All_Drugs string. Counts top-level
    semicolon-separated entries — semicolons inside parentheses (synonym lists
    like "Elotuzumab (BMS-901608; HuLuc63)") don't inflate the count."""
    def count_drugs(s):
        if not isinstance(s, str): return 0
        return len(_split_top_level(s))
    df["trial_n_drugs"] = df["All_Drugs"].apply(count_drugs).astype(int)
    df["is_combination"] = (df["trial_n_drugs"] > 1).astype(int)
    # is_multi_investigational: 2+ new drugs being tested together (distinct from
    # is_combination which fires for any backbone drug). Different failure mode —
    # multi-investigational arms test combined novel mechanisms.
    df["is_multi_investigational"] = (df["n_investigational"] > 1).astype(int)
    return df


def add_flag_columns(df: pd.DataFrame, sources_dir: Path) -> pd.DataFrame:
    """Add has_black_box, is_anti_pathogen, is_endogenous as arm-level OR across drugs."""
    import re
    import sqlite3
    ANTI_PATHOGEN = {
        # Antibiotics (narrow-spectrum / no known non-pathogen repurposing)
        "amoxicillin", "aztreonam", "ceftaroline fosamil", "dalbavancin", "delafloxacin",
        "linezolid", "meropenem", "omadacycline", "oxacillin", "tigecycline",
        "tedizolid", "tedizolid phosphate", "plazomicin",
        "ceftolozane", "caz-avi", "ceftazidime-avibactam", "sulopenem", "durlobactam",
        "cefiderocol", "nitazoxanide", "cxa-201",
        # HIV antiretrovirals
        "dolutegravir", "raltegravir", "bictegravir", "cabotegravir",
        "efavirenz", "rilpivirine", "nevirapine", "etravirine", "doravirine",
        "emtricitabine", "lamivudine", "abacavir",
        "tenofovir", "tenofovir disoproxil", "tenofovir alafenamide",
        "elvitegravir", "cobicistat", "maraviroc",
        # HCV direct-acting antivirals
        "daclatasvir", "asunaprevir", "ombitasvir", "paritaprevir", "dasabuvir",
        "elbasvir", "grazoprevir", "glecaprevir", "pibrentasvir", "velpatasvir",
        "ledipasvir", "boceprevir", "telaprevir", "simeprevir", "danoprevir",
        "sofosbuvir", "ribavirin",
        # HBV antivirals
        "telbivudine", "entecavir", "adefovir",
        # CMV antivirals
        "valganciclovir", "ganciclovir", "maribavir", "letermovir", "foscarnet",
        # Influenza antivirals
        "oseltamivir", "zanamivir", "baloxavir marboxil",
        # Antimalarials
        "primaquine", "tafenoquine", "artefenomel", "artesunate",
        "artemether", "lumefantrine", "atovaquone", "amodiaquine", "mefloquine",
        # Anti-tuberculosis
        "clofazimine", "delamanid", "pretomanid",
        "rifampicin", "isoniazid", "ethambutol", "pyrazinamide", "rifapentine",
        # Other antivirals / antiparasitics
        "abi-h2158", "inarigivir soproxil", "lcq908", "pbi-0451 (pomotrelvir)",
        "shionogi protease inhibitor (s-217622)", "favipiravir",
    }
    # Disease-gated: flag only when the arm's Disease also matches the pathogen context.
    # Used for drugs that have legitimate non-pathogen indications (e.g. HCQ for RA/lupus).
    ANTI_PATHOGEN_DISEASE_GATED: dict[str, str] = {
        "chloroquine":      r"covid|sars|corona|malaria|vivax|falciparum",
        "hydroxychloroquine": r"covid|sars|corona|malaria|vivax|falciparum",
    }
    ENDOGENOUS = {
        "testosterone", "testosterone enanthate", "testosterone undecanoate", "estradiol",
        "progesterone", "hydrocortisone", "cortisone acetate", "melatonin", "oxytocin",
        "epinephrine", "norepinephrine", "dopamine", "calcitriol", "vitamin d3",
        "vasopressin", "levothyroxine", "dinoprostone", "epoprostenol",
    }
    # Black box drugs from ChEMBL
    bbw_drugs = set()
    chembl_db = sources_dir.parent / "cache" / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"
    if chembl_db.exists():
        conn = sqlite3.connect(str(chembl_db))
        bbw_ids = pd.read_sql("SELECT chembl_id FROM molecule_dictionary WHERE black_box_warning = 1", conn)
        conn.close()
        chembl_lu = pd.read_csv(sources_dir / "chembl_smiles_lookup.csv")
        bbw_drugs = set(chembl_lu[chembl_lu["chembl_id"].isin(bbw_ids["chembl_id"])]["Drug_Clean"].str.lower())

    # Regex for word-boundary matching — handles "Dolutegravir, Abacavir"-style
    # comma-within-semicolon entries that defeat exact split matching.
    _ap_re = re.compile(
        r'\b(?:' + '|'.join(re.escape(t) for t in sorted(ANTI_PATHOGEN, key=len, reverse=True)) + r')\b',
        re.IGNORECASE,
    )
    _gated_res = {
        drug: (re.compile(r'\b' + re.escape(drug) + r'\b', re.IGNORECASE),
               re.compile(dis_pat, re.IGNORECASE))
        for drug, dis_pat in ANTI_PATHOGEN_DISEASE_GATED.items()
    }

    def any_match_exact(all_drugs, target_set):
        if not isinstance(all_drugs, str): return 0
        for d in all_drugs.split(";"):
            if d.strip().lower() in target_set:
                return 1
        return 0

    def is_anti_pathogen_flag(row):
        drugs = row.get("All_Drugs", "")
        if not isinstance(drugs, str):
            return 0
        if _ap_re.search(drugs):
            return 1
        disease = str(row.get("Disease", "") or "")
        for _drug_re, _dis_re in _gated_res.values():
            if _drug_re.search(drugs) and _dis_re.search(disease):
                return 1
        return 0

    df["has_black_box"] = df["All_Drugs"].apply(lambda s: any_match_exact(s, bbw_drugs))
    df["is_anti_pathogen"] = df.apply(is_anti_pathogen_flag, axis=1)
    df["is_endogenous"] = df["All_Drugs"].apply(lambda s: any_match_exact(s, ENDOGENOUS))
    return df


def _relabel_active_comparator_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Relabel ACTIVE_COMPARATOR arms that carry a FAIL outcome to PASS.

    When a trial is stopped for futility or safety, AACT records the failure
    outcome on ALL arms — including the active comparator arm whose drug was
    the benchmark the experimental arm was trying to beat. That FAIL label on
    the comparator arm is a mislabel: the comparator drug didn't fail; the
    experimental drug failed to surpass it. Labeling rosuvastatin or dexamethasone
    as FAIL_EFFICACY because a new drug couldn't beat them trains the model in the
    wrong direction. The correct signal is PASS — these drugs held their ground as
    the standard of care.

    Applies to all ACTIVE_COMPARATOR arms with Corrected_Outcome in
    {FAIL_EFFICACY, FAIL_SAFETY, FAIL_BOTH}.  PASS arms are untouched.
    """
    df = df.copy()
    if "Arm_Type" not in df.columns:
        return df

    fail_outcomes = {"FAIL_EFFICACY", "FAIL_SAFETY", "FAIL_BOTH"}
    mask = (df["Arm_Type"] == "ACTIVE_COMPARATOR") & df["Corrected_Outcome"].isin(fail_outcomes)
    n = int(mask.sum())
    if n:
        drugs = df.loc[mask, "Investigational_Drugs"].fillna(df.loc[mask, "All_Drugs"]).unique()
        print(f"  Active-comparator relabel: {n} FAIL → PASS "
              f"(drugs: {', '.join(str(d) for d in drugs[:8])}{'…' if len(drugs) > 8 else ''})")
        df.loc[mask, "Corrected_Outcome"] = "PASS"
    return df


def _recover_mislabeled_test_arms(df: pd.DataFrame) -> pd.DataFrame:
    """Re-type investigational arms that CT.gov mislabeled as non-EXPERIMENTAL.

    CT.gov's armGroupType is registrant-supplied and frequently wrong. The
    decompose step already detects these arms as the tested intervention and
    populates Investigational_Drugs, but it preserves the raw CT.gov type. That
    leaves two downstream bugs: (1) the training mask keeps only
    Arm_Type=='EXPERIMENTAL', silently dropping these test arms, and
    (2) _relabel_active_comparator_outcomes flips their genuine FAIL → PASS.

    Two recovery patterns (verified May 28 2026), keyed only on trial structure
    and the per-arm outcome already stamped from 12_trials:

      A. fallback_active (drug-vs-placebo): the trial has NO EXPERIMENTAL arm but
         HAS a placebo/sham arm, and this ACTIVE_COMPARATOR arm carries an
         investigational drug → the drug IS the experimental intervention tested
         against placebo. Recover ALL such arms (PASS and FAIL). 350 arms / 294
         trials, e.g. NCT02932475 Metformin, NCT00979121 Rosuvastatin.

      B. single_active_arm FAILs: the trial has exactly one drug-bearing arm, no
         EXPERIMENTAL arm and no placebo (single-arm / open-label test). The drug
         is the one being tested. Per user decision (May 28), recover ONLY the
         FAIL arms here — the PASS arms are established-drug single-arm studies
         whose inclusion would add survivorship-flavored PASS mass without new
         signal. ~4 arms (Gefitinib, Duvelisib, Carboplatin-combo …).

    Retypes Arm_Type → 'EXPERIMENTAL', preserves the raw type in arm_type_ctgov,
    and sets is_recovered_test_arm + recovery_reason for the WITH/WITHOUT
    sensitivity toggle. MUST run before the dosing-duplicate block and before
    _relabel_active_comparator_outcomes so the recovered arms (now EXPERIMENTAL)
    are excluded from the active-comparator relabel and keep their true outcome.
    Genuine comparators (an ACTIVE_COMPARATOR arm in a trial that DOES have an
    EXPERIMENTAL arm) are left untouched — their FAIL relabel is still correct.
    """
    df = df.copy()
    df["arm_type_ctgov"] = df["Arm_Type"]
    df["is_recovered_test_arm"] = False
    df["recovery_reason"] = ""

    PLAC = {"PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION"}
    FAIL = {"FAIL_EFFICACY", "FAIL_SAFETY", "FAIL_BOTH"}
    has_inv = df["n_investigational"].fillna(0) >= 1

    grp_type = df.groupby("NCT_ID")["Arm_Type"]
    nct_has_exp = grp_type.transform(lambda s: (s == "EXPERIMENTAL").any())
    nct_has_plac = grp_type.transform(lambda s: s.isin(PLAC).any())
    n_active = df.groupby("NCT_ID")["n_investigational"].transform(
        lambda s: (s.fillna(0) >= 1).sum())

    fallback_active = (
        has_inv
        & (df["Arm_Type"] == "ACTIVE_COMPARATOR")
        & (~nct_has_exp)
        & nct_has_plac
    )
    single_arm_fail = (
        has_inv
        & (~nct_has_exp)
        & (~nct_has_plac)
        & (n_active == 1)
        & df["Arm_Type"].isin(["ACTIVE_COMPARATOR", "OTHER", "UNKNOWN_NO_ARM_DATA"])
        & df["Corrected_Outcome"].isin(FAIL)
    )

    df.loc[fallback_active, "recovery_reason"] = "fallback_active_drug_vs_placebo"
    df.loc[single_arm_fail & ~fallback_active, "recovery_reason"] = "single_active_arm_fail"
    recovered = fallback_active | single_arm_fail
    df.loc[recovered, "is_recovered_test_arm"] = True
    df.loc[recovered, "Arm_Type"] = "EXPERIMENTAL"

    n_fa = int(fallback_active.sum())
    n_sa = int((single_arm_fail & ~fallback_active).sum())
    fa_out = df.loc[fallback_active, "Corrected_Outcome"].value_counts().to_dict()
    print(f"  Recovered test arms → EXPERIMENTAL: {n_fa} fallback-active {fa_out} "
          f"+ {n_sa} single-arm-FAIL = {int(recovered.sum())} total")
    return df


# Treatment-duration studies confirmed by Caline's manual review (May 27 2026).
# These trials test the optimal DURATION of treatment with the same drug, not the
# drug's efficacy/safety, so their arms are not independent drug-disease outcomes.
# is_treatment_duration_study arms are EXCLUDED from training (via retrain mask).
# Add NCT_IDs here only after a reviewer confirms the trial is duration-only (not
# population- or dose-varying). The auto-detector below marks broader candidates
# for review but does NOT exclude them.
CONFIRMED_DURATION_STUDIES = {
    "NCT01743989",  # Nilotinib 24-mo vs 36-mo, PH+ CML
    "NCT01756885",  # Varenicline 12-wk vs 24-wk duration; tests when to stop, not efficacy (Caline v4)
}


def _flag_duration_studies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag treatment-duration studies.

    - is_treatment_duration_study: reviewer-confirmed duration-only trials
      (CONFIRMED_DURATION_STUDIES) → excluded from training.
    - duration_study_candidate: auto-detected trials where ≥2 same-anchor arms
      differ only by a duration token (e.g. '12 weeks' vs '16 weeks'). Surfaced
      for human review in the xlsx; NOT excluded, because the detector also
      catches population/dose-varying regimens (HCV naive-vs-experienced,
      MDR-vs-DS-TB) that are legitimately distinct.
    """
    df = df.copy()
    df["is_treatment_duration_study"] = df["NCT_ID"].isin(CONFIRMED_DURATION_STUDIES)

    dur_re = re.compile(r"\b\d+\s*[- ]?(?:month|months|week|weeks|year|years|day|days|mo|wk)\b", re.I)
    df["duration_study_candidate"] = False
    has_dur = df["Arm_Label"].fillna("").apply(lambda s: bool(dur_re.search(s)))
    for (_nct, _ik), g in df[df["feature_anchor_IK14"].notna()].groupby(["NCT_ID", "feature_anchor_IK14"]):
        if len(g) < 2 or not has_dur.loc[g.index].any():
            continue
        stripped = g["Arm_Label"].fillna("").apply(lambda s: dur_re.sub("", str(s)).strip().lower())
        if stripped.nunique() <= max(1, len(g) - 1):
            df.loc[g.index, "duration_study_candidate"] = True

    n_conf = int(df["is_treatment_duration_study"].sum())
    n_cand = int(df["duration_study_candidate"].sum())
    print(f"  Duration studies: {n_conf} confirmed-excluded arms, "
          f"{n_cand} candidate arms flagged for review")
    return df


def _add_ivan_review_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Review-aid flags from Ivan's May 28 feedback. These mark arms/trials for
    HUMAN review in the xlsx; none change the training mask.

    - disease_needs_review: Disease column lists >4 terms (#3) — likely a vague
      or over-broad MeSH dump that needs a curated primary indication.
    - multidrug_needs_review: n_drugs > 4 (#2) — large regimens to hand-check for
      combination-vs-alternatives before trusting the arm structure.
    - summary_has_standard_of_care: brief_summary mentions 'standard of care' (#5).
    - summary_has_combination / summary_has_compare: keyword signals (#2) to help
      classify combination trials (group drugs) vs comparison/'multiple' trials
      (alternative drugs that may need splitting into separate rows).
    """
    df = df.copy()
    n_terms = df["Disease"].fillna("").apply(
        lambda s: len([t for t in str(s).split(";") if t.strip()]))
    df["disease_needs_review"] = n_terms > 4
    df["multidrug_needs_review"] = df["n_drugs"].fillna(0) > 4

    desc_path = CACHE / "trial_descriptions.json"
    if desc_path.exists():
        import json
        with open(desc_path) as f:
            desc = json.load(f)
        summ = {k: (v.get("brief_summary") or "").lower()
                for k, v in desc.items() if isinstance(v, dict)}
        s = df["NCT_ID"].map(lambda n: summ.get(n, ""))
        df["summary_has_standard_of_care"] = s.str.contains("standard of care", regex=False)
        df["summary_has_combination"] = s.str.contains("combination", regex=False)
        df["summary_has_compare"] = s.str.contains("compare", regex=False) | s.str.contains("comparison", regex=False)
    else:
        print("  WARNING: trial_descriptions.json not found — summary flags all-False")
        for c in ("summary_has_standard_of_care", "summary_has_combination", "summary_has_compare"):
            df[c] = False

    print(f"  Ivan review flags: disease>4={int(df.disease_needs_review.sum())}, "
          f"n_drugs>4={int(df.multidrug_needs_review.sum())}, "
          f"std-of-care={int(df.summary_has_standard_of_care.sum())}, "
          f"combination={int(df.summary_has_combination.sum())}, "
          f"compare={int(df.summary_has_compare.sum())}")
    return df


def _anchor_quality_audit(df: pd.DataFrame) -> None:
    """Warn about FAIL arms whose anchor drug predominantly appears in PASS arms.

    A high pass-rate anchor is a backbone signal: the drug is well-established and
    usually succeeds, so attributing a FAIL outcome to its feature profile is likely
    wrong drug assignment.  Thresholds (empirically calibrated May 2026):
      - anchor appears in ≥ 5 training-eligible arms
      - pass rate ≥ 0.85 across all those arms
      - the specific arm in question is a FAIL outcome

    Prints a table of suspicious arms so new additions are caught immediately.
    ADMINISTRATIVE_STOP and IN_PROCESS arms are excluded from the denominator
    (only PASS/FAIL outcomes count).
    """
    TRAIN_OUTCOMES = {"PASS", "FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"}
    excl_flags = [
        "inv_biologic_only", "inv_large_peptide", "inv_approved_biologic_coinv",
        "inv_investigational_biologic_coinv", "is_business_stop",
        "is_narrow_population", "is_wrong_drug_assignment",
    ]
    mask = (
        (df["n_investigational"].fillna(0) >= 1)
        & (df.get("is_methodology_study", 0) == 0)
        & df["Corrected_Outcome"].isin(TRAIN_OUTCOMES)
        & df["All_SMILES"].notna()
        & df["feature_anchor_IK14"].notna()
    )
    for flag in excl_flags:
        if flag in df.columns:
            mask &= ~df[flag].fillna(False)
    tr = df[mask].copy()

    # Per-anchor stats
    stats = tr.groupby("feature_anchor_IK14").agg(
        n_total=("Corrected_Outcome", "count"),
        n_pass=("Corrected_Outcome", lambda x: (x == "PASS").sum()),
    ).reset_index()
    stats["pass_rate"] = stats["n_pass"] / stats["n_total"]
    high_pass = stats[(stats["n_total"] >= 5) & (stats["pass_rate"] >= 0.85)]

    # FAIL arms whose anchor is high-pass
    fail_arms = tr[tr["Corrected_Outcome"].isin({"FAIL_SAFETY", "FAIL_EFFICACY", "FAIL_BOTH"})]
    suspects = fail_arms[fail_arms["feature_anchor_IK14"].isin(high_pass["feature_anchor_IK14"])].copy()
    suspects = suspects.merge(
        stats[["feature_anchor_IK14", "n_total", "pass_rate"]],
        on="feature_anchor_IK14", how="left",
    )
    suspects["anchor_drug"] = (
        suspects["Investigational_Drugs"].fillna("").str.split(";").str[0].str.strip()
    )

    print(f"\n=== ANCHOR QUALITY AUDIT ===")
    print(f"FAIL arms where anchor pass-rate ≥85% (n≥5): {len(suspects)}")
    if len(suspects):
        print(f"{'NCT_ID':12s}  {'anchor_drug':30s}  {'outcome':14s}  {'n_anchor':8s}  {'pass_rt':7s}")
        print("-" * 80)
        for _, r in suspects.sort_values("pass_rate", ascending=False).iterrows():
            print(
                f"{r['NCT_ID']:12s}  {r['anchor_drug']:30s}  "
                f"{r['Corrected_Outcome']:14s}  {int(r['n_total']):8d}  {r['pass_rate']:.2f}"
            )
    print("=== END AUDIT ===\n")


def _apply_drug_assignment_corrections(arms: pd.DataFrame) -> pd.DataFrame:
    """Fix known wrong investigational-drug assignments before feature aggregation.

    The arm-decomposer sometimes labels the wrong drug as 'investigational'
    (e.g. a backbone chemotherapy instead of the novel agent). For cases where
    the correct investigational drug EXISTS in our feature table, we patch the
    arm rows here so the anchor and feature aggregation use the right drug.
    Cases where the correct drug has no pipeline features are handled instead
    by is_wrong_drug_assignment (exclusion).

    Patched NCTs
    ------------
    NCT01205828  Veliparib+TMZ for liver cancer — TMZ listed first, should be Veliparib
    NCT01110876  Vorinostat+Erlotinib+TMZ for GBM — TMZ wrongly labeled investigational
    NCT06065059  TNG348+Olaparib — Olaparib wrongly labeled; TNG348 monotherapy arms
                 have n_inv=0 when TNG348 IS the investigational drug
    """
    arms = arms.copy()

    # NCT01205828: swap Veliparib to first position
    # Investigational_InChIKeys: "TMZ_IK; VEL_IK" → "VEL_IK; TMZ_IK"
    TMZ_IK = "BPEGJWRSRHCHSN-UHFFFAOYSA-N"
    VEL_IK = "JNAHVYVRKWKWKQ-CYBMUJFWSA-N"
    m = arms["NCT_ID"] == "NCT01205828"
    arms.loc[m, "Investigational_InChIKeys"] = f"{VEL_IK}; {TMZ_IK}"
    arms.loc[m, "Investigational_Drugs"] = "Veliparib; Temozolomide"
    arms.loc[m, "All_InChIKeys"] = f"{VEL_IK}; {TMZ_IK}"
    arms.loc[m, "All_Drugs"] = "Veliparib; Temozolomide"
    n = int(m.sum())
    print(f"  Drug correction NCT01205828 (Veliparib→anchor): {n} arm(s)")

    # NCT01110876: Vorinostat+Erlotinib are the investigational drugs, not TMZ
    VORI_IK = "WAEXFXRVDQXREF-UHFFFAOYSA-N"
    ERLO_IK = "AAKJLRGGTJKAMG-UHFFFAOYSA-N"
    # Arms with TMZ wrongly assigned as investigational
    m_tmz = (arms["NCT_ID"] == "NCT01110876") & (
        arms["Investigational_Drugs"].fillna("") == "Temozolomide"
    )
    arms.loc[m_tmz, "Investigational_InChIKeys"] = f"{VORI_IK}; {ERLO_IK}"
    arms.loc[m_tmz, "Investigational_Drugs"] = "Vorinostat; Erlotinib"
    arms.loc[m_tmz, "n_investigational"] = 2
    # Arms with n_inv=0 that should be Vorinostat+Erlotinib investigational
    m_nan = (arms["NCT_ID"] == "NCT01110876") & (arms["n_investigational"] == 0)
    arms.loc[m_nan, "Investigational_InChIKeys"] = f"{VORI_IK}; {ERLO_IK}"
    arms.loc[m_nan, "Investigational_Drugs"] = "Vorinostat; Erlotinib"
    arms.loc[m_nan, "n_investigational"] = 2
    n = int((arms["NCT_ID"] == "NCT01110876").sum())
    print(f"  Drug correction NCT01110876 (Vorinostat+Erlotinib→investigational): {n} arm(s)")

    # NCT06065059: TNG348 is investigational, Olaparib is backbone
    TNG_IK = "NKGSHRLGUQURMS-UHFFFAOYSA-N"
    # Combination arms: Olaparib wrongly labeled as investigational
    m_ola = (arms["NCT_ID"] == "NCT06065059") & (
        arms["Investigational_Drugs"].fillna("") == "Olaparib"
    )
    arms.loc[m_ola, "Investigational_InChIKeys"] = TNG_IK
    arms.loc[m_ola, "Investigational_Drugs"] = "TNG348"
    # Single-agent arms: TNG348 in All_Drugs but n_inv=0 (arm decomposer missed it)
    m_tng = (arms["NCT_ID"] == "NCT06065059") & (arms["n_investigational"] == 0)
    arms.loc[m_tng, "Investigational_InChIKeys"] = TNG_IK
    arms.loc[m_tng, "Investigational_Drugs"] = "TNG348"
    arms.loc[m_tng, "n_investigational"] = 1
    n = int((arms["NCT_ID"] == "NCT06065059").sum())
    print(f"  Drug correction NCT06065059 (TNG348→investigational): {n} arm(s)")

    # NCT03176277: ONO-7475 (Tamnorzatinib) is investigational; Venetoclax is backbone
    # Arm label "ONO-7475 … + Venetoclax" correctly names ONO-7475 first;
    # arm decomposer reversed the relationship.
    TAMNO_IK = "WHMMKPWGWNYYFE-UHFFFAOYSA-N"
    # Monotherapy arms have n_inv=0 — ONO-7475 IS the investigational drug
    m_mono = (arms["NCT_ID"] == "NCT03176277") & (arms["n_investigational"] == 0)
    arms.loc[m_mono, "Investigational_InChIKeys"] = TAMNO_IK
    arms.loc[m_mono, "Investigational_Drugs"] = "Tamnorzatinib"
    arms.loc[m_mono, "n_investigational"] = 1
    # Combination arm: Venetoclax wrongly labeled investigational
    m_vcl = (arms["NCT_ID"] == "NCT03176277") & (arms["n_investigational"] == 1)
    arms.loc[m_vcl, "Investigational_InChIKeys"] = TAMNO_IK
    arms.loc[m_vcl, "Investigational_Drugs"] = "Tamnorzatinib"
    n = int((arms["NCT_ID"] == "NCT03176277").sum())
    print(f"  Drug correction NCT03176277 (Tamnorzatinib→investigational): {n} arm(s)")

    return arms


def main():
    arms = pd.read_csv(SOURCES / "trial_arms.csv", low_memory=False)
    print(f"Loaded {len(arms)} arm rows across {arms.NCT_ID.nunique()} NCTs")

    arms = _apply_drug_assignment_corrections(arms)

    drug = load_drug_features()
    feature_cols = [c for c in drug.columns if c != "feature_IK"]

    print(f"Aggregating arms × {len(feature_cols)} features...")
    arm_feats = []
    n_found = 0
    n_partial = 0
    n_inv_scoped = 0
    for i, arm in arms.iterrows():
        feats, new_order, n_with, anchor_iks = aggregate_arm_features(
            arm.get("All_InChIKeys"),
            arm.get("Investigational_InChIKeys"),
            drug, feature_cols)
        rec = arm.to_dict()
        if feats is not None:
            n_found += 1
            inv_iks_str = arm.get("Investigational_InChIKeys")
            # True only if at least one investigational IK actually resolved
            # (co-indexed lists carry empty placeholders for unresolved biologics,
            # so a non-empty string alone no longer implies a real inv IK).
            scope_was_inv = isinstance(inv_iks_str, str) and any(
                p.strip() for p in inv_iks_str.split(";"))
            if scope_was_inv:
                n_inv_scoped += 1
            # Reorder parallel All_* lists so inv-feature-bearing drugs come
            # first, then any non-inv drugs that also have features, then
            # drugs missing from drug_features. This makes "first IK14 of
            # All_InChIKeys" reliably point to a feature-contributing
            # investigational drug whenever the arm has one.
            rec["All_Drugs"] = _reorder_list(rec.get("All_Drugs"), new_order)
            rec["All_InChIKeys"] = _reorder_list(rec.get("All_InChIKeys"), new_order)
            rec["All_SMILES"] = _reorder_list(rec.get("All_SMILES"), new_order)
            if n_with < len(new_order):
                n_partial += 1
            rec["n_drugs_with_features"] = n_with
            # First NON-EMPTY IK after reorder (reorder puts feature-bearing inv
            # drugs first; empty placeholders sort last). Robust to leading blanks.
            first_ik = next((p.strip() for p in (rec.get("All_InChIKeys") or "").split(";")
                             if p.strip()), "")
            rec["feature_anchor_IK14"] = first_ik[:14] if first_ik else None
            rec["feature_scope"] = "investigational" if scope_was_inv else "all_delivered"
            rec.update(feats.to_dict())
        else:
            rec["n_drugs_with_features"] = 0
            rec["feature_anchor_IK14"] = None
            rec["feature_scope"] = None
        arm_feats.append(rec)
        if (i+1) % 1000 == 0:
            print(f"  {i+1}/{len(arms)}")
    print(f"  Inv-only scope used in {n_inv_scoped} arms; all-delivered scope in {n_found - n_inv_scoped}")

    df = pd.DataFrame(arm_feats)
    print(f"Arms with ≥1 drug found in feature table: {n_found}/{len(arms)}")
    print(f"  Of those, partial-match arms (some drugs missing features): {n_partial}")

    # Compute arm-level context, design, flags
    df = add_disease_context(df)
    df = add_trial_design(df)
    df = add_flag_columns(df, SOURCES)
    df = add_special_trial_flags(df)

    # Flag arms where ALL investigational drugs are biologics (no IK14 in our
    # pipeline). Features for these arms come from backbone small molecules
    # (all_delivered scope), not the investigational drug — spurious for training.
    # All investigational IKs unresolved: with co-indexed lists this is a string of
    # empty placeholders (e.g. "; ;"), not "", so test that NO token is non-empty.
    _no_inv_ik = df["Investigational_InChIKeys"].fillna("").apply(
        lambda s: not any(p.strip() for p in str(s).split(";")))
    df["inv_biologic_only"] = (
        df["n_investigational"].fillna(0).astype(int) > 0
    ) & _no_inv_ik
    n_bio = int(df["inv_biologic_only"].sum())
    print(f"  inv_biologic_only arms flagged: {n_bio}")

    # Flag arms whose feature anchor is a high-MW peptide (>900 Da) — Binding
    # predictions for large peptides are not meaningful. Compute MW of anchor
    # drug from chembl_smiles_lookup.
    from rdkit import Chem as _Chem
    from rdkit.Chem.inchi import MolToInchi as _MolToInchi, InchiToInchiKey as _IK
    from rdkit.Chem.Descriptors import MolWt as _MolWt
    _chembl = pd.read_csv(SOURCES / "chembl_smiles_lookup.csv")
    _anchor_mw: dict[str, float] = {}
    for _, _r in _chembl.iterrows():
        _smi = _r.get("chembl_smiles", "")
        if not isinstance(_smi, str) or not _smi: continue
        _mol = _Chem.MolFromSmiles(_smi)
        if _mol:
            try:
                _ik = _IK(_MolToInchi(_mol))
                if _ik: _anchor_mw[_ik[:14]] = _MolWt(_mol)
            except: pass
    # Known large peptides/biologics the MW lookup misses: these resolve to a
    # peptide SMILES (so they carry an IK14 and slip past the no-IK14 biologic
    # detector) but are absent from chembl_smiles_lookup, so the MW>900 test
    # never fires. Flagged by investigational IK14 directly so the arm is caught
    # even when the stereo-mismatched anchor left feature_anchor_IK14 unset.
    KNOWN_LARGE_PEPTIDE_IK14 = {
        "HTQBXNHDCUEHJF",  # Exenatide — 39-aa GLP-1 peptide (~4.2 kDa); Binding not meaningful
    }
    _inv_has_known_peptide = df["Investigational_InChIKeys"].fillna("").apply(
        lambda s: any(p.strip()[:14] in KNOWN_LARGE_PEPTIDE_IK14
                      for p in str(s).split(";") if p.strip())
    )
    df["inv_large_peptide"] = df["feature_anchor_IK14"].apply(
        lambda ik: bool(ik and _anchor_mw.get(str(ik)[:14], 0) > 900)
    ) | _inv_has_known_peptide
    n_pep = int(df["inv_large_peptide"].sum())
    print(f"  inv_large_peptide arms flagged (anchor MW >900 or known peptide): {n_pep}")

    # Flag arms where an APPROVED biologic (e.g. an antibody with max_phase=4 in
    # ChEMBL) is co-investigational alongside a small molecule.  The trial outcome
    # is driven by the approved biologic's established mechanism, not the small
    # molecule anchor whose features we use — including these would teach the model
    # spurious associations (e.g. "drugs paired with checkpoint inhibitors pass").
    # Distinct from inv_biologic_only: that flag covers all-biologic investigational
    # groups; this flag covers mixed groups where the biologic happens to be approved.
    import sqlite3 as _sqlite3
    _chembl_db = SOURCES.parent / "cache" / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"
    _approved_bio_names: set[str] = set()
    if _chembl_db.exists():
        _conn = _sqlite3.connect(str(_chembl_db))
        _bio_df = pd.read_sql(
            """SELECT LOWER(pref_name) as name FROM molecule_dictionary
               WHERE max_phase = 4
                 AND molecule_type IN ('Antibody','Protein','Enzyme',
                                       'Oligopeptide','Oligonucleotide',
                                       'Antibody drug conjugate')""",
            _conn,
        )
        _conn.close()
        _approved_bio_names = set(_bio_df["name"].dropna())
    else:
        print("  WARNING: ChEMBL DB not found — inv_approved_biologic_coinv will be all-False")

    import re as _re
    _paren_re = _re.compile(r'\(([^)]+)\)')
    _num_prefix_re = _re.compile(r'^\d+\)\s*')

    def _candidate_names(drug_str: str) -> list[str]:
        """Yield candidate names from a drug entry: the entry itself, parenthetical
        synonyms, and entries after leading number prefixes like '1) basiliximab'."""
        names = [drug_str]
        # Extract names inside parentheses: "BIIB019 (Daclizumab)" → "Daclizumab"
        for m in _paren_re.finditer(drug_str):
            names.append(m.group(1).strip())
        # Strip number prefixes: "1) Basiliximab" → "Basiliximab"
        stripped = _num_prefix_re.sub("", drug_str).strip()
        if stripped != drug_str:
            names.append(stripped)
        return [n.lower() for n in names if n.strip()]

    # Brand names not in ChEMBL preferred names but known to be approved biologics
    _BIOLOGIC_BRAND_NAMES: set[str] = {
        "avastin",      # bevacizumab
        "herceptin",    # trastuzumab
        "perjeta",      # pertuzumab
        "kadcyla",      # ado-trastuzumab emtansine
        "keytruda",     # pembrolizumab
        "opdivo",       # nivolumab
        "tecentriq",    # atezolizumab
        "imfinzi",      # durvalumab
        "bavencio",     # avelumab
        "libtayo",      # cemiplimab
        "rituxan",      # rituximab
        "mabthera",     # rituximab
        "erbitux",      # cetuximab
        "vectibix",     # panitumumab
        "prolia",       # denosumab
        "xgeva",        # denosumab
        "soliris",      # eculizumab
        "benlysta",     # belimumab
        "actemra",      # tocilizumab
        "roactemra",    # tocilizumab
        "stelara",      # ustekinumab
        "humira",       # adalimumab
        "remicade",     # infliximab
        "simponi",      # golimumab
        "cimzia",       # certolizumab
        "orencia",      # abatacept
        "enspryng",     # satralizumab
        "zinplava",     # bezlotoxumab
        # G-CSF / granulocyte colony-stimulating factor variants (Caline v4: NCT01767714)
        "filgrastim", "pegfilgrastim", "lenograstim", "lipegfilgrastim",
        "neupogen", "neulasta", "granix", "zarxio", "nivestym", "stimufend",
        "g-csf", "granulocyte colony stimulating factor",
        "granulocyte-colony stimulating factor",
    }

    def _has_approved_bio_coinv(row) -> bool:
        # Only relevant when the arm has at least one small-molecule investigational drug
        if row.get("inv_biologic_only", False):
            return False
        inv_iks = str(row.get("Investigational_InChIKeys", "") or "")
        inv_drugs = str(row.get("Investigational_Drugs", "") or "")
        if not inv_drugs.strip():
            return False
        inv_drug_list = [d.strip() for d in inv_drugs.split(";") if d.strip()]
        n_with_ik = len([x for x in inv_iks.split(";") if x.strip()]) if inv_iks.strip() else 0
        # All investigational drugs have IK14 → all small molecules, no biologics
        if n_with_ik >= len(inv_drug_list):
            return False
        # Check each drug that lacks an IK14 against approved biologic set,
        # trying the full name, parenthetical synonyms, and first token.
        # Also check against hardcoded brand names not in ChEMBL.
        for drug in inv_drug_list:
            for candidate in _candidate_names(drug):
                if candidate in _approved_bio_names or candidate in _BIOLOGIC_BRAND_NAMES:
                    return True
                tok = candidate.split()[0]
                if tok and (tok in _approved_bio_names or tok in _BIOLOGIC_BRAND_NAMES):
                    return True
        return False

    df["inv_approved_biologic_coinv"] = df.apply(_has_approved_bio_coinv, axis=1)
    n_abc = int(df["inv_approved_biologic_coinv"].sum())
    print(f"  inv_approved_biologic_coinv arms flagged: {n_abc}")

    # --- is_business_stop ---
    # True when a FAIL_EFFICACY/FAIL_BOTH trial was stopped for strategic/commercial
    # reasons, not a scientific efficacy signal. The drug was never disproven; the
    # company made a portfolio decision. These arms are excluded from training.
    # Source: Why_Stopped text in 12_trials_corrected_outcomes.csv joined on NCT_ID.
    _corrected_src = SOURCES / "12_trials_corrected_outcomes.csv"
    if _corrected_src.exists():
        _why = (pd.read_csv(_corrected_src, usecols=["NCT_ID", "Why_Stopped"])
                .dropna(subset=["Why_Stopped"])
                .drop_duplicates("NCT_ID")
                .set_index("NCT_ID")["Why_Stopped"])
        _biz_re = re.compile(
            r"business reason|sponsor decision|strategic decision|"
            r"commercial decision|company decision|portfolio|"
            r"not due to safety or efficacy|not related to safety or efficacy",
            re.IGNORECASE,
        )
        def _is_biz_stop(row) -> bool:
            if row.get("Corrected_Outcome") not in ("FAIL_EFFICACY", "FAIL_BOTH"):
                return False
            why = _why.get(row["NCT_ID"], "")
            return bool(_biz_re.search(str(why)))
        df["is_business_stop"] = df.apply(_is_biz_stop, axis=1)
    else:
        print("  WARNING: 12_trials_corrected_outcomes.csv not found — is_business_stop will be all-False")
        df["is_business_stop"] = False
    n_biz = int(df["is_business_stop"].sum())
    print(f"  is_business_stop arms flagged: {n_biz}")

    # --- is_pediatric_trial ---
    # True when Trial_Title indicates the trial was designed for children/infants.
    # Flagged for reviewer visibility; kept in training by default.
    _ped_re = re.compile(
        r"\bpediatric\b|\bpaediatric\b|\bchildren\b|\bchild\b|"
        r"\bneonatal\b|\bneonate\b|\binfant\b|\badolescent\b|\bjuvenile\b",
        re.IGNORECASE,
    )
    df["is_pediatric_trial"] = df["Trial_Title"].apply(
        lambda t: bool(_ped_re.search(str(t))) if pd.notna(t) else False
    )
    n_ped = int(df["is_pediatric_trial"].sum())
    print(f"  is_pediatric_trial arms flagged: {n_ped}")

    # --- is_narrow_population ---
    # True for trials targeting an unusually narrow/specific population (twin
    # pregnancies, compassionate use, expanded access, named patient). These are
    # excluded from training — too niche to generalise from.
    _narrow_re = re.compile(
        r"\btwin\b|\btriplet\b|compassionate use|expanded access|named patient",
        re.IGNORECASE,
    )
    df["is_narrow_population"] = df["Trial_Title"].apply(
        lambda t: bool(_narrow_re.search(str(t))) if pd.notna(t) else False
    )
    n_narrow = int(df["is_narrow_population"].sum())
    print(f"  is_narrow_population arms flagged: {n_narrow}")

    # --- Manual review overrides (discovered during human review, May 2026) ---
    # Trials that pattern-matching cannot reliably detect but are known to be
    # problematic from direct expert inspection.
    #
    # NCT02927379: "Effect of Wound Infiltration by Ketamine vs Dexmedetomidine..." —
    #   tests wound-infiltration technique/route, not drug efficacy for a disease.
    #   Ivan's review: "METHODE TRIAL Wound Infiltration"
    _MANUAL_METHODOLOGY_NCTS = {
        "NCT02927379",  # Ketamine vs Dexmedetomidine wound infiltration technique study
        "NCT01814553",  # Afatinib + loperamide: timing of diarrhea management drug, not efficacy (Caline v4)
    }
    df.loc[df["NCT_ID"].isin(_MANUAL_METHODOLOGY_NCTS), "is_methodology_study"] = 1
    n_manual_meth = int(df["NCT_ID"].isin(_MANUAL_METHODOLOGY_NCTS).sum())
    if n_manual_meth:
        print(f"  is_methodology_study manual overrides: {n_manual_meth}")

    # Safety-label corrections (blind symmetric audit, Jun 7) — trials whose
    # FAIL_SAFETY label is not a genuine drug-attributable in-trial safety failure
    # (COVID-funding / investigator-death / post-marketing-temporal). Relabel to
    # EXCLUDE_NONDRUG_STOP so they drop from every task cohort. Provenance:
    # data/sources/safety_label_corrections_jun7.csv, notes/safety_label_audit_jun7.md.
    _corr_files = sorted(SOURCES.glob("*_label_corrections_*.csv"))
    if _corr_files:
        _scfix = {}
        for _f in _corr_files:
            _c = pd.read_csv(_f)
            _scfix.update(dict(zip(_c["NCT_ID"], _c["new_outcome"])))
        _scn = 0
        for _nct, _new in _scfix.items():
            _m = df["NCT_ID"] == _nct
            _scn += int(_m.sum())
            df.loc[_m, "Corrected_Outcome"] = _new
        if _scn:
            print(f"  label corrections applied: {_scn} rows -> EXCLUDE_NONDRUG_STOP "
                  f"(from {[f.name for f in _corr_files]})")

    # NCT02934698: Ivacaftor in CF patients with 2 splicing mutations — only 2
    #   patients enrolled (Ivan: "TRILA ON 2 PATIENTS ONLY"). Too small to train.
    _MANUAL_NARROW_POP_NCTS = {"NCT02934698"}
    df.loc[df["NCT_ID"].isin(_MANUAL_NARROW_POP_NCTS), "is_narrow_population"] = True
    n_manual_narrow = int(df["NCT_ID"].isin(_MANUAL_NARROW_POP_NCTS).sum())
    if n_manual_narrow:
        print(f"  is_narrow_population manual overrides: {n_manual_narrow}")

    # --- inv_investigational_biologic_coinv ---
    # True when an UNAPPROVED investigational biologic is co-investigational with a
    # small molecule. Like inv_approved_biologic_coinv, the outcome is confounded by
    # the biologic component whose molecular features we lack. Unlike the approved
    # case, the biologic is also experimental — both drugs are investigational.
    #
    # Known cases (manual review, May 2026):
    #   NCT01718158: Peginterferon Lambda-1a + Daclatasvir (Lambda NOT approved)
    #   NCT01741545: Pegylated-Interferon-lambda + Daclatasvir (same Lambda biologic)
    #   NCT03451773: M7824 (anti-PD-L1/TGF-β bifunctional fusion protein) + Gemcitabine
    #   NCT02910882: PEGPH20 (recombinant human hyaluronidase, investigational) + Gemcitabine
    _bio_coinv_re = re.compile(
        r"peginterferon lambda|pegylated.{0,5}interferon.{0,10}lambda"
        r"|MSB0011359C|M7824"       # anti-PD-L1/TGF-β bifunctional fusion protein
        r"|PEGPH20|hyaluronidase",  # recombinant hyaluronidase enzyme
        re.IGNORECASE,
    )
    inv_bio_coinv = (
        df["Investigational_Drugs"].fillna("").apply(lambda s: bool(_bio_coinv_re.search(s)))
        | df["All_Drugs"].fillna("").apply(lambda s: bool(_bio_coinv_re.search(s)))
    ) & ~df["inv_biologic_only"].fillna(False)
    df["inv_investigational_biologic_coinv"] = inv_bio_coinv
    n_ibc = int(inv_bio_coinv.sum())
    print(f"  inv_investigational_biologic_coinv arms flagged: {n_ibc}")

    # --- is_wrong_drug_assignment ---
    # Arms where the arm-decomposition algorithm assigned the WRONG drug as
    # "investigational" — i.e. the approved backbone chemo was marked as investigational
    # instead of the actual novel drug under study. These arms have wrong features and
    # wrong outcome attribution; they are excluded from training.
    #
    # NCT02711137 (INCB057643 BET-inhibitor safety study, Part 3):
    #   Combination arms show backbone chemos (Gemcitabine, Paclitaxel, Rucaparib,
    #   Abiraterone, Ruxolitinib, Azacitidine) as investigational. The safety failure
    #   was from INCB057643, not the backbones. Using backbone features for a FAIL_SAFETY
    #   outcome trains the model that established drugs caused safety failures here.
    # NCT06065059 (TNG348 USP1 inhibitor + Olaparib):
    #   Combination arms show Olaparib as investigational. The liver toxicity was from
    #   TNG348 (the new drug); Olaparib is the backbone PARP inhibitor.
    # NCT02711137 (INCB057643 BET-inhibitor safety study, Part 3):
    #   Combination arms show backbone chemos (Gemcitabine, Paclitaxel, Rucaparib,
    #   Abiraterone, Ruxolitinib, Azacitidine) as investigational. The safety failure
    #   was from INCB057643, not the backbones. INCB057643 has no pipeline features so
    #   the assignment cannot be fixed — arms are excluded.
    # NCT02393209 (Serabelisib/TAK-117 + Docetaxel, NSCLC, FAIL_EFFICACY):
    #   Docetaxel is the backbone; Serabelisib (PI3Kδ inhibitor) is the novel agent.
    #   Serabelisib has no pipeline features, so the anchor remains Docetaxel — wrong.
    # NCT01059448 (Brodalumab/AMG 827 in RA, FAIL_EFFICACY):
    #   Trial title and Why_Stopped both name Brodalumab (AMG 827, IL-17RA biologic antibody)
    #   but the intervention is coded as "Amg-517" (a TRPV1 small-molecule antagonist) — a
    #   completely different compound. Cannot fix: correct drug is a biologic.
    # NOT HERE (fixed by _apply_drug_assignment_corrections):
    #   NCT01110876, NCT01205828, NCT06065059
    # NOT HERE (fixed by biologic brand name detection):
    #   NCT01186406 (Avastin = bevacizumab → caught by inv_approved_biologic_coinv)
    _WRONG_DRUG_NCTS = {"NCT02711137", "NCT02393209", "NCT01059448"}
    df["is_wrong_drug_assignment"] = df["NCT_ID"].isin(_WRONG_DRUG_NCTS)
    n_wda = int(df["is_wrong_drug_assignment"].sum())
    print(f"  is_wrong_drug_assignment arms flagged: {n_wda}")

    # --- is_dosing_arm_duplicate ---
    # True for EXPERIMENTAL arms that share (NCT_ID, feature_anchor_IK14) with
    # another EXPERIMENTAL arm in the same trial — dose-escalation, dose-comparison,
    # or schedule-variant arms where the molecular features are identical to the
    # "primary" arm for that drug.
    # Flagged for reviewer visibility; NOT excluded from training.
    # Among arms sharing the same (NCT_ID, feature_anchor_IK14), the one that sorts
    # first by Arm_Label is the primary (is_dosing_arm_duplicate=False); the rest are True.
    #
    # Recover CT.gov-mislabeled test arms FIRST so they participate in dosing-dup
    # dedup as EXPERIMENTAL and are excluded from the active-comparator relabel.
    df = _recover_mislabeled_test_arms(df)

    # Flag treatment-duration studies (Caline review, May 28 2026).
    df = _flag_duration_studies(df)

    # Review-aid flags from Ivan's May 28 feedback (#3 disease, #5 SoC, #2 helpers).
    df = _add_ivan_review_flags(df)

    # Dedup key = (NCT_ID, set of investigational drug names). A "dosing/phase
    # duplicate" is the SAME treatment given at a different dose/schedule/study
    # phase — i.e. the same investigational drug SET. Keying on the inv-name set
    # (not just feature_anchor_IK14) is important: a monotherapy arm and a
    # combination arm that share an anchor (e.g. rucaparib mono vs
    # rucaparib+nivolumab) are DIFFERENT treatments and must NOT be collapsed.
    def _inv_set_key(s):
        if not isinstance(s, str) or not s.strip():
            return ""
        return "|".join(sorted(t.strip().lower() for t in s.split(";") if t.strip()))
    df["_inv_set_key"] = df["Investigational_Drugs"].apply(_inv_set_key)
    exp_sorted = df[df["Arm_Type"] == "EXPERIMENTAL"].sort_values("Arm_Label")
    primary_idx = exp_sorted[exp_sorted["_inv_set_key"] != ""].drop_duplicates(
        subset=["NCT_ID", "_inv_set_key"], keep="first"
    ).index
    df["is_dosing_arm_duplicate"] = False
    df.loc[
        (df["Arm_Type"] == "EXPERIMENTAL") &
        (df["_inv_set_key"] != "") &
        ~df.index.isin(primary_idx),
        "is_dosing_arm_duplicate"
    ] = True
    df = df.drop(columns=["_inv_set_key"])
    n_dup = int(df["is_dosing_arm_duplicate"].sum())
    print(f"  is_dosing_arm_duplicate arms flagged: {n_dup}")

    df = _relabel_active_comparator_outcomes(df)

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: shape={df.shape}")

    _anchor_quality_audit(df)


if __name__ == "__main__":
    main()
