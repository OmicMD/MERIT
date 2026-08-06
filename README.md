# MERIT

MEchanism-Resolved Inference of Trial outcomes (MERIT): Pre-trial mechanism predicts outcomes and nominates indications for failed drugs

This repository contains the analysis code accompanying the manuscript. It is provided for peer review and to document how the reported results were produced.

## What is included

- `notebooks/` - the nine entry-point notebooks, retaining their executed outputs so the reported numbers can be inspected without re-running the analyses.
- `scripts/` - the Python modules invoked by those notebooks, including the `benchmark/`, `phase1/`, `lincs/` and `strengthening/` sub-packages.
- `data/` - the trial-level and arm-level modelling cohorts, together with the feature and lookup tables the analyses read.
- `model/` - the pre-trained model bundle, a self-contained predictor, container definitions and a worked example.

## What is not included

- `prediction/` - the locked, outcome-blind prospective-registration predictions with their SHA-256 commitments. Archived on Zenodo at https://doi.org/10.5281/zenodo.21824277. The Zenodo record fixes these predictions to a date independently of this repository, which is the point of a prospective lock.
- `results/` - the model outputs: per-fold and headline metrics, out-of-fold predictions, benchmark comparisons and figure inputs. Archived on Zenodo under the same DOI, and restored by unpacking the archive into a `results/` directory at the repository root.
- The structure-to-target binding profiles produced by our STAR pipeline. These total approximately 3 TB, far beyond what a Git repository can host, and are available from the authors on request.
- The large third-party databases that the analyses query or download at run time, listed under [Data availability](#data-availability). Each carries its own licence and redistribution terms.
- `data/processed/` - the SIDER and DILIrank derivative tables used by the external safety-axis validation.
- The manuscript sources, figures and their build tooling.

## Repository layout

```
notebooks/
  00_reproduce_full_pipeline.ipynb            master workflow driver
  01_data_provenance_rebuild_executed.ipynb   dataset construction and provenance
  02_model_training_evaluation.ipynb          model training and headline evaluation
  03_supporting_analyses.ipynb                supporting and sensitivity analyses
  04_mechanism_signal_decomposition.ipynb     mechanism signal decomposition
  05_aact_scale_transition.ipynb              AACT-scale transition analysis
  06_prospective_registration.ipynb           prospective registration of predictions
  07_safety_head_outlier_investigation_jul6.ipynb  safety head outlier investigation
  08_table1_benchmark_significance.ipynb      Table 1 benchmark significance testing
scripts/
  benchmark/      benchmarking, AACT-scale and prospective-registration tooling
  phase1/         figure generation
  lincs/          LINCS directed-propagation analysis
  strengthening/  mechanism, endpoint and decision-layer analyses
  *.py            dataset construction, feature engineering and model retraining
data/
  sources/        modelling cohorts, feature and lookup tables
  herg.tab        hERG blockade labels
model/
  predict.py             self-contained predictor
  model_bundle.pkl       serialised pre-trained model
  serialize_model_bundle.py  rebuilds the bundle from trained artefacts
  Dockerfile, main.nf    container and Nextflow definitions
  test/                  worked example: three compounds and expected output
```

Notebook `00_reproduce_full_pipeline.ipynb` is the canonical driver: it documents the order in which the dataset-construction and retraining scripts are executed.

## Requirements

Python 3.12. The analysis code depends on:

```
pandas
numpy
scipy
scikit-learn
xgboost
lightgbm
networkx
matplotlib
openpyxl
rdkit
```

The versions used for the reported results were pinned as follows:

```
pandas==2.3.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.6.1
xgboost==3.0.3
lightgbm==4.6.0
networkx==3.5
```

`rdkit` is most reliably installed through conda:

```
conda install -c conda-forge rdkit
```

## Running the pre-trained model

`model/` holds the serialised bundle and a self-contained predictor. A worked example with three compounds is provided under `model/test/`.

```
pip install -r model/requirements.txt

python model/predict.py \
    --input-dir model/test/test_input \
    --output predictions.csv \
    --bundle model/model_bundle.pkl
```

An input directory holds `compounds.csv`, `biolprop_merged.tsv`, and `binding/` and `string/` subdirectories of per-compound feature tables. Predictions are returned per compound and disease as `P_FAIL_SAFETY`, `P_FAIL_EFFICACY` and `P_PASS`. Expected output for the supplied example is `model/test/test_output.csv`, and `model/test/run_test.sh` runs and validates it.

The bundle was serialised with the versions pinned in `model/requirements.txt`. Install those versions: scikit-learn does not guarantee correct results when estimators are unpickled under a different release.

To run through Docker and Nextflow instead:

```
docker build -t merit:latest model/
nextflow run model/main.nf --input_dir <dir> --output_dir <dir>
```

## Notes for reviewers

- Notebook outputs are retained deliberately, so the reported figures and metrics can be read directly from the executed cells.
- Absolute filesystem paths appearing in stored notebook outputs have been replaced with the placeholder `<repo>`. No other output content was altered.
- The "Arm-Level Pipeline" section of `01_data_provenance_rebuild_executed.ipynb` (later cells) invokes an earlier arm-level rebuild toolchain that has since been superseded and is not distributed in this repository. Those cells will not execute. The section is retained as a record of provenance; the canonical workflow is the one driven by notebook `00_reproduce_full_pipeline.ipynb`. All other notebooks reference only code included here.
- `scripts/` contains the dependency closure of the entry-point notebooks together with every script named in Supplementary Table S16.
- The locked prospective-registration predictions are archived on Zenodo (see [Data availability](#data-availability)) and carry SHA-256 commitments. Verify them with `sha256sum -c`, normalising line endings first if the files were checked out on Windows (`tr -d '\r' < FILE.csv | sha256sum`).

## Data availability

The modelling cohorts and feature tables needed to reproduce the reported analyses are included in `data/`. The model outputs and the locked prospective predictions are archived on Zenodo:

| Archive | Contents | DOI |
|---|---|---|
| MERIT frozen outputs | `results/` model outputs and `prediction/` locked prospective-registration predictions | https://doi.org/10.5281/zenodo.21824276 |

Download `MERIT.zip` and unpack it at the repository root, restoring `results/` and `prediction/`, before running notebooks that read existing outputs. `SHA256SUMS.txt` is a separate file on the same record and lists every file's SHA-256; place it at the repository root and verify with `sha256sum -c SHA256SUMS.txt`.

The resources below are the public databases and knowledgebases the `data/` tables were derived from; the analyses query or download several of them at run time. Each is subject to its own licence and terms of use. Construction steps are recorded in `01_data_provenance_rebuild_executed.ipynb`.

### Trials and clinical outcomes

| Resource | Use | Link |
|---|---|---|
| ClinicalTrials.gov | Trial registry records, retrieved through the v2 API | https://clinicaltrials.gov |
| AACT (Aggregate Analysis of ClinicalTrials.gov), CTTI | Relational snapshot of the registry | https://aact.ctti-clinicaltrials.org |
| repoDB | Approved and failed drug-indication pairs, for the repositioning recovery set | https://unmtid-shinyapps.net/shiny/repodb/ |

### Compounds, targets and pharmacology

| Resource | Use | Link |
|---|---|---|
| ChEMBL | Compound structures, bioactivity and development phase | https://www.ebi.ac.uk/chembl/ |
| PubChem | Compound identifiers and structure resolution | https://pubchem.ncbi.nlm.nih.gov |
| IUPHAR/BPS Guide to PHARMACOLOGY (GtoPdb) | Target families for the cardiac-axis analysis | https://www.guidetopharmacology.org |
| DruMAP | Predicted pharmacokinetic parameters | https://drumap.nibiohn.go.jp |

### Target-disease biology and genetics

| Resource | Use | Link |
|---|---|---|
| Open Targets Platform | Target-disease association channels, retrieved via the GraphQL API | https://platform.opentargets.org |
| STRING | Protein interaction networks and pathway enrichment, via the REST API | https://string-db.org |
| KEGG | Pathway membership | https://www.genome.jp/kegg/ |
| OmniPath | Directed signalling interactions | https://omnipathdb.org |
| ClinGen | Gene-disease validity classifications | https://clinicalgenome.org |
| DepMap | Gene dependency and essentiality | https://depmap.org/portal/ |
| gnomAD | LOEUF constraint scores | https://gnomad.broadinstitute.org |
| OncoKB | Cancer driver gene annotation | https://www.oncokb.org |
| Human Protein Atlas | Tissue-specific expression | https://www.proteinatlas.org |
| Ensembl | Gene and transcript identifier mapping | https://www.ensembl.org |
| LINCS L1000 (NIH LINCS Program) | Transcriptomic perturbation signatures; consensus signatures derive from the L1000 Connectivity Map, deposited in GEO as GSE92742 (Phase I) and GSE70138 (Phase II) | https://lincsproject.org |
| Drug Repurposing Hub, Broad Institute | Bridges LINCS L1000 signatures to compound structures | https://repo-hub.broadinstitute.org/repurposing |

### Safety and toxicity

| Resource | Use | Link |
|---|---|---|
| Therapeutics Data Commons (TDC) | hERG blockade labels (`data/herg.tab`) and the safety endpoint panel | https://tdcommons.ai |
| DILIrank, FDA Liver Toxicity Knowledge Base | Drug-induced liver injury concern categories | https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-dataset |
| SIDER (Side Effect Resource), EMBL | Clinical adverse-event labels | http://sideeffects.embl.de |
| FAERS | Post-marketing adverse-event reports | https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers |

### Ontologies and classifications

| Resource | Use | Link |
|---|---|---|
| MONDO Disease Ontology | Disease concept normalisation | https://mondo.monarchinitiative.org |
| EFO (Experimental Factor Ontology) | Disease mapping for Open Targets queries | https://www.ebi.ac.uk/efo/ |
| MeSH | Condition term mapping | https://www.nlm.nih.gov/mesh/ |
| WHO ATC | Drug class assignment | https://atcddd.fhi.no/atc_ddd_index/ |

### Comparison benchmarks

| Resource | Use | Link |
|---|---|---|
| HINT | Published trial-outcome prediction benchmark and cohort | https://github.com/futianfan/clinical-trial-outcome-prediction |
| TrialBench | AI-ready clinical trial prediction datasets | https://huyjj.github.io/Trialbench/ |

## License

Released under the MIT License. See [LICENSE](LICENSE).
