#!/usr/bin/env python3
"""
Self-contained prediction script for the trial-outcome prediction model.

Usage:
    python predict.py --input-dir ./input --output predictions.csv --bundle model_bundle.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# ============================================================================
# CONSTANTS — Gene lists (inlined from compute_*.py scripts)
# ============================================================================

FDR_THRESHOLD = 0.05

TISSUE_NAMES = [
    'brain', 'endocrine', 'female_tissues', 'gasrto_intestin', 'heart',
    'hematopoietic', 'immune', 'kidney', 'liver', 'male_tissues',
    'musculoskeletal', 'others', 'respiratory', 'sensory',
]

TISSUE_TO_AREA = {
    'brain': 'cns', 'heart': 'cardiovascular', 'liver': 'metabolic',
    'kidney': 'metabolic', 'immune': 'autoimmune', 'hematopoietic': 'autoimmune',
    'respiratory': 'respiratory', 'gasrto_intestin': 'metabolic',
    'endocrine': 'metabolic',
}

# ── Cancer type mapping ─────────────────────────────────────────────────────
CANCER_TYPE_MAP = {
    'breast': ['breast cancer', 'breast carcinoma', 'breast neoplasm', 'tnbc',
               'triple negative', 'her2', 'er+', 'er-positive', 'lobular',
               'ductal carcinoma', 'mammary'],
    'nsclc': ['non-small cell lung', 'nsclc', 'lung adenocarcinoma', 'lung squamous',
              'non small cell lung'],
    'sclc': ['small cell lung', 'sclc'],
    'lung': ['lung cancer', 'lung carcinoma', 'lung neoplasm'],
    'colorectal': ['colorectal', 'colon cancer', 'rectal cancer', 'colon carcinoma',
                   'colorectal cancer', 'bowel cancer', 'crc'],
    'pancreatic': ['pancreatic', 'pancreas cancer', 'pancreatic ductal',
                   'pancreatic adenocarcinoma'],
    'melanoma': ['melanoma', 'cutaneous melanoma', 'uveal melanoma'],
    'prostate': ['prostate cancer', 'prostate carcinoma', 'castration-resistant',
                 'crpc', 'prostate adenocarcinoma'],
    'ovarian': ['ovarian cancer', 'ovarian carcinoma', 'ovarian neoplasm',
                'fallopian tube', 'epithelial ovarian'],
    'aml': ['acute myeloid leukemia', 'aml', 'acute myelogenous'],
    'all': ['acute lymphoblastic leukemia', 'all', 'acute lymphocytic'],
    'cml': ['chronic myeloid leukemia', 'cml', 'chronic myelogenous'],
    'cll': ['chronic lymphocytic leukemia', 'cll'],
    'dlbcl': ['diffuse large b-cell', 'dlbcl', 'diffuse large b cell'],
    'follicular': ['follicular lymphoma'],
    'hodgkin': ['hodgkin lymphoma', 'hodgkin disease'],
    'myeloma': ['multiple myeloma', 'myeloma', 'plasma cell'],
    'renal': ['renal cell carcinoma', 'rcc', 'kidney cancer', 'renal cancer',
              'clear cell renal'],
    'glioblastoma': ['glioblastoma', 'gbm', 'glioblastoma multiforme'],
    'glioma': ['glioma', 'astrocytoma', 'oligodendroglioma', 'brain tumor',
               'brain cancer'],
    'hepatocellular': ['hepatocellular carcinoma', 'hcc', 'liver cancer',
                       'hepatocellular'],
    'gastric': ['gastric cancer', 'stomach cancer', 'gastric carcinoma',
                'gastric adenocarcinoma', 'gastroesophageal'],
    'esophageal': ['esophageal cancer', 'esophageal carcinoma', 'esophageal squamous'],
    'bladder': ['bladder cancer', 'urothelial carcinoma', 'urothelial cancer',
                'transitional cell', 'bladder carcinoma'],
    'head_neck': ['head and neck', 'squamous cell carcinoma of the head',
                  'head and neck squamous', 'hnscc', 'oral cancer',
                  'oropharyngeal', 'nasopharyngeal', 'laryngeal'],
    'thyroid': ['thyroid cancer', 'thyroid carcinoma', 'papillary thyroid',
                'anaplastic thyroid', 'medullary thyroid'],
    'endometrial': ['endometrial cancer', 'uterine cancer', 'endometrial carcinoma'],
    'cervical': ['cervical cancer', 'cervical carcinoma'],
    'sarcoma': ['sarcoma', 'osteosarcoma', 'soft tissue sarcoma', 'ewing sarcoma',
                'rhabdomyosarcoma', 'liposarcoma', 'leiomyosarcoma',
                'gastrointestinal stromal'],
    'mesothelioma': ['mesothelioma', 'malignant mesothelioma'],
    'neuroblastoma': ['neuroblastoma'],
    'mds': ['myelodysplastic', 'mds'],
}

# ── Cancer drivers (gene→role per cancer type) ──────────────────────────────
CANCER_DRIVERS = {
    'breast': {
        'PIK3CA': 'O', 'TP53': 'T', 'GATA3': 'T', 'CDH1': 'T', 'MAP3K1': 'T',
        'AKT1': 'O', 'PTEN': 'T', 'NF1': 'T', 'RB1': 'T', 'ERBB2': 'O',
        'BRCA1': 'T', 'BRCA2': 'T', 'ESR1': 'O', 'MYC': 'O', 'EGFR': 'O',
        'FOXA1': 'O', 'TBX3': 'T', 'CBFB': 'T', 'RUNX1': 'T', 'MTOR': 'O',
    },
    'nsclc': {
        'KRAS': 'O', 'EGFR': 'O', 'ALK': 'O', 'TP53': 'T', 'STK11': 'T',
        'KEAP1': 'T', 'NF1': 'T', 'BRAF': 'O', 'ROS1': 'O', 'RET': 'O',
        'MET': 'O', 'ERBB2': 'O', 'PIK3CA': 'O', 'CDKN2A': 'T', 'SMARCA4': 'T',
        'ARID1A': 'T', 'FGFR1': 'O', 'NFE2L2': 'O',
    },
    'sclc': {
        'TP53': 'T', 'RB1': 'T', 'NOTCH1': 'B', 'MYC': 'O', 'PTEN': 'T',
        'CREBBP': 'T', 'EP300': 'T', 'FGFR1': 'O',
    },
    'lung': {
        'KRAS': 'O', 'EGFR': 'O', 'ALK': 'O', 'TP53': 'T', 'STK11': 'T',
        'KEAP1': 'T', 'BRAF': 'O', 'ROS1': 'O', 'RET': 'O', 'MET': 'O',
        'ERBB2': 'O', 'PIK3CA': 'O', 'CDKN2A': 'T', 'RB1': 'T', 'NOTCH1': 'B',
        'MYC': 'O', 'NF1': 'T', 'FGFR1': 'O',
    },
    'colorectal': {
        'APC': 'T', 'TP53': 'T', 'KRAS': 'O', 'PIK3CA': 'O', 'BRAF': 'O',
        'SMAD4': 'T', 'FBXW7': 'T', 'NRAS': 'O', 'PTEN': 'T', 'CTNNB1': 'O',
        'ARID1A': 'T', 'RNF43': 'T', 'ERBB2': 'O', 'MLH1': 'T', 'MSH2': 'T',
    },
    'pancreatic': {
        'KRAS': 'O', 'TP53': 'T', 'CDKN2A': 'T', 'SMAD4': 'T', 'ARID1A': 'T',
        'TGFBR2': 'T', 'BRCA2': 'T', 'ATM': 'T', 'PALB2': 'T', 'STK11': 'T',
        'RNF43': 'T',
    },
    'melanoma': {
        'BRAF': 'O', 'NRAS': 'O', 'NF1': 'T', 'TP53': 'T', 'CDKN2A': 'T',
        'PTEN': 'T', 'KIT': 'O', 'RAC1': 'O', 'MAP2K1': 'O', 'IDH1': 'O',
    },
    'prostate': {
        'AR': 'O', 'PTEN': 'T', 'TP53': 'T', 'SPOP': 'T', 'FOXA1': 'O',
        'PIK3CA': 'O', 'RB1': 'T', 'BRCA2': 'T', 'BRCA1': 'T', 'ATM': 'T',
        'CDK12': 'T', 'MYC': 'O', 'ERG': 'O',
    },
    'ovarian': {
        'TP53': 'T', 'BRCA1': 'T', 'BRCA2': 'T', 'NF1': 'T', 'RB1': 'T',
        'CDK12': 'T', 'CCNE1': 'O', 'MYC': 'O', 'PTEN': 'T', 'PIK3CA': 'O',
        'KRAS': 'O', 'ARID1A': 'T',
    },
    'aml': {
        'FLT3': 'O', 'NPM1': 'O', 'DNMT3A': 'T', 'IDH1': 'O', 'IDH2': 'O',
        'TET2': 'T', 'RUNX1': 'T', 'CEBPA': 'T', 'TP53': 'T', 'KIT': 'O',
        'NRAS': 'O', 'KRAS': 'O', 'ASXL1': 'T',
    },
    'all': {
        'PAX5': 'T', 'IKZF1': 'T', 'CDKN2A': 'T', 'JAK2': 'O', 'NOTCH1': 'O',
        'FBXW7': 'T', 'PTEN': 'T', 'TP53': 'T', 'NRAS': 'O', 'KRAS': 'O',
        'CREBBP': 'T',
    },
    'cml': {'ABL1': 'O', 'BCR': 'O', 'ASXL1': 'T', 'RUNX1': 'T', 'IKZF1': 'T'},
    'cll': {
        'TP53': 'T', 'ATM': 'T', 'NOTCH1': 'O', 'SF3B1': 'O', 'BIRC3': 'T',
        'MYD88': 'O',
    },
    'dlbcl': {
        'MYD88': 'O', 'CD79A': 'O', 'CD79B': 'O', 'BCL2': 'O', 'BCL6': 'O',
        'MYC': 'O', 'EZH2': 'O', 'CREBBP': 'T', 'KMT2D': 'T', 'TP53': 'T',
        'CARD11': 'O',
    },
    'follicular': {'BCL2': 'O', 'KMT2D': 'T', 'CREBBP': 'T', 'EZH2': 'O'},
    'hodgkin': {'JAK2': 'O', 'SOCS1': 'T', 'B2M': 'T', 'STAT6': 'O'},
    'myeloma': {
        'KRAS': 'O', 'NRAS': 'O', 'BRAF': 'O', 'TP53': 'T', 'DIS3': 'T',
        'TRAF3': 'T', 'IRF4': 'O', 'MYC': 'O', 'RB1': 'T', 'FGFR3': 'O',
    },
    'renal': {
        'VHL': 'T', 'PBRM1': 'T', 'SETD2': 'T', 'BAP1': 'T', 'KDM5C': 'T',
        'MTOR': 'O', 'PIK3CA': 'O', 'PTEN': 'T', 'TP53': 'T', 'MET': 'O',
    },
    'glioblastoma': {
        'EGFR': 'O', 'TP53': 'T', 'PTEN': 'T', 'CDKN2A': 'T', 'NF1': 'T',
        'PIK3CA': 'O', 'RB1': 'T', 'CDK4': 'O', 'MDM2': 'O', 'PDGFRA': 'O',
        'IDH1': 'O',
    },
    'glioma': {
        'IDH1': 'O', 'IDH2': 'O', 'TP53': 'T', 'ATRX': 'T', 'EGFR': 'O',
        'PTEN': 'T', 'CDKN2A': 'T', 'NF1': 'T', 'PIK3CA': 'O',
    },
    'hepatocellular': {
        'TP53': 'T', 'CTNNB1': 'O', 'AXIN1': 'T', 'ARID1A': 'T', 'ARID2': 'T',
        'ALB': 'B', 'TERT': 'O', 'NFE2L2': 'O', 'KEAP1': 'T', 'RB1': 'T',
    },
    'gastric': {
        'TP53': 'T', 'PIK3CA': 'O', 'ARID1A': 'T', 'CDH1': 'T', 'ERBB2': 'O',
        'KRAS': 'O', 'RHOA': 'O', 'MYC': 'O', 'FBXW7': 'T',
    },
    'esophageal': {
        'TP53': 'T', 'CDKN2A': 'T', 'NFE2L2': 'O', 'NOTCH1': 'B', 'PIK3CA': 'O',
        'KMT2D': 'T', 'ERBB2': 'O',
    },
    'bladder': {
        'TP53': 'T', 'FGFR3': 'O', 'PIK3CA': 'O', 'CDKN2A': 'T', 'RB1': 'T',
        'ARID1A': 'T', 'KDM6A': 'T', 'ERBB2': 'O', 'STAG2': 'T',
        'TERT': 'O', 'HRAS': 'O',
    },
    'head_neck': {
        'TP53': 'T', 'CDKN2A': 'T', 'PIK3CA': 'O', 'NOTCH1': 'B', 'HRAS': 'O',
        'FBXW7': 'T', 'CASP8': 'T', 'FAT1': 'T', 'NFE2L2': 'O', 'EGFR': 'O',
    },
    'thyroid': {
        'BRAF': 'O', 'RAS': 'O', 'NRAS': 'O', 'HRAS': 'O', 'RET': 'O',
        'PAX8': 'O', 'PPARG': 'O', 'TP53': 'T', 'TERT': 'O', 'PIK3CA': 'O',
    },
    'endometrial': {
        'PTEN': 'T', 'PIK3CA': 'O', 'TP53': 'T', 'ARID1A': 'T', 'CTNNB1': 'O',
        'KRAS': 'O', 'PIK3R1': 'T', 'FBXW7': 'T', 'PPP2R1A': 'T',
    },
    'cervical': {
        'PIK3CA': 'O', 'PTEN': 'T', 'TP53': 'T', 'KRAS': 'O', 'FBXW7': 'T',
        'HLA-A': 'T', 'HLA-B': 'T', 'ERBB2': 'O', 'STK11': 'T',
    },
    'sarcoma': {
        'TP53': 'T', 'RB1': 'T', 'ATRX': 'T', 'MDM2': 'O', 'CDK4': 'O',
        'KIT': 'O', 'PDGFRA': 'O', 'EWSR1': 'O',
    },
    'mds': {
        'TET2': 'T', 'SF3B1': 'O', 'ASXL1': 'T', 'DNMT3A': 'T', 'RUNX1': 'T',
        'TP53': 'T', 'SRSF2': 'O', 'U2AF1': 'O', 'EZH2': 'T', 'STAG2': 'T',
    },
    'other_cancer': {
        'TP53': 'T', 'KRAS': 'O', 'PIK3CA': 'O', 'PTEN': 'T', 'RB1': 'T',
        'CDKN2A': 'T', 'MYC': 'O', 'BRAF': 'O', 'EGFR': 'O', 'ERBB2': 'O',
        'NF1': 'T', 'APC': 'T', 'NRAS': 'O',
    },
}

# ── Oncogenic pathways ──────────────────────────────────────────────────────
ONCOGENIC_PATHWAYS = {
    'MAPK_RAS': {
        'EGFR', 'ERBB2', 'ERBB3', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4',
        'MET', 'PDGFRA', 'KIT', 'ALK', 'ROS1', 'RET',
        'KRAS', 'NRAS', 'HRAS', 'NF1', 'BRAF', 'RAF1', 'MAP2K1', 'MAP2K2',
        'MAPK1', 'MAPK3',
    },
    'PI3K_AKT_MTOR': {
        'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3R1', 'PIK3R2',
        'PTEN', 'INPP4B', 'TSC1', 'TSC2',
        'AKT1', 'AKT2', 'AKT3', 'MTOR', 'RPTOR', 'RICTOR', 'STK11',
    },
    'Cell_Cycle': {
        'CDK1', 'CDK2', 'CDK4', 'CDK6',
        'CCNA1', 'CCNA2', 'CCNB1', 'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CCNE2',
        'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'RB1', 'E2F1', 'MDM2', 'MDM4', 'TP53',
    },
    'DDR': {
        'BRCA1', 'BRCA2', 'RAD51', 'RAD51C', 'RAD51D', 'PALB2',
        'ATM', 'ATR', 'CHEK1', 'CHEK2', 'TP53',
        'MLH1', 'MSH2', 'MSH6', 'PMS2', 'PARP1', 'PARP2',
    },
    'Chromatin': {
        'SMARCA4', 'SMARCB1', 'ARID1A', 'ARID1B', 'ARID2', 'PBRM1',
        'EZH2', 'KMT2A', 'KMT2C', 'KMT2D', 'SETD2', 'NSD1',
        'KDM5A', 'KDM5C', 'KDM6A', 'CREBBP', 'EP300', 'BRD4',
        'DNMT1', 'DNMT3A', 'DNMT3B', 'TET1', 'TET2', 'IDH1', 'IDH2', 'BAP1', 'ASXL1',
    },
    'Wnt': {'APC', 'AXIN1', 'AXIN2', 'CTNNB1', 'GSK3B', 'RNF43', 'ZNRF3'},
    'Notch': {
        'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4',
        'DLL1', 'DLL3', 'DLL4', 'JAG1', 'JAG2', 'FBXW7',
    },
    'JAK_STAT': {
        'JAK1', 'JAK2', 'JAK3', 'TYK2',
        'STAT1', 'STAT3', 'STAT5A', 'STAT5B', 'STAT6', 'SOCS1', 'SOCS3',
    },
    'Immune_Checkpoint': {
        'CD274', 'PDCD1', 'CTLA4', 'LAG3', 'TIGIT', 'HAVCR2',
        'B2M', 'HLA-A', 'HLA-B', 'HLA-C', 'TAP1', 'TAP2',
        'JAK1', 'JAK2', 'IFNGR1', 'IRF1',
    },
}

# ── Domain gene lists ───────────────────────────────────────────────────────
IMMUNE_CHECKPOINT_GENES = [
    'PDCD1', 'CD274', 'PDCD1LG2', 'CTLA4', 'CD80', 'CD86',
    'LAG3', 'HAVCR2', 'TIGIT', 'VSIR', 'CD276', 'VTCN1',
    'IDO1', 'IDO2', 'TDO2', 'CD47', 'SIRPA',
    'TNFRSF4', 'TNFRSF9', 'TNFRSF18',
]
T_CELL_GENES = [
    'CD3E', 'CD3D', 'CD3G', 'CD28', 'ICOS',
    'LCK', 'ZAP70', 'ITK', 'PLCG1',
    'IL2', 'IL2RA', 'IL2RB', 'IL15', 'IL15RA',
    'IFNG', 'TNF', 'GZMB', 'PRF1', 'FASLG',
    'CD8A', 'CD8B', 'CD4', 'EOMES', 'TBX21', 'FOXP3', 'BATF', 'IRF4',
]
NK_CELL_GENES = [
    'NCR1', 'NCR2', 'NCR3', 'KLRK1', 'KLRD1', 'KLRC1',
    'FCGR3A', 'SH2D1B', 'SLAMF7', 'KIR2DL1', 'KIR2DL3', 'KIR3DL1',
]
MACROPHAGE_GENES = [
    'CSF1R', 'CSF1', 'CD163', 'MRC1', 'CCR2', 'CCL2',
    'TREM2', 'SIGLEC10', 'ARG1', 'NOS2', 'MARCO', 'MSR1',
]
TME_GENES = [
    'VEGFA', 'KDR', 'FLT1', 'FLT4', 'PDGFRA', 'PDGFRB', 'PDGFB',
    'TGFB1', 'TGFB2', 'TGFBR1', 'TGFBR2', 'FAP', 'ACTA2',
    'MMP2', 'MMP9', 'MMP14', 'HIF1A', 'EPAS1', 'CXCL12', 'CXCR4',
]
DDR_GENES = [
    'BRCA1', 'BRCA2', 'PALB2', 'PARP1', 'PARP2',
    'ATM', 'ATR', 'CHEK1', 'CHEK2',
    'RAD51', 'RAD51C', 'RAD51D', 'WRN', 'BLM', 'FANCA', 'FANCD2',
    'TP53', 'MDM2', 'CDKN2A',
]
CELL_CYCLE_DRUG_TARGETS = [
    'CDK4', 'CDK6', 'CCND1', 'CDK2', 'CDK1', 'CCNE1',
    'PLK1', 'AURKA', 'AURKB', 'TTK', 'BUB1', 'BUB1B', 'CHEK1', 'WEE1',
]
ESSENTIAL_GENES_DOMAIN = [
    'RPL3', 'RPL4', 'RPL5', 'RPL7', 'RPL8', 'RPL11', 'RPL13', 'RPL14',
    'RPS2', 'RPS3', 'RPS5', 'RPS6', 'RPS8', 'RPS9', 'RPS13', 'RPS14',
    'PSMA1', 'PSMA2', 'PSMA3', 'PSMA4', 'PSMA5', 'PSMA6', 'PSMA7',
    'PSMB1', 'PSMB2', 'PSMB3', 'PSMB4', 'PSMB5',
    'SF3B1', 'SF3B3', 'SNRPD1', 'SNRPD2', 'PRPF8',
    'POLR2A', 'POLR2B', 'POLR2C',
    'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7',
    'PCNA', 'RFC1', 'RFC2', 'RFC3',
    'EIF3A', 'EIF3B', 'EIF4A1',
    'HSP90AA1', 'HSP90AB1', 'HSPA5', 'HSPA8',
    'ACTB', 'TUBB', 'GAPDH', 'UBA1',
]

# ── Vital organ genes ───────────────────────────────────────────────────────
VITAL_ORGAN_GENES = {
    'heart': [
        'TNNT2', 'MYH7', 'MYH6', 'MYBPC3', 'SCN5A', 'KCNH2', 'KCNQ1',
        'RYR2', 'CASQ2', 'PLN', 'ACTC1', 'MYL2', 'MYL3', 'TNNI3',
        'TNNC1', 'TTN', 'LMNA', 'DES', 'GJA1', 'GJA5',
        'HCN4', 'CACNA1C', 'CACNA1D', 'ATP2A2', 'SLC8A1',
        'NKX2-5', 'TBX5', 'GATA4', 'HAND2', 'MEF2C',
        'NPPA', 'NPPB', 'BNP', 'ANP', 'CORIN',
        'ADRB1', 'ADRB2', 'CHRM2', 'KCNJ2', 'KCNJ11',
        'SLC25A4', 'CKM', 'MB', 'MYL7', 'MYOZ2',
        'PKP2', 'DSP', 'DSG2', 'DSC2', 'JUP',
    ],
    'brain': [
        'SYN1', 'SYN2', 'SYP', 'SNAP25', 'STX1A', 'VAMP2',
        'GRIN1', 'GRIN2A', 'GRIN2B', 'GRIA1', 'GRIA2',
        'DRD1', 'DRD2', 'DRD3', 'DRD4', 'SLC6A3', 'SLC6A4', 'SLC6A2',
        'HTR1A', 'HTR2A', 'HTR2C', 'GABRA1', 'GABRG2', 'GABBR1', 'GAD1', 'GAD2',
        'MAPT', 'MAP2', 'NEFL', 'NEFM', 'NEFH',
        'CHAT', 'TH', 'DBH', 'TPH2', 'DDC',
        'BDNF', 'NTRK2', 'NGF', 'NTRK1', 'MBP', 'PLP1', 'MOG', 'MAG', 'MOBP',
        'RBFOX3', 'NEUROD1', 'ENO2', 'SLC17A7', 'SLC17A6',
        'ACHE', 'CHRNA4', 'CHRNB2', 'SCN1A', 'SCN2A',
    ],
    'liver': [
        'ALB', 'AFP', 'SERPINA1', 'FGA', 'FGB', 'FGG',
        'CYP3A4', 'CYP2D6', 'CYP2C9', 'CYP2C19', 'CYP1A2', 'CYP2E1', 'CYP2B6', 'CYP2A6',
        'UGT1A1', 'UGT2B7', 'UGT2B15', 'ABCB1', 'ABCB11', 'ABCC2', 'ABCG2',
        'SLC22A1', 'SLC22A7', 'SLCO1B1', 'SLCO1B3', 'HNF4A', 'HNF1A', 'CEBPA',
        'APOA1', 'APOB', 'APOC3', 'APOE', 'PCK1', 'G6PC', 'GCK', 'HMGCR',
        'F2', 'F5', 'F7', 'F8', 'F9', 'F10', 'PROC', 'SERPINC1',
        'TAT', 'ASS1', 'ASL', 'OTC', 'CPS1', 'ALDOB', 'HPD', 'FAH', 'GALT',
    ],
    'kidney': [
        'SLC22A6', 'SLC22A8', 'SLC22A2', 'SLC22A11',
        'AQP1', 'AQP2', 'AQP3', 'AQP4', 'UMOD', 'SLC12A1', 'SLC12A3', 'KCNJ1',
        'NPHS1', 'NPHS2', 'PODXL', 'WT1', 'CD2AP',
        'SLC5A1', 'SLC5A2', 'SLC2A2', 'REN', 'AGT', 'ACE', 'ACE2', 'AGTR1',
        'AVPR2', 'SCNN1A', 'SCNN1B', 'SCNN1G', 'SLC4A1', 'SLC4A4', 'CA2', 'CA4',
        'HSD11B2', 'NR3C2', 'CLCNKA', 'CLCNKB', 'BSND',
        'SLC34A1', 'SLC34A3', 'NKCC2', 'CUBN', 'AMN', 'LRP2',
        'TRPC6', 'TRPM6', 'CLDN16', 'PAX2', 'PAX8', 'HNF1B', 'SALL1', 'EPO', 'EPOR',
    ],
}

# ── Expanded essential genes (for oncology selectivity) ─────────────────────
ESSENTIAL_GENES_EXPANDED = [
    'RPL3', 'RPL4', 'RPL5', 'RPL6', 'RPL7', 'RPL8', 'RPL9', 'RPL10',
    'RPL11', 'RPL13', 'RPL14', 'RPL18', 'RPL23', 'RPL26', 'RPL27',
    'RPS2', 'RPS3', 'RPS5', 'RPS6', 'RPS8', 'RPS9', 'RPS13', 'RPS14',
    'RPS15', 'RPS19', 'RPS24',
    'PSMA1', 'PSMA2', 'PSMA3', 'PSMA4', 'PSMA5', 'PSMA6', 'PSMA7',
    'PSMB1', 'PSMB2', 'PSMB3', 'PSMB4', 'PSMB5', 'PSMB6', 'PSMB7',
    'SF3B1', 'SF3B3', 'SF3A1', 'SNRPD1', 'SNRPD2', 'SNRPD3',
    'SNRPE', 'SNRPF', 'PRPF8', 'PRPF31',
    'POLR2A', 'POLR2B', 'POLR2C', 'POLR2D', 'POLR2E', 'POLR2H',
    'CDK1', 'CDK2', 'PLK1', 'AURKA', 'AURKB', 'BUB1B', 'CDC20',
    'CCNA2', 'CCNB1', 'CCNE1',
    'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7',
    'PCNA', 'RFC1', 'RFC2', 'RFC3', 'RFC4', 'RFC5', 'POLA1', 'POLD1', 'POLE',
    'EIF3A', 'EIF3B', 'EIF3C', 'EIF4A1', 'EIF4G1', 'EIF2S1',
    'HSP90AA1', 'HSP90AB1', 'HSPA5', 'HSPA8', 'CCT2', 'CCT3', 'CCT5',
    'ACTB', 'TUBB', 'TUBA1B', 'GAPDH', 'UBA1', 'RAN', 'SUPT5H', 'NUP93', 'NUP107',
]

# ── Pathogen mapping (tuple format for domain features) ─────────────────────
PATHOGEN_MAP_TUPLE = {
    'hiv': (['hiv', 'aids', 'human immunodeficiency'], 'viral'),
    'hcv': (['hepatitis c', 'hcv'], 'viral'),
    'hbv': (['hepatitis b', 'hbv'], 'viral'),
    'covid': (['covid', 'sars-cov', 'coronavirus', '2019-ncov'], 'viral'),
    'influenza': (['influenza', 'flu'], 'viral'),
    'rsv': (['rsv', 'respiratory syncytial'], 'viral'),
    'herpes': (['herpes', 'hsv', 'cmv', 'ebv', 'cytomegalovirus', 'varicella', 'zoster'], 'viral'),
    'tuberculosis': (['tuberculosis', 'mycobacterium'], 'bacterial'),
    'staph': (['staphylococcus', 'mrsa'], 'bacterial'),
    'strep': (['streptococcus'], 'bacterial'),
    'ecoli': (['escherichia', 'e. coli'], 'bacterial'),
    'pseudomonas': (['pseudomonas'], 'bacterial'),
    'fungal': (['aspergill', 'candida', 'fungal', 'mycosis'], 'fungal'),
    'malaria': (['malaria', 'plasmodium'], 'parasitic'),
    'general_bacterial': (['bacteri', 'sepsis', 'pneumonia', 'antibiotic'], 'bacterial'),
    'general_infection': (['infect'], 'mixed'),
}

# ── Pathogen mapping (dict format for infectious features) ──────────────────
PATHOGEN_MAP_DICT = {
    'hiv': {'keywords': ['hiv', 'aids', 'human immunodeficiency'], 'class': 'viral', 'subtype': 'retrovirus'},
    'hcv': {'keywords': ['hepatitis c', 'hcv'], 'class': 'viral', 'subtype': 'flavivirus'},
    'hbv': {'keywords': ['hepatitis b', 'hbv'], 'class': 'viral', 'subtype': 'hepadnavirus'},
    'covid': {'keywords': ['covid', 'sars-cov', 'coronavirus', '2019-ncov', 'sars coronavirus'], 'class': 'viral', 'subtype': 'coronavirus'},
    'influenza': {'keywords': ['influenza', 'flu'], 'class': 'viral', 'subtype': 'orthomyxovirus'},
    'rsv': {'keywords': ['rsv', 'respiratory syncytial'], 'class': 'viral', 'subtype': 'paramyxovirus'},
    'herpes': {'keywords': ['herpes', 'hsv', 'cmv', 'ebv', 'cytomegalovirus', 'epstein-barr', 'varicella', 'zoster'], 'class': 'viral', 'subtype': 'herpesvirus'},
    'tuberculosis': {'keywords': ['tuberculosis', 'mycobacterium'], 'class': 'bacterial', 'subtype': 'mycobacterium'},
    'staph': {'keywords': ['staphylococcus', 'mrsa', 'staph'], 'class': 'bacterial', 'subtype': 'gram_positive'},
    'strep': {'keywords': ['streptococcus', 'strep'], 'class': 'bacterial', 'subtype': 'gram_positive'},
    'ecoli': {'keywords': ['escherichia', 'e. coli', 'e.coli'], 'class': 'bacterial', 'subtype': 'gram_negative'},
    'pseudomonas': {'keywords': ['pseudomonas'], 'class': 'bacterial', 'subtype': 'gram_negative'},
    'fungal': {'keywords': ['aspergill', 'candida', 'fungal', 'mycosis', 'antifungal'], 'class': 'fungal', 'subtype': 'fungal'},
    'malaria': {'keywords': ['malaria', 'plasmodium'], 'class': 'parasitic', 'subtype': 'protozoan'},
    'general_bacterial': {'keywords': ['bacteri', 'sepsis', 'pneumonia', 'antimicrob', 'antibiotic'], 'class': 'bacterial', 'subtype': 'general'},
    'general_infection': {'keywords': ['infect'], 'class': 'mixed', 'subtype': 'general'},
}

PATHOGEN_CHARS = {
    'hiv': (0.95, 8, 0.7), 'hcv': (0.80, 5, 0.3), 'hbv': (0.40, 4, 0.2),
    'covid': (0.70, 6, 0.3), 'influenza': (0.90, 3, 0.5), 'rsv': (0.30, 2, 0.1),
    'herpes': (0.20, 4, 0.2), 'tuberculosis': (0.30, 10, 0.6), 'staph': (0.60, 8, 0.7),
    'strep': (0.40, 6, 0.3), 'ecoli': (0.70, 10, 0.6), 'pseudomonas': (0.80, 8, 0.8),
    'fungal': (0.30, 5, 0.3), 'malaria': (0.60, 6, 0.5),
    'general_bacterial': (0.50, 8, 0.5), 'general_infection': (0.50, 5, 0.4),
}

PATHOGEN_CHARACTERISTICS = {
    'hiv': {'plasticity': 0.95, 'n_targets': 8, 'resistance': 0.7},
    'hcv': {'plasticity': 0.80, 'n_targets': 5, 'resistance': 0.3},
    'hbv': {'plasticity': 0.40, 'n_targets': 4, 'resistance': 0.2},
    'covid': {'plasticity': 0.70, 'n_targets': 6, 'resistance': 0.3},
    'influenza': {'plasticity': 0.90, 'n_targets': 3, 'resistance': 0.5},
    'rsv': {'plasticity': 0.30, 'n_targets': 2, 'resistance': 0.1},
    'herpes': {'plasticity': 0.20, 'n_targets': 4, 'resistance': 0.2},
    'tuberculosis': {'plasticity': 0.30, 'n_targets': 10, 'resistance': 0.6},
    'staph': {'plasticity': 0.60, 'n_targets': 8, 'resistance': 0.7},
    'strep': {'plasticity': 0.40, 'n_targets': 6, 'resistance': 0.3},
    'ecoli': {'plasticity': 0.70, 'n_targets': 10, 'resistance': 0.6},
    'pseudomonas': {'plasticity': 0.80, 'n_targets': 8, 'resistance': 0.8},
    'fungal': {'plasticity': 0.30, 'n_targets': 5, 'resistance': 0.3},
    'malaria': {'plasticity': 0.60, 'n_targets': 6, 'resistance': 0.5},
    'general_bacterial': {'plasticity': 0.50, 'n_targets': 8, 'resistance': 0.5},
    'general_infection': {'plasticity': 0.50, 'n_targets': 5, 'resistance': 0.4},
}

# ── Host entry factors ──────────────────────────────────────────────────────
HOST_ENTRY_FACTORS = {
    'hiv': ['CCR5', 'CXCR4', 'CD4', 'PSIP1', 'TNPO3', 'NUP153'],
    'hcv': ['CLDN1', 'OCLN', 'SCARB1', 'CD81', 'LDLR', 'EGFR', 'PPIA'],
    'hbv': ['SLC10A1', 'ADAR'],
    'covid': ['ACE2', 'TMPRSS2', 'CTSL', 'FURIN', 'NRP1', 'BSG'],
    'influenza': ['SLC35A1', 'ST3GAL4', 'ST6GAL1'],
    'rsv': ['ICAM1', 'CX3CR1', 'HSPG2', 'IGF1R'],
    'herpes': ['TNFRSF14', 'NECTIN1', 'NECTIN2'],
    'tuberculosis': ['TLR2', 'TLR4', 'IFNGR1', 'IFNGR2', 'NOD2'],
    'staph': ['TLR2', 'TLR6', 'ADAM10', 'NLRP3'],
    'strep': ['TLR2', 'TLR4', 'CD14'],
    'ecoli': ['TLR4', 'TLR5', 'CD14', 'NOD1', 'NOD2'],
    'pseudomonas': ['TLR4', 'TLR5', 'CFTR'],
    'fungal': ['CLEC7A', 'CLEC6A', 'TLR2', 'SYK', 'CARD9'],
    'malaria': ['GYPA', 'GYPB', 'CR1', 'BSG', 'ABCB6'],
    'general_bacterial': ['TLR2', 'TLR4', 'TLR5', 'NOD1', 'NOD2', 'NLRP3', 'CD14'],
    'general_infection': ['TLR2', 'TLR3', 'TLR4', 'TLR7', 'TLR9', 'MAVS', 'STING1'],
}

PATHOGEN_HOST_FACTORS = {
    'hiv': ['CCR5', 'CXCR4', 'CD4', 'TRIM5', 'APOBEC3G', 'BST2', 'SAMHD1', 'LEDGF', 'PSIP1', 'TNPO3', 'NUP153', 'CPSF6', 'CUL5', 'NEDD8', 'UBE2M'],
    'hcv': ['CLDN1', 'OCLN', 'SCARB1', 'CD81', 'LDLR', 'EGFR', 'PPIA', 'FKBP8', 'SEC14L2', 'DGAT1', 'MTP'],
    'hbv': ['NTCP', 'SLC10A1', 'ADAR', 'APOBEC3B', 'TLR2', 'TLR3', 'STING1', 'MAVS'],
    'covid': ['ACE2', 'TMPRSS2', 'CTSL', 'FURIN', 'NRP1', 'BSG', 'HSPA5', 'ADAM17', 'TMEM106B', 'RAB7A', 'PIK3C3'],
    'influenza': ['SLC35A1', 'ST3GAL4', 'ST6GAL1', 'IFITM1', 'IFITM2', 'IFITM3', 'ANP32A', 'ANP32B', 'IMPORTIN'],
    'rsv': ['ICAM1', 'CX3CR1', 'HSPG2', 'IGF1R', 'TLR4', 'TLR2', 'RIG1'],
    'herpes': ['HVEM', 'TNFRSF14', 'NECTIN1', 'NECTIN2', 'PILRA', 'MHC1', 'HLA-A', 'HLA-B', 'HLA-C', 'STING1', 'CGAS', 'IFI16'],
    'tuberculosis': ['TLR2', 'TLR4', 'TLR9', 'IFNGR1', 'IFNGR2', 'IL12RB1', 'IL23R', 'STAT1', 'STAT4', 'NOD2', 'CARD9', 'DC-SIGN', 'MTOR', 'ATG5', 'ATG7', 'BECN1', 'VDR', 'CAMP'],
    'staph': ['TLR2', 'TLR6', 'ADAM10', 'NLRP3', 'CASP1', 'IL1B', 'CD11B', 'ITGAM', 'CR3'],
    'strep': ['TLR2', 'TLR4', 'CD14', 'PLG', 'PLAU', 'PLAUR'],
    'ecoli': ['TLR4', 'TLR5', 'CD14', 'LBP', 'NOD1', 'NOD2', 'NLRP3'],
    'pseudomonas': ['TLR4', 'TLR5', 'CFTR', 'NLRC4', 'CASP1'],
    'fungal': ['CLEC7A', 'CLEC6A', 'TLR2', 'TLR4', 'CARD9', 'SYK', 'PLCG2', 'DECTIN1', 'MINCLE'],
    'malaria': ['GYPA', 'GYPB', 'GYPC', 'DARC', 'CR1', 'BSG', 'CD147', 'ABCB6', 'HBB', 'HBA1', 'G6PD'],
    'general_bacterial': ['TLR2', 'TLR4', 'TLR5', 'TLR9', 'NOD1', 'NOD2', 'NLRP3', 'CD14', 'LBP', 'CASP1'],
    'general_infection': ['TLR2', 'TLR3', 'TLR4', 'TLR7', 'TLR8', 'TLR9', 'MAVS', 'STING1', 'IFNAR1', 'IFNAR2', 'STAT1', 'IRF3', 'IRF7'],
}

# ── Defense gene lists ──────────────────────────────────────────────────────
ANTIVIRAL_DEFENSE = [
    'IFNA1', 'IFNA2', 'IFNB1', 'IFNAR1', 'IFNAR2', 'IRF3', 'IRF7', 'IRF9', 'STAT1', 'STAT2',
    'TBK1', 'IKBKE', 'MAVS', 'STING1', 'CGAS', 'MX1', 'MX2', 'OAS1', 'OAS2', 'OAS3',
    'ISG15', 'IFIT1', 'IFIT2', 'IFIT3', 'EIF2AK2', 'ADAR', 'APOBEC3G',
    'IFITM1', 'IFITM2', 'IFITM3', 'CD8A', 'PRF1', 'GZMA', 'GZMB', 'IFNG',
]
ANTIBACTERIAL_DEFENSE = [
    'TLR2', 'TLR4', 'TLR5', 'TLR9', 'NOD1', 'NOD2', 'NLRP3', 'CASP1', 'IL1B', 'IL18',
    'TNF', 'IL6', 'CXCL8', 'CYBB', 'NCF1', 'NCF2', 'MPO', 'CAMP', 'DEFA1', 'LYZ',
    'C3', 'C5', 'C3AR1', 'C5AR1', 'IL17A', 'IL17F', 'IL22', 'IL23A', 'CXCL1', 'CXCL2',
]
ANTIFUNGAL_DEFENSE = [
    'CLEC7A', 'CLEC6A', 'SYK', 'CARD9', 'TLR2', 'IL17A', 'IL17F', 'IL22', 'IL23A',
    'RORC', 'STAT3', 'CYBB', 'NCF1', 'MPO', 'TNF', 'IL1B', 'IL6', 'CAMP', 'DEFA1',
]
ANTIPARASITIC_DEFENSE = [
    'IFNG', 'IL12A', 'IL12B', 'STAT4', 'TBX21', 'TNF', 'NOS2',
    'IL4', 'IL5', 'IL13', 'IL33', 'STAT6', 'GATA3', 'IL5RA', 'CCR3', 'FCER1A', 'EPX', 'RNASE3',
]
DEFENSE_MAP = {
    'viral': ANTIVIRAL_DEFENSE, 'bacterial': ANTIBACTERIAL_DEFENSE,
    'fungal': ANTIFUNGAL_DEFENSE, 'parasitic': ANTIPARASITIC_DEFENSE,
}

HOST_PATHOGEN_HOMOLOGS = {
    'dhfr': ['DHFR', 'DHFR2'], 'topoisomerase': ['TOP1', 'TOP2A', 'TOP2B'],
    'ribosome': ['RPL3', 'RPL4', 'RPS12', 'RPS3'], 'rna_pol': ['POLR2A', 'POLR2B'],
    'protease': ['CTSL', 'CTSD', 'CTSB', 'FURIN', 'PCSK9'],
    'polymerase': ['POLA1', 'POLD1', 'POLE', 'POLG'],
    'kinase': ['PIK3CA', 'PIK3CB', 'PI4KA', 'CSNK2A1'], 'cyp51': ['CYP51A1'],
}
PATHOGEN_RELEVANT_HOMOLOGS = {
    'viral': ['protease', 'polymerase'], 'bacterial': ['dhfr', 'topoisomerase', 'ribosome', 'rna_pol'],
    'fungal': ['cyp51', 'kinase'], 'parasitic': ['dhfr', 'kinase', 'protease'],
    'mixed': ['dhfr', 'topoisomerase', 'protease', 'polymerase'],
}

# ── Immune pathway genes ────────────────────────────────────────────────────
IMMUNE_PATHWAY_GENES = {
    'viral': [
        'IFNA1', 'IFNA2', 'IFNB1', 'IFNAR1', 'IFNAR2', 'IRF3', 'IRF7', 'IRF9', 'STAT1', 'STAT2',
        'TBK1', 'IKBKE', 'MAVS', 'STING1', 'CGAS', 'MX1', 'MX2', 'OAS1', 'OAS2', 'OAS3',
        'ISG15', 'IFIT1', 'IFIT2', 'IFIT3', 'EIF2AK2', 'ADAR', 'APOBEC3G',
        'CD8A', 'CD8B', 'PRF1', 'GZMA', 'GZMB', 'FASLG', 'IFNG',
    ],
    'bacterial': [
        'TLR2', 'TLR4', 'TLR5', 'TLR9', 'NOD1', 'NOD2', 'NLRP3', 'CASP1', 'IL1B', 'IL18',
        'TNF', 'IL6', 'IL8', 'CXCL8', 'CYBB', 'NCF1', 'NCF2', 'MPO', 'C3', 'C5', 'C3AR1', 'C5AR1',
        'IL17A', 'IL17F', 'IL22', 'IL23A', 'RORC', 'STAT3', 'CXCL1', 'CXCL2', 'CXCL5',
    ],
    'fungal': [
        'CLEC7A', 'CLEC6A', 'SYK', 'CARD9', 'TLR2', 'TLR4',
        'IL17A', 'IL17F', 'IL22', 'IL23A', 'RORC', 'STAT3',
        'CYBB', 'NCF1', 'MPO', 'TNF', 'IL1B', 'IL6',
    ],
    'parasitic': [
        'IFNG', 'IL12A', 'IL12B', 'STAT4', 'TBX21',
        'IL4', 'IL5', 'IL13', 'IL33', 'STAT6', 'GATA3',
        'IL5RA', 'CCR3', 'FCER1A', 'EPX', 'RNASE3',
    ],
    'mixed': [
        'TLR2', 'TLR3', 'TLR4', 'TLR7', 'TLR9', 'IFNA1', 'IFNB1', 'IFNG',
        'TNF', 'IL1B', 'IL6', 'IL8', 'STAT1', 'IRF3', 'MAVS',
    ],
}

# ── Tumor-specific constants ────────────────────────────────────────────────
IMMUNE_INFILTRATION = {
    'melanoma': 0.90, 'nsclc': 0.65, 'lung': 0.65, 'sclc': 0.35,
    'renal': 0.75, 'bladder': 0.70, 'head_neck': 0.65, 'colorectal': 0.55,
    'gastric': 0.50, 'breast': 0.55, 'ovarian': 0.45, 'hepatocellular': 0.40,
    'prostate': 0.25, 'pancreatic': 0.15, 'glioblastoma': 0.20, 'glioma': 0.20,
    'sarcoma': 0.30, 'aml': 0.40, 'all': 0.35, 'cml': 0.25, 'cll': 0.50,
    'dlbcl': 0.60, 'follicular': 0.55, 'hodgkin': 0.85, 'myeloma': 0.30,
    'thyroid': 0.50, 'endometrial': 0.55, 'cervical': 0.65, 'esophageal': 0.50,
    'mds': 0.35, 'mesothelioma': 0.45, 'neuroblastoma': 0.25, 'other_cancer': 0.40,
}

IMMUNE_THERAPY_GENES = {
    'PDCD1', 'CD274', 'PDCD1LG2', 'CTLA4', 'LAG3', 'TIGIT', 'HAVCR2',
    'CD28', 'ICOS', 'CD80', 'CD86', 'B7-H3', 'VISTA',
    'IL2', 'IL2RA', 'IFNG', 'IFNGR1', 'TNF', 'TNFRSF9',
}

SYNTHETIC_LETHAL_PAIRS = {
    'BRCA1_loss': {'vulnerability': ['BRCA1'], 'partners': ['PARP1', 'PARP2', 'ATR', 'CHEK1', 'RAD51', 'RAD51C'], 'cancers': ['ovarian', 'breast', 'prostate', 'pancreatic']},
    'BRCA2_loss': {'vulnerability': ['BRCA2'], 'partners': ['PARP1', 'PARP2', 'ATR', 'CHEK1'], 'cancers': ['ovarian', 'breast', 'prostate', 'pancreatic']},
    'TP53_loss': {'vulnerability': ['TP53'], 'partners': ['CHEK1', 'WEE1', 'ATR', 'CDK1', 'PLK1', 'AURKA'], 'cancers': ['ovarian', 'pancreatic', 'sclc', 'aml']},
    'RB1_loss': {'vulnerability': ['RB1'], 'partners': ['CDK2', 'CHEK1', 'AURKA', 'AURKB', 'PLK1'], 'cancers': ['sclc', 'bladder', 'prostate']},
    'KRAS_gain': {'vulnerability': ['KRAS'], 'partners': ['MAP2K1', 'MAP2K2', 'BRAF', 'SHP2', 'SOS1', 'ERK1', 'ERK2', 'MAPK1', 'MAPK3'], 'cancers': ['nsclc', 'colorectal', 'pancreatic']},
    'PTEN_loss': {'vulnerability': ['PTEN'], 'partners': ['AKT1', 'AKT2', 'MTOR', 'PIK3CA', 'PIK3CB'], 'cancers': ['prostate', 'endometrial', 'glioblastoma', 'breast']},
    'VHL_loss': {'vulnerability': ['VHL'], 'partners': ['HIF1A', 'VEGFA', 'KDR', 'PDGFRA', 'PDGFRB', 'MTOR'], 'cancers': ['renal']},
    'ARID1A_loss': {'vulnerability': ['ARID1A'], 'partners': ['EZH2', 'BRD4', 'HDAC1', 'HDAC2'], 'cancers': ['ovarian', 'gastric', 'endometrial', 'bladder']},
    'MSI_high': {'vulnerability': ['MLH1', 'MSH2', 'MSH6', 'PMS2'], 'partners': ['PDCD1', 'CD274', 'CTLA4'], 'cancers': ['colorectal', 'endometrial', 'gastric']},
    'IDH_gain': {'vulnerability': ['IDH1', 'IDH2'], 'partners': ['IDH1', 'IDH2'], 'cancers': ['aml', 'glioma', 'glioblastoma']},
}

DRIVER_ADDICTION = {
    'cml': {'BCR-ABL1': {'genes': ['ABL1', 'BCR'], 'score': 0.99}},
    'melanoma': {'BRAF_V600': {'genes': ['BRAF'], 'score': 0.50}},
    'nsclc': {
        'EGFR_mut': {'genes': ['EGFR'], 'score': 0.35}, 'ALK_fusion': {'genes': ['ALK'], 'score': 0.08},
        'KRAS_G12C': {'genes': ['KRAS'], 'score': 0.13}, 'ROS1_fusion': {'genes': ['ROS1'], 'score': 0.03},
        'RET_fusion': {'genes': ['RET'], 'score': 0.02}, 'MET_amp': {'genes': ['MET'], 'score': 0.03},
    },
    'breast': {
        'HER2_amp': {'genes': ['ERBB2'], 'score': 0.20}, 'ESR1': {'genes': ['ESR1'], 'score': 0.70},
        'PIK3CA': {'genes': ['PIK3CA', 'AKT1', 'MTOR'], 'score': 0.40},
    },
    'colorectal': {'BRAF_V600E': {'genes': ['BRAF'], 'score': 0.10}, 'ERBB2_amp': {'genes': ['ERBB2'], 'score': 0.05}},
    'glioblastoma': {'EGFR_amp': {'genes': ['EGFR'], 'score': 0.50}, 'IDH_mut': {'genes': ['IDH1', 'IDH2'], 'score': 0.10}},
    'aml': {'FLT3_ITD': {'genes': ['FLT3'], 'score': 0.30}, 'IDH_mut': {'genes': ['IDH1', 'IDH2'], 'score': 0.20}},
    'renal': {'VHL_loss': {'genes': ['VHL', 'HIF1A'], 'score': 0.80}},
    'prostate': {'AR_dep': {'genes': ['AR'], 'score': 0.90}},
    'gastric': {'HER2_amp': {'genes': ['ERBB2'], 'score': 0.15}},
    'hepatocellular': {'CTNNB1': {'genes': ['CTNNB1'], 'score': 0.30}},
    'thyroid': {'BRAF_V600E': {'genes': ['BRAF'], 'score': 0.60}, 'RET_fusion': {'genes': ['RET'], 'score': 0.15}},
}

MECHANISM_KEYWORDS = {
    'dna_damage': ['dna repair', 'dna replication', 'dna damage', 'nucleotide excision',
                   'double-strand break', 'recombinational repair', 'mismatch repair', 'base-excision repair'],
    'cell_cycle': ['cell cycle', 'mitotic', 'mitosis', 'cell division', 'cytokinesis',
                   'spindle', 'chromosome segregation', 'dna replication'],
    'epigenetic': ['histone', 'chromatin', 'nucleosome', 'deacetylase', 'acetyltransferase',
                   'methyltransferase', 'chromatin remodeling', 'dna methylation'],
    'immune': ['immune', 'toll-like', 'defense response', 'inflammatory', 'cytokine',
               'interferon', 'antigen', 'nf-kappa', 'innate immune'],
    'apoptosis': ['apoptosis', 'apoptotic', 'programmed cell death', 'caspase'],
}


# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_indication(disease):
    if pd.isna(disease):
        return 'other'
    d = str(disease).lower()
    if any(w in d for w in ['cancer', 'tumor', 'carcinoma', 'lymphoma', 'leukemia',
                             'melanoma', 'sarcoma', 'myeloma', 'glioma', 'neoplasm',
                             'oncolog', 'metastat', 'malignant']):
        return 'oncology'
    if any(w in d for w in ['depress', 'schizo', 'alzheimer', 'parkinson', 'epilep',
                             'seizure', 'bipolar', 'anxiety', 'psycho', 'dementia',
                             'neurodegen', 'multiple sclerosis', 'neuropath', 'migraine',
                             'insomnia', 'adhd', 'autism']):
        return 'cns'
    if any(w in d for w in ['heart', 'cardiac', 'coronary', 'hypertens', 'atrial',
                             'arrhythm', 'angina', 'stroke', 'thromb', 'aneurysm',
                             'atheroscl', 'myocard']):
        return 'cardiovascular'
    if any(w in d for w in ['diabet', 'obesity', 'metabol', 'lipid', 'cholesterol',
                             'thyroid', 'osteopor', 'gout']):
        return 'metabolic'
    if any(w in d for w in ['infect', 'hiv', 'hepatitis', 'tuberculosis', 'malaria',
                             'bacteri', 'viral', 'fungal', 'sepsis', 'pneumonia',
                             'covid', 'influenza']):
        return 'infectious'
    if any(w in d for w in ['arthritis', 'lupus', 'crohn', 'colitis', 'psoria',
                             'autoimmun', 'inflamm', 'asthma', 'allerg', 'eczema',
                             'dermatitis', 'fibrosis']):
        return 'autoimmune'
    if any(w in d for w in ['lung', 'pulmonary', 'copd', 'respiratory', 'bronch']):
        return 'respiratory'
    return 'other'


def map_cancer_type(disease_str):
    if pd.isna(disease_str):
        return None
    d = str(disease_str).lower()
    for cancer_type, keywords in CANCER_TYPE_MAP.items():
        if any(k in d for k in keywords):
            return cancer_type
    cancer_kw = ['cancer', 'tumor', 'carcinoma', 'lymphoma', 'leukemia',
                 'melanoma', 'sarcoma', 'myeloma', 'glioma', 'neoplasm',
                 'malignant', 'metastat']
    if any(k in d for k in cancer_kw):
        return 'other_cancer'
    return None


def map_pathogen(disease):
    if pd.isna(disease):
        return None
    d = str(disease).lower()
    for pid, (keywords, pclass) in PATHOGEN_MAP_TUPLE.items():
        if pid.startswith('general_'):
            continue
        if any(kw in d for kw in keywords):
            return pid, pclass
    for pid in ['general_bacterial', 'general_infection']:
        keywords, pclass = PATHOGEN_MAP_TUPLE[pid]
        if any(kw in d for kw in keywords):
            return pid, pclass
    return None


def map_pathogen_type(disease):
    if pd.isna(disease):
        return None
    d = str(disease).lower()
    for pathogen_id, info in PATHOGEN_MAP_DICT.items():
        if pathogen_id.startswith('general_'):
            continue
        if any(kw in d for kw in info['keywords']):
            return pathogen_id, info['class'], info['subtype']
    for pathogen_id in ['general_bacterial', 'general_infection']:
        info = PATHOGEN_MAP_DICT[pathogen_id]
        if any(kw in d for kw in info['keywords']):
            return pathogen_id, info['class'], info['subtype']
    return None


# ============================================================================
# NETWORK ENRICHMENT FUNCTIONS (from integrate_network_enrichment.py)
# ============================================================================

def parse_enrichment(filepath):
    try:
        df = pd.read_csv(filepath, sep='\t')
    except Exception:
        return None, None
    if df.empty or 'fdr' not in df.columns:
        return None, None
    sig = df[df['fdr'] < FDR_THRESHOLD]
    return df, sig


def parse_interactions(filepath):
    try:
        df = pd.read_csv(filepath, sep='\t')
    except Exception:
        return None
    if df.empty or 'score' not in df.columns:
        return None
    return df


def build_enst_to_symbol(enrich_df):
    mapping = {}
    if enrich_df is None or 'inputGenes' not in enrich_df.columns:
        return mapping
    for _, row in enrich_df.iterrows():
        genes_str = row.get('inputGenes', '')
        names_str = row.get('preferredNames', '')
        if pd.notna(genes_str) and pd.notna(names_str):
            genes = str(genes_str).split(',')
            names = str(names_str).split(',')
            for g, n in zip(genes, names):
                mapping[g.strip()] = n.strip()
    return mapping


def build_ppi_graph(interact_df):
    G = nx.Graph()
    if interact_df is None:
        return G
    for _, row in interact_df.iterrows():
        a = row.get('preferredName_A')
        b = row.get('preferredName_B')
        if pd.notna(a) and pd.notna(b):
            G.add_edge(str(a), str(b),
                       score=row.get('score', 0), escore=row.get('escore', 0),
                       dscore=row.get('dscore', 0), tscore=row.get('tscore', 0))
    return G


def extract_network_features(enrich_df, sig_df, interact_df,
                              disease_targets_enst, drug_targets_enst):
    """Extract comprehensive network features for one compound."""
    feats = {}
    enst_to_sym = build_enst_to_symbol(enrich_df)
    disease_symbols = {enst_to_sym[e] for e in disease_targets_enst if e in enst_to_sym}
    drug_symbols = {enst_to_sym[e] for e in drug_targets_enst if e in enst_to_sym}
    n_disease = len(disease_targets_enst)
    has_disease = n_disease > 0

    # 1. Pathway-disease match
    pathway_gene_sets, pathway_fdrs = [], []
    all_pathway_genes_enst = set()
    for _, row in sig_df.iterrows():
        gs = row.get('inputGenes', '')
        if pd.notna(gs) and str(gs).strip():
            genes = set(str(gs).split(','))
            pathway_gene_sets.append(genes)
            pathway_fdrs.append(row['fdr'])
            all_pathway_genes_enst.update(genes)
    n_pathways = len(pathway_gene_sets)

    if has_disease and n_pathways > 0:
        disease_in_pathways = disease_targets_enst & all_pathway_genes_enst
        feats['net_n_disease_genes_in_pathways'] = len(disease_in_pathways)
        feats['net_frac_disease_covered'] = len(disease_in_pathways) / n_disease
        pathways_with_disease = [i for i, pg in enumerate(pathway_gene_sets) if pg & disease_targets_enst]
        n_pw_disease = len(pathways_with_disease)
        feats['net_n_pathways_with_disease'] = n_pw_disease
        feats['net_frac_pathways_with_disease'] = n_pw_disease / n_pathways
        if pathways_with_disease:
            disease_scores = [-np.log10(max(pathway_fdrs[i], 1e-300)) for i in pathways_with_disease]
            feats['net_disease_pathway_score_max'] = max(disease_scores)
            feats['net_disease_pathway_score_mean'] = np.mean(disease_scores)
        else:
            feats['net_disease_pathway_score_max'] = 0.0
            feats['net_disease_pathway_score_mean'] = 0.0
        best_idx = sig_df['fdr'].idxmin()
        best_gs = sig_df.loc[best_idx, 'inputGenes']
        best_genes = set(str(best_gs).split(',')) if pd.notna(best_gs) else set()
        feats['net_top_pathway_has_disease'] = int(bool(best_genes & disease_targets_enst))
        feats['net_has_disease_pathway_overlap'] = int(n_pw_disease > 0)
    else:
        for k in ['net_n_disease_genes_in_pathways', 'net_n_pathways_with_disease',
                   'net_top_pathway_has_disease', 'net_has_disease_pathway_overlap']:
            feats[k] = 0
        for k in ['net_frac_disease_covered', 'net_frac_pathways_with_disease',
                   'net_disease_pathway_score_max', 'net_disease_pathway_score_mean']:
            feats[k] = 0.0

    # 2. PPI reachability
    G = build_ppi_graph(interact_df)
    ppi_nodes = set(G.nodes())
    disease_in_ppi = disease_symbols & ppi_nodes
    feats['net_n_disease_in_ppi'] = len(disease_in_ppi)
    feats['net_frac_disease_in_ppi'] = len(disease_in_ppi) / n_disease if has_disease else 0.0
    drug_in_ppi = drug_symbols & ppi_nodes
    feats['net_n_drug_targets_in_ppi'] = len(drug_in_ppi)

    if drug_in_ppi and disease_symbols and len(G) > 0:
        neighbors_1hop = set()
        for dt in drug_in_ppi:
            neighbors_1hop.update(G.neighbors(dt))
        neighbors_1hop |= drug_in_ppi
        neighbors_2hop = set()
        for n in neighbors_1hop:
            if n in G:
                neighbors_2hop.update(G.neighbors(n))
        neighbors_2hop |= neighbors_1hop
        disease_1hop = disease_symbols & neighbors_1hop
        disease_2hop = disease_symbols & neighbors_2hop
        feats['net_disease_reachable_1hop'] = len(disease_1hop)
        feats['net_disease_reachable_2hop'] = len(disease_2hop)
        feats['net_frac_disease_reachable_1hop'] = len(disease_1hop) / n_disease if has_disease else 0.0
        feats['net_frac_disease_reachable_2hop'] = len(disease_2hop) / n_disease if has_disease else 0.0
        shortest_paths = []
        for dt in drug_in_ppi:
            for dg in disease_in_ppi:
                try:
                    sp = nx.shortest_path_length(G, dt, dg)
                    shortest_paths.append(sp)
                except nx.NetworkXNoPath:
                    pass
        if shortest_paths:
            feats['net_min_path_drug_to_disease'] = min(shortest_paths)
            feats['net_mean_path_drug_to_disease'] = np.mean(shortest_paths)
        else:
            feats['net_min_path_drug_to_disease'] = -1
            feats['net_mean_path_drug_to_disease'] = -1
    else:
        for k in ['net_disease_reachable_1hop', 'net_disease_reachable_2hop']:
            feats[k] = 0
        for k in ['net_frac_disease_reachable_1hop', 'net_frac_disease_reachable_2hop']:
            feats[k] = 0.0
        feats['net_min_path_drug_to_disease'] = -1
        feats['net_mean_path_drug_to_disease'] = -1

    # 3. Connection quality
    if drug_in_ppi and disease_in_ppi and len(G) > 0:
        direct_edges = []
        for dt in drug_in_ppi:
            for dg in disease_in_ppi:
                if G.has_edge(dt, dg):
                    direct_edges.append(G[dt][dg])
        if direct_edges:
            feats['net_n_direct_drug_disease_edges'] = len(direct_edges)
            feats['net_drug_disease_score_max'] = max(e['score'] for e in direct_edges)
            feats['net_drug_disease_experimental'] = np.mean([e['escore'] for e in direct_edges])
            feats['net_drug_disease_database'] = np.mean([e['dscore'] for e in direct_edges])
            feats['net_drug_disease_textmining'] = np.mean([e['tscore'] for e in direct_edges])
        else:
            for k in ['net_n_direct_drug_disease_edges']:
                feats[k] = 0
            for k in ['net_drug_disease_score_max', 'net_drug_disease_experimental',
                       'net_drug_disease_database', 'net_drug_disease_textmining']:
                feats[k] = 0.0
    else:
        feats['net_n_direct_drug_disease_edges'] = 0
        for k in ['net_drug_disease_score_max', 'net_drug_disease_experimental',
                   'net_drug_disease_database', 'net_drug_disease_textmining']:
            feats[k] = 0.0

    # 4. Off-target spread
    if n_pathways > 0:
        n_off_target = n_pathways - len([i for i, pg in enumerate(pathway_gene_sets)
                                          if pg & disease_targets_enst]) if has_disease else n_pathways
        feats['net_n_off_target_pathways'] = n_off_target
        feats['net_frac_off_target_pathways'] = n_off_target / n_pathways
        feats['net_n_unique_ppi_proteins'] = len(ppi_nodes)
        n_drug_t = len(drug_in_ppi) if drug_in_ppi else 1
        feats['net_spread_ratio'] = len(ppi_nodes) / n_drug_t
        if len(ppi_nodes) > 1:
            max_edges = len(ppi_nodes) * (len(ppi_nodes) - 1) / 2
            feats['net_network_density'] = len(G.edges()) / max_edges
        else:
            feats['net_network_density'] = 0.0
    else:
        feats['net_n_off_target_pathways'] = 0
        feats['net_frac_off_target_pathways'] = 0.0
        feats['net_n_unique_ppi_proteins'] = len(ppi_nodes)
        feats['net_spread_ratio'] = 0.0
        feats['net_network_density'] = 0.0

    # 5. Enrichment summary
    feats['net_n_enriched_total'] = n_pathways
    category_col = 'category' if 'category' in enrich_df.columns else enrich_df.columns[0]
    feats['net_enrichment_diversity'] = sig_df[category_col].nunique() if n_pathways > 0 else 0
    if n_pathways > 0:
        scores = -np.log10(sig_df['fdr'].clip(lower=1e-300))
        feats['net_max_enrichment_score'] = scores.max()
    else:
        feats['net_max_enrichment_score'] = 0.0

    # 6. Mechanism type
    if n_pathways > 0 and 'description' in sig_df.columns:
        descriptions = sig_df['description'].dropna().str.lower().tolist()
        fdrs = sig_df.loc[sig_df['description'].notna(), 'fdr'].values
        for mech_name, keywords in MECHANISM_KEYWORDS.items():
            matching_idx = [i for i, d in enumerate(descriptions) if any(k in d for k in keywords)]
            n_match = len(matching_idx)
            feats[f'net_mech_{mech_name}_n'] = n_match
            feats[f'net_mech_{mech_name}_frac'] = n_match / n_pathways
            if matching_idx:
                mech_scores = [-np.log10(max(fdrs[i], 1e-300)) for i in matching_idx]
                feats[f'net_mech_{mech_name}_score'] = max(mech_scores)
            else:
                feats[f'net_mech_{mech_name}_score'] = 0.0
    else:
        for mech_name in MECHANISM_KEYWORDS:
            feats[f'net_mech_{mech_name}_n'] = 0
            feats[f'net_mech_{mech_name}_frac'] = 0.0
            feats[f'net_mech_{mech_name}_score'] = 0.0

    return feats


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

def compute_features_for_compound(zinc_id, disease, target, input_dir,
                                   gene_to_enst, disease_targets_enst,
                                   disease_target_scores):
    """Compute all features for a single compound. Returns dict of features."""
    feats = {}
    input_dir = Path(input_dir)

    # ── Helper: bind gene set via ENST mapping ──────────────────────────────
    transcript_scores = {}
    scores_array = np.array([])

    def _bind(gene_set):
        vals = []
        for g in gene_set:
            enst = gene_to_enst.get(g)
            if enst and enst in transcript_scores:
                vals.append(transcript_scores[enst])
        return vals

    def _mean_bind(gene_set):
        vals = _bind(gene_set)
        return np.mean(vals) if vals else 0.0

    def _max_bind(gene_set):
        vals = _bind(gene_set)
        return max(vals) if vals else 0.0

    def _n_bound(gene_set, threshold=0.5):
        vals = _bind(gene_set)
        return sum(1 for v in vals if v > threshold)

    def _mean_bind_enst(enst_set):
        vals = [transcript_scores.get(e, 0) for e in enst_set]
        return np.mean(vals) if vals else 0.0

    def _max_bind_enst(enst_set):
        vals = [transcript_scores.get(e, 0) for e in enst_set]
        return max(vals) if vals else 0.0

    def _n_bound_enst(enst_set, threshold=0.5):
        return sum(1 for e in enst_set if transcript_scores.get(e, 0) > threshold)

    # ── A) Biolprop tissue features ─────────────────────────────────────────
    bioprop_file = input_dir / 'biolprop_merged.tsv'
    if bioprop_file.exists():
        bioprop = pd.read_csv(bioprop_file, sep='\t')
        match = bioprop[bioprop['Drug_ID'] == zinc_id]
        if len(match) > 0:
            data = match.iloc[0]
            for col in bioprop.columns:
                if col == 'Drug_ID':
                    continue
                val = data.get(col)
                if pd.notna(val):
                    try:
                        feats[col] = float(val)
                    except (ValueError, TypeError):
                        pass

    # Derived tissue features
    max_vals = [feats.get(f'max_{t}_interaction', 0.0) for t in TISSUE_NAMES]
    feats['max_tissue_interaction'] = max(max_vals)
    feats['mean_tissue_interaction'] = np.mean(max_vals)
    feats['n_tissues_active'] = sum(1 for v in max_vals if v > 0.5)

    # Disease-tissue features (user must provide DISEASE_TISSUE_MAP for custom diseases)
    disease_tissue_map = {
        'HIV': ['immune', 'hematopoietic', 'liver'],
        'Cancer and Viral Infections': ['immune', 'hematopoietic', 'liver'],
        'Cancer (Checkpoint Inhibitor)': ['immune', 'hematopoietic'],
        'Glaucoma': ['sensory', 'brain'],
        'Pain': ['brain', 'sensory'],
    }
    tissues = disease_tissue_map.get(disease, [])
    if tissues:
        dt_max = [feats.get(f'max_{t}_interaction', 0.0) for t in tissues]
        dt_enst = [feats.get(f'n_enst_{t}_above_target_interaction', 0.0) for t in tissues]
        feats['disease_tissue_max_interaction'] = max(dt_max)
        feats['disease_tissue_n_enst_above'] = sum(dt_enst)
        feats['disease_tissue_n_high_protein'] = sum(
            feats.get(f'n_high_protein_in_{t}', 0) for t in tissues)
        feats['disease_tissue_specificity'] = np.mean(dt_max) / (np.mean(max_vals) + 1e-8)
        feats['tissue_indication_match_score'] = np.mean(dt_max)
        feats['tissue_indication_max'] = max(dt_max)
        feats['tissue_indication_mean'] = np.mean(dt_max)

    # ── B) Binding scores ───────────────────────────────────────────────────
    binding_file = input_dir / 'binding' / f'{zinc_id}_patient_0_drug_scores.tsv'
    if binding_file.exists():
        sdf = pd.read_csv(binding_file, sep='\t')
        transcript_scores = dict(zip(sdf['Transcript'], sdf['Score']))
        scores_array = sdf['Score'].values

    # NOTE: Binding specificity (24), disease binding (7), drug-level binding (3),
    # and derived binding (2) features REMOVED per science team review (Mar 2026).
    # Δ AUC < 0.01 for both tasks. These were additional analysis layers on
    # Binding scores that are scientifically questionable.
    # transcript_scores and scores_array are still needed downstream for
    # cancer/oncology/infectious domain features.

    feats['target_coverage_ratio'] = np.nan

    return feats, transcript_scores, scores_array


def compute_domain_features(feats, disease, transcript_scores, scores_array,
                             gene_to_enst, disease_targets_enst, disease_target_scores,
                             input_dir):
    """Add cancer, infectious, tumor, network, pathway, and mechanism features."""
    input_dir = Path(input_dir)
    has_binding = len(transcript_scores) > 0

    # Helper closures
    def _bind(gene_set):
        vals = []
        for g in gene_set:
            enst = gene_to_enst.get(g)
            if enst and enst in transcript_scores:
                vals.append(transcript_scores[enst])
        return vals

    def _mean_bind(gene_set):
        vals = _bind(gene_set)
        return np.mean(vals) if vals else 0.0

    def _mean_bind_enst(enst_set):
        vals = [transcript_scores.get(e, 0) for e in enst_set]
        return np.mean(vals) if vals else 0.0

    def _max_bind_enst(enst_set):
        vals = [transcript_scores.get(e, 0) for e in enst_set]
        return max(vals) if vals else 0.0

    def _n_bound_enst(enst_set, threshold=0.5):
        return sum(1 for e in enst_set if transcript_scores.get(e, 0) > threshold)

    indication = classify_indication(disease)
    is_oncology = int(indication == 'oncology')
    is_infectious = int(indication == 'infectious')
    cancer_type = map_cancer_type(disease)

    # Build driver dicts
    ct_drivers_dict = {}
    if cancer_type and cancer_type in CANCER_DRIVERS:
        ct_drivers_dict = CANCER_DRIVERS[cancer_type]

    # Map gene sets to ENST
    checkpoint_enst = {gene_to_enst[g] for g in IMMUNE_CHECKPOINT_GENES if g in gene_to_enst}
    tcell_enst = {gene_to_enst[g] for g in T_CELL_GENES if g in gene_to_enst}
    nk_enst = {gene_to_enst[g] for g in NK_CELL_GENES if g in gene_to_enst}
    macrophage_enst = {gene_to_enst[g] for g in MACROPHAGE_GENES if g in gene_to_enst}
    tme_enst = {gene_to_enst[g] for g in TME_GENES if g in gene_to_enst}
    ddr_enst = {gene_to_enst[g] for g in DDR_GENES if g in gene_to_enst}
    cellcycle_enst = {gene_to_enst[g] for g in CELL_CYCLE_DRUG_TARGETS if g in gene_to_enst}
    essential_enst = {gene_to_enst[g] for g in ESSENTIAL_GENES_DOMAIN if g in gene_to_enst}
    all_immune_enst = checkpoint_enst | tcell_enst | nk_enst | macrophage_enst
    ct_driver_enst = {gene_to_enst[g] for g in ct_drivers_dict if g in gene_to_enst}

    # ── Cancer biology features ─────────────────────────────────────────────
    if cancer_type and has_binding:
        ct_driver_bind_vals = _bind(ct_drivers_dict.keys()) if ct_drivers_dict else []
        feats['cancer_driver_overlap'] = sum(1 for v in ct_driver_bind_vals if v > 0.5)
        feats['cancer_driver_specificity'] = np.mean(ct_driver_bind_vals) if ct_driver_bind_vals else 0
        feats['cancer_driver_n'] = len(ct_drivers_dict)
        feats['cancer_onco_frac'] = feats['cancer_driver_overlap'] / (len(scores_array) + 1) if len(scores_array) > 0 else 0
        feats['cancer_driver_weighted_score'] = sum(ct_driver_bind_vals) if ct_driver_bind_vals else 0
        feats['cancer_is_targeted_therapy'] = int(feats['cancer_driver_overlap'] > 0 and feats.get('binding_drug_frac_bound', 0) < 0.05)
    else:
        for c in ['cancer_driver_overlap', 'cancer_driver_specificity', 'cancer_driver_n',
                  'cancer_onco_frac', 'cancer_driver_weighted_score', 'cancer_is_targeted_therapy']:
            feats[c] = 0

    # Cancer disease-level features
    if cancer_type and ct_drivers_dict:
        feats['cancer_type_n_drivers'] = len(ct_drivers_dict)
        n_onco = sum(1 for role in ct_drivers_dict.values() if role == 'O')
        feats['cancer_type_onco_ratio'] = n_onco / len(ct_drivers_dict) if ct_drivers_dict else 0
        ct_driver_genes = set(ct_drivers_dict.keys())
        n_dominant = sum(1 for pw_genes in ONCOGENIC_PATHWAYS.values() if len(ct_driver_genes & pw_genes) >= 2)
        feats['cancer_type_n_pathways_dominant'] = n_dominant
        feats['cancer_pathway_n'] = 0

        if has_binding and ct_driver_enst:
            ct_scores = [transcript_scores.get(e, 0) for e in ct_driver_enst]
            n_ct_bound = sum(1 for s in ct_scores if s > 0.5)
            disease_enst_local = disease_targets_enst.get(disease, set())
            n_d = len(disease_enst_local)
            n_drivers_in_disease = len(ct_driver_enst & disease_enst_local) if disease_enst_local else 0
            feats['cancer_disease_driver_score_sum'] = sum(ct_scores)
            feats['cancer_disease_specific_driver_frac'] = n_ct_bound / len(ct_drivers_dict) if ct_drivers_dict else 0
            feats['cancer_disease_driver_frac'] = n_drivers_in_disease / n_d if n_d > 0 else 0
            feats['cancer_disease_onco_target_frac'] = n_drivers_in_disease / len(ct_drivers_dict) if ct_drivers_dict else 0
        else:
            for c in ['cancer_disease_driver_score_sum', 'cancer_disease_specific_driver_frac',
                      'cancer_disease_driver_frac', 'cancer_disease_onco_target_frac']:
                feats[c] = 0
    else:
        for c in ['cancer_type_n_drivers', 'cancer_type_onco_ratio', 'cancer_type_n_pathways_dominant',
                  'cancer_pathway_n', 'cancer_disease_driver_score_sum', 'cancer_disease_specific_driver_frac',
                  'cancer_disease_driver_frac', 'cancer_disease_onco_target_frac']:
            feats[c] = 0

    # Binding driver binding
    if has_binding and cancer_type and ct_driver_enst:
        ct_driver_scores = [transcript_scores.get(e, 0) for e in ct_driver_enst]
        feats['binding_driver_bind_max'] = max(ct_driver_scores) if ct_driver_scores else 0
        feats['binding_driver_bind_mean'] = np.mean(ct_driver_scores) if ct_driver_scores else 0
        n_driver_bound = sum(1 for s in ct_driver_scores if s > 0.5)
        feats['binding_driver_n_bound'] = n_driver_bound
        feats['binding_driver_frac_bound'] = n_driver_bound / len(ct_driver_scores) if ct_driver_scores else 0
        onco_scores = [transcript_scores.get(gene_to_enst.get(g, ''), 0) for g, role in ct_drivers_dict.items() if role == 'O' and g in gene_to_enst]
        tsg_scores = [transcript_scores.get(gene_to_enst.get(g, ''), 0) for g, role in ct_drivers_dict.items() if role == 'T' and g in gene_to_enst]
        feats['binding_onco_driver_bind_max'] = max(onco_scores) if onco_scores else 0
        feats['binding_tsg_driver_bind_max'] = max(tsg_scores) if tsg_scores else 0
        pw_means = []
        for pw_genes in ONCOGENIC_PATHWAYS.values():
            pw_enst = {gene_to_enst[g] for g in pw_genes if g in gene_to_enst}
            if pw_enst:
                pw_means.append(np.mean([transcript_scores.get(e, 0) for e in pw_enst]))
        total_pw = sum(pw_means) if pw_means else 0
        feats['binding_pathway_bind_concentration'] = max(pw_means) / (total_pw + 1e-8) if pw_means else 0
    else:
        for c in ['binding_driver_bind_max', 'binding_driver_bind_mean', 'binding_driver_n_bound',
                  'binding_driver_frac_bound', 'binding_onco_driver_bind_max',
                  'binding_tsg_driver_bind_max', 'binding_pathway_bind_concentration']:
            feats[c] = 0

    # Oncology selectivity
    essential_bind_mean = _mean_bind_enst(essential_enst) if has_binding else 0
    ct_driver_bind_mean = _mean_bind_enst(ct_driver_enst) if has_binding and ct_driver_enst else 0
    if cancer_type and has_binding:
        feats['selectivity_driver_vs_essential'] = ct_driver_bind_mean / (essential_bind_mean + 0.01)
        feats['essential_gene_bind_burden'] = essential_bind_mean
        feats['essential_gene_n_bound'] = _n_bound_enst(essential_enst)
        for organ, genes in VITAL_ORGAN_GENES.items():
            feats[f'vital_organ_bind_{organ}'] = _mean_bind(genes)
        feats['vital_organ_bind_max'] = max(feats.get(f'vital_organ_bind_{o}', 0) for o in VITAL_ORGAN_GENES)
        feats['selectivity_driver_vs_vital'] = ct_driver_bind_mean / (feats['vital_organ_bind_max'] + 0.01)
        n_targets_above_05 = (scores_array > 0.5).sum()
        n_drivers_bound = _n_bound_enst(ct_driver_enst) if ct_driver_enst else 0
        feats['binding_breadth_ratio'] = n_targets_above_05 / n_drivers_bound if n_drivers_bound > 0 else float(n_targets_above_05) if n_targets_above_05 > 0 else 0
        driver_signal = sum(transcript_scores.get(e, 0) for e in ct_driver_enst) if ct_driver_enst else 0
        feats['cancer_binding_concentration'] = driver_signal / (scores_array.sum() + 1e-8)
        feats['mechanism_type_score'] = 1.0 / (1.0 + n_targets_above_05)
    else:
        for c in ['selectivity_driver_vs_essential', 'essential_gene_bind_burden', 'essential_gene_n_bound',
                  'vital_organ_bind_max', 'selectivity_driver_vs_vital', 'binding_breadth_ratio',
                  'cancer_binding_concentration', 'mechanism_type_score']:
            feats[c] = 0
        for organ in VITAL_ORGAN_GENES:
            feats[f'vital_organ_bind_{organ}'] = 0

    # Domain oncology features
    if cancer_type and has_binding:
        feats['onco_checkpoint_bind'] = _mean_bind_enst(checkpoint_enst)
        feats['onco_tcell_bind'] = _mean_bind_enst(tcell_enst)
        feats['onco_nk_bind'] = _mean_bind_enst(nk_enst)
        feats['onco_macrophage_bind'] = _mean_bind_enst(macrophage_enst)
        feats['onco_tme_bind'] = _mean_bind_enst(tme_enst)
        feats['onco_ddr_bind'] = _mean_bind_enst(ddr_enst)
        feats['onco_cellcycle_bind'] = _mean_bind_enst(cellcycle_enst)
        mean_immune = _mean_bind_enst(all_immune_enst)
        mean_essential = _mean_bind_enst(essential_enst)
        mean_cytotoxic = (_mean_bind_enst(cellcycle_enst) + mean_essential) / 2
        feats['onco_immune_vs_essential'] = mean_immune / (mean_essential + 0.01) if mean_essential > 0.01 else 0
        feats['onco_immune_vs_cytotoxic'] = mean_immune / (mean_cytotoxic + 0.01) if mean_cytotoxic > 0.01 else 0
        feats['onco_driver_selectivity'] = ct_driver_bind_mean / (mean_essential + 0.01) if mean_essential > 0.01 else 0
        feats['onco_n_immune_bound'] = _n_bound_enst(all_immune_enst)
        compartments = [checkpoint_enst, tcell_enst, nk_enst, macrophage_enst]
        engaged = sum(1 for c in compartments if _max_bind_enst(c) > 0.5)
        feats['onco_immune_breadth'] = engaged / len(compartments)
        feats['onco_mechanism_class'] = (mean_immune - mean_cytotoxic) / (mean_immune + mean_cytotoxic) if mean_immune + mean_cytotoxic > 0.01 else 0
    else:
        for c in ['onco_checkpoint_bind', 'onco_tcell_bind', 'onco_nk_bind', 'onco_macrophage_bind',
                  'onco_tme_bind', 'onco_ddr_bind', 'onco_cellcycle_bind', 'onco_immune_vs_essential',
                  'onco_immune_vs_cytotoxic', 'onco_driver_selectivity', 'onco_n_immune_bound',
                  'onco_immune_breadth', 'onco_mechanism_class']:
            feats[c] = 0

    # ── Infectious features (from compute_infectious_features) ──────────────
    inf_pathogen_info = map_pathogen_type(disease)
    if inf_pathogen_info:
        inf_pid, inf_pclass, inf_subtype = inf_pathogen_info
        feats['is_infectious'] = 1
        feats['pathogen_class_viral'] = int(inf_pclass == 'viral')
        feats['pathogen_class_bacterial'] = int(inf_pclass == 'bacterial')
        feats['pathogen_class_fungal'] = int(inf_pclass == 'fungal')
        feats['pathogen_class_parasitic'] = int(inf_pclass == 'parasitic')
        inf_chars = PATHOGEN_CHARACTERISTICS.get(inf_pid, {})
        feats['pathogen_genome_plasticity'] = inf_chars.get('plasticity', 0.5)
        feats['pathogen_n_druggable_targets'] = inf_chars.get('n_targets', 5)
        if has_binding:
            hf_genes = PATHOGEN_HOST_FACTORS.get(inf_pid, [])
            hf_ensts = {gene_to_enst[g] for g in hf_genes if g in gene_to_enst}
            feats['host_factor_bind_score'] = _mean_bind_enst(hf_ensts) if hf_ensts else 0
            imm_genes = IMMUNE_PATHWAY_GENES.get(inf_pclass, IMMUNE_PATHWAY_GENES.get('mixed', []))
            imm_ensts = {gene_to_enst[g] for g in imm_genes if g in gene_to_enst}
            feats['immune_pathway_engagement'] = _mean_bind_enst(imm_ensts) if imm_ensts else 0
            total_mean = np.mean(scores_array) if len(scores_array) > 0 else 0
            feats['host_directed_therapy_score'] = feats['host_factor_bind_score'] / max(total_mean, 0.01)
        else:
            feats['host_factor_bind_score'] = 0
            feats['immune_pathway_engagement'] = 0
            feats['host_directed_therapy_score'] = 0
    else:
        for c in ['is_infectious', 'pathogen_class_viral', 'pathogen_class_bacterial',
                  'pathogen_class_fungal', 'pathogen_class_parasitic',
                  'pathogen_genome_plasticity', 'pathogen_n_druggable_targets',
                  'host_factor_bind_score', 'immune_pathway_engagement', 'host_directed_therapy_score']:
            feats[c] = 0

    # ── Infectious features (from compute_domain_features) ──────────────────
    pathogen_info = map_pathogen(disease)
    if pathogen_info:
        pid, pclass = pathogen_info
        feats['inf_pathogen_viral'] = int(pclass == 'viral')
        feats['inf_pathogen_bacterial'] = int(pclass == 'bacterial')
        feats['inf_pathogen_fungal'] = int(pclass == 'fungal')
        feats['inf_pathogen_parasitic'] = int(pclass == 'parasitic')
        feats['inf_is_infectious'] = 1
        chars = PATHOGEN_CHARS.get(pid, (0.5, 5, 0.4))
        feats['inf_plasticity'] = chars[0]
        feats['inf_n_known_targets'] = chars[1]
        feats['inf_resistance_risk'] = chars[2]
        if has_binding:
            host_entry_enst_set = {gene_to_enst[g] for g in HOST_ENTRY_FACTORS.get(pid, []) if g in gene_to_enst}
            feats['inf_host_entry_bind'] = _mean_bind_enst(host_entry_enst_set) if host_entry_enst_set else 0
            feats['inf_host_entry_n_bound'] = _n_bound_enst(host_entry_enst_set) if host_entry_enst_set else 0
            defense_gene_list = DEFENSE_MAP.get(pclass, ANTIVIRAL_DEFENSE)
            defense_enst = {gene_to_enst[g] for g in defense_gene_list if g in gene_to_enst}
            feats['inf_defense_pathway_bind'] = _mean_bind_enst(defense_enst) if defense_enst else 0
            feats['inf_defense_n_bound'] = _n_bound_enst(defense_enst) if defense_enst else 0
            all_defense_enst = set()
            for attr_genes in [ANTIVIRAL_DEFENSE, ANTIBACTERIAL_DEFENSE, ANTIFUNGAL_DEFENSE]:
                all_defense_enst |= {gene_to_enst[g] for g in attr_genes if g in gene_to_enst}
            all_defense_bind = _mean_bind_enst(all_defense_enst) if all_defense_enst else 0
            feats['inf_defense_specificity'] = feats['inf_defense_pathway_bind'] / (all_defense_bind + 0.01) if all_defense_bind > 0.01 else 0
            relevant_hclasses = PATHOGEN_RELEVANT_HOMOLOGS.get(pclass, [])
            homolog_set = set()
            for hc in relevant_hclasses:
                for g in HOST_PATHOGEN_HOMOLOGS.get(hc, []):
                    if g in gene_to_enst:
                        homolog_set.add(gene_to_enst[g])
            feats['inf_homolog_bind_score'] = _mean_bind_enst(homolog_set) if homolog_set else 0
            feats['inf_homolog_n_bound'] = _n_bound_enst(homolog_set) if homolog_set else 0
            total_mean = np.mean(scores_array) if len(scores_array) > 0 else 0
            feats['inf_host_directed_ratio'] = feats['inf_host_entry_bind'] / (total_mean + 0.01) if total_mean > 0.01 else 0
        else:
            for c in ['inf_host_entry_bind', 'inf_host_entry_n_bound', 'inf_defense_pathway_bind',
                      'inf_defense_n_bound', 'inf_defense_specificity', 'inf_homolog_bind_score',
                      'inf_homolog_n_bound', 'inf_host_directed_ratio']:
                feats[c] = 0
    else:
        for c in ['inf_pathogen_viral', 'inf_pathogen_bacterial', 'inf_pathogen_fungal',
                  'inf_pathogen_parasitic', 'inf_host_entry_bind', 'inf_host_entry_n_bound',
                  'inf_defense_pathway_bind', 'inf_defense_n_bound', 'inf_defense_specificity',
                  'inf_homolog_bind_score', 'inf_homolog_n_bound', 'inf_host_directed_ratio',
                  'inf_resistance_risk', 'inf_plasticity', 'inf_n_known_targets']:
            feats[c] = 0

    # ── Tumor-specific features ─────────────────────────────────────────────
    if cancer_type:
        infiltration = IMMUNE_INFILTRATION.get(cancer_type, 0)
        feats['tumor_immune_infiltration'] = infiltration
        if has_binding:
            ct_driver_genes_list = list(ct_drivers_dict.keys())
            n_driver_bound = sum(1 for g in ct_driver_genes_list
                                 if gene_to_enst.get(g) and transcript_scores.get(gene_to_enst[g], 0) > 0.3)
            feats['tumor_driver_bound_frac'] = n_driver_bound / max(len(ct_driver_genes_list), 1)
            immune_bind = sum(transcript_scores.get(gene_to_enst.get(g, ''), 0)
                             for g in IMMUNE_THERAPY_GENES
                             if gene_to_enst.get(g) and transcript_scores.get(gene_to_enst[g], 0) > 0.3)
            feats['tumor_immune_fitness'] = immune_bind * infiltration
            feats['tumor_immune_mismatch'] = immune_bind * (1 - infiltration)
            sl_scores = []
            for sl_info in SYNTHETIC_LETHAL_PAIRS.values():
                partners = sl_info.get('partners', [])
                partner_vals = _bind(partners)
                if partner_vals:
                    sl_scores.append(max(partner_vals))
            feats['tumor_sl_score'] = np.mean(sl_scores) if sl_scores else 0
            feats['tumor_sl_n_pairs'] = len(sl_scores)
            feats['tumor_sl_best'] = max(sl_scores) if sl_scores else 0
            feats['tumor_expr_bind_overlap'] = 0
            n_total_bound = (scores_array > 0.5).sum() if len(scores_array) > 0 else 0
            n_immune_bound = _n_bound_enst(all_immune_enst)
            is_targeted = n_total_bound <= 10
            is_cytotoxic = n_total_bound > 50
            is_immunotherapy = n_immune_bound >= 3
            if is_targeted and n_driver_bound > 0:
                feats['tumor_mechanism_fitness'] = 0.8
            elif is_immunotherapy and infiltration > 0.5:
                feats['tumor_mechanism_fitness'] = 0.7
            elif is_immunotherapy and infiltration < 0.3:
                feats['tumor_mechanism_fitness'] = -0.3
            elif is_cytotoxic:
                feats['tumor_mechanism_fitness'] = 0.2
            elif is_targeted and n_driver_bound == 0:
                feats['tumor_mechanism_fitness'] = -0.2
            else:
                feats['tumor_mechanism_fitness'] = 0
            addiction_best = 0
            if cancer_type in DRIVER_ADDICTION:
                for event, info in DRIVER_ADDICTION[cancer_type].items():
                    for g in info.get('genes', []):
                        enst = gene_to_enst.get(g)
                        if enst and transcript_scores.get(enst, 0) > 0.3:
                            addiction_best = max(addiction_best, info.get('score', 0) * transcript_scores[enst])
            feats['tumor_composite_score'] = (0.3 * addiction_best + 0.25 * feats['tumor_sl_best'] +
                0.25 * feats['tumor_mechanism_fitness'] + 0.1 * feats['tumor_immune_fitness'])
        else:
            for c in ['tumor_driver_bound_frac', 'tumor_immune_fitness', 'tumor_immune_mismatch',
                      'tumor_sl_score', 'tumor_sl_n_pairs', 'tumor_sl_best', 'tumor_expr_bind_overlap',
                      'tumor_mechanism_fitness', 'tumor_composite_score']:
                feats[c] = 0
    else:
        feats['tumor_immune_infiltration'] = 0
        for c in ['tumor_driver_bound_frac', 'tumor_immune_fitness', 'tumor_immune_mismatch',
                  'tumor_sl_score', 'tumor_sl_n_pairs', 'tumor_sl_best', 'tumor_expr_bind_overlap',
                  'tumor_mechanism_fitness', 'tumor_composite_score']:
            feats[c] = 0
    feats['tumor_immune_therapy_bind'] = _mean_bind(IMMUNE_THERAPY_GENES) if has_binding else 0

    # ── Indication flags ────────────────────────────────────────────────────
    feats['is_oncology'] = is_oncology
    feats['is_infectious'] = max(feats.get('is_infectious', 0), is_infectious)
    feats['is_cns'] = int(indication == 'cns')
    feats['is_cardiovascular'] = int(indication == 'cardiovascular')
    feats['is_metabolic'] = int(indication == 'metabolic')
    feats['is_autoimmune'] = int(indication == 'autoimmune')
    feats['is_respiratory'] = int(indication == 'respiratory')
    feats['high_risk_indication'] = is_oncology | feats['is_cns']
    feats['trial_is_oncology'] = is_oncology
    feats['trial_is_infectious'] = is_infectious
    feats['high_protein_binding'] = 0

    # ── STRING network enrichment ───────────────────────────────────────────
    zinc_id = feats.get('_zinc_id', '')
    enrich_path = input_dir / 'string' / f'{zinc_id}_patient_0_network_enrichment.tsv'
    interact_path = input_dir / 'string' / f'{zinc_id}_patient_0_network_interactions.tsv'
    if enrich_path.exists():
        enrich_df, sig_df = parse_enrichment(enrich_path)
        interact_df = parse_interactions(interact_path)
        if enrich_df is not None:
            net_feats = extract_network_features(
                enrich_df, sig_df, interact_df,
                disease_targets_enst.get(disease, set()), set())
            feats.update(net_feats)

    # ── Pathway interaction terms ───────────────────────────────────────────
    key_pw_shorts = ['n_pathways_with_disease', 'disease_pathway_score_max',
                     'has_disease_pathway_overlap', 'disease_reachable_1hop', 'disease_reachable_2hop']
    for short in key_pw_shorts:
        net_val = feats.get(f'net_{short}', 0)
        feats[f'pw_onco_x_{short}'] = is_oncology * net_val
        feats[f'pw_inf_x_{short}'] = is_infectious * net_val

    n_pw_dis = feats.get('net_n_pathways_with_disease', 0)
    n_pw_total = feats.get('net_n_enriched_total', 0)
    n_offtarget = feats.get('net_n_off_target_pathways', 0)
    feats['pw_relative_overlap'] = np.nan
    feats['pw_relative_score'] = np.nan
    feats['pw_exclusivity'] = np.nan
    feats['pw_pathway_exclusivity'] = n_pw_dis / (n_pw_total + 1) if n_pw_total > 0 else 0
    feats['pw_non_onco_pathway_overlap'] = (1 - is_oncology) * n_pw_dis
    feats['pw_non_onco_disease_score_max'] = (1 - is_oncology) * feats.get('net_disease_pathway_score_max', 0)
    feats['pw_off_target_exclusivity'] = n_offtarget / (n_pw_total + 1) if n_pw_total > 0 else 0
    feats['pw_mechanism_pathway_ratio'] = n_pw_dis / (n_offtarget + 1)
    feats['pw_pathway_concentration'] = feats.get('net_disease_pathway_score_max', 0) / (feats.get('net_max_enrichment_score', 0) + 1e-8)
    feats['pw_top_pathway_importance'] = feats.get('net_disease_pathway_score_max', 0) * (n_pw_dis / (n_pw_total + 1) if n_pw_total > 0 else 0)

    # ── Disease complexity ──────────────────────────────────────────────────
    dt_scores_list = list(disease_target_scores.get(disease, {}).values())
    if dt_scores_list:
        dt_scores_capped = sorted(dt_scores_list, reverse=True)[:20]
        feats['disease_complexity_n'] = len(dt_scores_capped)
        dt_arr = np.array(dt_scores_capped)
        feats['disease_complexity_score_entropy'] = -(dt_arr * np.log(dt_arr + 1e-10)).sum()
        feats['disease_complexity_top_frac'] = max(dt_scores_capped) / (sum(dt_scores_capped) + 1e-10)
    else:
        feats['disease_complexity_n'] = 0
        feats['disease_complexity_score_entropy'] = 0
        feats['disease_complexity_top_frac'] = 0
    feats['target_overlap_count'] = np.nan
    feats['target_overlap_weighted'] = np.nan

    # ── Mechanism/repurposing features ──────────────────────────────────────
    tissue_area_scores = {}
    for tissue in TISSUE_NAMES:
        area = TISSUE_TO_AREA.get(tissue)
        enst_val = feats.get(f'n_enst_{tissue}_above_target_interaction', 0)
        if area and enst_val > 0:
            tissue_area_scores[area] = tissue_area_scores.get(area, 0) + enst_val
    total_area = sum(tissue_area_scores.values())
    mech_areas = {a: s / total_area for a, s in tissue_area_scores.items()} if total_area > 0 else {}
    trial_area = indication
    if mech_areas:
        feats['mech_area_affinity'] = mech_areas.get(trial_area, 0)
        primary_area = max(mech_areas, key=mech_areas.get)
        feats['mech_primary_match'] = int(primary_area == trial_area)
        area_vals = np.array(list(mech_areas.values()))
        if len(area_vals) > 1 and area_vals.sum() > 0:
            probs = area_vals / area_vals.sum()
            ent = -np.sum(probs * np.log2(probs + 1e-10))
            feats['mech_specificity'] = 1.0 / (1.0 + ent)
        else:
            feats['mech_specificity'] = 1.0
        feats['mech_n_areas'] = len([a for a, s in mech_areas.items() if s > 0.1])
        feats['mech_onco_for_inf'] = int(mech_areas.get('oncology', 0) > 0.3 and trial_area == 'infectious')
    else:
        for c in ['mech_area_affinity', 'mech_primary_match', 'mech_specificity', 'mech_n_areas', 'mech_onco_for_inf']:
            feats[c] = 0

    return feats


# ============================================================================
# MODEL LOADING & PREDICTION
# ============================================================================

def load_bundle(bundle_path):
    """Load serialized model bundle."""
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)
    required = ['safety_model', 'safety_imputer', 'safety_feature_cols',
                'efficacy_gbm', 'efficacy_imputer', 'efficacy_feature_cols',
                'gene_to_enst', 'disease_targets_cache']
    missing = [k for k in required if k not in bundle]
    if missing:
        raise ValueError(f"Bundle missing keys: {missing}")
    return bundle


def resolve_disease_targets(disease, cache, gene_to_enst):
    """Resolve disease targets from cache. Returns (enst_set, scores_dict)."""
    # Try direct disease name as search term
    search_terms = [disease.lower()]
    # Common normalizations
    if 'cancer' in disease.lower():
        search_terms.append('cancer')
    if 'hiv' in disease.lower():
        search_terms.append('hiv infection')
    if 'glaucoma' in disease.lower():
        search_terms.append('glaucoma')
    if 'pain' in disease.lower():
        search_terms.append('pain')

    for term in search_terms:
        disease_id = cache.get(f'search:{term}')
        if disease_id:
            targets_data = cache.get(f'targets:{disease_id}')
            if targets_data:
                enst_set = set()
                scores = {}
                for t in targets_data['targets']:
                    symbol = t['symbol']
                    enst = gene_to_enst.get(symbol)
                    if enst:
                        enst_set.add(enst)
                    scores[symbol] = t['score']
                return enst_set, scores

    return set(), {}


def predict(input_dir, output_path, bundle_path):
    """Run the model over the input directory."""
    input_dir = Path(input_dir)
    bundle = load_bundle(bundle_path)

    gene_to_enst = bundle['gene_to_enst']
    cache = bundle['disease_targets_cache']

    # Read compounds
    compounds = pd.read_csv(input_dir / 'compounds.csv')
    print(f"Loaded {len(compounds)} compounds from {input_dir / 'compounds.csv'}")

    # Resolve disease targets for each unique disease
    unique_diseases = compounds['disease'].unique()
    disease_targets_enst = {}
    disease_target_scores = {}
    for d in unique_diseases:
        enst_set, scores = resolve_disease_targets(d, cache, gene_to_enst)
        disease_targets_enst[d] = enst_set
        disease_target_scores[d] = scores
        print(f"  Disease '{d}': {len(enst_set)} ENST targets")

    # Compute features for each compound
    all_feats = []
    for _, row in compounds.iterrows():
        zinc_id = row['zinc_id']
        disease = row['disease']
        target = row.get('target', '')

        feats, transcript_scores, scores_array = compute_features_for_compound(
            zinc_id, disease, target, input_dir,
            gene_to_enst, disease_targets_enst, disease_target_scores)

        # Store zinc_id for network file lookup
        feats['_zinc_id'] = zinc_id

        feats = compute_domain_features(
            feats, disease, transcript_scores, scores_array,
            gene_to_enst, disease_targets_enst, disease_target_scores,
            input_dir)

        # Remove internal key
        feats.pop('_zinc_id', None)
        all_feats.append(feats)

    new_feat = pd.DataFrame(all_feats)
    print(f"Computed {new_feat.shape[1]} raw features")

    # Predict for each task
    results = {}
    for task in ['safety', 'efficacy']:
        if task == 'safety':
            imputer = bundle['safety_imputer']
            imputer_cols = bundle.get('safety_imputer_cols', bundle['safety_feature_cols'])
            feature_cols = bundle['safety_feature_cols']
            models = [bundle['safety_model']]
        else:
            imputer = bundle['efficacy_imputer']
            imputer_cols = bundle.get('efficacy_imputer_cols', bundle['efficacy_feature_cols'])
            feature_cols = bundle['efficacy_feature_cols']
            models = [bundle['efficacy_gbm']]
            if 'efficacy_xgb' in bundle:
                models.append(bundle['efficacy_xgb'])
            if 'efficacy_lgbm' in bundle:
                models.append(bundle['efficacy_lgbm'])

        # Align to imputer columns (pre-dedup), impute, then select deduped cols
        X = pd.DataFrame(index=range(len(compounds)), columns=imputer_cols, dtype=float)
        n_available = 0
        for col in imputer_cols:
            if col in new_feat.columns:
                X[col] = new_feat[col].values
                if col in feature_cols:
                    n_available += 1
            else:
                X[col] = np.nan

        # Impute using training medians
        X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns)

        # Select deduped feature columns
        X_clean = X_imp[feature_cols]

        # Predict (average for ensemble)
        probas = np.zeros(len(compounds))
        for model in models:
            probas += model.predict_proba(X_clean)[:, 1]
        probas /= len(models)

        results[task] = probas
        coverage = n_available / len(feature_cols) * 100
        print(f"  {task}: {n_available}/{len(feature_cols)} features ({coverage:.0f}%), "
              f"P(fail) range [{probas.min():.3f}, {probas.max():.3f}]")

    # Build output
    out = compounds[['zinc_id', 'disease', 'target']].copy().reset_index(drop=True)
    if 'SMILES' in compounds.columns:
        out['SMILES'] = compounds['SMILES'].values
    out['indication'] = compounds['disease'].apply(classify_indication).values
    out['P_FAIL_SAFETY'] = results['safety']
    out['P_FAIL_EFFICACY'] = results['efficacy']
    out['P_PASS'] = 1 - out[['P_FAIL_SAFETY', 'P_FAIL_EFFICACY']].max(axis=1)

    safety_cols = bundle['safety_feature_cols']
    efficacy_cols = bundle['efficacy_feature_cols']
    n_safety = sum(1 for c in safety_cols if c in new_feat.columns)
    n_efficacy = sum(1 for c in efficacy_cols if c in new_feat.columns)
    out['feature_coverage_safety'] = f"{n_safety}/{len(safety_cols)}"
    out['feature_coverage_efficacy'] = f"{n_efficacy}/{len(efficacy_cols)}"

    out.to_csv(output_path, index=False)
    print(f"\nPredictions written to {output_path}")

    # Summary
    print(f"\n{'zinc_id':<25} {'disease':<35} {'P(safe)':<9} {'P(effic)':<9} {'P(pass)':<9}")
    print("-" * 90)
    for _, r in out.iterrows():
        print(f"{r['zinc_id']:<25} {r['disease']:<35} "
              f"{1 - r['P_FAIL_SAFETY']:<9.1%} {1 - r['P_FAIL_EFFICACY']:<9.1%} {r['P_PASS']:<9.1%}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Trial-outcome prediction model')
    parser.add_argument('--input-dir', required=True,
                        help='Directory with compounds.csv, binding/, string/, biolprop_merged.tsv')
    parser.add_argument('--output', default='predictions.csv',
                        help='Output CSV path (default: predictions.csv)')
    parser.add_argument('--bundle', required=True,
                        help='Path to model_bundle.pkl')
    args = parser.parse_args()

    predict(args.input_dir, args.output, args.bundle)


if __name__ == '__main__':
    main()
