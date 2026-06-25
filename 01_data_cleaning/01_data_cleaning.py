#!/usr/bin/env python3
"""
01_data_cleaning.py -- Data Loading and Cleaning pipeline for ARP spoofing detection
CICIoT2023: MITM-ArpSpoofing + Benign_Final

This is the first stage of the research pipeline: it loads the raw CICFlowMeter
CSVs, applies domain-grounded cleaning fixes, validates the result against
impossible values, and checks for outliers/duplicates. It produces the single
checkpoint dataset (final_dataset.csv) that every later stage builds on.

No feature engineering or feature selection happens in this file -- those
decisions depend on the train/validation/test split and must be fit on the
training partition only, so they belong in later pipeline stages, not here.

ROADMAP (this file)
---------------------------------------------------------------------------
  SECTION 2   Dataset Description
  SECTION 3   Data Loading and Cleaning
                (3.1 Rate Inf, 3.2 Std/Variance NaN, 3.3 IAT artifact)
  SECTION 4   Domain Validation & Artifact Removal
                (35 impossible-value checks, Header_Length=0 drop,
                 per-class IQR outlier analysis, duplicate analysis)
                -> saves data/final_dataset.csv (checkpoint)

INPUTS  (read from BASE, see "Paths" below)
  Benign_Final/*.csv      -- benign IoT traffic flows        (label = 0)
  MITM-ArpSpoofing/*.csv  -- ARP spoofing attack flows        (label = 1)
  Both are CICIoT2023 subsets exported by CICFlowMeter.

DATA ACCESS  (the raw data is NOT in this repository)
  CICIoT2023's usage terms do not allow redistributing the raw data, so it
  is not committed alongside this script. To reproduce the pipeline, get
  your own copy directly from the source below and place it under BASE.

  Source : CIC Research, University of New Brunswick
           https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/
  Cite   : Neto, E.C.P. et al., "CICIoT2023: A Real-Time Dataset and
           Benchmark for Large-Scale Attacks in IoT Environment,"
           Sensors, 23(13):5941, 2023.

  Steps:
    1. Download the CSV (flow-level) release from the link above -- this
       script needs the CICFlowMeter-derived CSVs, not the raw pcaps.
    2. The full download has one folder per attack category (DDoS-*,
       DoS-*, Mirai-*, Recon-*, MITM-ArpSpoofing, ...) plus benign traffic.
       This script only needs two of them:
         Benign_Final/      (4 files, e.g. BenignTraffic.pcap.csv, BenignTraffic1.pcap.csv, ...)
         MITM-ArpSpoofing/  (2 files, e.g. MITM-ArpSpoofing.pcap.csv, MITM-ArpSpoofing1.pcap.csv)
    3. Copy those two folders, unmodified, into BASE so the layout is:
         <BASE>/Benign_Final/*.csv
         <BASE>/MITM-ArpSpoofing/*.csv
       load_folder() (below) globs every *.csv in each folder, so extra or
       renamed files are fine as long as the two folder names match.

OUTPUTS
  data/final_dataset.csv      Section 4 checkpoint: cleaned and
                                           validated, before any feature
                                           engineering (1,405,749 rows).

This file is fully deterministic -- no sampling or randomness is involved,
so no random_state is needed here (later pipeline stages that do sample
data set random_state=42 for reproducibility).
"""
import sys, io as _sysio
sys.stdout = _sysio.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = _sysio.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import glob, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format

# ─────────────────────────────────────────────
# Paths -- every input/output location this script touches
# ─────────────────────────────────────────────
# Inputs (raw CICIoT2023 CSVs) -- not included in this repo, see the
# "DATA ACCESS" section in the module docstring above for how to get them.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO_ROOT, 'data')

# Output -- the cleaned/validated checkpoint consumed by every later stage.
OUT_PATH = os.path.join(BASE, 'final_dataset.csv')
# Section 4 checkpoint: cleaned + validated dataset, no engineered features yet.

# ─────────────────────────────────────────────
# SECTION 2: DATASET DESCRIPTION -- load & merge raw CSVs
# ─────────────────────────────────────────────
def load_folder(folder_path, label):
    files = sorted(glob.glob(folder_path + '/*.csv'))
    dfs = []
    for f in files:
        tmp = pd.read_csv(f)
        print(f"  Loaded {f.split('/')[-1]}: {len(tmp):,} rows")
        dfs.append(tmp)
    combined = pd.concat(dfs, ignore_index=True)
    combined['label'] = label
    return combined

print("Loading Benign traffic...")
benign = load_folder(BASE + '/Benign_Final', 0)
print(f"  → Benign total: {len(benign):,} rows\n")

print("Loading MITM-ArpSpoofing traffic...")
attack = load_folder(BASE + '/MITM-ArpSpoofing', 1)
print(f"  → Attack total: {len(attack):,} rows\n")

df = pd.concat([benign, attack], ignore_index=True)
feature_cols = [c for c in df.columns if c != 'label']

# ─────────────────────────────────────────────
# SECTION 2 (cont.): raw dataset overview, before any cleaning
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: DATASET DESCRIPTION -- RAW DATASET OVERVIEW (BEFORE CLEANING)")
print("=" * 70)

n_total_raw  = len(df)
n_benign_raw = (df['label'] == 0).sum()
n_attack_raw = (df['label'] == 1).sum()
print(f"Shape         : {df.shape}")
print(f"Features      : {len(feature_cols)}")
print(f"Total flows   : {n_total_raw:,}")
print(f"Benign flows  : {n_benign_raw:,}  ({n_benign_raw/n_total_raw*100:.1f}%)")
print(f"Attack flows  : {n_attack_raw:,}  ({n_attack_raw/n_total_raw*100:.1f}%)")
print(f"\nFeature list  : {feature_cols}")

raw_stats = df[feature_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
raw_stats['missing'] = df[feature_cols].isnull().sum()
raw_stats['inf'] = [
    int(np.isinf(df[c]).sum()) if df[c].dtype in ['float64', 'int64'] else 0
    for c in feature_cols
]
raw_stats['skewness'] = df[feature_cols].skew()
print("\n" + raw_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
                         'missing', 'inf', 'skewness']].to_string())

# ─────────────────────────────────────────────
# SECTION 3: DATA LOADING AND CLEANING
# Three targeted, domain-grounded fixes, applied before any
# analysis. None of these are statistical imputation -- each value below is
# derived from what the quantity means, not from the data's own distribution.
# ───────────────────────────────────
# 3.1  Rate: replace Inf with SmartNIC MAX_INT
# ─────────────────────────────────────────────
# Rate = Number / flow_duration. When duration = 0 (all packets in the same
# clock tick), Rate → ∞. This is not missing data — it is a real zero-duration
# burst whose rate exceeded measurement resolution. On a SmartNIC the integer
# guard "if duration == 0: rate = MAX_INT" is the standard fix. We use 2^32 - 1
# (32-bit unsigned integer maximum), the natural ceiling for SmartNIC counters.
SMARTNIC_MAX_INT = 2**32 - 1   # 4,294,967,295

n_inf_rate = int(np.isinf(df['Rate']).sum())
df['Rate'] = df['Rate'].replace(np.inf, SMARTNIC_MAX_INT)
print(f"\nRate: replaced {n_inf_rate} Inf values with SmartNIC MAX_INT ({SMARTNIC_MAX_INT:,})")

# 3.2  Std and Variance: replace NaN with 0 for single-packet flows
# Both are undefined when Number = 1 (no spread exists for a single value).
# Zero is mathematically exact — not an approximation — because a single packet
# has zero deviation from itself. Volume/rate info is already in Rate and Number.
n_nan_std = int(df['Std'].isnull().sum())
n_nan_var = int(df['Variance'].isnull().sum())
df['Std']      = df['Std'].fillna(0)
df['Variance'] = df['Variance'].fillna(0)
print(f"Std:      replaced {n_nan_std} NaN values with 0 (single-packet flows)")
print(f"Variance: replaced {n_nan_var} NaN values with 0 (single-packet flows)")

# 3.3  IAT: set to 0 for single-packet flows
# IAT = mean inter-arrival time between consecutive packets. With Number = 1
# there are no consecutive pairs — IAT is undefined exactly like Std/Variance.
# The CICIoT2023 extractor produced small non-zero artifact values (0.000001 to
# 0.084 s) for all 83 single-packet flows instead of 0 or NaN. Setting to 0 is
# mathematically correct and consistent with the Std/Variance treatment above.
# No data leakage risk — 0 is domain-defined, not derived from the data.
single_mask = df['Number'] == 1
n_iat_fix   = int((single_mask & (df['IAT'] > 0)).sum())
df.loc[single_mask, 'IAT'] = 0
print(f"IAT:      set to 0 for {n_iat_fix} single-packet flows (artifact non-zero values)")

n_total  = len(df)
n_benign = (df['label'] == 0).sum()
n_attack = (df['label'] == 1).sum()
pct_ben  = n_benign / n_total * 100
pct_atk  = n_attack / n_total * 100

print("\n" + "=" * 70)
print("SECTION 3: DATA LOADING AND CLEANING -- post-cleaning shape")
print("=" * 70)
print(f"Shape         : {df.shape}")
print(f"Features      : {len(feature_cols)}")
print(f"Total flows   : {n_total:,}")
print(f"Benign flows  : {n_benign:,}  ({pct_ben:.1f}%)")
print(f"Attack flows  : {n_attack:,}  ({pct_atk:.1f}%)")
print(f"\nFeature list  : {feature_cols}")

# ─────────────────────────────────────────────
# SECTION 3 (cont.): post-cleaning descriptive statistics
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("OVERALL FEATURE STATISTICS")
print("=" * 70)

stats = df[feature_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
stats['missing'] = df[feature_cols].isnull().sum()
stats['inf']     = [
    int(np.isinf(df[c]).sum()) if df[c].dtype in ['float64', 'int64'] else 0
    for c in feature_cols
]
stats['skewness'] = df[feature_cols].skew()

print(stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
             'missing', 'inf', 'skewness']].to_string())

# ─────────────────────────────────────────────
# SECTION 3 (cont.): post-cleaning data-quality check (NaN/Inf, zero-variance)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: DATA LOADING AND CLEANING -- DATA QUALITY ISSUES (should be none)")
print("=" * 70)

dq = []
for col in feature_cols:
    n_miss = int(df[col].isnull().sum())
    n_inf  = int(np.isinf(df[col]).sum()) if df[col].dtype in ['float64', 'int64'] else 0
    if n_miss > 0 or n_inf > 0:
        dq.append({'Feature': col, 'NaN': n_miss, 'Inf': n_inf})

if dq:
    print(pd.DataFrame(dq).to_string(index=False))
else:
    print("  None")

# Zero-variance features
zero_var = [c for c in feature_cols if df[c].std() == 0]
print(f"\nZero-variance features (all values identical): {zero_var}")

# ─────────────────────────────────────────────
# SECTION 4: DOMAIN VALIDATION AND ARTIFACT REMOVAL -- impossible value checks
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: DOMAIN VALIDATION AND ARTIFACT REMOVAL")
print("=" * 70)

violations = []

def report(check, valid_mask, note=''):
    # valid_mask is True for rows that PASS the check — violations are the inverse
    n = int((~valid_mask).sum())
    status = 'FAIL' if n > 0 else 'OK'
    violations.append({'Check': check, 'Status': status, 'Violations': n, 'Note': note})

# No feature should be negative
for col in feature_cols:
    report(f'{col} >= 0', df[col] >= 0)

# Ratio features must be in [0, 1]
ratio_cols = [
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC'
]
for col in ratio_cols:
    report(f'{col} in [0,1]', (df[col] >= 0) & (df[col] <= 1))

# TTL and Protocol Type bounded by IP header spec
report('Time_To_Live in [0, 255]',  (df['Time_To_Live']  >= 0) & (df['Time_To_Live']  <= 255))
report('Protocol Type in [0, 255]', (df['Protocol Type'] >= 0) & (df['Protocol Type'] <= 255))

# A flow must have at least 1 packet
report('Number >= 1', df['Number'] >= 1)

# Packet size ordering must be consistent
report('Min <= AVG',  df['Min'] <= df['AVG'])
report('AVG <= Max',  df['AVG'] <= df['Max'])
report('Min <= Max',  df['Min'] <= df['Max'])

# Spread features must be non-negative (already replaced NaN with 0 above)
report('Std >= 0',      df['Std']      >= 0)
report('Variance >= 0', df['Variance'] >= 0)

# Payload cannot exceed total bytes
report('Tot size <= Tot sum', df['Tot size'] <= df['Tot sum'])

# IAT must be non-negative
report('IAT >= 0', df['IAT'] >= 0)

val_df = pd.DataFrame(violations)
fails = val_df[val_df['Status'] == 'FAIL']
ok_count = (val_df['Status'] == 'OK').sum()

print(f"\n  Checks passed : {ok_count} / {len(val_df)}")
if len(fails) > 0:
    print(f"  Checks FAILED : {len(fails)}")
    print(fails[['Check', 'Violations', 'Note']].to_string(index=False))
else:
    print("  All checks passed — no impossible values found.")

# Header_Length = 0: expected only for pure ARP flows (no IP header).
# ARP is Layer 2 — it has no IP header, so Header_Length = 0 is correct for
# pure ARP flows (ARP = 1.0). Any row with Header_Length = 0 and ARP < 1.0
# is contradictory: it claims to carry non-ARP traffic (which has an IP header)
# yet reports zero header length. These are feature extractor artifacts.
#
# Investigation found 2 such rows:
#   Row 1207256 — pure ICMP flow (ICMP=1.0, IPv=1.0) with Header_Length=0.
#                 IPv4+ICMP must have an IP header; zero is physically impossible.
#                 Likely a malformed/crafted packet the extractor could not parse.
#   Row 1228595 — mixed ARP+ICMP flow (ARP=0.9, ICMP=0.1) with Header_Length=0
#                 AND Time_To_Live=6.4 (fractional TTL — impossible for a real
#                 packet; TTL is a single integer byte). Both anomalies together
#                 confirm this is a CICIoT2023 extractor artifact, not real traffic.
#
# Decision: DROP both rows. They represent 2 / 1,405,751 (0.00014%) of the dataset
# and are structurally impossible, not edge cases. Keeping them would introduce
# noise with no physical meaning.
hl_zero  = df[df['Header_Length'] == 0]
non_arp  = hl_zero[hl_zero['ARP'] < 1.0]
print(f"\n  Header_Length = 0 rows : {len(hl_zero)}")
print(f"    Of which ARP = 1.0   : {int((hl_zero['ARP'] == 1.0).sum())}  (expected — ARP has no IP header)")
print(f"    Of which ARP < 1.0   : {len(non_arp)}  (extractor artifacts — dropping)")

drop_idx = non_arp.index
df = df.drop(index=drop_idx).reset_index(drop=True)
print(f"\n  Dropped {len(drop_idx)} artifact rows.")
print(f"  Dataset size after drop: {len(df):,} rows")

# ─────────────────────────────────────────────
# SECTION 4 (cont.): Outlier Analysis -- Per-Class IQR Detection
# ─────────────────────────────────────────────
#
# WHY NOT GLOBAL OUTLIER REMOVAL:
# In this dataset the "outliers" ARE the label. Attack flows will always appear
# statistically extreme on features like Rate, ARP, and IAT — that separation is
# exactly what the classifier exploits. A global outlier removal (e.g. drop anything
# beyond 3 sigma or Q3+1.5*IQR across the full dataset) would delete the most
# informative attack samples and actively degrade the model.
#
# THE RIGHT QUESTION is not "are there outliers?" but "are there values extreme
# even within their own class?" There are two meaningful concerns:
#
#   1. Benign flows that look like attack traffic — a benign flow with very high
#      Rate or ARP ratio is suspicious. It could be a mislabeled flow or a rare
#      but legitimate IoT burst. These should be inspected, not automatically dropped.
#
#   2. Within-class extremes that suggest extractor errors — values so extreme they
#      could not plausibly come from any real traffic even of that class. These are
#      the ones worth investigating (like the Header_Length=0 + ICMP case above).
#
# APPROACH: IQR-based detection PER CLASS with a wide threshold (Q1 - 3*IQR and
# Q3 + 3*IQR). The 3x multiplier is intentionally conservative — we are looking
# for structural impossibilities and gross errors, not routine high-value traffic.
#
# IMPORTANT NOTE ON ZERO-INFLATED FEATURES:
# Features like syn_count, fin_count, rst_count have IQR = 0 (most flows = 0).
# When IQR = 0, the fence collapses to Q1/Q3 itself, so ANY non-zero value is
# flagged. This is a known property of zero-inflated features and does NOT indicate
# a data problem. These flags should be read as "this feature has non-zero values"
# not as "these rows are errors."
#
# CONCLUSION FROM THE ANALYSIS (run below):
# No rows are dropped from the outlier analysis. All findings fall into one of:
#   a) Legitimate domain behavior — high Rate/IAT in attack, large packets in benign
#   b) Zero-inflated feature artifact — IQR=0 makes every non-zero value look extreme
#   c) Already handled — Rate=MAX_INT single-packet flows (65 benign, 18 attack)
#      are structurally valid and already fixed above
# The 1,777 benign flows exceeding the attack 99th percentile Rate are high-speed
# legitimate IoT bursts. The model will separate them using multi-feature boundaries,
# not a single Rate threshold. Dropping them would bias the benign class distribution.

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS — PER-CLASS IQR (threshold: Q1-3*IQR / Q3+3*IQR)")
print("=" * 70)

continuous_cols = [
    'Rate', 'IAT', 'Header_Length', 'Time_To_Live', 'Tot sum',
    'Min', 'Max', 'AVG', 'Std', 'Variance',
    'Number', 'ack_count', 'syn_count', 'fin_count', 'rst_count'
]

benign_df = df[df['label'] == 0]
attack_df = df[df['label'] == 1]

outlier_results = []
for feat in continuous_cols:
    for lbl, name, sub in [(0, 'Benign', benign_df), (1, 'Attack', attack_df)]:
        Q1  = sub[feat].quantile(0.25)
        Q3  = sub[feat].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - 3 * IQR
        hi  = Q3 + 3 * IQR
        n_lo = int((sub[feat] < lo).sum())
        n_hi = int((sub[feat] > hi).sum())
        if n_lo > 0 or n_hi > 0:
            outlier_results.append({
                'Feature'       : feat,
                'Class'         : name,
                'Below fence'   : n_lo,
                'Above fence'   : n_hi,
                'Lower fence'   : round(lo, 2),
                'Upper fence'   : round(hi, 2),
                'Actual min'    : round(sub[feat].min(), 4),
                'Actual max'    : round(sub[feat].max(), 4),
            })

out_df = pd.DataFrame(outlier_results)
print(out_df.to_string(index=False))

# Cross-class inspection: benign flows exceeding attack extremes
print("\n--- Cross-class check: benign flows in attack-level territory ---")
rate_99_atk = attack_df['Rate'].quantile(0.99)
iat_99_atk  = attack_df['IAT'].quantile(0.99)
n_ben_high_rate = int((benign_df['Rate'] > rate_99_atk).sum())
n_ben_high_iat  = int((benign_df['IAT']  > iat_99_atk).sum())
n_ben_max_rate  = int((benign_df['Rate'] == 2**32 - 1).sum())
n_atk_max_rate  = int((attack_df['Rate'] == 2**32 - 1).sum())

print(f"  Attack 99th pct Rate           : {rate_99_atk:,.1f} pkt/s")
print(f"  Benign flows above that Rate   : {n_ben_high_rate:,}  "
      f"(legitimate IoT bursts — retain, model uses multi-feature boundaries)")
print(f"  Attack 99th pct IAT            : {iat_99_atk:.6f} s")
print(f"  Benign flows above that IAT    : {n_ben_high_iat:,}  "
      f"(infrequent IoT devices with long gaps — retain)")
print(f"  Rate=MAX_INT in Benign/Attack  : {n_ben_max_rate} / {n_atk_max_rate}  "
      f"(single-packet zero-duration bursts — already handled)")

print("\n  Decision: NO rows dropped from outlier analysis.")
print("  All flagged values are legitimate domain behavior or zero-inflated")
print("  feature artifacts. Dropping them would remove discriminative signal.")

# ─────────────────────────────────────────────
# SECTION 4 (cont., supplementary): Cardinality, Duplicates & Near-Duplicates
# ─────────────────────────────────────────────
#
# This block is supplementary data-quality investigation that feeds two downstream
# decisions: the 557 label-conflicting duplicate groups documented here are
# the irreducible-overlap argument for why SMOTE was not used when modelling, and the cardinality flags inform which features
# look "near-constant" before the feature engineering stage's drop list is finalised.
#
# CARDINALITY:
# Cardinality = the number of distinct values a feature takes. Unexpected
# cardinality can signal encoding errors (a continuous feature that should
# have thousands of values but only has 3) or near-constant features that
# add no information to the model.
#
# Expected cardinality in this dataset:
#   - Truly continuous (high cardinality): Rate, IAT, Std, Variance, AVG,
#     Tot sum, Tot size, Header_Length, Time_To_Live
#   - Discretised ratios (medium): flag_number and protocol ratio features —
#     values are k/N where N = Number (1-10), so at most ~33 unique values
#   - Integer counts (low): ack_count, syn_count, fin_count, rst_count (0-10)
#   - Near-constant / constant: Telnet=0, SMTP=0 (already noted as zero-variance)
#     IRC: 1,092 non-zero rows out of 1.4M (0.08%) — value is only 0 or 0.1
#     IGMP: 3,578 non-zero rows (0.25%) — values 0, 0.1, 0.2, 0.3
#     SSH: 25,965 non-zero rows (1.8%) — sparse but not constant
#   - Protocol Type: only 4 values (0=none/ARP, 1=ICMP, 6=TCP, 17=UDP)
#     This is correct — CICIoT2023 uses standard IANA IP protocol numbers.
#
# IRC and IGMP are so dominated by zeros that they carry almost no information.
# Whether to drop them is a modelling decision (feature selection step), not a
# data quality issue. Flag them here; decide in the feature engineering stage.
#
# EXACT DUPLICATES — DECISION: RETAIN ALL
# 30,932 rows where ALL 39 features AND the label are identical.
# The majority are attack flows (28,804 attack vs 6,106 benign).
#
# Retention rationale: In ARP spoofing, the attacker sends the same spoofed
# ARP reply repeatedly to poison caches across the network. The repetition of
# an identical flow IS the attack pattern — scripted tooling sends the same
# packet template in a loop, producing identical feature vectors on every
# iteration. Deduplicating would erase exactly the signal that distinguishes
# automated attack behaviour from one-off legitimate traffic.
# The same logic applies to benign duplicates: IoT sensors stream identical
# periodic readings (e.g. a temperature sensor sending the same payload every
# second). That repetition is real traffic, not an artifact.
#
# Timestamp note: raw timestamps are NOT present in the CICIoT2023 pre-processed
# CSVs, so direct temporal features cannot be built from this file. However, a
# FLOW REPETITION COUNT feature can be engineered as a proxy — for each row,
# count how many other rows share the same feature vector. A high count signals
# scripted/repeated behaviour. This must be computed on the training split only
# and then mapped to test to avoid data leakage. Flag for feature engineering stage.
#
# LABEL-CONFLICTING DUPLICATES — DECISION: RETAIN, DOCUMENT AS LIMITATION
# 557 groups where all 39 feature values are identical but the label differs
# (some copies benign, others attack). This affects 22,881 rows.
# Root cause: some attack flows are statistically indistinguishable from
# legitimate traffic using these features alone. In ARP spoofing, the attacker
# also generates normal-looking TCP/UDP follow-on flows after poisoning the cache
# — these can be feature-identical to benign flows from the same device type.
# This creates IRREDUCIBLE classification error: no model can separate these
# rows from features alone. This is a fundamental dataset limitation, not an
# error — and it is stated explicitly in the paper as
# a bound on achievable classification performance.
# Retaining them is the honest choice: it preserves the real ambiguity rather
# than artificially inflating accuracy by removing hard cases.
#
# NEAR-DUPLICATES — DECISION: RETAIN ALL
# When continuous features (Rate, IAT, Std, Variance, etc.) are rounded to
# 2 decimal places, 28,319 additional near-duplicates appear beyond exact matches.
# These represent flows with nearly identical statistical summaries but slightly
# different floating-point values (e.g. Rate=229.40 vs 229.41 from slightly
# different burst timings). Like exact duplicates, these are real repeated attack
# flows with minor timing variation — the near-identical pattern across many rows
# is itself a signal of scripted behaviour. For tree-based models (Random Forest,
# Decision Tree) this is not a concern — small differences in continuous features
# create valid distinct splits. Near-duplicate removal is only relevant for
# distance-based models (KNN, SVM) where near-identical rows inflate density
# estimates.

print("\n" + "=" * 70)
print("CARDINALITY CHECK")
print("=" * 70)

card_rows = []
for col in feature_cols:
    n_unique = df[col].nunique()
    pct_zero = (df[col] == 0).mean() * 100
    card_rows.append({'Feature': col, 'Unique values': n_unique,
                      '% zero': round(pct_zero, 2)})
card_df = pd.DataFrame(card_rows)
print(card_df.to_string(index=False))

# Flag near-constant features (>99% of rows are the same value)
print("\n  Near-constant features (>99% identical value):")
for col in feature_cols:
    top_freq = df[col].value_counts(normalize=True).iloc[0]
    if top_freq > 0.99:
        top_val = df[col].value_counts().index[0]
        print(f"    {col}: {top_freq*100:.2f}% = {top_val}")

print("\n" + "=" * 70)
print("DUPLICATE & NEAR-DUPLICATE ANALYSIS")
print("=" * 70)

# Exact duplicates (all features + label)
n_exact_all   = int(df.duplicated().sum())
exact_all_df  = df[df.duplicated(keep=False)]
n_exact_ben   = int((exact_all_df['label'] == 0).sum())
n_exact_atk   = int((exact_all_df['label'] == 1).sum())
print(f"\n  Exact duplicates (features + label)  : {n_exact_all:,} rows")
print(f"    Benign rows involved               : {n_exact_ben:,}")
print(f"    Attack rows involved               : {n_exact_atk:,}")
print(f"    % of total dataset                 : {n_exact_all/len(df)*100:.2f}%")

# Exact duplicates (features only — catches label conflicts)
n_feat_dup    = int(df.duplicated(subset=feature_cols).sum())
feat_dup_df   = df[df.duplicated(subset=feature_cols, keep=False)]
label_nuniq   = feat_dup_df.groupby(feature_cols)['label'].nunique()
n_conflict    = int((label_nuniq > 1).sum())
n_conflict_rows = int(feat_dup_df[
    feat_dup_df.set_index(feature_cols).index.isin(label_nuniq[label_nuniq > 1].index)
].shape[0])
print(f"\n  Feature-identical rows (any label)   : {n_feat_dup:,} rows")
print(f"    Groups with conflicting labels     : {n_conflict:,} groups")
print(f"    Total rows in conflicting groups   : {n_conflict_rows:,}")

# Near-duplicates (continuous features rounded to 2dp)
continuous_high = ['Rate', 'IAT', 'Std', 'Variance', 'AVG', 'Tot size',
                   'Header_Length', 'Time_To_Live', 'Tot sum']
df_rounded = df.copy()
for col in continuous_high:
    df_rounded[col] = df_rounded[col].round(2)
n_near = int(df_rounded.duplicated(subset=feature_cols).sum())
print(f"\n  Near-duplicates (continuous rounded  ")
print(f"  to 2 decimal places)                 : {n_near:,} rows")
print(f"    Of which already exact duplicates  : {n_feat_dup:,}")
print(f"    Additional near-duplicates         : {n_near - n_feat_dup:,}")

print(f"\n  Decisions (see comments for full rationale):")
print(f"    Exact duplicates (same label)      : RETAIN — repetition is the attack signal")
print(f"    Label-conflicting groups           : RETAIN — irreducible dataset limitation,")
print(f"                                         documents the classification boundary")
print(f"    Near-duplicates                    : RETAIN — same rationale as exact dups;")
print(f"                                         revisit only if using distance-based models")
print(f"    Flow repetition count feature      : FLAG for feature engineering stage")

# ─────────────────────────────────────────────
# SECTION 4 (checkpoint): save final cleaned dataset
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: DOMAIN VALIDATION AND ARTIFACT REMOVAL -- SAVING CHECKPOINT")
print("=" * 70)
print("  This is the cleaned + validated dataset, BEFORE any")
print("  feature engineering. It is NOT the modelling dataset -- see later pipeline stages.")
print("  OUT_PATH is defined once, in the Paths section above.")

df.to_csv(OUT_PATH, index=False)

print(f"\n  Rows    : {len(df):,}")
print(f"  Columns : {df.shape[1]} ({len(feature_cols)} features + label)")
print(f"  NaN     : {df.isnull().sum().sum()}")
n_inf_total = sum(int(np.isinf(df[c]).sum()) for c in df.columns if df[c].dtype == 'float64')
print(f"  Inf     : {n_inf_total}")
print(f"  Saved to: {OUT_PATH}")
