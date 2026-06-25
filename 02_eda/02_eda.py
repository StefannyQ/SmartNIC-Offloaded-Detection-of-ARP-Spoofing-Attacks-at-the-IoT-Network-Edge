#!/usr/bin/env python3
"""
02_eda.py -- Descriptive Exploratory Data Analysis for ARP spoofing detection
CICIoT2023: MITM-ArpSpoofing + Benign_Final

This is the second stage of the research pipeline. It reads the cleaned
checkpoint produced by 01_data_cleaning.py and performs purely descriptive
analysis: distribution shape, class separation, class balance, per-class
distributions, and unsupervised structure (PCA / K-Means / t-SNE). Every
finding here is for understanding the data and for the methodology write-up
-- nothing in this file produces a feature drop/keep recommendation, so it
is safe to run on the complete dataset before any train/validation/test
split exists.

ROADMAP (this file)
---------------------------------------------------------------------------
  SECTION 5         Distribution and Shape Analysis
  SECTION 6         Class Balance and Per-Class Distributions
  SECTION 7         Unsupervised Exploration (Clustering)

This file is purely descriptive: distribution shape, class separation,
class balance, per-class distributions, and unsupervised structure. It
does not compute any label-dependent statistic that recommends dropping
or engineering a specific feature -- any such recommendation must be fit
on the training partition only, so that work happens in a separate stage
that runs after the train/validation/test split.

INPUTS
  data/final_dataset.csv   Output of 01_data_cleaning.py
                                        (cleaned, validated, no engineered
                                        features, 1,405,749 rows).

OUTPUTS
  plots/*.png   Every figure produced by Sections 5, 6, and 7.
  No CSV is written by this file -- it does not modify the dataset.

Every sampling step below uses random_state=42 for reproducibility.
"""
import sys, io as _sysio
sys.stdout = _sysio.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = _sysio.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, ks_2samp
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves to file, no display window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format

# ─────────────────────────────────────────────
# Paths -- every input/output location this script touches
# ─────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(REPO_ROOT, 'data', 'final_dataset.csv')
# Checkpoint produced by 01_data_cleaning.py.

PLOT_DIR = os.path.join(REPO_ROOT, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load the cleaned checkpoint
# ─────────────────────────────────────────────
print("Loading cleaned dataset...")
df = pd.read_csv(IN_PATH)
feature_cols = [c for c in df.columns if c != 'label']
print(f"  Loaded {IN_PATH}")
print(f"  Shape: {df.shape}  ({len(feature_cols)} features + label)")

# ─────────────────────────────────────────────
# SECTION 5: DISTRIBUTION AND SHAPE ANALYSIS
# ─────────────────────────────────────────────
#
# PURPOSE: Understand the shape of each feature's distribution to:
#   1. Decide whether transformations (log1p, Yeo-Johnson) are needed
#   2. Detect multimodality — multiple peaks often indicate hidden sub-populations
#      or data quality issues
#   3. Quantify how well each feature separates the two classes
#
# TRANSFORMATION NOTE — CRITICAL FOR THIS RESEARCH:
# Random Forest and other tree-based models are INVARIANT to monotonic
# transformations (log, sqrt, Box-Cox). The splits they learn are based on
# rank order, not absolute values, so a skewed distribution does not harm them.
# Transformations are only needed if you evaluate:
#   - Logistic Regression, SVM, KNN (distance/linearity sensitive)
#   - Neural networks (gradient stability benefits from normalisation)
# For this project's primary model (Random Forest), the skewness findings below
# are DOCUMENTED for future reference only — no transformation is applied here.
# If a linear baseline is evaluated, apply log1p to all right-skewed features
# with skewness > 1 and non-negative values, and Yeo-Johnson to left-skewed ones.
#
# SKEWNESS SEVERITY THRESHOLDS USED:
#   |skew| <= 0.5  → OK (approximately symmetric)
#   0.5 < |skew| <= 3 → MODERATE
#   3 < |skew| <= 10  → HIGH
#   |skew| > 10       → EXTREME
#
# MODALITY INTERPRETATION:
# Peak detection uses Gaussian KDE on a 50,000-row sample (clipped at 99th pct
# for KDE stability), with a prominence threshold of 5% of the max density.
#
# Multimodal features fall into THREE distinct categories with different causes:
#
# CATEGORY 1 — DISCRETIZATION ARTIFACT (no hidden sub-populations):
#   psh_flag_number, ack_flag_number, ack_count, TCP, UDP, DNS
#   These are ratio features computed as k/N where N = Number (packets per flow,
#   typically 1–10). The only possible values are 0/N, 1/N, 2/N, ..., N/N, so
#   values land on a grid at 0.1 intervals and KDE produces a peak at each grid
#   point. This is a pure measurement artifact — the peaks carry no semantic
#   meaning about hidden sub-populations or data quality issues.
#   → No feature engineering warranted. Tree models handle the discrete grid
#     natively through their split-finding process.
#
# CATEGORY 2 — GENUINE BIMODAL: protocol presence vs absence:
#   ARP, IPv, LLC
#   Two real populations exist: flows that carry the protocol (value → 1.0) and
#   flows that don't (value = 0). ARP spoofing flows form a structurally distinct
#   population from TCP/UDP flows. This is real signal, not an artifact.
#   DNS also fits this pattern but appears borderline due to sparse non-zero values.
#   → The existing ratio feature already captures this distinction. A binary
#     indicator (is_arp_flow, has_dns) would add marginal value for linear models
#     only — tree models discover the threshold automatically. Decision: do NOT
#     engineer additional binary features; the ratio already encodes the information.
#
# CATEGORY 3 — GENUINE PACKET SIZE CLUSTERS:
#   Max (4 peaks), Std (2 peaks)
#   Max (maximum packet size in the flow) has 4 peaks because fundamentally
#   different protocol types produce packets of distinct sizes:
#     ~60 bytes   → ARP, pure ACK, SYN (pure control packets)
#     ~100-300 B  → DNS, ICMP, small control payloads
#     ~500-1000 B → small data transfers, handshake payloads
#     ~1500 bytes → TCP full-MTU data segments
#   Std has two peaks: Std=0 (constant-size flows, e.g. ARP which always sends
#   exactly the same frame) vs Std>0 (flows with variable packet sizes).
#   → These are real sub-populations with physical meaning. A tree model will
#     discover the split points automatically. A packet_size_category bin could
#     improve interpretability in the paper (naming the clusters explicitly), but
#     adds no accuracy benefit and is not needed for the primary Random Forest model.
#     Decision: do NOT add engineered features from these peaks.
#
# SPIKE DISTRIBUTIONS (KDE failure):
#   ece_flag_number, cwr_flag_number, IRC, IGMP
#   >99% of values are exactly 0 with a tiny non-zero tail. KDE bandwidth collapses
#   on near-constant data. These are classified as SPIKE (zero-inflated), not
#   multimodal. The near-zero variance means these features carry almost no
#   discriminative information — they are candidates for removal in the feature
#   selection phase (evaluated on the training split only, in a later stage).
#
# CONCLUSION ON FEATURE ENGINEERING FROM PEAKS:
#   No new features will be engineered based on peak structure. All multimodal
#   patterns are either mathematical artifacts, already-captured protocol ratios,
#   or cluster boundaries that tree models discover natively. The findings are
#   documented here for reference and for the paper's discussion section.
#
# CLASS SEPARATION METRICS:
#   KS statistic (Kolmogorov-Smirnov): max distance between the two CDFs.
#     Range [0,1]. Higher = better separation. Threshold: >0.3 STRONG, >0.1 MODERATE.
#   Cohen's d: mean difference normalised by pooled standard deviation.
#     Commonly: d>0.8 large, d>0.5 medium, d>0.2 small effect.
#   Both metrics are computed on the full dataset (no sampling needed).

print("\n" + "=" * 70)
print("SECTION 5: DISTRIBUTION AND SHAPE ANALYSIS")
print("=" * 70)

numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
skip_constant = ['Telnet', 'SMTP']
benign_df = df[df['label'] == 0]
attack_df = df[df['label'] == 1]

# ── 5.1  Skewness & Kurtosis ────────────────────
print("\n--- 5.1  Skewness & Kurtosis ---")
sk_rows = []
for col in numeric_cols:
    s = df[col].skew()
    k = df[col].kurtosis()
    mn = df[col].min()
    if abs(s) <= 0.5:
        severity, tx = 'OK',      'None needed (tree models unaffected regardless)'
    elif abs(s) <= 3:
        severity = 'MODERATE'
        tx = 'Yeo-Johnson (left skew)' if s < 0 else 'log1p or Yeo-Johnson'
    elif abs(s) <= 10:
        severity = 'HIGH'
        tx = 'Yeo-Johnson (left skew)' if s < 0 else ('log1p' if mn >= 0 else 'Yeo-Johnson')
    else:
        severity = 'EXTREME'
        tx = 'Yeo-Johnson (left skew)' if s < 0 else ('log1p' if mn >= 0 else 'Yeo-Johnson')
    sk_rows.append({'Feature': col, 'Skewness': round(s, 3), 'Kurtosis (excess)': round(k, 3),
                    'Severity': severity, 'Transform (if linear model)': tx})

sk_df = pd.DataFrame(sk_rows).sort_values('Skewness', ascending=False)
print(sk_df.to_string(index=False))

# ── 5.2  Modality ───────────────────────────────
print("\n--- 5.2  Modality (KDE peak detection on 50k sample, clipped at 99th pct) ---")
sample = df.sample(50000, random_state=42)
modal_rows = []
for col in numeric_cols:
    if col in skip_constant:
        modal_rows.append({'Feature': col, 'Peaks': 0, 'Modality': 'CONSTANT',
                           'Interpretation': 'Zero variance — carries no information'}); continue
    vals = sample[col].dropna()
    if col == 'Rate':
        vals = vals[vals < 2**32 - 1]
    if vals.std() == 0:
        modal_rows.append({'Feature': col, 'Peaks': 1, 'Modality': 'unimodal',
                           'Interpretation': 'Effectively constant in sample'}); continue
    p99 = vals.quantile(0.99)
    vals_clip = vals.clip(upper=p99)
    try:
        kde = gaussian_kde(vals_clip, bw_method='scott')
        x   = np.linspace(vals_clip.min(), vals_clip.max(), 500)
        y   = kde(x)
        peaks, _ = find_peaks(y, prominence=y.max() * 0.05)
        n_peaks  = max(len(peaks), 1)
        if n_peaks == 1:
            label, interp = 'unimodal', 'Single population — straightforward'
        else:
            label = f'MULTIMODAL ({n_peaks} peaks)'
            pct_zero = (df[col] == 0).mean() * 100
            if pct_zero > 80:
                interp = f'Discretization artifact or zero-inflated ({pct_zero:.0f}% zeros)'
            elif col in ['ARP', 'IPv', 'LLC']:
                interp = 'Genuine: flows with vs without the protocol'
            elif col == 'Std':
                interp = 'Genuine: constant-size flows (peak@0) vs variable-size'
            elif col == 'Max':
                interp = 'Genuine: distinct packet size clusters (control/handshake/data)'
            else:
                interp = 'Likely discretization artifact (ratio feature k/N, N=1-10)'
    except Exception:
        n_peaks, label = -1, 'SPIKE (zero-inflated)'
        interp = 'KDE failed — >99% zeros; near-constant spike distribution'
    modal_rows.append({'Feature': col, 'Peaks': n_peaks, 'Modality': label,
                       'Interpretation': interp})

modal_df = pd.DataFrame(modal_rows)
print(modal_df.to_string(index=False))

# ── 5.3  Class Density Separation ───────────────
print("\n--- 5.3  Class Density Separation (KS statistic + Cohen's d) ---")
print("    KS > 0.3 = STRONG | 0.1-0.3 = MODERATE | < 0.1 = WEAK")
print("    Cohen d > 0.8 = large | 0.5-0.8 = medium | 0.2-0.5 = small | < 0.2 = negligible\n")
sep_rows = []
for col in numeric_cols:
    b = benign_df[col].dropna()
    a = attack_df[col].dropna()
    ks_stat, _ = ks_2samp(b, a)
    pooled_std  = np.sqrt((b.std()**2 + a.std()**2) / 2)
    cohen_d     = abs(b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0
    ks_label    = 'STRONG' if ks_stat > 0.3 else 'MODERATE' if ks_stat > 0.1 else 'WEAK'
    d_label     = 'large' if cohen_d > 0.8 else 'medium' if cohen_d > 0.5 else 'small' if cohen_d > 0.2 else 'negligible'
    sep_rows.append({'Feature': col, 'KS stat': round(ks_stat, 4), 'KS label': ks_label,
                     "Cohen's d": round(cohen_d, 4), 'd label': d_label})

sep_df = pd.DataFrame(sep_rows).sort_values('KS stat', ascending=False)
print(sep_df.to_string(index=False))

# Summary
strong   = sep_df[sep_df['KS label'] == 'STRONG']['Feature'].tolist()
moderate = sep_df[sep_df['KS label'] == 'MODERATE']['Feature'].tolist()
weak     = sep_df[sep_df['KS label'] == 'WEAK']['Feature'].tolist()
print(f"\n  STRONG separators  ({len(strong)})  : {strong}")
print(f"  MODERATE separators ({len(moderate)}): {moderate}")
print(f"  WEAK separators    ({len(weak)})  : {weak}")

# ─────────────────────────────────────────────
# SECTION 6: CLASS BALANCE AND PER-CLASS DISTRIBUTIONS
# ─────────────────────────────────────────────
#
# PURPOSE: Go beyond aggregate statistics and understand how the two classes
# differ structurally. This is the most directly actionable descriptive EDA
# section — it answers the question "what does the model actually have to learn?"
#
# SUBSECTIONS:
#
#   1. Class balance analysis
#      Imbalance matters because most classifiers optimise accuracy, which means
#      they can achieve high accuracy by always predicting the majority class.
#      The imbalance ratio (majority / minority) determines how aggressively to
#      compensate:
#        ratio < 1.5   → BALANCED      — no special treatment needed
#        ratio 1.5–3   → MILD          — class_weight='balanced' usually sufficient
#        ratio 3–10    → MODERATE      — class_weight='balanced' or mild oversampling
#        ratio > 10    → HIGH          — SMOTE or undersampling may help; report
#                                        precision/recall per class, not just accuracy
#      Minority class patterns: which feature values are characteristic of the
#      minority class — useful for understanding what the model must detect.
#
#   2. Per-class distributions
#      Mean, median, and std for each feature broken down by class. Median is
#      preferred over mean for skewed features (less influenced by the extreme
#      values that dominate many features here). The relative shift between classes
#      on each feature is what the classifier exploits: large shifts = easy splits,
#      small shifts = the model must combine multiple features.
#      Per-class boxplots saved for the 8 strongest KS separators — these give
#      visual intuition of where the decision boundary sits for each feature.
#      Rate is clipped at its 99th percentile for plotting (MAX_INT values would
#      collapse the axis scale, hiding all meaningful variation).
#
# NOTE: feature-importance-driven drop recommendations (e.g. from an
# exploratory model's importance ranking) are intentionally not computed
# here. Any statistic that recommends dropping or engineering a feature
# must be fit on the training partition only, so that work belongs in a
# separate stage that runs after the train/validation/test split.

print("\n" + "=" * 70)
print("SECTION 6: CLASS BALANCE AND PER-CLASS DISTRIBUTIONS")
print("=" * 70)

# ── 6.1  Class Balance ──────────────────────────
n_total  = len(df)
n_ben    = int((df['label'] == 0).sum())
n_atk    = int((df['label'] == 1).sum())
pct_ben  = n_ben / n_total * 100
pct_atk  = n_atk / n_total * 100
majority = max(n_ben, n_atk)
minority = min(n_ben, n_atk)
imb_ratio = majority / minority
minority_label = 0 if n_ben < n_atk else 1

if imb_ratio < 1.5:
    balance_status = 'BALANCED — no special treatment needed'
elif imb_ratio < 3:
    balance_status = 'MILDLY IMBALANCED — class_weight="balanced" usually sufficient'
elif imb_ratio < 10:
    balance_status = 'MODERATELY IMBALANCED — class_weight="balanced" or mild oversampling'
else:
    balance_status = 'HIGHLY IMBALANCED — consider SMOTE; report per-class precision/recall'

print("\n--- 6.1  Class Balance ---")
print(f"  Benign (0) : {n_ben:>10,}  ({pct_ben:.2f}%)")
print(f"  Attack (1) : {n_atk:>10,}  ({pct_atk:.2f}%)")
print(f"  Total      : {n_total:>10,}")
print(f"  Imbalance ratio (majority / minority) : {imb_ratio:.2f} : 1")
print(f"  Status     : {balance_status}")

# CLASS BALANCE FINDINGS:
#   Benign (0): 1,098,191 rows  (78.12%)
#   Attack (1):   307,558 rows  (21.88%)
#   Imbalance ratio: 3.57 : 1  → MODERATELY IMBALANCED
#
#   Implication for modelling:
#     A naive classifier that always predicts Benign would achieve 78% accuracy —
#     this is why accuracy alone is a misleading metric here. Always report
#     precision, recall, and F1 per class, plus ROC-AUC.
#     Chosen mitigation: class_weight='balanced' in all sklearn estimators.
#     This reweights each sample inversely proportional to class frequency, so
#     the 307k attack flows are treated as if they were as numerous as benign.
#     SMOTE (synthetic oversampling) can be an alternative approach but must be applied ONLY
#     on the training split after the split is made — never on the full dataset.

# Minority class patterns — which features have the most extreme values in the
# minority class (median and std relative to the full-dataset median)
print(f"\n  Minority class (label={minority_label}) feature profile:")
minority_df = df[df['label'] == minority_label]
pattern_rows = []
for col in numeric_cols:
    full_med  = df[col].median()
    minor_med = minority_df[col].median()
    shift     = minor_med - full_med
    pct_shift = (shift / full_med * 100) if full_med != 0 else float('nan')
    pattern_rows.append({'Feature': col,
                         'Full median': round(full_med, 4),
                         'Minority median': round(minor_med, 4),
                         'Shift': round(shift, 4),
                         '% shift': round(pct_shift, 1)})
pattern_df = pd.DataFrame(pattern_rows)
pattern_df = pattern_df.reindex(pattern_df['% shift'].abs().sort_values(ascending=False).index)
print(pattern_df.to_string(index=False))

# ── 6.2  Per-class Distributions ────────────────
print("\n--- 6.2  Per-class Distributions (mean / median / std) ---")
stat_rows = []
for col in numeric_cols:
    b = benign_df[col]
    a = attack_df[col]
    stat_rows.append({
        'Feature'   : col,
        'Ben mean'  : round(b.mean(),   4),
        'Atk mean'  : round(a.mean(),   4),
        'Ben median': round(b.median(), 4),
        'Atk median': round(a.median(), 4),
        'Ben std'   : round(b.std(),    4),
        'Atk std'   : round(a.std(),    4),
    })
stat_df = pd.DataFrame(stat_rows)
print(stat_df.to_string(index=False))

# Per-class boxplots for the 8 strongest KS separators
TOP_BOX = ['IAT', 'Max', 'Rate', 'Min', 'Time_To_Live', 'Std', 'Header_Length', 'TCP']
N_BOX = 5000
box_b = benign_df.sample(n=min(N_BOX, len(benign_df)), random_state=42)
box_a = attack_df.sample(n=min(N_BOX, len(attack_df)), random_state=42)
box_samp = pd.concat([box_b, box_a]).reset_index(drop=True)
box_samp['Class'] = box_samp['label'].map({0: 'Benign', 1: 'Attack'})
box_samp['Rate']  = box_samp['Rate'].clip(upper=box_samp['Rate'].quantile(0.99))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
palette_box = {'Benign': '#2196F3', 'Attack': '#F44336'}
for ax, feat in zip(axes.flat, TOP_BOX):
    sns.boxplot(data=box_samp, x='Class', y=feat, ax=ax,
                palette=palette_box, fliersize=2, linewidth=1)
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel('')
plt.suptitle('Per-class Boxplots — Top 8 Features by KS Separation (stratified 10k sample)',
             fontsize=13)
plt.tight_layout()
box_path = os.path.join(PLOT_DIR, 'per_class_boxplots.png')
plt.savefig(box_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Boxplots saved → {box_path}")

# PER-CLASS DISTRIBUTION FINDINGS:
#
# ATTACK MINORITY CLASS PROFILE (features with the largest median shift from the
# full-dataset median, sorted by magnitude):
#
#   Rate       +279%  Attack flows arrive at nearly 4× the median rate. The most
#                     extreme shift in the dataset — attack tooling sends spoofed
#                     ARP replies in rapid bursts, driving up packet rate sharply.
#
#   UDP        -100%  Attack flows have median UDP=0 vs full-dataset median of 0.1.
#                     ARP spoofing operates over Ethernet (Layer 2) or TCP-heavy
#                     follow-on flows — UDP is largely absent in attack traffic.
#
#   IAT         -74%  Median inter-arrival time drops from 0.005 s (full dataset)
#                     to 0.001 s in attack flows. Packets arrive ~5× more tightly
#                     packed — consistent with scripted loop-based attack tooling.
#
#   Tot sum /
#   AVG         +46%  Attack flows carry more bytes per packet on average. Likely
#                     reflects the mix of ARP replies (60 bytes each, but high
#                     volume) combined with large TCP data flows the attacker
#                     intercepts after cache poisoning succeeds.
#
#   Time_To_Live -17% Attack flows have a lower median TTL (80 vs 97). TTL reflects
#                     the originating OS and network hop count. A consistent
#                     downward shift suggests attack traffic originates from a
#                     different OS or traverses more hops than typical benign IoT
#                     devices. Useful discriminator in combination with other features.
#
#   Max         -12%  Attack max packet size is slightly smaller (1292 vs 1462
#                     bytes). The attacker's traffic does not reach full MTU as
#                     consistently as benign data transfers.
#
# Features with zero shift (median identical in both classes):
#   Header_Length, Protocol Type, psh_flag_number, LLC, ack_count, TCP,
#   ack_flag_number, IPv, Min, Number — and all zero-inflated spike features.
#   These features do not shift at the median level but may still contribute via
#   distributional shape differences (std, tails) — visible in boxplots.

# ─────────────────────────────────────────────
# SECTION 7: UNSUPERVISED EXPLORATION (CLUSTERING)
# ─────────────────────────────────────────────
#
# PURPOSE: Explore the feature space without using labels to ask:
#   1. Is the dataset naturally separable into two groups that align with
#      the attack/benign boundary?
#   2. Are there hidden sub-populations within each class that a single
#      model may struggle to capture uniformly?
#   3. What are the dominant axes of variation in the data?
#
# PREPROCESSING:
#   Stratified 20k sample (proportional to class balance: ~78% benign, ~22% attack).
#   Each feature is clipped at its 99th percentile before scaling — this handles
#   MAX_INT Rate values and other extreme outliers without affecting the shape of
#   the majority distribution. StandardScaler then centres and scales each feature.
#   Clipping + scaling is applied to the sample only; the saved dataset is unchanged.
#
# METHOD RATIONALE:
#   PCA: linear dimensionality reduction. Reveals which feature combinations
#     explain the most variance and how many independent dimensions the data has.
#     The loading vectors identify which original features drive each component.
#
#   K-Means: centroid-based clustering. Assumes spherical, equally-sized clusters.
#     Applied on PCA-reduced space (95% variance components) for speed and to
#     remove noise dimensions. Optimal k selected by silhouette score (geometric
#     compactness) cross-validated against the elbow in inertia.
#     Cluster alignment with true labels evaluated via:
#       ARI (Adjusted Rand Index): 0=random, 1=perfect. Corrects for chance.
#       NMI (Normalised Mutual Information): 0=independent, 1=perfect.
#
#   t-SNE: nonlinear 2D embedding. Run on 5k-row subset with PCA-39 input
#     (standard practice: PCA pre-reduction speeds t-SNE and denoises).
#     perplexity=30: balances local vs global structure. max_iter=1000.
#     Used for qualitative visual inspection only — t-SNE distances are not
#     interpretable as actual feature-space distances.
#
#   Cluster profiles: per-cluster median of the 10 most discriminative features
#     (top RF MDI features), Z-scored across clusters so relative shifts are
#     visible regardless of scale. Dominant class label (majority vote) assigned
#     to each cluster for interpretability.
#
# ALL PLOTS SAVED TO: plots/
#   pca_scatter.png       — scree plot + PC1/PC2 scatter coloured by label and PC3
#   kmeans_elbow.png      — inertia elbow + silhouette score vs k
#   kmeans_pca.png        — K-Means clusters vs true labels in PCA space
#   tsne_scatter.png      — t-SNE coloured by true label and by cluster
#   cluster_profiles.png  — heatmap of Z-scored cluster medians

print("\n" + "=" * 70)
print("SECTION 7: UNSUPERVISED EXPLORATION (CLUSTERING)")
print("=" * 70)

# ── Preprocessing ─────────────────────────────
N_CLUST = 20_000
ratio   = len(benign_df) / len(df)
n_b_c   = int(N_CLUST * ratio); n_a_c = N_CLUST - n_b_c
clust_samp = pd.concat([
    benign_df.sample(n=min(n_b_c, len(benign_df)), random_state=42),
    attack_df.sample(n=min(n_a_c, len(attack_df)), random_state=42)
]).reset_index(drop=True)
y_true = clust_samp['label'].values

X_raw = clust_samp[numeric_cols].copy()
for col in numeric_cols:
    X_raw[col] = X_raw[col].clip(upper=X_raw[col].quantile(0.99))

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"\n  Sample: {N_CLUST:,} rows (stratified) | clip 99th pct | StandardScaler")

# ── 7.1  PCA ────────────────────────────────────
print("\n--- 7.1  PCA ---")
pca   = PCA(random_state=42)
X_pca = pca.fit_transform(X_scaled)
ev    = pca.explained_variance_ratio_
cumev = np.cumsum(ev)
n_90  = int(np.searchsorted(cumev, 0.90)) + 1
n_95  = int(np.searchsorted(cumev, 0.95)) + 1
n_99  = int(np.searchsorted(cumev, 0.99)) + 1

print(f"\n  PCs for 90% variance : {n_90}  |  95%: {n_95}  |  99%: {n_99}  (of {len(numeric_cols)} features)")
print(f"\n  Top 10 component explained variance:")
print(f"  {'PC':<6} {'Var%':>7} {'Cum%':>7}")
for i in range(min(10, len(ev))):
    print(f"  PC{i+1:<4} {ev[i]*100:>6.2f}%  {cumev[i]*100:>6.2f}%")

loadings = pd.DataFrame(pca.components_[:4].T, index=numeric_cols,
                        columns=[f'PC{i+1}' for i in range(4)])
print(f"\n  Top 5 loadings on PC1: {loadings['PC1'].abs().nlargest(5).index.tolist()}")
print(f"  Top 5 loadings on PC2: {loadings['PC2'].abs().nlargest(5).index.tolist()}")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
ax = axes[0]
ax.bar(range(1, 21), ev[:20] * 100, color='#42A5F5', alpha=0.8)
ax.plot(range(1, 21), cumev[:20] * 100, 'o-', color='#E53935', linewidth=1.5, markersize=4)
ax.axhline(95, linestyle='--', color='gray', alpha=0.6, label='95% threshold')
ax.set_xlabel('PC'); ax.set_ylabel('Explained Variance (%)'); ax.set_title('Scree Plot'); ax.legend()

palette_lbl = {0: '#2196F3', 1: '#F44336'}
ax = axes[1]
for lbl, name in [(0, 'Benign'), (1, 'Attack')]:
    mask = y_true == lbl
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=palette_lbl[lbl], label=name,
               alpha=0.3, s=4, rasterized=True)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title('PCA — True Label'); ax.legend(markerscale=3)

ax = axes[2]
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=X_pca[:, 2],
                cmap='coolwarm', alpha=0.3, s=4, rasterized=True)
fig.colorbar(sc, ax=ax, label='PC3')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title('PCA — PC3 as colour (hidden structure)')
plt.suptitle(f'PCA — {N_CLUST:,} stratified sample (clipped + standardised)', fontsize=13)
plt.tight_layout()
pca_path = os.path.join(PLOT_DIR, 'pca_scatter.png')
plt.savefig(pca_path, dpi=150, bbox_inches='tight'); plt.close()
print(f"\n  Saved → {pca_path}")

# ── 7.2  K-Means ────────────────────────────────
print("\n--- 7.2  K-Means Clustering (elbow + silhouette, k=2..8) ---")
K_RANGE    = range(2, 9)
inertias   = []; sil_scores = []
X_pca_95   = X_pca[:, :n_95]

for k in K_RANGE:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    lbl = km.fit_predict(X_pca_95)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_pca_95, lbl, sample_size=5000, random_state=42)
    sil_scores.append(sil)
    print(f"  k={k}  inertia={km.inertia_:>12.0f}  silhouette={sil:.4f}")

best_k = list(K_RANGE)[int(np.argmax(sil_scores))]
print(f"\n  Best k by silhouette: {best_k}")

km_best       = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
cluster_labels = km_best.fit_predict(X_pca_95)
ari = adjusted_rand_score(y_true, cluster_labels)
nmi = normalized_mutual_info_score(y_true, cluster_labels)
print(f"  ARI : {ari:.4f}  (1.0=perfect, 0=random)")
print(f"  NMI : {nmi:.4f}  (1.0=perfect, 0=random)")

comp_df = pd.DataFrame({'cluster': cluster_labels, 'label': y_true})
comp    = comp_df.groupby('cluster')['label'].value_counts(normalize=True).unstack(fill_value=0)
comp.columns = ['% Benign', '% Attack']
comp['n_flows'] = comp_df.groupby('cluster').size()
comp[['% Benign','% Attack']] = (comp[['% Benign','% Attack']] * 100).round(1)
print(f"\n  Cluster composition (k={best_k}):")
print(comp.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K_RANGE), inertias, 'o-', color='#42A5F5')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].set_title('K-Means Elbow')
axes[1].plot(list(K_RANGE), sil_scores, 'o-', color='#66BB6A')
axes[1].axvline(best_k, linestyle='--', color='gray', alpha=0.6, label=f'best k={best_k}')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette'); axes[1].set_title('Silhouette vs k')
axes[1].legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'kmeans_elbow.png'), dpi=150, bbox_inches='tight'); plt.close()

# plt.colormaps[...].resampled(n) is the modern replacement for the removed
# plt.cm.get_cmap(name, lut) API (deprecated since Matplotlib 3.7).
cmap_clust = plt.colormaps['tab10'].resampled(best_k)
fig, axes  = plt.subplots(1, 2, figsize=(16, 6))
for c in range(best_k):
    mask = cluster_labels == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], color=cmap_clust(c),
                    label=f'C{c} (n={mask.sum()})', alpha=0.3, s=4, rasterized=True)
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
axes[0].set_title(f'K-Means (k={best_k})'); axes[0].legend(markerscale=3)
for lbl, name in [(0, 'Benign'), (1, 'Attack')]:
    mask = y_true == lbl
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=palette_lbl[lbl], label=name,
                    alpha=0.3, s=4, rasterized=True)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('True Labels'); axes[1].legend(markerscale=3)
plt.suptitle('K-Means Clusters vs True Labels in PCA Space', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'kmeans_pca.png'), dpi=150, bbox_inches='tight'); plt.close()
print(f"\n  Plots saved → {PLOT_DIR}")

# ── 7.3  t-SNE ──────────────────────────────────
print("\n--- 7.3  t-SNE (5k subset, PCA input, perplexity=30) ---")
N_TSNE   = 5000
rng      = np.random.RandomState(42)
tsne_idx = rng.choice(N_CLUST, N_TSNE, replace=False)
X_tsne_in = X_pca[tsne_idx, :]
y_tsne    = y_true[tsne_idx]
c_tsne    = cluster_labels[tsne_idx]

print(f"  Running t-SNE on {N_TSNE:,} rows × {X_tsne_in.shape[1]} PCs...")
tsne   = TSNE(n_components=2, random_state=42, perplexity=30,
              max_iter=1000, learning_rate='auto', init='pca')
X_tsne = tsne.fit_transform(X_tsne_in)
print(f"  Done. KL divergence: {tsne.kl_divergence_:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for lbl, name in [(0, 'Benign'), (1, 'Attack')]:
    mask = y_tsne == lbl
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=palette_lbl[lbl], label=name,
                    alpha=0.4, s=5, rasterized=True)
axes[0].set_title('t-SNE — True Labels'); axes[0].legend(markerscale=3)
axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
for c in range(best_k):
    mask = c_tsne == c
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=cmap_clust(c),
                    label=f'C{c}', alpha=0.4, s=5, rasterized=True)
axes[1].set_title(f't-SNE — K-Means Clusters (k={best_k})')
axes[1].legend(markerscale=3); axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')
plt.suptitle(f't-SNE (perplexity=30, {N_TSNE:,} rows)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'tsne_scatter.png'), dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved → {PLOT_DIR}/tsne_scatter.png")

# ── 7.4  Cluster profiles ───────────────────────
print(f"\n--- 7.4  Cluster Profiles (k={best_k}) ---")
PROFILE_FEATS = ['Max', 'Min', 'IAT', 'Time_To_Live', 'Rate',
                 'Header_Length', 'Std', 'TCP', 'HTTPS', 'psh_flag_number']
prof_df = clust_samp[PROFILE_FEATS].copy()
prof_df['Rate']    = prof_df['Rate'].clip(upper=prof_df['Rate'].quantile(0.99))
prof_df['cluster'] = cluster_labels
prof_df['label']   = y_true

medians = prof_df.groupby('cluster')[PROFILE_FEATS].median().round(3)
medians['dominant_class'] = comp_df.groupby('cluster')['label'].apply(
    lambda x: 'Benign' if (x == 0).mean() > 0.5 else 'Attack')
print(f"\n  Median per cluster:")
print(medians.to_string())

med_vals = prof_df.groupby('cluster')[PROFILE_FEATS].median()
med_z    = (med_vals - med_vals.mean()) / (med_vals.std() + 1e-9)
fig, ax  = plt.subplots(figsize=(14, max(4, best_k * 1.5)))
sns.heatmap(med_z, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Z-score of cluster median'})
ax.set_title(f'Cluster Profiles — Z-scored Median (k={best_k})')
ax.set_xlabel('Feature'); ax.set_ylabel('Cluster'); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'cluster_profiles.png'), dpi=150, bbox_inches='tight'); plt.close()
print(f"\n  Profile heatmap saved → {PLOT_DIR}/cluster_profiles.png")

# ═══════════════════════════════════════════════════════════════════════
# SEGMENTATION & CLUSTERING — FINDINGS
# ═══════════════════════════════════════════════════════════════════════
#
# ── 7.1  PCA Findings ───────────────────────────────────────────────────
# 39 features reduce to just 16 PCs for 95% of variance (19 for 99%).
# This confirms the extensive collinearity documented separately in the
# feature-engineering stage's correlation analysis: many features are
# near-linear combinations of others. The dataset is not full-rank — the
# effective dimensionality is roughly half the nominal feature count.
#
# PC1 (24.9% variance) — Protocol / connection type axis:
#   Top loadings: ack_flag_number, ack_count, TCP, Header_Length, UDP
#   This component separates TCP-heavy flows (high ACK rate, large header) from
#   non-TCP flows (UDP, ARP, ICMP). It is the dominant axis of variation in the
#   dataset, reflecting the diversity of IoT protocols. It is NOT the attack axis.
#
# PC2 (13.0% variance) — Packet size axis:
#   Top loadings: Max, AVG, Tot size, Tot sum, Std
#   Separates large-packet flows (data transfers, video) from small-packet flows
#   (ARP, DNS, control packets). Again, not directly the attack axis.
#
# IMPLICATION: The two largest variance axes describe TRAFFIC TYPE, not ATTACK
# vs BENIGN. The attack signal is not the dominant source of variation in the
# feature space — it is a secondary signal embedded within each traffic type.
# This is why unsupervised methods struggle (see K-Means below).
#
# ── 7.2  K-Means Findings ──────────────────────────────────────────────
# Best k=2 by silhouette score (0.26 — low, indicating overlapping clusters).
# ARI=0.025, NMI=0.004 — NEAR-ZERO alignment with true labels.
#
# Both clusters have nearly identical class composition:
#   Cluster 0: 79.9% Benign, 20.1% Attack  (n=13,885)
#   Cluster 1: 74.1% Benign, 25.9% Attack  (n=6,115)
# Neither cluster corresponds to the attack class. K-Means with k=2 does NOT
# recover the benign/attack partition.
#
# What the clusters DO represent (from profile medians):
#   Cluster 0: Max=1514B, Rate=428 pkt/s, TCP=1.0, HTTPS=0.9, IAT=0.003s
#     → Large TCP/HTTPS flows (established sessions, data transfers).
#       These are high-rate, large-packet, pure-TCP flows.
#   Cluster 1: Max=284B,  Rate=98  pkt/s, TCP=0.5, HTTPS=0.3, IAT=0.011s
#     → Mixed smaller/non-TCP flows (ARP, UDP, ICMP, short control packets).
#       Lower rate, smaller packets, mixed protocols.
#
# K-Means found a TRAFFIC TYPE partition, not an attack partition. Attack flows
# are distributed across both traffic types because the attacker generates both
# types of traffic (ARP poisoning + MITM TCP relay).
#
# KEY INSIGHT: The attack is NOT an anomaly in the global feature space.
# It is embedded within the normal traffic distribution, distinguished only by
# SUBTLE COMBINATIONS of features (Rate × IAT × TTL × Max simultaneously shifted)
# that a linear centroid-based method cannot detect. This is precisely why:
#   a) Supervised learning with labels is necessary — unsupervised cannot detect it
#   b) The 557 label-conflicting duplicate groups are irreducible (documented in
#      01_data_cleaning.py)
#   c) A Random Forest (which can learn nonlinear feature combinations) is more
#      appropriate than logistic regression for this problem
#
# ── 7.3  t-SNE Findings ────────────────────────────────────────────────
# KL divergence=0.734 (reasonable fit for 5k points).
# Visual inspection of tsne_scatter.png shows:
#   - True label plot: Benign and Attack points are HEAVILY MIXED — no clean
#     separation into distinct regions. Attack points appear throughout the
#     benign cloud, confirming the unsupervised inseparability finding above.
#   - Cluster plot: K-Means clusters follow a spatial split that does NOT
#     correspond to the attack/benign boundary — consistent with ARI≈0.
#   - Some local pockets of attack concentration are visible at the periphery
#     of the t-SNE embedding — these likely correspond to the extreme Rate/IAT
#     flows (the STRONG KS separators from Section 5), which are separable in
#     isolation but represent only a fraction of all attack flows.
#
# ── 7.4  Cluster Profile Findings ──────────────────────────────────────
# The Z-scored heatmap (cluster_profiles.png) shows:
#   Cluster 0 has high Z-scores on Max, Std, Rate, Tot sum — high-intensity,
#     large-packet flows. Contains the majority of high-Rate attack flows.
#   Cluster 1 has high Z-score on IAT, low on Rate and Max — slower, smaller
#     flows. Contains the ARP-level and control-packet flows.
# Neither cluster is attack-dominated. The dominant class in both is Benign.
#
# ── OVERALL CONCLUSION ───────────────────────────────────────────────
# Unsupervised methods confirm that ARP spoofing attack detection REQUIRES
# supervised learning. The attack is not a global anomaly — it is a subtle
# shift in a combination of features that only becomes distinguishable when
# the label is used to guide the separation. The findings here strengthen
# the case for the Random Forest approach and validate that label-aware
# feature analysis (carried out in the feature-engineering stage) is
# necessary, since simple clustering cannot surface the relevant signal
# on its own.

print("\n" + "=" * 70)
print("SECTION 5 / 6 / 7 COMPLETE")
print("=" * 70)
print(f"  Plots saved to: {PLOT_DIR}")
print(f"  No CSV written -- this stage is purely descriptive.")
