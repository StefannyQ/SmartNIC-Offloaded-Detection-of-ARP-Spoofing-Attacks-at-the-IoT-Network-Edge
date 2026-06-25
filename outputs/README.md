# Outputs

This file describes what each pipeline stage writes and where. Nothing
here is committed to the repo (see `.gitignore`) -- it's all regenerated
by running the scripts in order. Locations are relative to the repo root.

## `data/` (stage 1)

| File | Produced by | Contents |
|---|---|---|
| `final_dataset.csv` | `01_data_cleaning.py` | Cleaned, validated dataset, before any feature engineering (1,405,749 rows: `Benign_Final` + `MITM-ArpSpoofing`, label 0/1). Every later stage reads this. |

## `plots/` (stage 2)

PNG plots from `02_eda.py`'s purely descriptive analysis: distribution
shape, class balance, per-class distributions, and unsupervised structure
(PCA / K-Means / t-SNE).

## `modeling/` (stage 3)

| Path | Contents |
|---|---|
| `modeling/results/preprocessing_summary.json` | Per-scenario feature lists (S1-S4), train-only statistical-reduction diagnostics, Rate/IAT ablation decisions, engineered-feature AUROC screen |
| `modeling/results/S{1,2,3,4}_results.pkl` | Tuned hyperparameters + val/test metrics per model per scenario (model objects are *not* in these -- see `modeling/models/`) |
| `modeling/results/S4_threshold_extraction.json` | Integer scale factors derived from the fitted S4 tree, and verification that the retrained integer tree's thresholds are exact integers |
| `modeling/results/cross_scenario.pkl` | S1-S4 best-model test metrics, for the cross-scenario comparison |
| `modeling/results/lift_tables.pkl` | Per-scenario lift / cumulative-gains tables (top-N% review-budget capture rates) |
| `modeling/results/threshold_results.pkl` | F2-optimal decision threshold per scenario and the val/test sweep behind it |
| `modeling/models/S{1,2}_{Logistic_Regression,Decision_Tree,Random_Forest}.joblib` | Fitted sklearn models, S1 float / S2 integer-retrained |
| `modeling/models/S{3,4}_Decision_Tree.joblib` | Fitted depth-3 SmartNIC-constrained trees, integer-retrained. `S4_Decision_Tree.joblib` is the model `04_generate_p4.py` turns into a P4 program. |
| `modeling/plots/*.png` | ROC curves, confusion matrices, feature importance (MDI + permutation), threshold analysis, cumulative-gains curves -- per scenario and cross-scenario |

## `p4_validation/` (stage 4)

| File | Produced by | Contents |
|---|---|---|
| `arp_detector.p4` | `04_generate_p4.py` | P4_16 program encoding the fitted S4 tree as match-action logic |
| `flow_test_vectors.csv` | `04_generate_p4.py` | Full test set (210,863 rows) with the Python model's prediction per row -- the ground truth the P4 verdicts are compared against |
| `test_harness.py` | `04_generate_p4.py` | Scapy script: sends each row through BMv2, compares the verdict to `python_pred` |
| `table_entries.txt`, `README.md` | `04_generate_p4.py` | BMv2 startup notes / quick reference (the full guide is `04_p4_validation/vm_setup.md`) |
| `p4_validation_results.csv` | `test_harness.py`, copied back from the VM | Per-row P4 verdict vs. Python prediction, plus match/mismatch/timeout status |

**Actual result from the full-test-set run** (see `04_p4_validation/vm_setup.md`
for the full setup and explanation): 209,093/210,863 exact matches
(**99.16%**), **zero mismatches**. The remaining 1,770 rows timed out
(VM load artifact, not a classification disagreement -- no verdict was
issued for those rows).
