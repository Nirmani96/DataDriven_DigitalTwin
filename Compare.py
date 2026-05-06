# -*- coding: utf-8 -*-
"""
Compare.py
----------
Statistical Comparison: Digital Twin vs DES Baseline (Arrivals).

This script compares the arrival forecasting performance of the
database-driven Digital Twin (DT) against the DES baseline model
(arrivals_simulation_validation_12_hours.py) using per-replication
validation metrics.

STATISTICAL TESTING APPROACH:
    For each metric, the script first checks normality using the
    Shapiro-Wilk test. Based on the result:
      - Both normal  → Paired t-test (parametric)
      - Non-normal   → Wilcoxon signed-rank test (non-parametric)

    The paired approach is used because both models ran exactly
    100 replications on the same forecast period, so each replication
    pair (DT run i vs DES run i) is a natural matched pair.

    A p-value < 0.05 indicates a statistically significant difference
    in performance between the DT and DES models.

INPUTS:
    - DT output CSV:  data/output/<timestamp>/
                      dt_fixed_validation_per_replication_10min.csv
    - DES output CSV: Compare DT and DES/arrivals_<timestamp>.csv

    Both files must contain columns:
        MAE_arrivals, RMSE_arrivals, MAPE_arrivals

OUTPUTS:
    - Console output: Shapiro-Wilk p-values, test used, p-value, conclusion
    - Console output: Median ± IQR summary for each metric
    - Scatter plots:  DT vs DES per-replication values (identity line shown)
    - Violin plots:   Distribution comparison for DT vs DES

AUTHORS:
    Nirmani Amarasinghe  (ORCID: 0009-0001-9719-6366)
    Laura Boyle          (ORCID: 0000-0001-9651-1363)
    Adele H. Marshall    (ORCID: 0000-0001-5306-2756)

    Mathematical Science Research Centre, Queen's University Belfast
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_rel, wilcoxon

# ====================================================================
# LOAD DATA
# ====================================================================
# Load per-replication metrics from both models.
# Update these paths to match the forecast timestamp you want to compare.
# ====================================================================

base_dir = os.path.dirname(__file__)  # script location
file_path = os.path.join(
    base_dir,
    "data",
    "output",
    "2007-09-30_12-00-00",
    "dt_fixed_validation_per_replication_10min.csv"
)

dt = pd.read_csv(file_path)
des = pd.read_csv("arrivals_2007-09-30_12-00.csv")

# Metrics to compare — arrivals only in this script
metrics = ["MAE_arrivals", "RMSE_arrivals", "MAPE_arrivals"]

# Significance level for all hypothesis tests
alpha = 0.05

# ====================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ====================================================================
# For each metric:
#   1. Test normality of both DT and DES distributions (Shapiro-Wilk)
#   2. Choose appropriate paired test based on normality results
#   3. Report whether the difference is statistically significant
# ====================================================================

for metric in metrics:
    print("\n" + "=" * 30)
    print(f"Metric: {metric}")
    print("=" * 30)

    dt_values  = dt[metric].dropna()
    des_values = des[metric].dropna()

    # Shapiro-Wilk normality test
    # H0: data is normally distributed
    # p > 0.05 → fail to reject H0 → treat as normal
    dt_p  = shapiro(dt_values)[1]
    des_p = shapiro(des_values)[1]

    print(f"DT  Shapiro p-value: {dt_p:.5f}")
    print(f"DES Shapiro p-value: {des_p:.5f}")

    normal_dt  = dt_p  > alpha
    normal_des = des_p > alpha

    if normal_dt and normal_des:
        # Both normal → use parametric paired t-test
        print("→ Both normal → Using Paired t-test")
        stat, p_value = ttest_rel(dt_values, des_values)
        test_name     = "Paired t-test"
    else:
        # At least one non-normal → use non-parametric Wilcoxon test
        print("→ Not normal → Using Wilcoxon signed-rank test")
        stat, p_value = wilcoxon(dt_values, des_values)
        test_name     = "Wilcoxon signed-rank"

    print(f"{test_name} p-value: {p_value:.5f}")

    if p_value < alpha:
        print("→ Significant difference between DT and DES")
    else:
        print("→ No significant difference between DT and DES")


# ====================================================================
# DESCRIPTIVE STATISTICS: MEDIAN ± IQR
# ====================================================================
# The median and interquartile range (IQR) are used rather than
# mean ± SD because metric distributions may be skewed (non-normal).
# IQR = 75th percentile − 25th percentile.
# ====================================================================

print("\n" + "=" * 35)
print("Median ± IQR Summary")
print("=" * 35 + "\n")

for metric in metrics:
    if metric not in dt.columns or metric not in des.columns:
        print(f"{metric} not found in data.\n")
        continue

    dt_vals  = dt[metric].dropna()
    des_vals = des[metric].dropna()

    dt_median  = np.median(dt_vals)
    dt_iqr     = np.percentile(dt_vals, 75) - np.percentile(dt_vals, 25)
    des_median = np.median(des_vals)
    des_iqr    = np.percentile(des_vals, 75) - np.percentile(des_vals, 25)

    print(f"{metric}")
    print(f"  DT  → Median: {dt_median:.4f},  IQR: {dt_iqr:.4f}")
    print(f"  DES → Median: {des_median:.4f}, IQR: {des_iqr:.4f}\n")

    # ------------------------------------------------------------------
    # SCATTER PLOT: DT vs DES per-replication values
    # Points above the identity line (slope=1) indicate DES > DT for
    # that replication; points below indicate DT > DES.
    # ------------------------------------------------------------------
    plt.figure()
    plt.scatter(dt_vals, des_vals, alpha=0.6)
    plt.xlabel(f"DT — {metric}")
    plt.ylabel(f"DES — {metric}")
    plt.title(f"Scatter: DT vs DES\n{metric}")
    plt.axline(
        (0, 0), slope=1,
        color="red", linestyle="--", linewidth=1, label="Identity line"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # VIOLIN PLOT: Distribution shape comparison
    # Wider sections indicate more frequent values.
    # The horizontal line shows the median.
    # ------------------------------------------------------------------
    plt.figure()
    plt.violinplot([dt_vals, des_vals], showmedians=True)
    plt.xticks([1, 2], ["DT", "DES"])
    plt.title(f"Violin Plot: DT vs DES\n{metric}")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()
