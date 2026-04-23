# -*- coding: utf-8 -*-
"""
LOS.py
------
Length-of-Stay (LOS) Parameter Estimation for the ED Digital Twin.

This script connects to Microsoft Access (.mdb) databases containing
historical ED visit records, extracts patient length-of-stay data from
the 2 weeks prior to the forecast date, and fits Gaussian Mixture Models
(GMMs) to the LOS distributions separately for each day of the week.

The fitted GMM parameters are saved as a JSON file which is then read
by the main simulation (Digital_Twin_10_minute.py) to sample realistic
patient LOS values during the forward simulation.

WHY GMM?
    Patient LOS in an ED is typically multi-modal — for example, short
    stays for minor injuries and longer stays for complex cases. A GMM
    can capture this multi-modal shape by combining multiple Gaussian
    components, unlike a single normal or exponential distribution.

WHY WEEKDAY-SPECIFIC?
    ED patient behaviour and case-mix varies by day of the week
    (e.g. weekends vs weekdays), so separate distributions are fitted
    for each weekday to capture this variation.

Authors:
    Nirmani Amarasinghe  (ORCID: 0009-0001-9719-6366)
    Laura Boyle          (ORCID: 0000-0001-9651-1363)
    Adele H. Marshall    (ORCID: 0000-0001-5306-2756)

    Mathematical Science Research Centre, Queen's University Belfast

Inputs:
    - .mdb database files in data/input/<subfolder>/
    - config.py (FORECAST_DATE)

Outputs:
    - data/input/parameters/los_parameters_weekday_<YYYY-MM>.json

Data source:
    Armony, M. et al. (2015). On patient flow in hospitals: A data-based
    queueing-science perspective. Stochastic Systems, 5(1), 146-194.
    DOI: 10.1214/14-SSY153
"""

import os
import glob
import pyodbc
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats
from datetime import datetime, timedelta
import json
from config import FORECAST_DATE
from pathlib import Path

# --------------------------------------------------------------------
# DATABASE PATH SETUP
# --------------------------------------------------------------------
# Locate all .mdb files under data/input/ relative to this script.
# Using Path(__file__) ensures the path works regardless of where
# the script is called from.
# --------------------------------------------------------------------

base_db_dir = Path(__file__).resolve().parent / "data" / "input"
db_paths = glob.glob(os.path.join(base_db_dir, "*", "*.mdb"))

# --------------------------------------------------------------------
# TIME WINDOW
# --------------------------------------------------------------------
# Extract LOS data from the 2 weeks immediately before the forecast
# date. A 2-week window balances recency (capturing current patterns)
# against having enough data to fit reliable distributions.
# --------------------------------------------------------------------

all_los_records = []

forecast_ts = pd.Timestamp(FORECAST_DATE)
start_ts    = forecast_ts - pd.Timedelta(weeks=2)  # 14-day lookback window

# --------------------------------------------------------------------
# DATA EXTRACTION
# --------------------------------------------------------------------
# Loop through each .mdb database, connect via ODBC, and extract
# completed visits (entry_group=1 AND exit_group=1) that started
# within the 2-week lookback window. Calculate LOS in hours.
# --------------------------------------------------------------------

for db_path in db_paths:
    if not os.path.exists(db_path):
        continue
    try:
        conn_str = (
            r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
            f"DBQ={db_path};"
        )
        with pyodbc.connect(conn_str) as conn:
            # Only select completed visits (both entry and exit recorded)
            df = pd.read_sql(
                "SELECT entry_date, exit_date FROM visits "
                "WHERE entry_group = 1 AND exit_group = 1;",
                conn
            )
            df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
            df['exit_date']  = pd.to_datetime(df['exit_date'],  errors='coerce')

            # Drop rows where dates could not be parsed
            df = df.dropna(subset=['entry_date', 'exit_date'])

            # Filter to the 2-week lookback window
            df = df[
                (df['entry_date'] >= start_ts) &
                (df['entry_date'] <  forecast_ts)
            ]

            # Calculate LOS in hours
            df['los_hours'] = (
                (df['exit_date'] - df['entry_date']).dt.total_seconds() / 3600
            )

            # Remove any zero or negative LOS values (data quality filter)
            df = df[df['los_hours'] > 0]
            all_los_records.append(df)

    except Exception as e:
        print(f"Error reading {db_path}: {e}")

# Raise an error early if no data was found rather than silently failing
if len(all_los_records) == 0:
    raise ValueError("No LOS data found in the last 2 weeks!")

# Combine records from all databases into a single DataFrame
all_los_df = pd.concat(all_los_records)

# Extract day of week (0 = Monday, 6 = Sunday) for weekday-specific fitting
all_los_df['weekday'] = all_los_df['entry_date'].dt.dayofweek
weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


# --------------------------------------------------------------------
# GMM CUMULATIVE DISTRIBUTION FUNCTION (CDF)
# --------------------------------------------------------------------
# Helper function to compute the CDF of a fitted GMM at given points x.
# The GMM CDF is a weighted sum of the CDFs of its Gaussian components.
# This is used in the Kolmogorov-Smirnov (KS) goodness-of-fit test.
# --------------------------------------------------------------------

def gmm_cdf(x, gmm):
    """
    Compute the CDF of a Gaussian Mixture Model at values x.

    Parameters
    ----------
    x   : array-like, points at which to evaluate the CDF
    gmm : fitted GaussianMixture object

    Returns
    -------
    cdf_vals : numpy array of CDF values at each point in x
    """
    cdf_vals = np.zeros_like(x, dtype=float)
    for weight, mean, cov in zip(
        gmm.weights_.flatten(),
        gmm.means_.flatten(),
        gmm.covariances_.flatten()
    ):
        std       = np.sqrt(cov)
        cdf_vals += weight * stats.norm.cdf(x, loc=mean, scale=std)
    return cdf_vals


# --------------------------------------------------------------------
# GMM FITTING — WEEKDAY-SPECIFIC
# --------------------------------------------------------------------
# For each day of the week, fit a GMM to the LOS data using the
# minimum number of components (k) that passes a KS goodness-of-fit
# test at the 5% significance level.
#
# KS TEST LOGIC:
#   - Try k = 1, 2, 3, ... up to 10 components
#   - If p-value > 0.05, the GMM fits well enough → stop and use this k
#   - If no k passes the test, fall back to k=10 (maximum complexity)
#
# This automated model selection avoids overfitting (too many components)
# while ensuring the fitted distribution is statistically adequate.
# --------------------------------------------------------------------

weekday_gmm_params = {}

for wd in range(7):
    # Extract LOS values for this weekday only
    los_data = all_los_df[all_los_df['weekday'] == wd]['los_hours'].values

    # Skip this weekday if no data is available
    if len(los_data) == 0:
        continue

    # Reshape to (n_samples, 1) as required by scikit-learn GaussianMixture
    X = los_data.reshape(-1, 1)

    # Build empirical CDF for the KS test
    los_sorted    = np.sort(los_data)
    empirical_cdf = np.arange(1, len(los_sorted) + 1) / len(los_sorted)

    best_gmm = None

    # Try increasing numbers of GMM components until the KS test passes
    for k in range(1, 11):
        gmm_k = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42       # Fixed seed for reproducibility
        )
        gmm_k.fit(X)

        # Compute KS statistic: max difference between empirical and model CDF
        model_cdf = gmm_cdf(los_sorted, gmm_k)
        ks_stat   = np.max(np.abs(empirical_cdf - model_cdf))

        # p-value for the KS statistic
        p_val = stats.kstwo.sf(ks_stat, len(los_sorted))

        # Accept this k if the fit is statistically adequate (p > 0.05)
        if p_val > 0.05:
            best_gmm = gmm_k
            best_k   = k
            break

    # If no k passed the KS test, use the maximum (k=10) as a fallback
    if best_gmm is None:
        best_k   = 10
        best_gmm = GaussianMixture(n_components=10, random_state=42)
        best_gmm.fit(X)

    # Store the fitted parameters for this weekday
    # stds = sqrt(covariances) converts variance back to standard deviation
    weekday_gmm_params[weekday_names[wd]] = {
        "k":       best_k,
        "weights": best_gmm.weights_.tolist(),
        "means":   best_gmm.means_.flatten().tolist(),
        "stds":    np.sqrt(best_gmm.covariances_.flatten()).tolist()
    }


# --------------------------------------------------------------------
# SAVE PARAMETERS TO JSON
# --------------------------------------------------------------------
# The output filename encodes the parameter month (3 months before
# the forecast date) to ensure the correct file is loaded by the
# simulation even when running forecasts across different time periods.
# --------------------------------------------------------------------

los_param_month = (
    forecast_ts.replace(day=1) - pd.DateOffset(months=3)
).strftime("%Y-%m")

output_path = (
    f"data/input/parameters/los_parameters_weekday_{los_param_month}.json"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(weekday_gmm_params, f, indent=4)

print(
    f"Weekday-specific LOS GMM parameters (last 2 weeks) saved to:\n"
    f"{output_path}"
)
