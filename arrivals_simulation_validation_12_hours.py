# -*- coding: utf-8 -*-
"""
arrivals_simulation_validation_12_hours.py
------------------------------------------
Discrete-Event Simulation (DES) Baseline for Arrival Forecasting.

This script implements a standalone DES model for patient arrivals using
SimPy. It serves as a BASELINE/COMPARISON model against the database-driven
Digital Twin (Digital_Twin_10_minute.py).

PURPOSE:
    The DES model simulates patient arrivals over a 12-hour forecast horizon
    at 10-minute resolution using 100 stochastic replications. Its outputs
    (per-replication metrics) are saved as CSV files that are then compared
    against the Digital Twin outputs using Compare.py.

DIFFERENCE FROM DIGITAL TWIN:
    - This DES model uses FIXED hardcoded GMM parameters (LOS_GMM_BY_DAY)
      rather than dynamically fitting them from the database
    - This DES model does NOT model discharges or census
    - This DES model does NOT load currently-present patients as initial state
    - It is a simpler, purely arrival-focused simulation

AUTHORS:
    Nirmani Amarasinghe  (ORCID: 0009-0001-9719-6366)
    Laura Boyle          (ORCID: 0000-0001-9651-1363)
    Adele H. Marshall    (ORCID: 0000-0001-5306-2756)

    Mathematical Science Research Centre, Queen's University Belfast

INPUTS:
    - database/<folder>/<folder>.mdb   (visit records)
    - best_fits_arrivals.csv           (best distribution per weekday/hour)

OUTPUTS:
    - Compare DT and DES/arrivals_<timestamp>.csv
        Per-replication validation metrics (MAE, RMSE, MAPE, sMAPE, Aggregate PE)
    - Compare DT and DES/summary_10min_<timestamp>.csv
        Actual vs mean/median forecast with 95% CI at 10-min resolution
    - des_plots/forecast.png
        Plot of forecast CI, mean, and actual arrivals

DATA SOURCE:
    Armony, M. et al. (2015). On patient flow in hospitals: A data-based
    queueing-science perspective. Stochastic Systems, 5(1), 146-194.
    DOI: 10.1214/14-SSY153
"""

import os
import pandas as pd
import numpy as np
import simpy
import pyodbc
import ast
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import gamma, norm, weibull_min, lognorm

# Fixed random seed for full reproducibility across runs
np.random.seed(42)

# ====================================================================
# SIMULATION PARAMETERS
# ====================================================================

STEP_MINUTES   = 10                          # Time resolution (minutes)
STEPS_PER_HOUR = 60 // STEP_MINUTES          # = 6 steps per hour
STEP_SIZE_H    = STEP_MINUTES / 60.0         # Step size in hours (1/6)
FORECAST_HOURS = 12                          # Forecast horizon
FORECAST_STEPS = FORECAST_HOURS * STEPS_PER_HOUR   # = 72 total steps

# ====================================================================
# DATA LOADING
# ====================================================================
# Load all visit records from .mdb databases in the database/ folder.
# Each subfolder is expected to contain an .mdb file with the same name
# as the folder (e.g. database/ED1/ED1.mdb).
# ====================================================================

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
all_data = []

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        mdb_file = os.path.join(folder_path, f"{folder}.mdb")
        if os.path.isfile(mdb_file):
            conn_str = (
                f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};"
                f"DBQ={mdb_file};"
            )
            try:
                conn    = pyodbc.connect(conn_str)
                df_tmp  = pd.read_sql("SELECT * FROM visits", conn)
                conn.close()
                all_data.append(df_tmp)
                print(f"Loaded: {mdb_file}")
            except Exception as e:
                print(f"Failed to load {mdb_file}: {e}")

if not all_data:
    raise ValueError("No data loaded.")

# Combine all database records into a single DataFrame
df = pd.concat(all_data, ignore_index=True)

# Parse entry dates and filter to ED pathway patients only (entry_group == 1)
df["entry_date"] = pd.to_datetime(df["entry_date"])
df = df[df["entry_group"] == 1].dropna(subset=["entry_date"])
df = df.sort_values("entry_date")

# ====================================================================
# LENGTH-OF-STAY GMM PARAMETERS (HARDCODED)
# ====================================================================
# These are pre-fitted Gaussian Mixture Model parameters for patient
# length of stay (LOS), organised by day of week (0=Monday, 6=Sunday).
#
# Each component has:
#   mean   : mean LOS in hours for this Gaussian component
#   var    : variance of LOS for this component
#   weight : mixture weight (all weights sum to 1 for each day)
#
# NOTE: These are fixed parameters. For dynamically-fitted parameters
# that update based on recent data, see LOS.py.
# ====================================================================

LOS_GMM_BY_DAY = {
    0: [{"mean": 1.65, "var": 0.13, "weight": 0.23},
        {"mean": 3.63, "var": 0.23, "weight": 0.154},
        {"mean": 6.79, "var": 0.53, "weight": 0.061},
        {"mean": 4.96, "var": 0.37, "weight": 0.102},
        {"mean": 0.79, "var": 0.12, "weight": 0.249},
        {"mean": 2.58, "var": 0.15, "weight": 0.204}],
    1: [{"mean": 7.17, "var": 0.23, "weight": 0.033},
        {"mean": 2.09, "var": 0.09, "weight": 0.177},
        {"mean": 1.35, "var": 0.08, "weight": 0.203},
        {"mean": 3.76, "var": 0.13, "weight": 0.11},
        {"mean": 0.68, "var": 0.09, "weight": 0.207},
        {"mean": 4.74, "var": 0.16, "weight": 0.07},
        {"mean": 2.9,  "var": 0.1,  "weight": 0.149},
        {"mean": 5.85, "var": 0.21, "weight": 0.051}],
    2: [{"mean": 1.26, "var": 0.08, "weight": 0.211},
        {"mean": 6.59, "var": 0.41, "weight": 0.052},
        {"mean": 3.78, "var": 0.18, "weight": 0.123},
        {"mean": 2.83, "var": 0.12, "weight": 0.156},
        {"mean": 0.6,  "var": 0.07, "weight": 0.18},
        {"mean": 5.01, "var": 0.29, "weight": 0.083},
        {"mean": 2.0,  "var": 0.09, "weight": 0.195}],
    3: [{"mean": 0.75, "var": 0.12, "weight": 0.242},
        {"mean": 4.86, "var": 0.36, "weight": 0.104},
        {"mean": 2.5,  "var": 0.15, "weight": 0.195},
        {"mean": 6.67, "var": 0.6,  "weight": 0.066},
        {"mean": 3.54, "var": 0.22, "weight": 0.158},
        {"mean": 1.59, "var": 0.12, "weight": 0.235}],
    4: [{"mean": 4.37, "var": 0.29, "weight": 0.1},
        {"mean": 1.33, "var": 0.1,  "weight": 0.24},
        {"mean": 5.94, "var": 0.4,  "weight": 0.059},
        {"mean": 3.16, "var": 0.19, "weight": 0.161},
        {"mean": 2.17, "var": 0.13, "weight": 0.216},
        {"mean": 0.64, "var": 0.08, "weight": 0.224}],
    5: [{"mean": 2.23, "var": 0.11, "weight": 0.199},
        {"mean": 0.69, "var": 0.08, "weight": 0.276},
        {"mean": 4.27, "var": 0.25, "weight": 0.089},
        {"mean": 3.14, "var": 0.15, "weight": 0.142},
        {"mean": 1.43, "var": 0.09, "weight": 0.242},
        {"mean": 5.72, "var": 0.3,  "weight": 0.052}],
    6: [{"mean": 3.04, "var": 0.13, "weight": 0.166},
        {"mean": 0.68, "var": 0.1,  "weight": 0.184},
        {"mean": 5.31, "var": 0.32, "weight": 0.085},
        {"mean": 1.38, "var": 0.09, "weight": 0.203},
        {"mean": 7.01, "var": 0.47, "weight": 0.051},
        {"mean": 4.06, "var": 0.2,  "weight": 0.12},
        {"mean": 2.15, "var": 0.1,  "weight": 0.191}]
}


def sample_los(day):
    """
    Sample a patient length-of-stay (hours) from the hardcoded GMM
    for the given day of the week.

    Parameters
    ----------
    day : int — day of week (0=Monday, 6=Sunday)

    Returns
    -------
    float — LOS in hours (minimum 0.1 to avoid zero/negative values)
    """
    comps = LOS_GMM_BY_DAY[day]
    w     = [c["weight"] for c in comps]
    # Select a GMM component proportional to mixture weights
    c     = np.random.choice(len(comps), p=w)
    # Sample from the selected Gaussian, using sqrt(var) as std dev
    return max(np.random.normal(comps[c]["mean"], np.sqrt(comps[c]["var"])), 0.1)


# ====================================================================
# ARRIVAL MODEL
# ====================================================================
# Load pre-fitted arrival distribution parameters from CSV.
# The 'params_continuous' column is stored as a string representation
# of a dict, so ast.literal_eval is used to safely parse it back.
# ====================================================================

arrival_fits = pd.read_csv("best_fits_arrivals.csv")
arrival_fits["params_continuous"] = arrival_fits["params_continuous"].apply(
    ast.literal_eval
)

# Supported scipy distribution objects, keyed by name string
DIST_MAP = {
    "gamma":       gamma,
    "norm":        norm,
    "weibull_min": weibull_min,
    "lognorm":     lognorm
}


def sample_arrivals_10min(wd, hour):
    """
    Sample arrivals for a single 10-minute step by scaling down
    the fitted hourly distribution parameters.

    SCALING LOGIC:
        Arrival distributions are fitted to hourly counts.
        To get 10-minute counts (1/6 of an hour):
          - norm     : divide loc and scale by 6
          - lognorm  : subtract log(6) from loc (log-scale shift)
          - gamma / weibull : divide scale by 6 (mean scales linearly)

    Parameters
    ----------
    wd   : int — weekday (0=Monday)
    hour : int — hour of day (0-23)

    Returns
    -------
    int — number of arrivals in this 10-minute step (minimum 0)
    """
    row    = arrival_fits[
        (arrival_fits.weekday == wd) & (arrival_fits.hour == hour)
    ].iloc[0]
    dist   = DIST_MAP[row["best_continuous"]]
    params = dict(row["params_continuous"])

    # Scale hourly parameters down to 10-minute resolution
    if row["best_continuous"] == "norm":
        params["loc"]   /= STEPS_PER_HOUR
        params["scale"] /= STEPS_PER_HOUR
    elif row["best_continuous"] == "lognorm":
        # In log-space, dividing by n = subtracting log(n)
        params["loc"] -= np.log(STEPS_PER_HOUR)
    else:
        if "scale" in params:
            params["scale"] /= STEPS_PER_HOUR

    return max(0, int(round(dist.rvs(**params))))


# ====================================================================
# SIMPY SIMULATION
# ====================================================================

def simulate_replication_10min(weekday, start_hour):
    """
    Run a single SimPy replication of the arrival process over 12 hours
    at 10-minute resolution.

    The SimPy environment steps through FORECAST_STEPS time steps.
    At each step, arrivals are sampled and appended to a list.
    SimPy's event-driven framework is used here in a simple time-stepped
    mode (timeout at each step) rather than event-driven patient logic.

    Parameters
    ----------
    weekday    : int — forecast day of week (0=Monday)
    start_hour : int — hour at which the forecast starts

    Returns
    -------
    numpy array of shape (FORECAST_STEPS,) — arrivals per 10-min step
    """
    def process(env, rows):
        for step in range(FORECAST_STEPS):
            # Advance hour as simulation progresses through the day
            hour = (start_hour + step // STEPS_PER_HOUR) % 24
            n    = sample_arrivals_10min(weekday, hour)
            rows.append(n)
            yield env.timeout(STEP_SIZE_H)   # Advance time by 10 minutes

    env  = simpy.Environment()
    rows = []
    env.process(process(env, rows))
    env.run(until=FORECAST_HOURS)
    return np.array(rows)


# ====================================================================
# FORECAST CONFIGURATION
# ====================================================================

replications   = 100

# Set the forecast start time — change this to run a different period
FORECAST_START = pd.Timestamp("2007-09-18 00:00")
FORECAST_START = FORECAST_START.floor(f"{STEP_MINUTES}min")

period_start = FORECAST_START
period_end   = period_start + pd.Timedelta(hours=FORECAST_HOURS)
weekday      = period_start.weekday()
start_hour   = period_start.hour

# ====================================================================
# ACTUAL DATA EXTRACTION
# ====================================================================
# Count actual observed arrivals in each 10-minute slot within the
# forecast window. Used as ground truth for validation metrics.
# ====================================================================

df_period    = df[(df.entry_date >= period_start) & (df.entry_date < period_end)]
actual_10min = np.zeros(FORECAST_STEPS)

for s in range(FORECAST_STEPS):
    t0 = period_start + pd.Timedelta(minutes=s * STEP_MINUTES)
    t1 = t0 + pd.Timedelta(minutes=STEP_MINUTES)
    actual_10min[s] = (
        (df_period.entry_date >= t0) & (df_period.entry_date < t1)
    ).sum()

# Aggregate 10-min actuals to hourly bins for hourly metric reporting
actual_hourly = actual_10min.reshape(FORECAST_HOURS, STEPS_PER_HOUR).sum(axis=1)


# ====================================================================
# VALIDATION METRIC FUNCTIONS
# ====================================================================

def mape(p, a):
    """
    Mean Absolute Percentage Error.
    Only computed at steps where actual > 0 to avoid division by zero.
    """
    m = a > 0
    return np.mean(np.abs(p[m] - a[m]) / a[m]) * 100 if m.any() else np.nan


def smape(p, a):
    """
    Symmetric Mean Absolute Percentage Error.
    More robust than MAPE when actual values are near zero.
    Formula: (1/n) * Σ 2|F-A| / (|F|+|A|) * 100
    """
    return np.mean(2 * np.abs(p - a) / (np.abs(p) + np.abs(a) + 1e-6)) * 100


# ====================================================================
# PRELIMINARY RUN — AGGREGATE METRICS ONLY
# ====================================================================
# First pass: run all replications and compute aggregate statistics
# for a quick overview before the detailed per-replication analysis.
# ====================================================================

all_preds = []

for _ in range(replications):
    all_preds.append(simulate_replication_10min(weekday, start_hour))

preds      = np.array(all_preds)
pred_mean  = preds.mean(axis=0)
pred_p025  = np.percentile(preds, 2.5,  axis=0)   # Lower 95% CI bound
pred_p975  = np.percentile(preds, 97.5, axis=0)   # Upper 95% CI bound

# Print aggregate validation metrics at 10-minute resolution
print("\n=== 10-MIN METRICS ===")
print("MAE:",  mean_absolute_error(actual_10min, pred_mean))
print("RMSE:", np.sqrt(mean_squared_error(actual_10min, pred_mean)))
print("MAPE:", mape(pred_mean, actual_10min))
print("sMAPE:", smape(pred_mean, actual_10min))

# Aggregate 10-min predictions to hourly for hourly metric reporting
pred_hourly = pred_mean.reshape(FORECAST_HOURS, STEPS_PER_HOUR).sum(axis=1)

print("\n=== HOURLY METRICS ===")
print("MAE:",  mean_absolute_error(actual_hourly, pred_hourly))
print("RMSE:", np.sqrt(mean_squared_error(actual_hourly, pred_hourly)))
print("MAPE:", mape(pred_hourly, actual_hourly))

# ====================================================================
# QUICK DIAGNOSTIC PLOT
# ====================================================================
# Save a forecast plot showing the 95% CI band, mean forecast,
# and actual observed arrivals.
# ====================================================================

PLOT_DIR = "des_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

time_axis = [
    period_start + pd.Timedelta(minutes=s * STEP_MINUTES)
    for s in range(FORECAST_STEPS)
]

plt.figure(figsize=(14, 5))
plt.fill_between(time_axis, pred_p025, pred_p975, alpha=0.2, label="95% CI")
plt.plot(time_axis, pred_mean, label="Simulated Mean")
plt.plot(time_axis, actual_10min, '--', label="Actual")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Arrivals per 10 min")
plt.title(f"DES Arrival Forecast vs Actual | {period_start}")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "forecast.png"))
plt.close()

print("\nDiagnostic plot saved.")

# ====================================================================
# PER-REPLICATION VALIDATION
# ====================================================================
# Second pass: re-run replications and save per-replication metrics.
# These CSV files are used by Compare.py to statistically compare
# the DES baseline against the Digital Twin outputs.
# ====================================================================

rep_results = []
all_preds   = []

for rep in range(1, replications + 1):
    pred = simulate_replication_10min(weekday, start_hour)
    all_preds.append(pred)

    # Aggregate percentage error: how close is total simulated count
    # to total actual count over the full 12-hour window
    agg_pe = (
        abs(pred.sum() - actual_10min.sum()) / actual_10min.sum() * 100
        if actual_10min.sum() > 0 else np.nan
    )

    rep_results.append({
        "Replication":      rep,
        "MAE_arrivals":     mean_absolute_error(actual_10min, pred),
        "RMSE_arrivals":    np.sqrt(mean_squared_error(actual_10min, pred)),
        "MAPE_arrivals":    mape(pred, actual_10min),
        "sMAPE (%)":        smape(pred, actual_10min),
        "Aggregate_PE (%)": agg_pe
    })

rep_df        = pd.DataFrame(rep_results)
timestamp_str = period_start.strftime("%Y-%m-%d_%H-%M")

# Save per-replication metrics CSV for use in Compare.py
file_name = (
    f"C:/PhD/Pilot/ed_digital_twin/Compare DT and DES/"
    f"arrivals_{timestamp_str}.csv"
)
rep_df.to_csv(file_name, index=False)
print(f"Saved per-replication metrics: {file_name}")

# ====================================================================
# SUMMARY CSV (ACTUAL vs SIMULATED PER TIME STEP)
# ====================================================================
# Save a detailed time-series summary with actual, mean, median,
# and 95% CI bounds at each 10-minute step.
# This can be used for plotting or further analysis.
# ====================================================================

preds        = np.array(all_preds)
mean_10min   = preds.mean(axis=0)
median_10min = np.median(preds, axis=0)

time_axis = [
    period_start + pd.Timedelta(minutes=s * STEP_MINUTES)
    for s in range(FORECAST_STEPS)
]

summary_df = pd.DataFrame({
    "timestamp":       time_axis,
    "actual_arrivals": actual_10min,
    "mean_arrivals":   mean_10min,
    "median_arrivals": median_10min,
    "p025":            np.percentile(preds, 2.5,  axis=0),
    "p975":            np.percentile(preds, 97.5, axis=0),
})

summary_file = (
    f"C:/PhD/Pilot/ed_digital_twin/Compare DT and DES/"
    f"summary_10min_{timestamp_str}.csv"
)
summary_df.to_csv(summary_file, index=False)
print(f"Saved time-series summary: {summary_file}")
