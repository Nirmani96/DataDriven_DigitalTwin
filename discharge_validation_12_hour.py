# -*- coding: utf-8 -*-
"""
discharge_validation_12_hour.py
--------------------------------
Discrete-Event Simulation (DES) Baseline for Discharge Forecasting.

This script implements a SimPy-based DES model for patient discharges,
serving as a BASELINE/COMPARISON model against the database-driven
Digital Twin (Digital_Twin_10_minute.py).

PURPOSE:
    Simulates patient discharges over a 12-hour forecast window at
    10-minute resolution using 100 stochastic replications. Per-replication
    metrics are saved as CSV files for statistical comparison against the
    Digital Twin using Compare_Discharge.py.

HOW DISCHARGE SIMULATION WORKS:
    Unlike the arrival model (arrivals_simulation_validation_12_hours.py),
    this script models the FULL patient pathway:
        1. An 8-hour WARM-UP period generates patients who are already
           in the ED when the forecast window starts. This avoids the
           unrealistic "empty ED" problem at time zero.
        2. Patient arrivals continue throughout the warm-up and forecast
           periods. Each arriving patient independently joins the ED
           pathway with probability 0.675 (ED_PROBABILITY).
        3. Each ED patient holds a sampled LOS, and is discharged when
           their LOS timer expires.
        4. Only discharges occurring within the 12-hour FORECAST window
           are counted and compared to actuals.

AUTHORS:
    Nirmani Amarasinghe  (ORCID: 0009-0001-9719-6366)
    Laura Boyle          (ORCID: 0000-0001-9651-1363)
    Adele H. Marshall    (ORCID: 0000-0001-5306-2756)

    Mathematical Science Research Centre, Queen's University Belfast

INPUTS:
    - database/<folder>/<folder>.mdb   (visit records)
    - best_fits_arrivals.csv           (best distribution per weekday/hour)

OUTPUTS:
    - Compare DT and DES/discharge_<timestamp>.csv
        Per-replication validation metrics (MAE, RMSE, MAPE, sMAPE, Aggregate PE)
    - Compare DT and DES/discharge_summary_10min_<timestamp>.csv
        Actual vs mean/median/CI discharge counts at 10-min resolution
    - des_plots/discharge_plot_<timestamp>.png
        Forecast vs actual discharge plot

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
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import gamma, norm, weibull_min, lognorm

# Fixed random seed for reproducibility
np.random.seed(42)

# ====================================================================
# SIMULATION PARAMETERS
# ====================================================================

STEP_MINUTES   = 10
STEPS_PER_HOUR = 6
STEP_SIZE_H    = STEP_MINUTES / 60.0         # 10 minutes in hours
FORECAST_HOURS = 12
FORECAST_STEPS = FORECAST_HOURS * STEPS_PER_HOUR   # 72 steps

# Output directory for plots
PLOT_DIR = "des_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ====================================================================
# DATA LOADING
# ====================================================================
# Load visit records from all .mdb databases under database/ folder.
# ====================================================================

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
all_data = []

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        mdb_file = os.path.join(folder_path, f"{folder}.mdb")
        if os.path.isfile(mdb_file):
            try:
                conn = pyodbc.connect(
                    r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
                    f"DBQ={mdb_file};"
                )
                df_tmp = pd.read_sql("SELECT * FROM visits", conn)
                conn.close()
                all_data.append(df_tmp)
                print(f"Loaded: {mdb_file}")
            except Exception as e:
                print(f"Failed: {mdb_file} | {e}")

df = pd.concat(all_data, ignore_index=True)

# Parse exit dates and remove:
#   - Records with implausible future dates (year > 2099 = missing/sentinel)
#   - Records within a known data quality gap (July-Aug 2006)
df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
df = df[df["exit_date"] < "2099-01-01"]
df = df[~(
    (df["exit_date"] >= "2006-07-16") & (df["exit_date"] <= "2006-08-16")
)]

# Keep only completed ED visits (arrived and discharged through ED pathway)
df_discharge = df[
    (df["entry_group"] == 1) & (df["exit_group"] == 1)
].dropna(subset=["exit_date"])

# ====================================================================
# LENGTH-OF-STAY GMM PARAMETERS (HARDCODED)
# ====================================================================
# Pre-fitted Gaussian Mixture Model parameters for patient LOS,
# by day of week (0=Monday, 6=Sunday).
# See arrivals_simulation_validation_12_hours.py for full explanation.
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
    Sample a patient length-of-stay (hours) from the GMM for the given weekday.

    Parameters
    ----------
    day : int — day of week (0=Monday, 6=Sunday)

    Returns
    -------
    float — sampled LOS in hours (minimum 0.1)
    """
    comps = LOS_GMM_BY_DAY[day]
    w     = [c["weight"] for c in comps]
    c     = np.random.choice(len(comps), p=w)
    return max(np.random.normal(comps[c]["mean"], np.sqrt(comps[c]["var"])), 0.1)


# ====================================================================
# ARRIVAL MODEL
# ====================================================================

arrival_fits = pd.read_csv("best_fits_arrivals.csv")
arrival_fits["params_continuous"] = arrival_fits["params_continuous"].apply(
    ast.literal_eval
)

DIST_MAP = {
    "gamma":       gamma,
    "norm":        norm,
    "weibull_min": weibull_min,
    "lognorm":     lognorm
}


def sample_arrivals_10min(wd, hour):
    """
    Sample arrivals for a single 10-minute step by scaling the
    fitted hourly distribution parameters down by STEPS_PER_HOUR (6).

    Parameters
    ----------
    wd   : int — weekday (0=Monday)
    hour : int — hour of day (0-23)

    Returns
    -------
    int — arrivals in this 10-minute step (minimum 0)
    """
    row    = arrival_fits[
        (arrival_fits.weekday == wd) & (arrival_fits.hour == hour)
    ].iloc[0]
    params = dict(row["params_continuous"])
    dist   = DIST_MAP[row["best_continuous"]]

    if row["best_continuous"] == "norm":
        params["loc"]   /= STEPS_PER_HOUR
        params["scale"] /= STEPS_PER_HOUR
    elif row["best_continuous"] == "lognorm":
        params["loc"] -= np.log(STEPS_PER_HOUR)
    else:
        if "scale" in params:
            params["scale"] /= STEPS_PER_HOUR

    return max(0, int(round(dist.rvs(**params))))


# ====================================================================
# DISCHARGE SIMULATION WITH WARM-UP
# ====================================================================

# Warm-up period: 8 hours before the forecast window.
# Purpose: populate the ED with patients who were already admitted
# before the forecast start time, so the simulation does not begin
# from an unrealistically empty department.
WARMUP_HOURS = 8
WARMUP_STEPS = WARMUP_HOURS * STEPS_PER_HOUR   # = 48 warm-up steps


def simulate_discharges_10min(weekday, start_hour):
    """
    Run a single SimPy replication of the discharge process.

    Simulation runs for (WARMUP_HOURS + FORECAST_HOURS) total.
    Only discharges occurring after the warm-up period are recorded
    and counted towards the FORECAST_STEPS output array.

    PATIENT PATHWAY:
        1. Arrivals process generates patients every 10-minute step
           (covering both warm-up and forecast periods)
        2. Each arrival independently becomes an ED patient with
           probability 0.675 (ED_PROBABILITY)
        3. Each ED patient is processed as a SimPy process that
           waits for their sampled LOS, then increments the
           discharge slot counter

    Parameters
    ----------
    weekday    : int — forecast day of week (0=Monday)
    start_hour : int — hour at which the forecast window starts

    Returns
    -------
    numpy array of shape (FORECAST_STEPS,) — discharges per 10-min step
    """
    # Shared array to accumulate discharge counts per forecast slot
    discharge_slots = np.zeros(FORECAST_STEPS)

    # The forecast window starts at simulation time = WARMUP_HOURS
    WINDOW_START = WARMUP_HOURS

    def patient(env):
        """
        SimPy process for a single patient.
        Waits for their LOS to expire, then records a discharge
        in the appropriate 10-minute forecast slot.
        """
        los = sample_los(weekday)
        yield env.timeout(los)       # Patient stays for their LOS duration
        t    = env.now               # Current simulation time (hours)

        # Only count discharges that fall within the forecast window
        if t >= WINDOW_START:
            slot = int((t - WINDOW_START) / STEP_SIZE_H)
            if 0 <= slot < FORECAST_STEPS:
                discharge_slots[slot] += 1

    def arrivals(env):
        """
        SimPy process that generates patient arrivals every 10-minute step.
        Covers the full simulation (warm-up + forecast) period.
        The hour is adjusted to account for the warm-up period offset.
        """
        for step in range(WARMUP_STEPS + FORECAST_STEPS):
            # Calculate real hour of day, adjusted for warm-up offset
            hour = (start_hour + step // STEPS_PER_HOUR - WARMUP_HOURS) % 24
            n    = sample_arrivals_10min(weekday, hour)

            for _ in range(n):
                # Each arrival independently joins the ED pathway
                if np.random.rand() < 0.675:
                    env.process(patient(env))

            yield env.timeout(STEP_SIZE_H)   # Advance by 10 minutes

    env = simpy.Environment()
    env.process(arrivals(env))
    env.run(until=WARMUP_HOURS + FORECAST_HOURS)
    return discharge_slots


# ====================================================================
# FORECAST CONFIGURATION
# ====================================================================

replications = 100

# Set forecast start time — change this to run a different period
FORECAST_START = pd.Timestamp("2007-09-18 00:00")
FORECAST_START = FORECAST_START.floor(f"{STEP_MINUTES}min")

period_start  = FORECAST_START
period_end    = period_start + timedelta(hours=FORECAST_HOURS)
weekday       = period_start.weekday()
start_hour    = period_start.hour
timestamp_str = period_start.strftime("%Y-%m-%d_%H-%M")

# ====================================================================
# ACTUAL DISCHARGE EXTRACTION
# ====================================================================
# Count observed discharges in each 10-minute slot in the forecast
# window from the database. Used as ground truth for validation.
# ====================================================================

df_window    = df_discharge[
    (df_discharge.exit_date >= period_start) &
    (df_discharge.exit_date <  period_end)
]
actual_10min = np.zeros(FORECAST_STEPS)

for s in range(FORECAST_STEPS):
    t0 = period_start + timedelta(minutes=s * STEP_MINUTES)
    t1 = t0 + timedelta(minutes=STEP_MINUTES)
    actual_10min[s] = (
        (df_window.exit_date >= t0) & (df_window.exit_date < t1)
    ).sum()

# Aggregate to hourly for hourly metric reporting
actual_hourly = actual_10min.reshape(FORECAST_HOURS, STEPS_PER_HOUR).sum(axis=1)


# ====================================================================
# VALIDATION METRIC FUNCTIONS
# ====================================================================

def mape(p, a):
    """Mean Absolute Percentage Error (ignores zero-actual steps)."""
    m = a > 0
    return np.mean(np.abs(p[m] - a[m]) / a[m]) * 100 if m.any() else np.nan


def smape(p, a):
    """Symmetric Mean Absolute Percentage Error."""
    return np.mean(2 * np.abs(p - a) / (np.abs(p) + np.abs(a) + 1e-6)) * 100


# ====================================================================
# PER-REPLICATION SIMULATION AND METRICS
# ====================================================================

rep_results = []
all_preds   = []

for rep in range(1, replications + 1):
    pred = simulate_discharges_10min(weekday, start_hour)
    all_preds.append(pred)

    rep_results.append({
        "Replication":       rep,
        "MAE_discharges":    mean_absolute_error(actual_10min, pred),
        "RMSE_discharges":   np.sqrt(mean_squared_error(actual_10min, pred)),
        "MAPE_discharges":   mape(pred, actual_10min),
        "sMAPE (%)":         smape(pred, actual_10min),
        "Aggregate_PE (%)":  (
            abs(pred.sum() - actual_10min.sum()) / actual_10min.sum() * 100
            if actual_10min.sum() > 0 else np.nan
        )
    })

rep_df   = pd.DataFrame(rep_results)
file_rep = (
    f"C:/PhD/Pilot/ed_digital_twin/Compare DT and DES/"
    f"discharge_{timestamp_str}.csv"
)
rep_df.to_csv(file_rep, index=False)
print(f"Saved per-replication metrics: {file_rep}")

# ====================================================================
# AGGREGATE STATISTICS AND PLOT
# ====================================================================

preds      = np.array(all_preds)
pred_mean  = preds.mean(axis=0)
pred_p025  = np.percentile(preds, 2.5,  axis=0)
pred_p975  = np.percentile(preds, 97.5, axis=0)

time_axis = [
    period_start + timedelta(minutes=s * STEP_MINUTES)
    for s in range(FORECAST_STEPS)
]

plt.figure(figsize=(14, 5))
plt.fill_between(time_axis, pred_p025, pred_p975, alpha=0.2, label="95% CI")
plt.plot(time_axis, pred_mean, label="Simulated Mean")
plt.plot(time_axis, actual_10min, '--', label="Actual")
plt.xlabel("Time")
plt.ylabel("Discharges per 10 min")
plt.title(f"DES Discharge Forecast vs Actual | {period_start}")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()

plot_file = os.path.join(PLOT_DIR, f"discharge_plot_{timestamp_str}.png")
plt.savefig(plot_file)
plt.close()
print(f"Plot saved: {plot_file}")

# ====================================================================
# SUMMARY CSV
# ====================================================================

mean_10min   = preds.mean(axis=0)
median_10min = np.median(preds, axis=0)

summary_df = pd.DataFrame({
    "timestamp":          time_axis,
    "actual_discharges":  actual_10min,
    "mean_discharges":    mean_10min,
    "median_discharges":  median_10min,
    "p025":               pred_p025,
    "p975":               pred_p975,
})

summary_file = (
    f"C:/PhD/Pilot/ed_digital_twin/Compare DT and DES/"
    f"discharge_summary_10min_{timestamp_str}.csv"
)
summary_df.to_csv(summary_file, index=False)
print(f"Saved discharge summary: {summary_file}")
