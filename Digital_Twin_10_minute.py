# -*- coding: utf-8 -*-
"""
Digital_Twin_10_minute.py
--------------------------
Database-Driven Discrete-Event Digital Twin of an Emergency Department.

This is the main simulation script. It builds a stochastic digital twin
of an ED that forecasts patient arrivals and discharges over a 12-hour
horizon at 10-minute resolution.

KEY DESIGN DECISIONS IN THIS VERSION:
    1. Arrivals are sampled INDEPENDENTLY at each 10-minute step by
       scaling hourly distribution parameters by 1/6. This avoids the
       Multinomial split problem where the median forecast collapses
       to zero for low-arrival slots.

    
    2. For plots and CSVs, the forecast MEAN is used (not median) to
       avoid the median=0 collapse problem in low-count slots. The TRUE
       MEDIAN is still used for DTW validation in validation_12h.csv.

HOW IT WORKS:
    1. Reads historical visit data from .mdb databases to:
       - Fit arrival rate distributions per weekday/hour (60-day lookback)
       - Load patients currently in the ED at the forecast time
       - Estimate ED capacity bounds from recent 14-day census data
    2. Runs NUM_RUNS stochastic replications of the 12-hour simulation,
       each with a different random seed for independence
    3. At each 10-minute step:
       - Samples new arrivals independently from the scaled distribution
       - Each arrival joins the ED pathway with probability ED_PROBABILITY
       - Each new ED patient is assigned a LOS sampled from the weekday GMM
       - Patients whose exceed the capacity assign a new discharge time
       - Patients whose exit time falls in this step are discharged
    4. Validates simulated results against actual observed data
    5. Saves validation metrics, CSVs, and plots

AUTHORS:
    Nirmani Amarasinghe  (ORCID: 0009-0001-9719-6366)
    Laura Boyle          (ORCID: 0000-0001-9651-1363)
    Adele H. Marshall    (ORCID: 0000-0001-5306-2756)

    Mathematical Science Research Centre, Queen's University Belfast

INPUTS:
    - config.py                                    (FORECAST_DATE)
    - data/input/<subdir>/*.mdb                    (visit records)
    - data/input/best_fits_arrivals.csv            (best distribution per weekday/hour)
    - data/input/parameters/los_parameters_weekday_<YYYY-MM>.json

OUTPUTS:
    - data/output/<forecast_datetime>/
        validation_12h.csv
        dt_actual_vs_simulated_arrivals_10min.csv
        dt_actual_vs_simulated_discharges_10min.csv
    - data/plots_12h/<forecast_datetime>/
        arrivals_10min.png
        arrivals_hourly.png
        discharges_10min.png
        discharges_hourly.png

DATA SOURCE:
    Armony, M. et al. (2015). On patient flow in hospitals: A data-based
    queueing-science perspective. Stochastic Systems, 5(1), 146-194.
    DOI: 10.1214/14-SSY153
"""

import os
import glob
import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from pathlib import Path
import pyodbc
from scipy.stats import gamma as sp_gamma, weibull_min, truncnorm
from scipy.special import gamma as gamma_function
from dtaidistance import dtw as dtw_distance
from config import FORECAST_DATE

# ====================================================================
# SIMULATION PARAMETERS
# ====================================================================

NUM_RUNS       = 100     # Number of stochastic replications
FORECAST_HOURS = 12      # Forecast horizon in hours
ED_PROBABILITY = 0.675   # Probability an arrival enters the ED pathway
                         

# 10-minute time resolution
STEP_MINUTES   = 10
STEPS_PER_HOUR = 60 // STEP_MINUTES          # = 6 steps per hour
FORECAST_STEPS = FORECAST_HOURS * STEPS_PER_HOUR  # = 72 steps total
STEP_TD        = timedelta(minutes=STEP_MINUTES)

# ====================================================================
# FILE PATHS
# ====================================================================
# NOTE: BASE_DB_DIR uses a relative path via Path(__file__) so the
# script works regardless of where it is called from. Update the
# path below if your database folder is in a different location.
# ====================================================================

BASE_DB_DIR       = r"C:\PhD\Pilot\ed_digital_twin\data\input"
ARRIVAL_RATES_CSV = "data/input/best_fits_arrivals.csv"

# Output directories are named by forecast timestamp for traceability
forecast_str = pd.Timestamp(FORECAST_DATE).strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR   = Path(f"data/output/{forecast_str}")
PLOT_DIR     = Path(f"data/plots_12h/{forecast_str}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ====================================================================
# FORECAST TIME WINDOW
# ====================================================================

FORECAST_TS     = pd.Timestamp(FORECAST_DATE)
LOOKBACK_DAYS   = 60   # Days of history used to fit arrival distributions
LOOKBACK_START  = FORECAST_TS - pd.Timedelta(days=LOOKBACK_DAYS)
LOOKBACK_START2 = FORECAST_TS - pd.Timedelta(days=14)  # For capacity estimation

# Find all .mdb database files under the input directory
db_paths = glob.glob(os.path.join(BASE_DB_DIR, "*", "*.mdb"))

# ====================================================================
# LOS MODEL — WEEKDAY-LEVEL GMM FROM JSON
# ====================================================================
# Load pre-fitted Gaussian Mixture Model parameters generated by LOS.py.
# The filename encodes the parameter month (3 months before forecast)
# to ensure consistent loading across different forecast periods.
# ====================================================================

los_param_month = (
    FORECAST_TS.replace(day=1) - pd.DateOffset(months=3)
).strftime("%Y-%m")

with open(
    f"data/input/parameters/los_parameters_weekday_{los_param_month}.json"
) as f:
    los_params = json.load(f)

# Map integer weekday (0=Monday) to the string keys used in the JSON
weekday_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def sample_los(wd):
    """
    Sample a patient length-of-stay (hours) from the weekday-specific GMM.

    A GMM component is selected by sampling from the mixture weights,
    then a value is drawn from the corresponding Gaussian component.
    Negative or zero samples are clipped to 0.01 hours minimum.

    Parameters
    ----------
    wd : int — weekday index (0=Monday, 6=Sunday)

    Returns
    -------
    float — sampled LOS in hours (always > 0)
    """
    p = los_params[weekday_map[wd]]
    # Select which Gaussian component to draw from (weighted random choice)
    k = np.random.choice(len(p["weights"]), p=p["weights"])
    return max(np.random.normal(p["means"][k], p["stds"][k]), 0.01)


# ====================================================================
# ARRIVAL MODEL — METHOD OF MOMENTS FROM RAW WEEKLY COUNTS
# ====================================================================

def build_arrival_models():
    """
    Build per-(weekday, hour) arrival distributions using Method of Moments.

    The best-fitting distribution TYPE for each (weekday, hour) cell is
    read from best_fits_arrivals.csv. Distribution PARAMETERS are then
    estimated from the raw 60-day historical arrival counts using the
    Method of Moments (MoM):

        Gamma  : shape a = mean²/var,  scale = var/mean
        Weibull: shape c solved numerically from the CV equation,
                 scale = mean / Γ(1 + 1/c)
        Normal : loc = mean,  scale = sample std
        Poisson: mu = mean
                 (upgraded to Gamma automatically if var > 1.5 * mean,
                  indicating overdispersion)

    WHY MoM RATHER THAN MLE?
        MoM is computationally simpler, interpretable, and adequate
        when sample sizes are moderate (weekly counts over 60 days).

    Returns
    -------
    dict — keys are (weekday, hour) tuples,
           values are (distribution_name, parameters_dict) tuples
    """
    from scipy.optimize import brentq

    # ------------------------------------------------------------------
    # Load raw arrival records from the 60-day lookback window
    # ------------------------------------------------------------------
    all_dfs = []
    for db in db_paths:
        try:
            conn = pyodbc.connect(
                f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db};"
            )
            df = pd.read_sql(
                "SELECT entry_date FROM visits WHERE entry_group=1;", conn
            )
            conn.close()
            df["entry_date"] = pd.to_datetime(df["entry_date"])
            df = df[
                (df.entry_date >= LOOKBACK_START) &
                (df.entry_date <  FORECAST_TS)
            ]
            all_dfs.append(df)
        except:
            continue

    raw            = pd.concat(all_dfs)
    raw["weekday"] = raw.entry_date.dt.weekday
    raw["hour"]    = raw.entry_date.dt.hour

    # Assign each record to its ISO week start date (Monday)
    raw["week"] = raw.entry_date.dt.to_period("W").apply(
        lambda r: r.start_time
    )

    # Aggregate to weekly counts per (weekday, hour) cell.
    # This gives us a sample of "how many arrivals typically occur
    # in this weekday/hour slot across different weeks".
    weekly = (
        raw.groupby(["weekday", "hour", "week"])
        .size()
        .reset_index(name="count")
    )

    # ------------------------------------------------------------------
    # Load pre-computed best distribution type per (weekday, hour)
    # ------------------------------------------------------------------
    best_fit  = pd.read_csv(ARRIVAL_RATES_CSV)
    best_type = {
        (int(r.weekday), int(r.hour)): r.best_continuous
        for _, r in best_fit.iterrows()
    }

    MIN_OBS = 5    # Minimum observations needed to fit a distribution
    models  = {}

    for wd in range(7):
        for hr in range(24):
            sub  = weekly[(weekly.weekday == wd) & (weekly.hour == hr)]["count"]
            n    = len(sub)
            mean = sub.mean() if n > 0 else 0.1
            mean = max(mean, 0.1)           # Prevent zero mean
            var  = sub.var(ddof=1) if n > 1 else mean
            var  = max(var, 1e-6)           # Prevent zero variance
            std  = np.sqrt(var)

            dist = best_type.get((wd, hr), "poisson")

            # ----------------------------------------------------------
            # GAMMA distribution
            # MoM: shape = mean²/var,  scale = var/mean
            # ----------------------------------------------------------
            if dist == "gamma":
                if n >= MIN_OBS and var > 0:
                    a_fit     = max(mean ** 2 / var, 0.1)
                    scale_fit = max(var / mean, 0.01)
                else:
                    # Fallback if insufficient data
                    a_fit, scale_fit = 2.0, mean / 2.0
                models[(wd, hr)] = ("gamma", {"a": a_fit, "scale": scale_fit})

            # ----------------------------------------------------------
            # WEIBULL distribution
            # Shape c is found by numerically solving:
            #   sqrt( Γ(1+2/c)/Γ(1+1/c)² − 1 ) = CV
            # where CV = std/mean is the coefficient of variation.
            # Brent's method is used for the root-finding.
            # ----------------------------------------------------------
            elif dist == "weibull_min":
                if n >= MIN_OBS and var > 0 and mean > 0:
                    cv = std / mean
                    def weibull_cv_eq(c):
                        g1 = gamma_function(1 + 1 / c)
                        g2 = gamma_function(1 + 2 / c)
                        return np.sqrt(max(g2 / g1 ** 2 - 1, 0)) - cv
                    try:
                        c_fit = brentq(weibull_cv_eq, 0.1, 100)
                    except Exception:
                        c_fit = 1.4   # Fallback shape if solver fails
                    scale_fit = max(
                        mean / gamma_function(1 + 1 / c_fit), 0.01
                    )
                else:
                    c_fit     = 1.4   # Moderate right skew (typical for arrivals)
                    scale_fit = mean / gamma_function(1 + 1 / 1.4)
                models[(wd, hr)] = (
                    "weibull_min", {"c": c_fit, "scale": scale_fit}
                )

            # ----------------------------------------------------------
            # NORMAL distribution (truncated at zero)
            # ----------------------------------------------------------
            elif dist == "norm":
                std_fit = (
                    std if n >= MIN_OBS and std > 0
                    else max(np.sqrt(mean), 1.0)
                )
                models[(wd, hr)] = (
                    "norm", {"loc": mean, "scale": max(std_fit, 0.01)}
                )

            # ----------------------------------------------------------
            # POISSON distribution (default)
            # If var > 1.5 * mean, the data is overdispersed and a
            # Gamma distribution is used instead, which can model
            # variance that exceeds the mean.
            # ----------------------------------------------------------
            else:
                if n >= MIN_OBS and var > mean * 1.5:
                    # Overdispersed — upgrade from Poisson to Gamma
                    a_fit     = max(mean ** 2 / var, 0.1)
                    scale_fit = max(var / mean, 0.01)
                    models[(wd, hr)] = (
                        "gamma", {"a": a_fit, "scale": scale_fit}
                    )
                else:
                    models[(wd, hr)] = ("poisson", {"mu": mean})

    return models


# Build arrival models once at module load time
arrival_models = build_arrival_models()


def sample_arrivals_10min(wd, hr):
    """
    Sample arrivals for a single 10-minute step by scaling down the
    fitted hourly distribution parameters.

    WHY NOT MULTINOMIAL SPLITTING:
        If we draw one hourly total (e.g. 2 arrivals) and split across
        6 slots via Multinomial, most slots get 0. The MEDIAN across
        100 runs is then 0 for most slots — the forecast median line
        collapses to zero and plots look broken.

    CORRECT APPROACH — independent scaled sampling:
        Scale the hourly distribution parameters down to a 10-minute
        rate and draw INDEPENDENTLY at every step. Each slot has its
        own positive expected value (hourly_mean / 6), so the mean
        across runs stays positive and tracks reality.

    SCALING RULES (mean-preserving, shape-preserving):
        Poisson : mu_10min    = mu / 6
        Gamma   : scale_10min = scale / 6    (shape 'a' unchanged)
        Weibull : scale_10min = scale / 6    (shape 'c' unchanged)
        Normal  : loc_10min   = loc / 6,
                  scale_10min = scale / 6

    Parameters
    ----------
    wd : int — weekday (0=Monday)
    hr : int — hour of day (0-23)

    Returns
    -------
    int — number of arrivals in this 10-minute step (minimum 0)
    """
    model, p = arrival_models[(wd, hr)]

    if model == "gamma":
        return max(int(sp_gamma(
            a=p["a"],
            scale=p["scale"] / STEPS_PER_HOUR
        ).rvs()), 0)

    if model == "weibull_min":
        return max(int(weibull_min(
            c=p["c"],
            scale=p["scale"] / STEPS_PER_HOUR
        ).rvs()), 0)

    if model == "norm":
        loc_s   = p["loc"]   / STEPS_PER_HOUR
        scale_s = max(p["scale"] / STEPS_PER_HOUR, 1e-6)
        lo      = (0 - loc_s) / scale_s
        return max(
            int(truncnorm(lo, np.inf, loc=loc_s, scale=scale_s).rvs()), 0
        )

    # Poisson: closed under thinning — mu scales linearly
    return np.random.poisson(p["mu"] / STEPS_PER_HOUR)


# ====================================================================
# INITIAL STATE — PATIENTS CURRENTLY IN THE ED
# ====================================================================

def load_inpatients():
    """
    Load patients who are present in the ED at the forecast start time.

    These are patients who arrived before FORECAST_TS and whose
    expected exit time is after FORECAST_TS. They are added to the
    simulation as 'in progress' patients with their remaining LOS
    pre-calculated from the database.

    This ensures the simulation starts from a realistic, populated
    ED state rather than an empty department.

    Returns
    -------
    list of dicts, each with key 'exit': expected discharge datetime
    """
    patients = []
    for db in db_paths:
        try:
            conn = pyodbc.connect(
                f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db};"
            )
            df = pd.read_sql(
                "SELECT entry_date, exit_date FROM visits "
                "WHERE entry_group=1 AND exit_group=1;",
                conn
            )
            conn.close()
            df["entry_date"] = pd.to_datetime(df["entry_date"])
            df["exit_date"]  = pd.to_datetime(df["exit_date"])

            # Select patients currently in the ED at the forecast time
            df = df[
                (df.entry_date <= FORECAST_TS) &
                (df.exit_date   >  FORECAST_TS)
            ]

            for _, r in df.iterrows():
                remaining = (
                    r.exit_date - FORECAST_TS
                ).total_seconds() / 3600
                if remaining > 0:
                    patients.append({
                        "exit": FORECAST_TS + timedelta(hours=remaining)
                    })
        except:
            continue
    return patients


# ====================================================================
# CAPACITY ESTIMATION
# ====================================================================

def estimate_capacity():
    """
    Estimate operational capacity bounds from the last 14 days of census.

    The hourly census (number of patients simultaneously present) is
    computed for each hour in the 14-day lookback window. The 25th and
    95th percentiles of the resulting distribution are used as:
      - MIN_CAPACITY (25th pct): typical low occupancy
      - MAX_CAPACITY (95th pct): near-peak occupancy

   

    Returns
    -------
    (int, int) — (MIN_CAPACITY, MAX_CAPACITY)
    """
    all_patients = []
    for db in db_paths:
        try:
            conn = pyodbc.connect(
                f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db};"
            )
            df = pd.read_sql(
                "SELECT entry_date, exit_date FROM visits "
                "WHERE entry_group=1 AND exit_group=1;",
                conn
            )
            conn.close()
            df["entry_date"] = pd.to_datetime(df["entry_date"])
            df["exit_date"]  = pd.to_datetime(df["exit_date"])

            # Exclude records with sentinel exit dates (year 9999 = missing)
            df = df[df.exit_date.dt.year != 9999]

            # Keep records that overlap the 14-day lookback window
            df = df[
                (df.entry_date < FORECAST_TS) &
                (df.exit_date   > LOOKBACK_START2)
            ]
            all_patients.append(df)
        except:
            continue

    hist  = pd.concat(all_patients)
    hours = pd.date_range(LOOKBACK_START2, FORECAST_TS, freq="H")

    # Count patients simultaneously present at each hour
    census = [
        ((hist.entry_date <= h) & (hist.exit_date > h)).sum()
        for h in hours
    ]
    return int(np.percentile(census, 25)), int(np.percentile(census, 95))


# Estimate capacity bounds once at startup (used for logging only)
MIN_CAPACITY, MAX_CAPACITY = estimate_capacity()
print(f"Estimated Capacity Range: {MIN_CAPACITY} – {MAX_CAPACITY}")


# ====================================================================
# MAIN SIMULATION — 10-MINUTE RESOLUTION
# ====================================================================

def run_simulation():
    """
    Run NUM_RUNS stochastic replications of the 12-hour ED simulation.

    Each replication:
        1. Seeds the RNG with the run index (reproducible + independent)
        2. Loads currently-present patients from the database
        3. Steps forward in 10-minute increments (FORECAST_STEPS = 72)
        4. At each step:
           a. Samples new arrivals independently using scaled distribution
           b. Each arrival joins the ED with probability ED_PROBABILITY
           c. New ED patients are assigned LOS from the weekday GMM
           d. Patients whose exceed the capacity assign a new discharge time
           e. Patients whose exit time falls in this step are discharged
        5. Records arrivals, discharges, and census per step

    CAPACITY ADJUSTMENT:
        A previous version adjusted discharge timing based on current
        census relative to MIN/MAX capacity:
            census > MAX_CAPACITY → factor = 1.2 (slower discharges)
            census < MIN_CAPACITY → factor = 0.7 (faster discharges)
        

    Returns
    -------
    pd.DataFrame with columns: datetime, arrivals, discharges, census, run
    """
    all_runs = []

    for run in range(NUM_RUNS):
        np.random.seed(run)    # Different seed per run → independence
        patients = load_inpatients()
        rows     = []

        for s in range(FORECAST_STEPS):
            now    = FORECAST_TS + s * STEP_TD
            wd, hr = now.weekday(), now.hour

            # Sample new arrivals for this 10-minute step
            step_arrivals = sample_arrivals_10min(wd, hr)

            # Each arrival independently enters ED pathway
            for _ in range(step_arrivals):
                if np.random.rand() < ED_PROBABILITY:
                    patients.append({
                        "exit": now + timedelta(hours=sample_los(wd))
                    })

            census = len(patients)

            if census > MAX_CAPACITY:
                factor = 1.2
            elif census < MIN_CAPACITY:
                factor = 0.7
            else:
                factor = 1.0

            discharges = 0
            remaining  = []
            for p in patients:
                adj_exit = p["exit"] + timedelta(minutes=(1 - factor) * 0.5 * 60)
                if now <= adj_exit < now + STEP_TD:
                    discharges += 1
                else:
                    remaining.append(p)

            patients = remaining

            rows.append({
                "datetime":   now,
                "arrivals":   step_arrivals,
                "discharges": discharges,
                "census":     len(patients),
                "run":        run
            })

        all_runs.append(pd.DataFrame(rows))

    return pd.concat(all_runs)


# ====================================================================
# VALIDATION METRICS
# ====================================================================

def mape_aggregate(f_total, a_total):
    """
    Aggregate Percentage Error over the full 12-hour horizon.

    NOTE: This is NOT the standard step-level MAPE. It compares the
    total simulated count vs total actual count over the entire window:
        |sum(Forecast) - sum(Actual)| / sum(Actual) * 100

    This is a useful single-number summary of overall bias, but does
    not capture how well the temporal pattern is reproduced.

    Parameters
    ----------
    f_total : float — total simulated count over 12 hours
    a_total : float — total actual count over 12 hours

    Returns
    -------
    float — aggregate percentage error (%)
    """
    return (
        np.abs(f_total - a_total) / a_total * 100
        if a_total > 0 else np.nan
    )


def dtw_similarity(v1, v2, clip_percentile=95):
    """
    Dynamic Time Warping (DTW) similarity score in [0, 1].

    DTW measures similarity between two time series, allowing for
    temporal shifts — useful when simulated peaks occur slightly
    earlier or later than actual peaks.

    NORMALISATION:
        The raw DTW distance is normalised by series length and
        converted to a similarity score:
            similarity = max(1 - dtw_dist / max(len1, len2), 0)
        Values above 0.7 = good; above 0.4 = moderate; below 0.4 = poor.

    CLIPPING:
        Both series are clipped at the 95th percentile before computing
        DTW to prevent single spikes from dominating the metric.

    SCALING:
        Both series are normalised to [0, 1] by dividing by the global
        maximum, so DTW operates on comparable scales.

    NOTE ON MEDIAN vs MEAN:
        DTW in validation_12h.csv is computed on the TRUE MEDIAN
        forecast (not the mean). The median is the statistically
        correct central tendency. The mean is only used for plotting
        to avoid the median=0 collapse in low-count slots.

    Parameters
    ----------
    v1, v2           : array-like — time series to compare
    clip_percentile  : int — percentile for outlier clipping (default 95)

    Returns
    -------
    float — similarity score in [0, 1]
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    l1, l2 = len(v1), len(v2)

    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0

    # Clip outliers at the 95th percentile of the combined series
    combined   = np.concatenate([v1, v2])
    cap        = max(np.percentile(combined, clip_percentile), 1e-6)
    v1         = np.clip(v1, 0, cap)
    v2         = np.clip(v2, 0, cap)

    # Normalise both series to [0, 1] for comparable DTW computation
    global_max = max(v1.max(), v2.max(), 1e-6)
    v1_bar     = v1 / global_max
    v2_bar     = v2 / global_max

    dtw_dist = dtw_distance.distance(v1_bar, v2_bar, use_pruning=True)
    return max(1.0 - dtw_dist / max(l1, l2), 0.0)


# ====================================================================
# PLOTTING UTILITIES
# ====================================================================

def _style_time_ax(ax, minor=True):
    """
    Apply consistent time-axis formatting to a matplotlib Axes object.
    Shows hours on the major axis, optionally half-hours on the minor.
    """
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    if minor:
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.25)


def plot_10min(fine, actual_10min, flow, colour, fname):
    """
    Plot 10-minute resolution forecast mean vs actual.

    Shows:
        - Shaded 95% confidence interval band
        - Forecast mean line (NOTE: mean used, not median — see design notes)
        - Actual observed values (dashed red)

    Parameters
    ----------
    fine         : DataFrame with forecast mean and CI bounds
    actual_10min : DataFrame with actual observed counts
    flow         : 'arrivals' or 'discharges'
    colour       : line/fill colour for the forecast
    fname        : output file path
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(
        f"{flow.capitalize()}  —  12h Forecast vs Actual  |  10-min resolution\n"
        f"{forecast_str}",
        fontsize=12, fontweight="bold"
    )

    t = actual_10min["datetime"]

    # 95% confidence interval band
    ax.fill_between(
        t, fine[f"{flow}_l"], fine[f"{flow}_u"],
        color=colour, alpha=0.20, label="95% CI"
    )
    # Forecast mean line
    ax.plot(
        t, fine[f"{flow}_median"],   # Column named _median but contains mean
        color=colour, linewidth=1.8, label="Forecast mean"
    )
    # Actual observed
    ax.plot(
        t, actual_10min[f"actual_{flow}"],
        color="crimson", linewidth=1.4, linestyle="--", label="Actual"
    )

    ax.set_ylabel("Count per 10 min", fontsize=11)
    ax.set_xlabel("Time", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    _style_time_ax(ax)
    plt.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_hourly(hourly, actual_hourly, flow, colour, fname):
    """
    Plot hourly aggregated forecast mean vs actual.

    Same layout as plot_10min but with data aggregated to hourly bins,
    giving a cleaner view of the overall trend across the 12-hour window.

    Parameters
    ----------
    hourly        : DataFrame with hourly forecast mean and CI bounds
    actual_hourly : DataFrame with hourly actual observed counts
    flow          : 'arrivals' or 'discharges'
    colour        : line/fill colour for the forecast
    fname         : output file path
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"{flow.capitalize()}  —  12h Forecast vs Actual  |  Hourly resolution\n"
        f"{forecast_str}",
        fontsize=12, fontweight="bold"
    )

    t = actual_hourly["datetime"]

    ax.fill_between(
        t, hourly[f"{flow}_l"], hourly[f"{flow}_u"],
        color=colour, alpha=0.20, label="95% CI"
    )
    ax.plot(
        t, hourly[f"{flow}_median"],   # Column named _median but contains mean
        color=colour, linewidth=2.0, marker="o", markersize=5,
        label="Forecast mean"
    )
    ax.plot(
        t, actual_hourly[f"actual_{flow}"],
        color="crimson", linewidth=1.6, linestyle="--",
        marker="o", markersize=5, label="Actual"
    )

    ax.set_ylabel("Count per hour", fontsize=11)
    ax.set_xlabel("Time", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    _style_time_ax(ax, minor=False)
    plt.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def make_plots(fine, actual_10min, hourly, actual_hourly):
    """
    Generate and save all four forecast plots:
        arrivals_10min.png
        discharges_10min.png
        arrivals_hourly.png
        discharges_hourly.png
    """
    cfg = [
        ("arrivals",   "steelblue"),
        ("discharges", "darkorange"),
    ]
    for flow, colour in cfg:
        plot_10min(
            fine, actual_10min, flow, colour,
            PLOT_DIR / f"{flow}_10min.png"
        )
        plot_hourly(
            hourly, actual_hourly, flow, colour,
            PLOT_DIR / f"{flow}_hourly.png"
        )


# ====================================================================
# MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":

    print(
        f"\nRunning simulation at {STEP_MINUTES}-min resolution "
        f"({FORECAST_STEPS} steps × {NUM_RUNS} runs)…"
    )

    # ------------------------------------------------------------------
    # STEP 1: Run the stochastic simulation
    # ------------------------------------------------------------------
    sim = run_simulation()

    # ------------------------------------------------------------------
    # STEP 2: Build actual observed data arrays from the databases
    # ------------------------------------------------------------------

    # 10-minute resolution actual data
    step_range   = [FORECAST_TS + s * STEP_TD for s in range(FORECAST_STEPS)]
    actual_10min = pd.DataFrame({"datetime": step_range})
    actual_10min["actual_arrivals"]   = 0
    actual_10min["actual_discharges"] = 0

    # Hourly actual data
    hour_range    = [FORECAST_TS + timedelta(hours=h) for h in range(FORECAST_HOURS)]
    actual_hourly = pd.DataFrame({"datetime": hour_range})
    actual_hourly["actual_arrivals"]   = 0
    actual_hourly["actual_discharges"] = 0

    # Count actual arrivals and discharges from each database
    for db in Path(BASE_DB_DIR).rglob("*.mdb"):
        try:
            conn = pyodbc.connect(
                f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db};"
            )
            a = pd.read_sql(
                "SELECT entry_date FROM visits WHERE entry_group=1", conn
            )
            d = pd.read_sql(
                "SELECT exit_date FROM visits "
                "WHERE entry_group=1 AND exit_group=1", conn
            )
            conn.close()
            a["entry_date"] = pd.to_datetime(a["entry_date"])
            d["exit_date"]  = pd.to_datetime(d["exit_date"])

            # Bin into 10-minute slots
            for s, dt in enumerate(step_range):
                nxt = dt + STEP_TD
                actual_10min.loc[s, "actual_arrivals"]   += (
                    a[(a.entry_date >= dt) & (a.entry_date < nxt)].shape[0]
                )
                actual_10min.loc[s, "actual_discharges"] += (
                    d[(d.exit_date >= dt) & (d.exit_date < nxt)].shape[0]
                )

            # Bin into hourly slots
            for h, dt in enumerate(hour_range):
                nxt = dt + timedelta(hours=1)
                actual_hourly.loc[h, "actual_arrivals"]   += (
                    a[(a.entry_date >= dt) & (a.entry_date < nxt)].shape[0]
                )
                actual_hourly.loc[h, "actual_discharges"] += (
                    d[(d.exit_date >= dt) & (d.exit_date < nxt)].shape[0]
                )
        except:
            continue

    # ------------------------------------------------------------------
    # STEP 3: Aggregate forecast statistics for plots and export CSVs
    # ------------------------------------------------------------------
    # NOTE: MEAN is used here (not median) to avoid the median=0
    # collapse problem in low-count 10-min slots. The column is named
    # '_median' for compatibility with the plotting functions but
    # contains the mean. See design notes in the module docstring.
    # ------------------------------------------------------------------

    fine_plot = sim.groupby("datetime").agg(
        arrivals_median  =("arrivals",   "mean"),   # mean, not median
        arrivals_l       =("arrivals",   lambda x: np.percentile(x, 2.5)),
        arrivals_u       =("arrivals",   lambda x: np.percentile(x, 97.5)),
        discharges_median=("discharges", "mean"),   # mean, not median
        discharges_l     =("discharges", lambda x: np.percentile(x, 2.5)),
        discharges_u     =("discharges", lambda x: np.percentile(x, 97.5)),
    ).reset_index()

    # ------------------------------------------------------------------
    # STEP 4: Compute CI coverage and aggregate validation metrics
    # ------------------------------------------------------------------

    # CI coverage: proportion of actual 10-min values within the 95% CI
    fine_plot_merged = fine_plot.merge(
        actual_10min[["datetime", "actual_arrivals", "actual_discharges"]],
        on="datetime", how="left"
    )
    ci_coverage_percent = {
        "arrivals": (
            (fine_plot_merged["arrivals_l"]   <= fine_plot_merged["actual_arrivals"]) &
            (fine_plot_merged["actual_arrivals"]   <= fine_plot_merged["arrivals_u"])
        ).mean() * 100,
        "discharges": (
            (fine_plot_merged["discharges_l"] <= fine_plot_merged["actual_discharges"]) &
            (fine_plot_merged["actual_discharges"] <= fine_plot_merged["discharges_u"])
        ).mean() * 100,
    }

    # 12-hour aggregate totals across all replications
    agg         = sim.groupby("run")[["arrivals", "discharges"]].sum()
    actuals_agg = actual_10min[["actual_arrivals", "actual_discharges"]].sum()

    # ------------------------------------------------------------------
    # STEP 5: Compute DTW on TRUE MEDIAN forecast vs actual
    # ------------------------------------------------------------------
    # DTW uses the true statistical MEDIAN across replications (not
    # the mean used for plotting). The median is the correct central
    # tendency for DTW validation — it represents the most typical
    # simulation outcome, not the average which can be pulled by
    # extreme replications.
    # ------------------------------------------------------------------

    fine_median = sim.groupby("datetime").agg(
        arrivals_median  =("arrivals",   "median"),   # True median for DTW
        discharges_median=("discharges", "median"),
    ).reset_index()

    dtw_arr = round(dtw_similarity(
        actual_10min["actual_arrivals"].values,
        fine_median["arrivals_median"].values
    ), 3)
    dtw_dis = round(dtw_similarity(
        actual_10min["actual_discharges"].values,
        fine_median["discharges_median"].values
    ), 3)

    # ------------------------------------------------------------------
    # STEP 6: Save aggregate validation metrics CSV
    # ------------------------------------------------------------------

    metrics = pd.DataFrame([{
        "MAPE_arrivals":              mape_aggregate(
                                          agg["arrivals"].median(),
                                          actuals_agg.actual_arrivals),
        "MAPE_discharges":            mape_aggregate(
                                          agg["discharges"].median(),
                                          actuals_agg.actual_discharges),
        "CI_coverage_arrivals_pct":   ci_coverage_percent["arrivals"],
        "CI_coverage_discharges_pct": ci_coverage_percent["discharges"],
        "DTW_arrivals":               dtw_arr,
        "DTW_discharges":             dtw_dis,
    }])
    metrics.to_csv(OUTPUT_DIR / "validation_12h.csv", index=False)
    print(f"Validation CSV saved: {OUTPUT_DIR / 'validation_12h.csv'}")

    # ------------------------------------------------------------------
    # STEP 7: Save actual vs simulated time-series CSVs
    # ------------------------------------------------------------------

    # Arrivals: actual vs forecast mean + 95% CI per 10-min step
    arrivals_export = pd.DataFrame({
        "datetime":       fine_plot["datetime"],
        "actual":         actual_10min["actual_arrivals"].values,
        "simulated_mean": np.round(fine_plot["arrivals_median"].values, 2),
        "sim_ci_lower":   np.round(fine_plot["arrivals_l"].values,      2),
        "sim_ci_upper":   np.round(fine_plot["arrivals_u"].values,      2),
    })
    arrivals_export.to_csv(
        OUTPUT_DIR / "dt_actual_vs_simulated_arrivals_10min.csv", index=False
    )
    print(f"Arrivals CSV saved:   "
          f"{OUTPUT_DIR / 'dt_actual_vs_simulated_arrivals_10min.csv'}")

    # Discharges: actual vs forecast mean + 95% CI per 10-min step
    discharges_export = pd.DataFrame({
        "datetime":       fine_plot["datetime"],
        "actual":         actual_10min["actual_discharges"].values,
        "simulated_mean": np.round(fine_plot["discharges_median"].values, 2),
        "sim_ci_lower":   np.round(fine_plot["discharges_l"].values,      2),
        "sim_ci_upper":   np.round(fine_plot["discharges_u"].values,      2),
    })
    discharges_export.to_csv(
        OUTPUT_DIR / "dt_actual_vs_simulated_discharges_10min.csv", index=False
    )
    print(f"Discharges CSV saved: "
          f"{OUTPUT_DIR / 'dt_actual_vs_simulated_discharges_10min.csv'}")

    # ------------------------------------------------------------------
    # STEP 8: Aggregate to hourly resolution for hourly plots
    # ------------------------------------------------------------------
    # Sum all 10-min steps within each hour across all replications,
    # then compute the mean and 95% CI across replications per hour.
    # ------------------------------------------------------------------

    sim["hour_dt"] = sim["datetime"].dt.floor("H")
    hourly_sim = sim.groupby(["run", "hour_dt"]).agg(
        arrivals  =("arrivals",   "sum"),
        discharges=("discharges", "sum"),
    ).reset_index()

    hourly = hourly_sim.groupby("hour_dt").agg(
        arrivals_median  =("arrivals",   "mean"),   # mean for plotting
        arrivals_l       =("arrivals",   lambda x: np.percentile(x, 2.5)),
        arrivals_u       =("arrivals",   lambda x: np.percentile(x, 97.5)),
        discharges_median=("discharges", "mean"),   # mean for plotting
        discharges_l     =("discharges", lambda x: np.percentile(x, 2.5)),
        discharges_u     =("discharges", lambda x: np.percentile(x, 97.5)),
    ).reset_index().rename(columns={"hour_dt": "datetime"})

    # ------------------------------------------------------------------
    # STEP 9: Generate and save all four plots
    # ------------------------------------------------------------------
    print("\nGenerating plots…")
    make_plots(fine_plot, actual_10min, hourly, actual_hourly)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Plots saved to:   {PLOT_DIR}")
