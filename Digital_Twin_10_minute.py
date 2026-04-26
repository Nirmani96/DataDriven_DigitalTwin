# -*- coding: utf-8 -*-
"""
Digital_Twin_10_minute.py
--------------------------
Database-Driven Digital Twin of an Emergency Department.

This is the main simulation script. It builds a stochastic digital twin
of an ED that forecasts patient arrivals and discharges over a 12-hour
horizon at 10-minute resolution.

HOW IT WORKS:
    1. Reads historical visit data from .mdb databases to:
       - Fit arrival rate distributions per weekday/hour
       - Load patients currently in the ED at the forecast time
       - Estimate ED capacity bounds from recent census data
    2. Runs NUM_RUNS stochastic replications of the 12-hour simulation,
       each with a different random seed
    3. At each 10-minute step:
       - Samples new arrivals from the fitted distribution
       - Assigns each new patient a length-of-stay (from LOS GMM)
       - Applies a capacity adjustment factor to discharge timing
       - Records arrivals, discharges, and census count
    4. Validates simulated results against actual observed data
    5. Saves validation metrics, CSVs, and plots

AUTHORS:
    Nirmani Amarasinghe  (ORCID: 0009-0001-9719-6366)
    Laura Boyle          (ORCID: 0000-0001-9651-1363)
    Adele H. Marshall    (ORCID: 0000-0001-5306-2756)

    Mathematical Science Research Centre, Queen's University Belfast

INPUTS:
    - config.py                          (FORECAST_DATE)
    - data/input/<subdir>/*.mdb          (visit records)
    - data/input/best_fits_arrivals.csv  (best distribution per weekday/hour)
    - data/input/parameters/los_parameters_weekday_<YYYY-MM>.json

OUTPUTS:
    - data/output/<forecast_datetime>/
        dt_fixed_validation_per_replication_10min.csv
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
# These control the core behaviour of the simulation.
# Adjust these if you want to change the forecast horizon, resolution,
# or number of replications.
# ====================================================================

NUM_RUNS       = 100    # Number of stochastic replications
FORECAST_HOURS = 12     # Forecast horizon in hours
ED_PROBABILITY = 0.675  # Probability a sampled arrival enters the ED pathway
                        # (remaining 0.325 are assumed to go elsewhere)

# Time resolution settings — 10-minute steps
STEP_MINUTES   = 10
STEPS_PER_HOUR = 60 // STEP_MINUTES   # = 6 steps per hour
FORECAST_STEPS = FORECAST_HOURS * STEPS_PER_HOUR  # = 72 steps total
STEP_TD        = timedelta(minutes=STEP_MINUTES)

# ====================================================================
# FILE PATHS
# ====================================================================

BASE_DB_DIR       = Path(__file__).resolve().parent / "data" / "input"
BASE_DIR          = Path(__file__).resolve().parent
ARRIVAL_RATES_CSV = BASE_DIR / "data" / "input" / "best_fits_arrivals.csv"

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

# ====================================================================
# DATABASE DISCOVERY
# ====================================================================
# Find all .mdb files recursively under the input data directory

db_paths = glob.glob(os.path.join(BASE_DB_DIR, "*", "*.mdb"))

# ====================================================================
# LOAD LOS PARAMETERS
# ====================================================================
# Load the pre-fitted GMM parameters generated by LOS.py.
# The filename encodes the parameter month (3 months before forecast)
# to allow consistent loading across different forecast periods.
# ====================================================================

los_param_month = (
    FORECAST_TS.replace(day=1) - pd.DateOffset(months=3)
).strftime("%Y-%m")

with open(
    f"data/input/parameters/los_parameters_weekday_{los_param_month}.json"
) as f:
    los_params = json.load(f)

# Map integer weekday (0=Mon) to the string keys used in the JSON file
weekday_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def sample_los(wd):
    """
    Sample a patient length-of-stay (in hours) from the fitted GMM
    for the given weekday.

    The GMM component is selected by sampling from the mixture weights,
    then a value is drawn from the corresponding Gaussian component.
    Negative samples are clipped to a minimum of 0.01 hours.

    Parameters
    ----------
    wd : int — weekday index (0=Monday, 6=Sunday)

    Returns
    -------
    float — sampled LOS in hours (always > 0)
    """
    p = los_params[weekday_map[wd]]
    # Select which GMM component to sample from (weighted random choice)
    k = np.random.choice(len(p["weights"]), p=p["weights"])
    # Sample from the selected Gaussian component
    return max(np.random.normal(p["means"][k], p["stds"][k]), 0.01)


# ====================================================================
# ARRIVAL MODEL FITTING
# ====================================================================

def build_arrival_models():
    """
    Fit statistical distributions to historical hourly arrival counts,
    separately for each combination of weekday (0-6) and hour (0-23).

    The best-fitting distribution type for each weekday/hour is read
    from best_fits_arrivals.csv (pre-computed externally). Supported
    distributions are: Gamma, Weibull, Normal, and Poisson.

    WHY NOT JUST POISSON?
        Poisson assumes the variance equals the mean. ED arrivals are
        often overdispersed (variance > mean), in which case Gamma or
        Negative Binomial fits better.

    Returns
    -------
    models : dict
        Keys are (weekday, hour) tuples.
        Values are (distribution_name, parameters_dict) tuples.
    """
    from scipy.optimize import brentq

    # Load all arrival records from the 60-day lookback window
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

    raw             = pd.concat(all_dfs)
    raw["weekday"]  = raw.entry_date.dt.weekday
    raw["hour"]     = raw.entry_date.dt.hour
    raw["week"]     = raw.entry_date.dt.to_period("W").apply(
        lambda r: r.start_time
    )

    # Aggregate to weekly counts per (weekday, hour) cell
    # This gives us the distribution of "how many arrivals in this
    # weekday/hour slot across different weeks"
    weekly = (
        raw.groupby(["weekday", "hour", "week"])
        .size()
        .reset_index(name="count")
    )

    # Load the pre-computed best distribution type per (weekday, hour)
    best_fit = pd.read_csv(ARRIVAL_RATES_CSV)
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
            # Method of moments: shape = mean²/var, scale = var/mean
            # ----------------------------------------------------------
            if dist == "gamma":
                if n >= MIN_OBS and var > 0:
                    a_fit     = max(mean ** 2 / var, 0.1)
                    scale_fit = max(var / mean, 0.01)
                else:
                    a_fit, scale_fit = 2.0, mean / 2.0
                models[(wd, hr)] = ("gamma", {"a": a_fit, "scale": scale_fit})

            # ----------------------------------------------------------
            # WEIBULL distribution
            # Shape parameter c is estimated by numerically solving the
            # coefficient of variation (CV) equation using Brent's method
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
                        c_fit = 1.4   # Fallback shape parameter
                    scale_fit = max(
                        mean / gamma_function(1 + 1 / c_fit), 0.01
                    )
                else:
                    c_fit     = 1.4
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
            # If overdispersion is detected (var > 1.5 * mean),
            # fall back to Gamma which handles overdispersion better
            # ----------------------------------------------------------
            else:
                if n >= MIN_OBS and var > mean * 1.5:
                    a_fit     = max(mean ** 2 / var, 0.1)
                    scale_fit = max(var / mean, 0.01)
                    models[(wd, hr)] = (
                        "gamma", {"a": a_fit, "scale": scale_fit}
                    )
                else:
                    models[(wd, hr)] = ("poisson", {"mu": mean})

    return models


# Build arrival models at module load time (used by both sampling functions)
arrival_models = build_arrival_models()


def sample_arrivals(wd, hr):
    """
    Sample hourly arrival count for a given weekday and hour.
    Kept for reference and diagnostics — the simulation uses
    sample_arrivals_10min() instead.

    Parameters
    ----------
    wd : int — weekday (0=Monday)
    hr : int — hour of day (0-23)

    Returns
    -------
    int — sampled number of arrivals in one hour
    """
    model, p = arrival_models[(wd, hr)]
    if model == "gamma":
        return max(int(sp_gamma(a=p["a"], scale=p["scale"]).rvs()), 0)
    if model == "weibull_min":
        return max(int(weibull_min(c=p["c"], scale=p["scale"]).rvs()), 0)
    if model == "norm":
        lo = (0 - p["loc"]) / p["scale"]
        return max(
            int(truncnorm(lo, np.inf, loc=p["loc"], scale=p["scale"]).rvs()),
            0
        )
    return np.random.poisson(p["mu"])


def sample_arrivals_10min(wd, hr):
    """
    Sample arrivals for a single 10-minute step.

    SCALING RATIONALE:
        The arrival models are fitted to hourly counts. To scale down
        to 10-minute intervals (1/6 of an hour) while preserving the
        distributional shape:
          - Poisson : mu_10min   = mu / 6
            (Poisson is closed under scaling: if X ~ Pois(mu) over 1h,
             then X_10min ~ Pois(mu/6))
          - Gamma   : scale_10min = scale / 6  (shape 'a' unchanged)
            (Gamma mean = a*scale, so dividing scale by 6 scales the mean)
          - Weibull : scale_10min = scale / 6  (shape 'c' unchanged)
          - Normal  : loc_10min  = loc/6, scale_10min = scale/6

    Parameters
    ----------
    wd : int — weekday (0=Monday)
    hr : int — hour of day (0-23)

    Returns
    -------
    int — sampled number of arrivals in one 10-minute step
    """
    model, p = arrival_models[(wd, hr)]

    if model == "gamma":
        return max(int(sp_gamma(
            a=p["a"],
            scale=p["scale"] / STEPS_PER_HOUR   # Scale mean down by 1/6
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

    # Poisson: closed under thinning
    return np.random.poisson(p["mu"] / STEPS_PER_HOUR)


# ====================================================================
# INITIAL STATE: PATIENTS CURRENTLY IN THE ED
# ====================================================================

def load_inpatients():
    """
    Load patients who are currently in the ED at the forecast time,
    i.e. they arrived before FORECAST_TS and have not yet been discharged.

    These patients are initialised into the simulation as 'in progress'
    with their remaining LOS calculated from their expected exit time.
    This ensures the simulation starts from a realistic state rather
    than an empty ED.

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

            # Select patients who arrived before and depart after the forecast time
            df = df[
                (df.entry_date <= FORECAST_TS) &
                (df.exit_date   >  FORECAST_TS)
            ]

            # Calculate remaining time in ED (hours from forecast time to exit)
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
    Estimate the operational capacity range of the ED from the last
    14 days of census data.

    The census at each hour is computed as the number of patients
    simultaneously present. The 25th and 95th percentiles of this
    hourly census distribution are used as:
      - MIN_CAPACITY (25th pct): typical low occupancy level
      - MAX_CAPACITY (95th pct): near-peak occupancy level

    WHY PERCENTILES?
        Using percentiles rather than min/max makes the estimates
        robust to outliers (e.g. data errors, unusually quiet nights).

    CAPACITY ADJUSTMENT IN SIMULATION:
        When census exceeds MAX_CAPACITY → factor = 1.2
            (patients stay slightly longer — ED is overcrowded)
        When census falls below MIN_CAPACITY → factor = 0.7
            (patients discharged slightly sooner — ED is quiet)
        Otherwise → factor = 1.0 (no adjustment)

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

            # Exclude records with implausible exit dates (year 9999 = missing)
            df = df[df.exit_date.dt.year != 9999]

            # Keep records overlapping the 14-day lookback window
            df = df[
                (df.entry_date < FORECAST_TS) &
                (df.exit_date   > LOOKBACK_START2)
            ]
            all_patients.append(df)
        except:
            continue

    hist  = pd.concat(all_patients)

    # Count patients present at each hour in the lookback window
    hours  = pd.date_range(LOOKBACK_START2, FORECAST_TS, freq="H")
    census = [
        ((hist.entry_date <= h) & (hist.exit_date > h)).sum()
        for h in hours
    ]

    return int(np.percentile(census, 25)), int(np.percentile(census, 95))


# Estimate capacity bounds once at startup
MIN_CAPACITY, MAX_CAPACITY = estimate_capacity()
print(f"Estimated Capacity Range: {MIN_CAPACITY} – {MAX_CAPACITY}")


# ====================================================================
# MAIN SIMULATION
# ====================================================================

def run_simulation():
    """
    Run NUM_RUNS stochastic replications of the 12-hour ED simulation.

    Each replication:
        1. Seeds the random number generator with the run index
           (ensures reproducibility and independence between runs)
        2. Initialises the patient list with currently-present patients
        3. Steps forward in 10-minute increments for FORECAST_STEPS steps
        4. At each step:
           a. Samples new arrivals from the fitted distribution
           b. Each arrival enters the ED with probability ED_PROBABILITY
           c. Each new ED patient is assigned a LOS from the GMM
           d. A capacity adjustment factor modifies discharge timing
           e. Patients whose adjusted exit time falls in this step
              are counted as discharges and removed from the list
        5. Records arrivals, discharges, and census at each step

    Returns
    -------
    pd.DataFrame with columns:
        datetime, arrivals, discharges, census, run
    """
    all_runs = []

    for run in range(NUM_RUNS):
        np.random.seed(run)   # Reproducible but independent replications
        patients = load_inpatients()
        rows     = []

        for s in range(FORECAST_STEPS):
            now    = FORECAST_TS + s * STEP_TD
            wd, hr = now.weekday(), now.hour

            # Sample new arrivals for this 10-minute step
            step_arrivals = sample_arrivals_10min(wd, hr)

            # Each arrival independently enters the ED with probability ED_PROBABILITY
            for _ in range(step_arrivals):
                if np.random.rand() < ED_PROBABILITY:
                    patients.append({
                        "exit": now + timedelta(hours=sample_los(wd))
                    })

            # Current census (number of patients in ED at start of step)
            census = len(patients)

            # Capacity adjustment factor:
            # Overcrowded → slower discharges (factor > 1 delays exit)
            # Undercrowded → faster discharges (factor < 1 brings exit forward)
            if census > MAX_CAPACITY:
                factor = 1.2
            elif census < MIN_CAPACITY:
                factor = 0.7
            else:
                factor = 1.0

            # Process discharges for this time step
            discharges = 0
            remaining  = []
            for p in patients:
                # Adjust exit time by capacity factor
                # (1 - factor) * 0.5h shifts exit: positive = delay, negative = early
                adj_exit = p["exit"] + timedelta(minutes=(1 - factor) * 0.5 * 60)

                # Discharge if adjusted exit falls within this 10-minute window
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

def mape(forecast, actual):
    """
    Mean Absolute Percentage Error (step-level).
    Only computed at steps where actual > 0 to avoid division by zero.

    Formula: (1/n) * Σ |F_t - A_t| / A_t * 100
    """
    f    = np.array(forecast, dtype=float)
    a    = np.array(actual,   dtype=float)
    mask = a > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(f[mask] - a[mask]) / a[mask]) * 100


def mape_aggregate(f_total, a_total):
    """
    Aggregate Percentage Error over the full 12-hour horizon.
    Compares total simulated count vs total actual count.

    Formula: |sum(F) - sum(A)| / sum(A) * 100
    """
    return (
        np.abs(f_total - a_total) / a_total * 100
        if a_total > 0 else np.nan
    )


def smape(forecast, actual):
    """
    Symmetric Mean Absolute Percentage Error (step-level).
    More balanced than MAPE when actual values are close to zero.

    Formula: (1/n) * Σ 2*|F_t - A_t| / (|F_t| + |A_t|) * 100
    """
    f = np.array(forecast, dtype=float)
    a = np.array(actual,   dtype=float)
    return np.mean(
        2 * np.abs(f - a) / (np.abs(f) + np.abs(a) + 1e-6)
    ) * 100


def smape_aggregate(f_total, a_total):
    """Aggregate sMAPE over the full 12-hour horizon."""
    return (
        2 * np.abs(f_total - a_total) /
        (np.abs(f_total) + np.abs(a_total) + 1e-6) * 100
    )


def mae(forecast, actual):
    """Mean Absolute Error — average absolute difference per time step."""
    f = np.array(forecast, dtype=float)
    a = np.array(actual,   dtype=float)
    return np.mean(np.abs(f - a))


def rmse(forecast, actual):
    """Root Mean Squared Error — penalises large errors more than MAE."""
    f = np.array(forecast, dtype=float)
    a = np.array(actual,   dtype=float)
    return np.sqrt(np.mean((f - a) ** 2))


def dtw_similarity(v1, v2, clip_percentile=95):
    """
    Dynamic Time Warping (DTW) similarity score in [0, 1].

    DTW measures similarity between two time series allowing for
    temporal shifts — useful when the simulated peaks occur slightly
    earlier or later than actual peaks.

    NORMALISATION:
        Raw DTW distance is normalised by series length and converted
        to a similarity score: 1 = perfect match, 0 = no similarity.
        Values above 0.7 are considered good; above 0.4 moderate.

    CLIPPING:
        Extreme outliers are clipped at the 95th percentile before
        computing DTW to prevent single spikes dominating the metric.

    Parameters
    ----------
    v1, v2          : array-like time series to compare
    clip_percentile : percentile for outlier clipping (default 95)

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

    # Clip outliers to the 95th percentile of the combined series
    combined   = np.concatenate([v1, v2])
    cap        = max(np.percentile(combined, clip_percentile), 1e-6)
    v1         = np.clip(v1, 0, cap)
    v2         = np.clip(v2, 0, cap)

    # Normalise both series to [0, 1] before computing DTW
    global_max = max(v1.max(), v2.max(), 1e-6)
    v1_bar     = v1 / global_max
    v2_bar     = v2 / global_max

    dtw_dist = dtw_distance.distance(v1_bar, v2_bar, use_pruning=True)

    # Convert distance to similarity: higher = more similar
    return max(1.0 - dtw_dist / max(l1, l2), 0.0)


# ====================================================================
# PLOTTING UTILITIES
# ====================================================================

def _style_time_ax(ax, minor=True):
    """
    Apply consistent time-axis formatting to a matplotlib Axes object.
    Shows hours on the major axis and optionally half-hours on the minor.
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
    Plot 10-minute resolution forecast vs actual for arrivals or discharges.

    Shows:
        - 95% confidence interval band (shaded)
        - Forecast median line
        - Actual observed values (dashed red)

    Parameters
    ----------
    fine        : DataFrame with forecast median and CI bounds
    actual_10min: DataFrame with actual observed counts
    flow        : 'arrivals' or 'discharges'
    colour      : line/fill colour for the forecast
    fname       : output file path
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(
        f"{flow.capitalize()}  —  12h Forecast vs Actual  |  10-min resolution\n"
        f"{forecast_str}",
        fontsize=12, fontweight="bold"
    )

    t = actual_10min["datetime"]

    # Shaded 95% confidence interval
    ax.fill_between(
        t, fine[f"{flow}_l"], fine[f"{flow}_u"],
        color=colour, alpha=0.20, label="95% CI"
    )
    # Forecast median
    ax.plot(
        t, fine[f"{flow}_median"],
        color=colour, linewidth=1.8, label="Forecast median"
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
    Plot hourly aggregated forecast vs actual for arrivals or discharges.

    Same layout as plot_10min but with data aggregated to hourly bins,
    giving a clearer view of the overall trend.

    Parameters
    ----------
    hourly       : DataFrame with hourly forecast median and CI bounds
    actual_hourly: DataFrame with hourly actual observed counts
    flow         : 'arrivals' or 'discharges'
    colour       : line/fill colour for the forecast
    fname        : output file path
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
        t, hourly[f"{flow}_median"],
        color=colour, linewidth=2.0, marker="o", markersize=5,
        label="Forecast median"
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
        arrivals_10min.png, arrivals_hourly.png,
        discharges_10min.png, discharges_hourly.png
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

    # ------------------------------------------------------------------
    # STEP 1: Run the stochastic simulation
    # ------------------------------------------------------------------
    sim = run_simulation()

    # ------------------------------------------------------------------
    # STEP 2: Build actual observed data arrays from the databases
    # Initialise with zeros, then populate from the .mdb files
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

    # Count actual arrivals and discharges from the database
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

            # Bin actual arrivals into 10-minute steps
            for s, dt in enumerate(step_range):
                nxt = dt + STEP_TD
                actual_10min.loc[s, "actual_arrivals"]   += (
                    a[(a.entry_date >= dt) & (a.entry_date < nxt)].shape[0]
                )
                actual_10min.loc[s, "actual_discharges"] += (
                    d[(d.exit_date >= dt) & (d.exit_date < nxt)].shape[0]
                )

            # Bin actual arrivals into hourly buckets
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
    # STEP 3: Compute per-replication validation metrics
    # ------------------------------------------------------------------
    per_run_metrics = []

    for run_id in sim["run"].unique():
        run_df = sim[sim["run"] == run_id].sort_values("datetime")

        forecast_arr = run_df["arrivals"].values
        forecast_dis = run_df["discharges"].values
        actual_arr   = actual_10min["actual_arrivals"].values
        actual_dis   = actual_10min["actual_discharges"].values

        per_run_metrics.append({
            "run":                       run_id,
            "MAE_arrivals":              mae(forecast_arr,  actual_arr),
            "MAE_discharges":            mae(forecast_dis,  actual_dis),
            "RMSE_arrivals":             rmse(forecast_arr, actual_arr),
            "RMSE_discharges":           rmse(forecast_dis, actual_dis),
            "MAPE_arrivals":             mape(forecast_arr, actual_arr),
            "MAPE_discharges":           mape(forecast_dis, actual_dis),
            "SMAPE_arrivals":            smape(forecast_arr, actual_arr),
            "SMAPE_discharges":          smape(forecast_dis, actual_dis),
            "AggregatePE_arrivals_12h":  mape_aggregate(
                                             forecast_arr.sum(), actual_arr.sum()),
            "AggregatePE_discharges_12h":mape_aggregate(
                                             forecast_dis.sum(), actual_dis.sum()),
            "DTW_arrivals":              dtw_similarity(actual_arr, forecast_arr),
            "DTW_discharges":            dtw_similarity(actual_dis, forecast_dis),
        })

    per_run_df = pd.DataFrame(per_run_metrics)
    per_run_df.to_csv(
        OUTPUT_DIR / "dt_fixed_validation_per_replication_10min.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # STEP 4: Print summary metrics across all replications
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("MEAN METRICS ACROSS REPLICATIONS  (12h @ 10-min)")
    print("=" * 55)
    summary_cols = [
        "MAE_arrivals",              "MAE_discharges",
        "RMSE_arrivals",             "RMSE_discharges",
        "MAPE_arrivals",             "MAPE_discharges",
        "SMAPE_arrivals",            "SMAPE_discharges",
        "AggregatePE_arrivals_12h",  "AggregatePE_discharges_12h",
        "DTW_arrivals",              "DTW_discharges",
    ]
    labels = {
        "MAE_arrivals":               "MAE arrivals (per 10-min)",
        "MAE_discharges":             "MAE discharges (per 10-min)",
        "RMSE_arrivals":              "RMSE arrivals (per 10-min)",
        "RMSE_discharges":            "RMSE discharges (per 10-min)",
        "MAPE_arrivals":              "MAPE arrivals (step-level %)",
        "MAPE_discharges":            "MAPE discharges (step-level %)",
        "SMAPE_arrivals":             "sMAPE arrivals (step-level %)",
        "SMAPE_discharges":           "sMAPE discharges (step-level %)",
        "AggregatePE_arrivals_12h":   "Aggregate PE arrivals 12h (%)",
        "AggregatePE_discharges_12h": "Aggregate PE discharges 12h (%)",
        "DTW_arrivals":               "DTW arrivals",
        "DTW_discharges":             "DTW discharges",
    }
    for col in summary_cols:
        val  = per_run_df[col].mean()
        note = ""
        if "DTW" in col:
            note = (
                "  good "     if val > 0.7 else
                "  moderate"  if val > 0.4 else
                "  poor "
            )
        print(f"  {labels[col]:<36}: {val:.4f}{note}")

    # ------------------------------------------------------------------
    # STEP 5: Compute aggregate statistics and CI coverage
    # ------------------------------------------------------------------

    # Aggregate forecast statistics across runs (median + 95% CI)
    fine = sim.groupby("datetime").agg(
        arrivals_median  =("arrivals",   "median"),
        arrivals_l       =("arrivals",   lambda x: np.percentile(x, 2.5)),
        arrivals_u       =("arrivals",   lambda x: np.percentile(x, 97.5)),
        discharges_median=("discharges", "median"),
        discharges_l     =("discharges", lambda x: np.percentile(x, 2.5)),
        discharges_u     =("discharges", lambda x: np.percentile(x, 97.5)),
    ).reset_index()

    # Merge actual data to check CI coverage
    fine = fine.merge(
        actual_10min[["datetime", "actual_arrivals", "actual_discharges"]],
        on="datetime", how="left"
    )

    # CI coverage: proportion of actual values falling within the 95% CI
    fine["arrivals_ci_cover"]   = (
        (fine["arrivals_l"]   <= fine["actual_arrivals"]) &
        (fine["actual_arrivals"]   <= fine["arrivals_u"])
    )
    fine["discharges_ci_cover"] = (
        (fine["discharges_l"] <= fine["actual_discharges"]) &
        (fine["actual_discharges"] <= fine["discharges_u"])
    )
    ci_coverage_percent = {
        "arrivals":   fine["arrivals_ci_cover"].mean()   * 100,
        "discharges": fine["discharges_ci_cover"].mean() * 100,
    }

    print(f"\n  CI coverage (arrivals)  : {ci_coverage_percent['arrivals']:.1f}%")
    print(f"  CI coverage (discharges): {ci_coverage_percent['discharges']:.1f}%")

    # 12-hour aggregate totals across runs
    agg     = sim.groupby("run")[["arrivals", "discharges"]].sum()
    summary = pd.Series({
        "arrivals_median":   agg["arrivals"].median(),
        "arrivals_l":        np.percentile(agg["arrivals"],   2.5),
        "arrivals_u":        np.percentile(agg["arrivals"],  97.5),
        "discharges_median": agg["discharges"].median(),
        "discharges_l":      np.percentile(agg["discharges"],  2.5),
        "discharges_u":      np.percentile(agg["discharges"], 97.5),
    })
    actuals_agg = actual_10min[["actual_arrivals", "actual_discharges"]].sum()

    # Save aggregate validation metrics to CSV
    metrics = pd.DataFrame([{
        "MAPE_arrivals":              mape_aggregate(
                                          summary.arrivals_median,
                                          actuals_agg.actual_arrivals),
        "MAPE_discharges":            mape_aggregate(
                                          summary.discharges_median,
                                          actuals_agg.actual_discharges),
        "SMAPE_arrivals":             smape_aggregate(
                                          summary.arrivals_median,
                                          actuals_agg.actual_arrivals),
        "SMAPE_discharges":           smape_aggregate(
                                          summary.discharges_median,
                                          actuals_agg.actual_discharges),
        "CI_coverage_arrivals_pct":   ci_coverage_percent["arrivals"],
        "CI_coverage_discharges_pct": ci_coverage_percent["discharges"],
    }])

    # Compute DTW similarity on the median forecast vs actual series
    dtw_arr = round(dtw_similarity(
        actual_10min["actual_arrivals"].values,
        fine["arrivals_median"].values
    ), 3)
    dtw_dis = round(dtw_similarity(
        actual_10min["actual_discharges"].values,
        fine["discharges_median"].values
    ), 3)

    print(f"\n DTW Similarity (median forecast vs actual, 10-min series):")
    print(f"  DTW Arrivals   : {dtw_arr:.3f}  ", end="")
    print(" good " if dtw_arr > 0.7 else ("moderate " if dtw_arr > 0.4 else "← poor ✘"))
    print(f"  DTW Discharges : {dtw_dis:.3f}  ", end="")
    print("good " if dtw_dis > 0.7 else (" moderate " if dtw_dis > 0.4 else "← poor ✘"))

    metrics["DTW_arrivals"]   = dtw_arr
    metrics["DTW_discharges"] = dtw_dis
    metrics.to_csv(OUTPUT_DIR / "validation_12h.csv", index=False)
    print(f"\nAggregate validation CSV saved: {OUTPUT_DIR / 'validation_12h.csv'}")

    # ------------------------------------------------------------------
    # STEP 6: Export detailed CSVs (actual vs simulated per time step)
    # ------------------------------------------------------------------

    arrivals_export = pd.DataFrame({
        "datetime":         fine["datetime"],
        "actual":           fine["actual_arrivals"],
        "simulated_mean":   np.round(
                                sim.groupby("datetime")["arrivals"]
                                .mean().values, 2),
        "simulated_median": np.round(fine["arrivals_median"].values, 2),
        "sim_ci_lower":     np.round(fine["arrivals_l"].values,      2),
        "sim_ci_upper":     np.round(fine["arrivals_u"].values,      2),
    })
    arrivals_export.to_csv(
        OUTPUT_DIR / "dt_actual_vs_simulated_arrivals_10min.csv", index=False
    )
    print(f"Arrivals CSV saved:   "
          f"{OUTPUT_DIR / 'dt_actual_vs_simulated_arrivals_10min.csv'}")

    discharges_export = pd.DataFrame({
        "datetime":         fine["datetime"],
        "actual":           fine["actual_discharges"],
        "simulated_mean":   np.round(
                                sim.groupby("datetime")["discharges"]
                                .mean().values, 2),
        "simulated_median": np.round(fine["discharges_median"].values, 2),
        "sim_ci_lower":     np.round(fine["discharges_l"].values,      2),
        "sim_ci_upper":     np.round(fine["discharges_u"].values,      2),
    })
    discharges_export.to_csv(
        OUTPUT_DIR / "dt_actual_vs_simulated_discharges_10min.csv", index=False
    )
    print(f"Discharges CSV saved: "
          f"{OUTPUT_DIR / 'dt_actual_vs_simulated_discharges_10min.csv'}")

    # ------------------------------------------------------------------
    # STEP 7: Aggregate to hourly resolution for plotting
    # ------------------------------------------------------------------

    fine_plot = sim.groupby("datetime").agg(
        arrivals_median  =("arrivals",   "mean"),
        arrivals_l       =("arrivals",   lambda x: np.percentile(x, 2.5)),
        arrivals_u       =("arrivals",   lambda x: np.percentile(x, 97.5)),
        discharges_median=("discharges", "mean"),
        discharges_l     =("discharges", lambda x: np.percentile(x, 2.5)),
        discharges_u     =("discharges", lambda x: np.percentile(x, 97.5)),
    ).reset_index()

    sim["hour_dt"] = sim["datetime"].dt.floor("H")
    hourly_sim = sim.groupby(["run", "hour_dt"]).agg(
        arrivals  =("arrivals",   "sum"),
        discharges=("discharges", "sum"),
    ).reset_index()

    hourly = hourly_sim.groupby("hour_dt").agg(
        arrivals_median  =("arrivals",   "mean"),
        arrivals_l       =("arrivals",   lambda x: np.percentile(x, 2.5)),
        arrivals_u       =("arrivals",   lambda x: np.percentile(x, 97.5)),
        discharges_median=("discharges", "mean"),
        discharges_l     =("discharges", lambda x: np.percentile(x, 2.5)),
        discharges_u     =("discharges", lambda x: np.percentile(x, 97.5)),
    ).reset_index().rename(columns={"hour_dt": "datetime"})

    # ------------------------------------------------------------------
    # STEP 8: Generate and save all plots
    # ------------------------------------------------------------------
    print("\nGenerating plots…")
    make_plots(fine_plot, actual_10min, hourly, actual_hourly)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Plots saved to:   {PLOT_DIR}")
    print(f"\nNext step: re-run your statistical comparison with this fixed DT.")
