# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:55:47 2026

@author: 40464988
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

NUM_RUNS       = 100
FORECAST_HOURS = 12
ED_PROBABILITY = 0.675

STEP_MINUTES   = 10
STEPS_PER_HOUR = 60 // STEP_MINUTES
FORECAST_STEPS = FORECAST_HOURS * STEPS_PER_HOUR  
STEP_TD        = timedelta(minutes=STEP_MINUTES)

BASE_DB_DIR       = r"C:\PhD\Pilot\ed_digital_twin\data\input"
ARRIVAL_RATES_CSV = "data/input/best_fits_arrivals.csv"

forecast_str = pd.Timestamp(FORECAST_DATE).strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR   = Path(f"data/output/{forecast_str}")
PLOT_DIR     = Path(f"data/plots_12h/{forecast_str}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_TS     = pd.Timestamp(FORECAST_DATE)
LOOKBACK_DAYS   = 60
LOOKBACK_START  = FORECAST_TS - pd.Timedelta(days=LOOKBACK_DAYS)
LOOKBACK_START2 = FORECAST_TS - pd.Timedelta(days=14)

db_paths = glob.glob(os.path.join(BASE_DB_DIR, "*", "*.mdb"))

los_param_month = (
    FORECAST_TS.replace(day=1) - pd.DateOffset(months=3)
).strftime("%Y-%m")

with open(
    f"data/input/parameters/los_parameters_weekday_{los_param_month}.json"
) as f:
    los_params = json.load(f)

weekday_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def sample_los(wd):
    p = los_params[weekday_map[wd]]
    k = np.random.choice(len(p["weights"]), p=p["weights"])
    return max(np.random.normal(p["means"][k], p["stds"][k]), 0.01)


def build_arrival_models():
    
    from scipy.optimize import brentq

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

    raw = pd.concat(all_dfs)
    raw["weekday"] = raw.entry_date.dt.weekday
    raw["hour"]    = raw.entry_date.dt.hour
    raw["week"]    = raw.entry_date.dt.to_period("W").apply(
        lambda r: r.start_time
    )

    weekly = (
        raw.groupby(["weekday", "hour", "week"])
        .size()
        .reset_index(name="count")
    )

    best_fit = pd.read_csv(ARRIVAL_RATES_CSV)
    best_type = {
        (int(r.weekday), int(r.hour)): r.best_continuous
        for _, r in best_fit.iterrows()
    }

    MIN_OBS = 5
    models  = {}

    for wd in range(7):
        for hr in range(24):
            sub  = weekly[(weekly.weekday == wd) & (weekly.hour == hr)]["count"]
            n    = len(sub)
            mean = sub.mean() if n > 0 else 0.1
            mean = max(mean, 0.1)
            var  = sub.var(ddof=1) if n > 1 else mean
            var  = max(var, 1e-6)
            std  = np.sqrt(var)

            dist = best_type.get((wd, hr), "poisson")

            # Gamma
            if dist == "gamma":
                if n >= MIN_OBS and var > 0:
                    a_fit     = max(mean ** 2 / var, 0.1)
                    scale_fit = max(var / mean, 0.01)
                else:
                    a_fit, scale_fit = 2.0, mean / 2.0
                models[(wd, hr)] = ("gamma", {"a": a_fit, "scale": scale_fit})

            # Weibull
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
                        c_fit = 1.4
                    scale_fit = max(mean / gamma_function(1 + 1 / c_fit), 0.01)
                else:
                    c_fit     = 1.4
                    scale_fit = mean / gamma_function(1 + 1 / 1.4)
                models[(wd, hr)] = ("weibull_min", {"c": c_fit, "scale": scale_fit})

            # Normal
            elif dist == "norm":
                std_fit = std if n >= MIN_OBS and std > 0 else max(np.sqrt(mean), 1.0)
                models[(wd, hr)] = ("norm", {"loc": mean, "scale": max(std_fit, 0.01)})

            # Poisson
            else:
                if n >= MIN_OBS and var > mean * 1.5:
                    a_fit     = max(mean ** 2 / var, 0.1)
                    scale_fit = max(var / mean, 0.01)
                    models[(wd, hr)] = ("gamma", {"a": a_fit, "scale": scale_fit})
                else:
                    models[(wd, hr)] = ("poisson", {"mu": mean})

    return models


arrival_models = build_arrival_models()


def sample_arrivals(wd, hr):
    """Sample hourly arrivals (kept for reference / diagnostics)."""
    model, p = arrival_models[(wd, hr)]
    if model == "gamma":
        return max(int(sp_gamma(a=p["a"], scale=p["scale"]).rvs()), 0)
    if model == "weibull_min":
        return max(int(weibull_min(c=p["c"], scale=p["scale"]).rvs()), 0)
    if model == "norm":
        lo = (0 - p["loc"]) / p["scale"]
        return max(int(truncnorm(lo, np.inf, loc=p["loc"], scale=p["scale"]).rvs()), 0)
    return np.random.poisson(p["mu"])


def sample_arrivals_10min(wd, hr):
    """
    Scaling rules (mean-preserving, shape-preserving):
      Poisson : mu_10   = mu / 6
      Gamma   : scale_10 = scale / 6   (shape 'a' unchanged)
      Weibull : scale_10 = scale / 6   (shape 'c' unchanged)
      Normal  : loc_10  = loc / 6,  scale_10 = scale / 6
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
        return max(int(truncnorm(lo, np.inf, loc=loc_s, scale=scale_s).rvs()), 0)

    # Poisson
    return np.random.poisson(p["mu"] / STEPS_PER_HOUR)


def load_inpatients():
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
            df = df[
                (df.entry_date <= FORECAST_TS) &
                (df.exit_date   >  FORECAST_TS)
            ]
            for _, r in df.iterrows():
                remaining = (r.exit_date - FORECAST_TS).total_seconds() / 3600
                if remaining > 0:
                    patients.append({"exit": FORECAST_TS + timedelta(hours=remaining)})
        except:
            continue
    return patients

def estimate_capacity():
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
            df = df[df.exit_date.dt.year != 9999]
            df = df[
                (df.entry_date < FORECAST_TS) &
                (df.exit_date   > LOOKBACK_START2)
            ]
            all_patients.append(df)
        except:
            continue

    hist  = pd.concat(all_patients)
    hours = pd.date_range(LOOKBACK_START2, FORECAST_TS, freq="H")
    census = [
        ((hist.entry_date <= h) & (hist.exit_date > h)).sum()
        for h in hours
    ]
    return int(np.percentile(census, 25)), int(np.percentile(census, 95))


MIN_CAPACITY, MAX_CAPACITY = estimate_capacity()
print(f"Estimated Capacity Range: {MIN_CAPACITY} – {MAX_CAPACITY}")

def run_simulation():
    
    all_runs = []

    for run in range(NUM_RUNS):
        np.random.seed(run)
        patients = load_inpatients()
        rows     = []

        for s in range(FORECAST_STEPS):
            now    = FORECAST_TS + s * STEP_TD
            wd, hr = now.weekday(), now.hour

            
            step_arrivals = sample_arrivals_10min(wd, hr)

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


def print_parameter_comparison(sample_slots=5):
    print("\n" + "=" * 65)
    print("ARRIVAL PARAMETER COMPARISON (Old Hardcoded vs New Fitted)")
    print("=" * 65)
    print(f"{'Slot':<20} {'Dist':<12} {'Old Shape':>12} {'New Shape':>12}  Change")
    print("-" * 65)

    count = 0
    for (wd, hr), (model, p) in arrival_models.items():
        if count >= sample_slots * 7:
            break
        if hr not in [8, 10, 12, 14, 16]:
            continue
        day_name = weekday_map[wd]
        if model == "gamma":
            old_shape = 2.0
            new_shape = round(p["a"], 3)
            changed   = "CHANGED" if abs(new_shape - old_shape) > 0.2 else "  same"
            print(f"{day_name} {hr:02d}:00  {model:<12} {old_shape:>12.3f} "
                  f"{new_shape:>12.3f}  {changed}")
        elif model == "weibull_min":
            old_shape = 1.4
            new_shape = round(p["c"], 3)
            changed   = "CHANGED" if abs(new_shape - old_shape) > 0.2 else "  same"
            print(f"{day_name} {hr:02d}:00  {model:<12} {old_shape:>12.3f} "
                  f"{new_shape:>12.3f}  {changed}")
        count += 1
    print("=" * 65)


def mape(forecast, actual):
    """
    MAPE = (1/n) * Σ |F_t - A_t| / A_t * 100

    """
    f = np.array(forecast, dtype=float)
    a = np.array(actual,   dtype=float)
    mask = a > 0                          
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(f[mask] - a[mask]) / a[mask]) * 100


def mape_aggregate(f_total, a_total):
    """
    |sum(F) - sum(A)| / sum(A) * 100
    """
    return np.abs(f_total - a_total) / a_total * 100 if a_total > 0 else np.nan


def smape(forecast, actual):
    """

    sMAPE = (1/n) * Σ 2*|F_t - A_t| / (|F_t| + |A_t|) * 100
    """
    f = np.array(forecast, dtype=float)
    a = np.array(actual,   dtype=float)
    return np.mean(2 * np.abs(f - a) / (np.abs(f) + np.abs(a) + 1e-6)) * 100


def smape_aggregate(f_total, a_total):
    
    return 2 * np.abs(f_total - a_total) / (np.abs(f_total) + np.abs(a_total) + 1e-6) * 100


def mae(forecast, actual):
    
    f = np.array(forecast, dtype=float)
    a = np.array(actual,   dtype=float)
    return np.mean(np.abs(f - a))


def rmse(forecast, actual):
    
    f = np.array(forecast, dtype=float)
    a = np.array(actual,   dtype=float)
    return np.sqrt(np.mean((f - a) ** 2))

def dtw_similarity(v1, v2, clip_percentile=95):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    l1, l2 = len(v1), len(v2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    combined   = np.concatenate([v1, v2])
    cap        = max(np.percentile(combined, clip_percentile), 1e-6)
    v1         = np.clip(v1, 0, cap)
    v2         = np.clip(v2, 0, cap)
    global_max = max(v1.max(), v2.max(), 1e-6)
    v1_bar     = v1 / global_max
    v2_bar     = v2 / global_max
    dtw_dist   = dtw_distance.distance(v1_bar, v2_bar, use_pruning=True)
    return max(1.0 - dtw_dist / max(l1, l2), 0.0)


def _style_time_ax(ax, minor=True):
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    if minor:
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.25)


def plot_10min(fine, actual_10min, flow, colour, fname):
    
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(
        f"{flow.capitalize()}  —  12h Forecast vs Actual  |  10-min resolution\n"
        f"{forecast_str}",
        fontsize=12, fontweight="bold"
    )

    t = actual_10min["datetime"]

    ax.fill_between(
        t, fine[f"{flow}_l"], fine[f"{flow}_u"],
        color=colour, alpha=0.20, label="95% CI"
    )
    ax.plot(
        t, fine[f"{flow}_median"],
        color=colour, linewidth=1.8, label="Forecast median"
    )
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


if __name__ == "__main__":

    print_parameter_comparison()
    print(f"\nRunning simulation at {STEP_MINUTES}-min resolution "
          f"({FORECAST_STEPS} steps × {NUM_RUNS} runs)…")

    sim = run_simulation()

    
    step_range   = [FORECAST_TS + s * STEP_TD for s in range(FORECAST_STEPS)]
    actual_10min = pd.DataFrame({"datetime": step_range})
    actual_10min["actual_arrivals"]   = 0
    actual_10min["actual_discharges"] = 0

   
    hour_range    = [FORECAST_TS + timedelta(hours=h) for h in range(FORECAST_HOURS)]
    actual_hourly = pd.DataFrame({"datetime": hour_range})
    actual_hourly["actual_arrivals"]   = 0
    actual_hourly["actual_discharges"] = 0

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

           
            for s, dt in enumerate(step_range):
                nxt = dt + STEP_TD
                actual_10min.loc[s, "actual_arrivals"]   += (
                    a[(a.entry_date >= dt) & (a.entry_date < nxt)].shape[0]
                )
                actual_10min.loc[s, "actual_discharges"] += (
                    d[(d.exit_date >= dt) & (d.exit_date < nxt)].shape[0]
                )

           
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

    
    per_run_metrics = []

    for run_id in sim["run"].unique():
        run_df = sim[sim["run"] == run_id].sort_values("datetime")

        forecast_arr = run_df["arrivals"].values
        forecast_dis = run_df["discharges"].values
        actual_arr   = actual_10min["actual_arrivals"].values
        actual_dis   = actual_10min["actual_discharges"].values

        per_run_metrics.append({
            "run":                       run_id,
            "MAE_arrivals":              mae(forecast_arr, actual_arr),
            "MAE_discharges":            mae(forecast_dis, actual_dis),
            "RMSE_arrivals":             rmse(forecast_arr, actual_arr),
            "RMSE_discharges":           rmse(forecast_dis, actual_dis),
       
            "MAPE_arrivals":             mape(forecast_arr, actual_arr),
            "MAPE_discharges":           mape(forecast_dis, actual_dis),
            
            "SMAPE_arrivals":            smape(forecast_arr, actual_arr),
            "SMAPE_discharges":          smape(forecast_dis, actual_dis),
            
            "AggregatePE_arrivals_12h":  mape_aggregate(forecast_arr.sum(), actual_arr.sum()),
            "AggregatePE_discharges_12h":mape_aggregate(forecast_dis.sum(), actual_dis.sum()),
            "DTW_arrivals":              dtw_similarity(actual_arr, forecast_arr),
            "DTW_discharges":            dtw_similarity(actual_dis, forecast_dis),
        })

    per_run_df = pd.DataFrame(per_run_metrics)
    per_run_df.to_csv(
        OUTPUT_DIR / "dt_fixed_validation_per_replication_10min.csv",
        index=False
    )

   
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
        val = per_run_df[col].mean()
        if "DTW" in col:
            note = (
                "  good "     if val > 0.7 else
                "  moderate" if val > 0.4 else
                "  poor "
            )
        else:
            note = ""
        print(f"  {labels[col]:<36}: {val:.4f}{note}")

    
    fine = sim.groupby("datetime").agg(
        arrivals_median  =("arrivals",   "median"),
        arrivals_l       =("arrivals",   lambda x: np.percentile(x, 2.5)),
        arrivals_u       =("arrivals",   lambda x: np.percentile(x, 97.5)),
        discharges_median=("discharges", "median"),
        discharges_l     =("discharges", lambda x: np.percentile(x, 2.5)),
        discharges_u     =("discharges", lambda x: np.percentile(x, 97.5)),
    ).reset_index()

    
    fine = fine.merge(
        actual_10min[["datetime", "actual_arrivals", "actual_discharges"]],
        on="datetime", how="left"
    )
    fine["arrivals_ci_cover"]   = (
        (fine["arrivals_l"]    <= fine["actual_arrivals"])   &
        (fine["actual_arrivals"]   <= fine["arrivals_u"])
    )
    fine["discharges_ci_cover"] = (
        (fine["discharges_l"]  <= fine["actual_discharges"]) &
        (fine["actual_discharges"] <= fine["discharges_u"])
    )
    ci_coverage_percent = {
        "arrivals":   fine["arrivals_ci_cover"].mean()   * 100,
        "discharges": fine["discharges_ci_cover"].mean() * 100,
    }

    print(f"\n  CI coverage (arrivals)  : {ci_coverage_percent['arrivals']:.1f}%")
    print(f"  CI coverage (discharges): {ci_coverage_percent['discharges']:.1f}%")

   
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

   
    arrivals_export = pd.DataFrame({
        "datetime":              fine["datetime"],
        "actual":                fine["actual_arrivals"],
        "simulated_mean":        np.round(
                                     sim.groupby("datetime")["arrivals"]
                                     .mean().values, 2),
        "simulated_median":      np.round(fine["arrivals_median"].values, 2),
        "sim_ci_lower":          np.round(fine["arrivals_l"].values,      2),
        "sim_ci_upper":          np.round(fine["arrivals_u"].values,      2),
    })
    arrivals_export.to_csv(
        OUTPUT_DIR / "dt_actual_vs_simulated_arrivals_10min.csv", index=False
    )
    print(f"Arrivals CSV saved:   {OUTPUT_DIR / 'dt_actual_vs_simulated_arrivals_10min.csv'}")

    discharges_export = pd.DataFrame({
        "datetime":              fine["datetime"],
        "actual":                fine["actual_discharges"],
        "simulated_mean":        np.round(
                                     sim.groupby("datetime")["discharges"]
                                     .mean().values, 2),
        "simulated_median":      np.round(fine["discharges_median"].values, 2),
        "sim_ci_lower":          np.round(fine["discharges_l"].values,      2),
        "sim_ci_upper":          np.round(fine["discharges_u"].values,      2),
    })
    discharges_export.to_csv(
        OUTPUT_DIR / "dt_actual_vs_simulated_discharges_10min.csv", index=False
    )
    print(f"Discharges CSV saved: {OUTPUT_DIR / 'dt_actual_vs_simulated_discharges_10min.csv'}")

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

    print("\nGenerating plots…")
    make_plots(fine_plot, actual_10min, hourly, actual_hourly)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Plots saved to:   {PLOT_DIR}")
    print(f"\nNext step: re-run your statistical comparison with this fixed DT.")