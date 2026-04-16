# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:54:07 2026

@author: 40464988
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

base_db_dir = r"C:\PhD\Pilot\ed_digital_twin\data\input"
db_paths = glob.glob(os.path.join(base_db_dir, "*", "*.mdb"))

all_los_records = []

forecast_ts = pd.Timestamp(FORECAST_DATE)
start_ts = forecast_ts - pd.Timedelta(weeks=2)

for db_path in db_paths:
    if not os.path.exists(db_path):
        continue
    try:
        conn_str = (
            r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
            f"DBQ={db_path};"
        )
        with pyodbc.connect(conn_str) as conn:
            df = pd.read_sql(
                "SELECT entry_date, exit_date FROM visits WHERE entry_group = 1 AND exit_group = 1;",
                conn
            )
            df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
            df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')
            df = df.dropna(subset=['entry_date','exit_date'])

            df = df[(df['entry_date'] >= start_ts) & (df['entry_date'] < forecast_ts)]

            df['los_hours'] = (df['exit_date'] - df['entry_date']).dt.total_seconds() / 3600
            df = df[df['los_hours'] > 0]
            all_los_records.append(df)
    except Exception as e:
        print(f"Error reading {db_path}: {e}")

if len(all_los_records) == 0:
    raise ValueError("No LOS data found in the last 2 weeks!")

all_los_df = pd.concat(all_los_records)
all_los_df['weekday'] = all_los_df['entry_date'].dt.dayofweek  # 0=Monday
weekday_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']


def gmm_cdf(x, gmm):
    cdf_vals = np.zeros_like(x, dtype=float)
    for weight, mean, cov in zip(gmm.weights_.flatten(),
                                 gmm.means_.flatten(),
                                 gmm.covariances_.flatten()):
        std = np.sqrt(cov)
        cdf_vals += weight * stats.norm.cdf(x, loc=mean, scale=std)
    return cdf_vals


weekday_gmm_params = {}

for wd in range(7):
    los_data = all_los_df[all_los_df['weekday'] == wd]['los_hours'].values
    
    if len(los_data) == 0:
        continue

    X = los_data.reshape(-1, 1)   

    
    los_sorted = np.sort(los_data)
    empirical_cdf = np.arange(1, len(los_sorted)+1) / len(los_sorted)

    best_gmm = None

    for k in range(1, 11):
        gmm_k = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm_k.fit(X)

        model_cdf = gmm_cdf(los_sorted, gmm_k)

        ks_stat = np.max(np.abs(empirical_cdf - model_cdf))
        p_val = stats.kstwo.sf(ks_stat, len(los_sorted))

        if p_val > 0.05:
            best_gmm = gmm_k
            best_k = k
            break

    if best_gmm is None:
        best_k = 10
        best_gmm = GaussianMixture(n_components=10, random_state=42)
        best_gmm.fit(X)
   
    weekday_gmm_params[weekday_names[wd]] = {
        "k": best_k,
        "weights": best_gmm.weights_.tolist(),
        "means": best_gmm.means_.flatten().tolist(),
        "stds": np.sqrt(best_gmm.covariances_.flatten()).tolist()
    }


los_param_month = (forecast_ts.replace(day=1)-pd.DateOffset(months=3)).strftime("%Y-%m")
output_path = f"data/input/parameters/los_parameters_weekday_{los_param_month}.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path,"w") as f:
    json.dump(weekday_gmm_params, f, indent=4)

print(f"Weekday-specific LOS GMM parameters (last 2 weeks) saved to:\n{output_path}")
