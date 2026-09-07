"""
Live Data Fetcher for German Electricity Market (Bundesnetzagentur SMARD.de API)
Author: Ritik Ghoghari
Project: German Energy Price Forecasting
"""

import requests
import concurrent.futures
import pandas as pd
import numpy as np
import os
from datetime import datetime

# SMARD.de API configuration
SMARD_SERIES_CONFIG = {
    'price': {'filter': '4169', 'region': 'DE-LU', 'name': 'Day_Ahead_Price'},
    'total_load': {'filter': '410', 'region': 'DE', 'name': 'Total_Load'},
    'solar': {'filter': '4068', 'region': 'DE', 'name': 'Solar'},
    'wind_onshore': {'filter': '4067', 'region': 'DE', 'name': 'Wind_Onshore'},
    'wind_offshore': {'filter': '4066', 'region': 'DE', 'name': 'Wind_Offshore'}
}

BASE_URL = "https://www.smard.de/app/chart_data"

def fetch_chunk(filter_id: str, region: str, resolution: str, timestamp_ms: int) -> list:
    """Fetches a single weekly data chunk from SMARD API."""
    url = f"{BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp_ms}.json"
    headers = {'User-Agent': 'German-Energy-Price-Forecaster/1.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get('series', [])
    except Exception as e:
        pass
    return []

def fetch_smard_series(filter_id: str, region: str, resolution: str = "hour", n_weeks: int = 12) -> pd.DataFrame:
    """
    Fetches the latest n_weeks of hourly data for a specific SMARD filter.
    Returns a DataFrame indexed by UTC datetime.
    """
    index_url = f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    try:
        idx_resp = requests.get(index_url, timeout=10)
        if idx_resp.status_code != 200:
            return pd.DataFrame()
        timestamps = idx_resp.json().get('timestamps', [])
    except Exception:
        return pd.DataFrame()

    if not timestamps:
        return pd.DataFrame()

    selected_timestamps = timestamps[-n_weeks:]
    all_data = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_chunk, filter_id, region, resolution, ts) for ts in selected_timestamps]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                all_data.extend(res)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp_ms', 'value'])
    df = df.dropna(subset=['value'])
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df = df.set_index('timestamp')
    return df[['value']]

def fetch_live_market_data(n_weeks: int = 12) -> pd.DataFrame:
    """
    Fetches all 5 key market series from SMARD.de in parallel,
    merges them into a single hourly DataFrame.
    """
    series_dfs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(
                fetch_smard_series,
                cfg['filter'],
                cfg['region'],
                'hour',
                n_weeks
            ): key for key, cfg in SMARD_SERIES_CONFIG.items()
        }
        for future in concurrent.futures.as_completed(future_map):
            key = future_map[future]
            try:
                series_dfs[key] = future.result()
            except Exception:
                series_dfs[key] = pd.DataFrame()

    # Verify all series fetched
    valid_dfs = [df for df in series_dfs.values() if not df.empty]
    if len(valid_dfs) < len(SMARD_SERIES_CONFIG):
        # Return fallback or empty
        return pd.DataFrame()

    # Rename columns
    for key, cfg in SMARD_SERIES_CONFIG.items():
        if key in series_dfs and not series_dfs[key].empty:
            series_dfs[key].columns = [cfg['name']]

    # Inner join on timestamp index
    merged = None
    for key, cfg in SMARD_SERIES_CONFIG.items():
        col_df = series_dfs[key]
        if merged is None:
            merged = col_df
        else:
            merged = merged.join(col_df, how='inner')

    merged = merged.sort_index()
    return merged

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs fundamental energy, calendar, and autoregressive lag features
    strictly respecting the day-ahead forecasting constraints (>24h offset).
    """
    df = df.copy()

    # 1. Fundamental Energy Features
    if 'Solar' in df.columns and 'Wind_Onshore' in df.columns and 'Wind_Offshore' in df.columns:
        df['Renewable_Total'] = df['Solar'] + df['Wind_Onshore'] + df['Wind_Offshore']
    else:
        df['Renewable_Total'] = 0.0

    if 'Total_Load' in df.columns:
        df['Residual_Load'] = df['Total_Load'] - df['Renewable_Total']
    else:
        df['Residual_Load'] = 0.0

    # 2. Calendar & Temporal Features
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.day_of_week
    df['Month'] = df.index.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # 3. Autoregressive Lags & Rolling Statistics
    target = 'Day_Ahead_Price'
    if target in df.columns:
        df['Lag_24'] = df[target].shift(24)
        df['Lag_48'] = df[target].shift(48)
        df['Lag_168'] = df[target].shift(168)
        df['Rolling_Mean_24'] = df[target].shift(24).rolling(window=24).mean()
        df['Rolling_Std_24'] = df[target].shift(24).rolling(window=24).std()

    return df

FEATURE_COLUMNS = [
    'Total_Load', 'Renewable_Total', 'Residual_Load',
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'Lag_24', 'Lag_48', 'Lag_168', 'Rolling_Mean_24', 'Rolling_Std_24'
]

if __name__ == "__main__":
    print("Fetching live German market data from SMARD.de API...")
    df_raw = fetch_live_market_data(n_weeks=4)
    print(f"Fetched raw rows: {len(df_raw)}")
    if not df_raw.empty:
        df_feat = engineer_features(df_raw).dropna()
        print(f"Engineered dataset rows: {len(df_feat)}")
        print(df_feat[FEATURE_COLUMNS].head())
