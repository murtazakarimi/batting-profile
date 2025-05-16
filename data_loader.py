import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
import pandas as pd
import pybaseball
from pybaseball import statcast, batting_stats, pitching_stats
from datetime import datetime

def load_baseball_data():
    """Load batting, pitching, and Statcast data for 2023-2025 seasons with daily cache refresh.

    Returns:
        tuple: (batter_df, pitcher_df, statcast_df) as Pandas DataFrames.
    """
    pybaseball.cache.enable()
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache file paths
    batter_file = f"{cache_dir}/batter_2025.parquet"
    pitcher_file = f"{cache_dir}/pitcher_2025.parquet"
    statcast_file = f"{cache_dir}/statcast_2023_2025.parquet"
    metadata_file = f"{cache_dir}/cache_metadata.json"
    
    # Get current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Check cache metadata for last refresh date
    refresh_cache = True
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            last_refresh = metadata.get('last_refresh', '')
            if last_refresh == current_date:
                refresh_cache = False
        except (json.JSONDecodeError, KeyError):
            print("Warning: Corrupted cache metadata. Forcing cache refresh.")
    
    # Load or fetch batting data
    if not refresh_cache and os.path.exists(batter_file):
        batter_df = pd.read_parquet(batter_file)
    else:
        try:
            batter_df = batting_stats(2025, end_season=None, league='all', qual=1, ind=1)
        except Exception as e:
            print(f"Warning: Failed to fetch 2025 batting stats: {e}. Using placeholder data.")
            batter_df = pd.DataFrame(columns=['IDfg', 'Name', 'Team', 'HR', 'ISO', 'Barrel%', 'HardHit%', 'HR/FB', 'wRC+', 'AVG', 'FB%', 'Pull%', 'K%', 'BB%'])
        batter_df.to_parquet(batter_file)
    
    # Load or fetch pitching data
    if not refresh_cache and os.path.exists(pitcher_file):
        pitcher_df = pd.read_parquet(pitcher_file)
    else:
        try:
            pitcher_df = pitching_stats(2025, end_season=None, league='all', qual=1, ind=1)
        except Exception as e:
            print(f"Warning: Failed to fetch 2025 pitching stats: {e}. Using placeholder data.")
            pitcher_df = pd.DataFrame(columns=['IDfg', 'Name', 'HR/9'])
        pitcher_df.to_parquet(pitcher_file)
    
    # Load or fetch Statcast data with additional columns
    if not refresh_cache and os.path.exists(statcast_file):
        statcast_df = pd.read_parquet(statcast_file)
    else:
        seasons = [
            ('2023-03-10', '2023-11-01'),
            ('2024-03-10', '2024-11-01'),
            ('2025-03-10', current_date)
        ]
        statcast_dfs = []
        required_columns = [
            'batter', 'pitcher', 'events', 'launch_speed', 'launch_angle', 'pitch_type',
            'game_date', 'home_team', 'player_name', 'stand', 'p_throws', 'game_pk',
            'at_bat_number', 'bb_type', 'woba_value', 'hit_distance_sc',
            'plate_x', 'plate_z', 'zone', 'description', 'sz_top', 'sz_bot',
            'release_speed', 'release_spin_rate', 'release_pos_x', 'release_pos_z',
            'pfx_x', 'pfx_z', 'hit_location', 'inning'
        ]
        for start_dt, end_dt in seasons:
            try:
                print(f"Fetching Statcast data for {start_dt} to {end_dt}")
                df = statcast(start_dt=start_dt, end_dt=end_dt, team=None)
                available_columns = [col for col in required_columns if col in df.columns]
                if len(available_columns) < len(required_columns):
                    missing = set(required_columns) - set(available_columns)
                    print(f"Warning: Missing Statcast columns for {start_dt} to {end_dt}: {missing}")
                df = df[available_columns]
                statcast_dfs.append(df)
            except Exception as e:
                print(f"Error fetching Statcast data for {start_dt} to {end_dt}: {e}")
        
        if not statcast_dfs:
            print("No Statcast data retrieved. Falling back to empty DataFrame.")
            statcast_df = pd.DataFrame()
        else:
            statcast_df = pd.concat(statcast_dfs, ignore_index=True)
            statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'])
            missing_batters = statcast_df['batter'].isna().sum()
            missing_pitches = statcast_df['pitch_type'].isna().sum()
            missing_events = statcast_df['events'].isna().sum()
            if missing_batters > 0 or missing_pitches > 0 or missing_events > 0:
                print(f"Warning: Statcast data issues - Missing batters: {missing_batters}, "
                      f"Missing pitch types: {missing_pitches}, Missing events: {missing_events}")
            for col in ['release_speed', 'release_spin_rate', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'inning']:
                if col not in statcast_df.columns:
                    print(f"Warning: Statcast column {col} not retrieved. Filling with default values.")
                    statcast_df[col] = np.nan if col != 'inning' else 1
            statcast_df.to_parquet(statcast_file)
    
    if refresh_cache:
        metadata = {'last_refresh': current_date}
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
    
    return batter_df, pitcher_df, statcast_df