import pandas as pd
import numpy as np
from datetime import datetime

def get_bullpen_matchup(statcast_df, pitcher_team, prediction_date='2025-05-15'):
    """Calculate batter HR probability and matchup score against the opposing team's bullpen."""
    statcast_df = statcast_df.copy()
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'])
    prediction_date = pd.to_datetime(prediction_date)
    
    # Filter to 2023-2025 data
    statcast_df = statcast_df[statcast_df['game_date'] >= '2023-01-01']
    
    # Identify relief pitchers (assuming non-starters, e.g., entering after 1st inning)
    if 'inning' not in statcast_df.columns:
        print(f"Warning: 'inning' column missing in statcast_df for {pitcher_team}. Using all pitchers as fallback.")
        relief_pitches = statcast_df[statcast_df['home_team'] == pitcher_team]
    else:
        relief_pitches = statcast_df[
            (statcast_df['inning'] > 1) &  # Exclude starters (typically pitch 1st inning)
            (statcast_df['home_team'] == pitcher_team)  # Bullpen of the opposing team
        ]
    
    if relief_pitches.empty:
        print(f"Warning: No relief pitcher data for {pitcher_team}. Using default values.")
        return pd.DataFrame({'batter': statcast_df['batter'].unique(), 'bp_hr_probability': 0.035, 'bp_matchup_score': 0.035})
    
    # Compute recency-weighted HR rate and wOBA against bullpen
    relief_pitches['time_weight'] = np.exp(-((prediction_date - relief_pitches['game_date']).dt.days / 30))
    bullpen_stats = relief_pitches.groupby('batter').agg({
        'events': [
            lambda x: (x == 'home_run').sum(),
            'count'  # Total PAs
        ],
        'woba_value': lambda x: (x * relief_pitches.loc[x.index, 'time_weight']).sum() / relief_pitches.loc[x.index, 'time_weight'].sum(),
        'time_weight': 'sum'
    })
    
    bullpen_stats.columns = ['hr_count', 'pa_count', 'bullpen_woba', 'total_weight']
    bullpen_stats = bullpen_stats.reset_index()
    
    # Calculate bullpen HR probability
    bullpen_stats['bp_hr_probability'] = bullpen_stats['hr_count'] / bullpen_stats['pa_count'].replace(0, np.nan)
    bullpen_stats['bp_hr_probability'] = bullpen_stats['bp_hr_probability'].fillna(0.035)
    
    # Calculate bullpen matchup score (simplified: weighted HR rate + wOBA)
    bullpen_stats['bp_matchup_score'] = (
        0.7 * bullpen_stats['bp_hr_probability'] +
        0.3 * bullpen_stats['bullpen_woba'].fillna(0.3)
    )
    
    # Filter for batters with sufficient PAs (e.g., >= 5)
    bullpen_stats = bullpen_stats[bullpen_stats['pa_count'] >= 5]
    
    return bullpen_stats[['batter', 'bp_hr_probability', 'bp_matchup_score']]