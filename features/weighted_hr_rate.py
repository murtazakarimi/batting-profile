import pandas as pd
import numpy as np

def get_weighted_hr_rate(statcast_df, prediction_date='2025-05-14'):
    """Calculate time-weighted HR rate."""
    statcast_df = statcast_df.copy()
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'])
    prediction_date = pd.to_datetime(prediction_date)
    statcast_df['days_ago'] = (prediction_date - statcast_df['game_date']).dt.days
    statcast_df['time_weight'] = np.exp(-statcast_df['days_ago'] / 30)
    weighted_hr = statcast_df[statcast_df['events'] == 'home_run'].groupby('batter').agg({
        'time_weight': 'sum'
    }).rename(columns={'time_weight': 'weighted_hr_sum'}).reset_index()
    weighted_pa = statcast_df.groupby('batter').agg({
        'time_weight': 'sum'
    }).rename(columns={'time_weight': 'weighted_pa_sum'}).reset_index()
    weighted_hr_data = weighted_hr.merge(weighted_pa, on='batter', how='outer').fillna(0)
    weighted_hr_data['weighted_hr_rate'] = weighted_hr_data['weighted_hr_sum'] / weighted_hr_data['weighted_pa_sum'].replace(0, 1)
    weighted_hr_data['weighted_hr_rate'] = weighted_hr_data['weighted_hr_rate'].fillna(0.035)
    return weighted_hr_data