import pandas as pd
import numpy as np

def get_matchup_history(pitcher_statcast, prediction_date='2025-05-15', include_minor_league=False):
    """Calculate batter-pitcher historical matchup data with recency weighting."""
    pitcher_statcast = pitcher_statcast.copy()
    pitcher_statcast['game_date'] = pd.to_datetime(pitcher_statcast['game_date'])
    prediction_date = pd.to_datetime(prediction_date)
    pitcher_statcast['days_ago'] = (prediction_date - pitcher_statcast['game_date']).dt.days
    pitcher_statcast['time_weight'] = np.exp(-pitcher_statcast['days_ago'] / 365)
    
    matchup_history = pitcher_statcast.groupby('batter').agg({
        'events': lambda x: ((x == 'home_run') * pitcher_statcast.loc[x.index, 'time_weight']).sum(),
        'at_bat_number': lambda x: pitcher_statcast.loc[x.index, 'time_weight'].sum(),
        'woba_value': lambda x: (x * pitcher_statcast.loc[x.index, 'time_weight']).sum() / pitcher_statcast.loc[x.index, 'time_weight'].sum()
    }).rename(columns={
        'events': 'matchup_hr_weighted',
        'at_bat_number': 'matchup_pa_weighted',
        'woba_value': 'matchup_woba'
    }).reset_index()
    matchup_history['matchup_hr_rate'] = matchup_history['matchup_hr_weighted'] / matchup_history['matchup_pa_weighted'].replace(0, 1)
    matchup_history['matchup_woba'] = matchup_history['matchup_woba'].fillna(0.3)
    matchup_history['matchup_pa'] = pitcher_statcast.groupby('batter')['at_bat_number'].count().reset_index()['at_bat_number']
    
    if include_minor_league:
        # Placeholder: Fetch minor league data (requires external source like Baseball-Reference or MiLB API)
        minor_league_data = pd.DataFrame()
        if not minor_league_data.empty:
            minor_matchup = minor_league_data.groupby('batter').agg({
                'events': lambda x: (x == 'home_run').sum(),
                'at_bat_number': 'count'
            }).rename(columns={
                'events': 'matchup_hr_weighted',
                'at_bat_number': 'matchup_pa_weighted'
            })
            minor_matchup['matchup_hr_rate'] = minor_matchup['matchup_hr_weighted'] / minor_matchup['matchup_pa_weighted'].replace(0, 1)
            matchup_history = matchup_history.merge(minor_matchup[['batter', 'matchup_hr_rate']], on='batter', how='left')
            matchup_history['matchup_hr_rate'] = matchup_history['matchup_hr_rate_x'].fillna(matchup_history['matchup_hr_rate_y']).fillna(0.035)
    
    return matchup_history