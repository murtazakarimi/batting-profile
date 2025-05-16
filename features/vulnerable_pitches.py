import pandas as pd
import numpy as np

def get_vulnerable_pitches(pitcher_statcast, prediction_date='2025-05-15'):
    """Identify pitch types vulnerable to HRs or barrels with recency weighting."""
    pitcher_statcast = pitcher_statcast.copy()
    pitcher_statcast['game_date'] = pd.to_datetime(pitcher_statcast['game_date'])
    prediction_date = pd.to_datetime(prediction_date)
    pitcher_statcast['days_ago'] = (prediction_date - pitcher_statcast['game_date']).dt.days
    pitcher_statcast['time_weight'] = np.exp(-pitcher_statcast['days_ago'] / 365)
    
    pitcher_statcast['is_barrel'] = (
        (pitcher_statcast['launch_speed'] >= 95) & 
        (pitcher_statcast['launch_angle'].between(10, 40))
    ).fillna(False).astype(int)
    
    total_pitches_hand = pitcher_statcast.groupby(['pitch_type', 'batter_hand']).agg({
        'time_weight': 'sum'
    }).rename(columns={'time_weight': 'total_weighted_pitches'})
    hr_by_pitch_hand = pitcher_statcast[pitcher_statcast['events'] == 'home_run'].groupby(['pitch_type', 'batter_hand']).agg({
        'time_weight': 'sum'
    }).rename(columns={'time_weight': 'hr_weighted_count'})
    barrel_by_pitch_hand = pitcher_statcast[pitcher_statcast['is_barrel'] == 1].groupby(['pitch_type', 'batter_hand']).agg({
        'time_weight': 'sum'
    }).rename(columns={'time_weight': 'barrel_weighted_count'})
    
    hr_rate_pitch_hand = (hr_by_pitch_hand['hr_weighted_count'] / total_pitches_hand['total_weighted_pitches']).fillna(0).reset_index(name='hr_rate')
    barrel_rate_pitch_hand = (barrel_by_pitch_hand['barrel_weighted_count'] / total_pitches_hand['total_weighted_pitches']).fillna(0).reset_index(name='barrel_rate')
    
    pitch_vulnerability = hr_rate_pitch_hand.merge(barrel_rate_pitch_hand, on=['pitch_type', 'batter_hand'], how='outer').fillna(0)
    vulnerable_pitches = pitch_vulnerability[
        (pitch_vulnerability['hr_rate'] > 0) | (pitch_vulnerability['barrel_rate'] > 0.05)
    ][['pitch_type', 'batter_hand']].drop_duplicates()
    
    all_pitches = pitcher_statcast['pitch_type'].unique()
    for hand in ['L', 'R']:
        for pitch in all_pitches:
            if not ((vulnerable_pitches['pitch_type'] == pitch) & (vulnerable_pitches['batter_hand'] == hand)).any():
                vulnerable_pitches = pd.concat([
                    vulnerable_pitches,
                    pd.DataFrame({'pitch_type': [pitch], 'batter_hand': [hand]})
                ], ignore_index=True)
    
    if vulnerable_pitches.empty:
        vulnerable_pitches = pd.DataFrame({
            'pitch_type': all_pitches,
            'batter_hand': ['R'] * len(all_pitches)
        })
    
    return vulnerable_pitches