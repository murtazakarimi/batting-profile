import pandas as pd
import numpy as np

def get_swing_discipline_score(statcast_df, prediction_date='2025-05-14'):
    """Calculate swing discipline metrics and score."""
    recent_statcast = statcast_df[
        (statcast_df['game_date'] >= '2025-05-01') & 
        (statcast_df['game_date'] <= '2025-05-14')
    ].copy()
    recent_statcast['time_weight'] = np.exp(-(pd.to_datetime(prediction_date) - recent_statcast['game_date']).dt.days / 14)
    
    # Z-Swing%
    z_swing = recent_statcast[
        (recent_statcast['zone'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])) & 
        (recent_statcast['description'].isin(['swinging_strike', 'hit_into_play', 'foul']))
    ].groupby('batter').agg({'time_weight': 'sum'}).reset_index()
    z_total = recent_statcast[
        recent_statcast['zone'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])
    ].groupby('batter').agg({'time_weight': 'sum'}).reset_index()
    z_swing_rate = z_swing.merge(z_total, on='batter', how='outer', suffixes=('_swing', '_total')).fillna(0)
    z_swing_rate['Z_Swing_Rate'] = z_swing_rate['time_weight_swing'] / z_swing_rate['time_weight_total'].replace(0, 1)
    
    # O-Swing%
    o_swing = recent_statcast[
        (recent_statcast['zone'].isin([11, 12, 13, 14])) & 
        (recent_statcast['description'].isin(['swinging_strike', 'hit_into_play', 'foul']))
    ].groupby('batter').agg({'time_weight': 'sum'}).reset_index()
    o_total = recent_statcast[
        recent_statcast['zone'].isin([11, 12, 13, 14])
    ].groupby('batter').agg({'time_weight': 'sum'}).reset_index()
    o_swing_rate = o_swing.merge(o_total, on='batter', how='outer', suffixes=('_swing', '_total')).fillna(0)
    o_swing_rate['O_Swing_Rate'] = o_swing_rate['time_weight_swing'] / o_swing_rate['time_weight_total'].replace(0, 1)
    
    # BB/K Ratio
    walks = recent_statcast[recent_statcast['events'] == 'walk'].groupby('batter').agg({'time_weight': 'sum'}).reset_index()
    strikeouts = recent_statcast[recent_statcast['events'].isin(['strikeout', 'strikeout_double_play'])].groupby('batter').agg({'time_weight': 'sum'}).reset_index()
    bb_k_ratio = walks.merge(strikeouts, on='batter', how='outer', suffixes=('_walk', '_strikeout')).fillna(0)
    bb_k_ratio['BB_K_Ratio'] = bb_k_ratio['time_weight_walk'] / bb_k_ratio['time_weight_strikeout'].replace(0, 1)
    
    # Combine into Swing_Discipline_Score
    swing_discipline = z_swing_rate[['batter', 'Z_Swing_Rate']].merge(
        o_swing_rate[['batter', 'O_Swing_Rate']], on='batter', how='outer'
    ).merge(bb_k_ratio[['batter', 'BB_K_Ratio']], on='batter', how='outer').fillna(0)
    swing_discipline['Swing_Discipline_Score'] = (
        swing_discipline['Z_Swing_Rate'] * 0.4 - 
        swing_discipline['O_Swing_Rate'] * 0.4 + 
        swing_discipline['BB_K_Ratio'] * 0.2
    )
    swing_discipline['Swing_Discipline_Score'] = (
        swing_discipline['Swing_Discipline_Score'] - swing_discipline['Swing_Discipline_Score'].min()
    ) / (swing_discipline['Swing_Discipline_Score'].max() - swing_discipline['Swing_Discipline_Score'].min() + 1e-6)
    
    return swing_discipline