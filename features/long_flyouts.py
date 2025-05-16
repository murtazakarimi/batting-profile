import pandas as pd
from datetime import timedelta, datetime

def get_long_flyout_count(statcast_df, prediction_date=None, days_back=21, min_count=2):
    """Calculate count of long flyouts (>300 feet) in the past 21 days."""
    statcast_df = statcast_df.copy()
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'])
    
    # Use today's date if prediction_date is not provided
    prediction_date = pd.to_datetime(datetime.now().date()) if prediction_date is None else pd.to_datetime(prediction_date)
    
    # Dynamic date range: prediction_date - days_back to prediction_date
    start_date = prediction_date - timedelta(days=days_back)
    recent_statcast = statcast_df[
        (statcast_df['game_date'] >= start_date) & 
        (statcast_df['game_date'] <= prediction_date)
    ]
    
    # long_flyouts = recent_statcast[
    #     (recent_statcast['bb_type'] == 'fly_ball') & 
    #     (recent_statcast['events'] == 'field_out') & 
    #     (recent_statcast['hit_distance_sc'] > 320)
    # ].groupby('batter').size().reset_index(name='Long_Flyout_Count')
    
    long_flyouts = statcast_df[
        (statcast_df['events'] == 'flyout') &
        (statcast_df['hit_distance_sc'] > 350)
    ].groupby('batter').size().reset_index(name='Long_Flyout_Count')

    # Filter for minimum count
    long_flyouts = long_flyouts[long_flyouts['Long_Flyout_Count'] >= min_count]
    
    return long_flyouts