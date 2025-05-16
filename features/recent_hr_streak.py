import pandas as pd

def get_recent_hr_streak(statcast_df, prediction_date='2025-05-15', window_days=3):
    """Calculate normalized HR count in the last 3 days."""
    statcast_df = statcast_df.copy()
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'])
    prediction_date = pd.to_datetime(prediction_date)
    recent_statcast = statcast_df[
        (statcast_df['game_date'] >= prediction_date - pd.Timedelta(days=window_days)) &
        (statcast_df['game_date'] <= prediction_date)
    ]
    hr_streak = recent_statcast[recent_statcast['events'] == 'home_run'].groupby('batter').size().reset_index(name='Recent_HR_Count')
    # Normalize by plate appearances to account for opportunity
    pa_count = recent_statcast.groupby('batter').size().reset_index(name='PA_Count')
    hr_streak = hr_streak.merge(pa_count, on='batter', how='left')
    hr_streak['Recent_HR_Score'] = hr_streak['Recent_HR_Count'] / hr_streak['PA_Count'].replace(0, 1)
    hr_streak = hr_streak[hr_streak['Recent_HR_Count'] >= 1]
    return hr_streak[['batter', 'Recent_HR_Score']]