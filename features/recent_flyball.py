import pandas as pd

def get_recent_flyball_rate(statcast_df):
    """Calculate recent flyball percentage (past 21 days)."""
    max_date = statcast_df['game_date'].max()
    min_date = max_date - pd.Timedelta(days=20)
    recent_flyball_statcast = statcast_df[
        (statcast_df['game_date'] >= min_date) & 
        (statcast_df['game_date'] <= max_date)
    ].copy()
    recent_flyball_statcast['is_flyball'] = (
        (recent_flyball_statcast['bb_type'] == 'fly_ball') | 
        (recent_flyball_statcast['launch_angle'] >= 25)
    ).fillna(False).astype(int)
    recent_flyball = recent_flyball_statcast.groupby('batter').agg({
        'is_flyball': 'mean',
        'events': 'count'
    }).rename(columns={'is_flyball': 'Recent_Flyball%'}).reset_index()
    recent_flyball = recent_flyball[recent_flyball['events'] >= 5]
    return recent_flyball