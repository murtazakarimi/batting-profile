import pandas as pd
from utils import standardize_team_short_name

def get_team_pitching_metrics(pitcher_df):
    """Calculate team-level pitching metrics."""
    pitcher_df = pitcher_df.copy()
    pitcher_df['Team'] = pitcher_df['Team'].apply(standardize_team_short_name)
    
    team_pitching = pitcher_df.groupby('Team').agg({
        'ERA': 'mean',
        'FIP': 'mean',
        'K/9': 'mean',
        'BB/9': 'mean',
        'HR/9': 'mean',
        'WHIP': 'mean'
    }).reset_index()
    
    team_pitching = team_pitching.fillna({
        'ERA': 4.00,
        'FIP': 4.00,
        'K/9': 8.00,
        'BB/9': 3.50,
        'HR/9': 1.20,
        'WHIP': 1.30
    })
    
    return team_pitching