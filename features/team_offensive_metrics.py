import pandas as pd
from utils import standardize_team_short_name

def get_team_offensive_metrics(batter_df):
    """Calculate team-level offensive metrics."""
    batter_df = batter_df.copy()
    batter_df['Team'] = batter_df['Team'].apply(standardize_team_short_name)
    
    team_offense = batter_df.groupby('Team').agg({
        'wRC+': 'mean',
        'AVG': 'mean',
        'OBP': 'mean',
        'SLG': 'mean',
        'HR': 'sum',
        'PA': 'sum',
        'Barrel%': 'mean',
        'HardHit%': 'mean'
    }).reset_index()
    
    team_offense['HR_Rate'] = team_offense['HR'] / team_offense['PA'].replace(0, 1)
    team_offense = team_offense[
        ['Team', 'wRC+', 'AVG', 'OBP', 'SLG', 'HR_Rate', 'Barrel%', 'HardHit%']
    ].fillna({
        'wRC+': 100,
        'AVG': 0.250,
        'OBP': 0.320,
        'SLG': 0.400,
        'HR_Rate': 0.035,
        'Barrel%': 0.05,
        'HardHit%': 0.30
    })
    
    return team_offense