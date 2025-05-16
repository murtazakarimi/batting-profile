import pandas as pd
import os
import json
import traceback
from datetime import datetime
from pybaseball import standings
from utils import standardize_team_short_name, TEAM_ID_MAPPING

def get_team_standings(season=2025, date='05/14/2025'):
    """Fetch team standings for the specified season using pybaseball."""
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/standings_{season}.parquet"
    metadata_file = f"{cache_dir}/standings_metadata.json"
    
    # Check cache
    current_date = datetime.now().strftime('%Y-%m-%d')
    use_cache = False
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            if metadata.get('last_refresh') == current_date and os.path.exists(cache_file):
                use_cache = True
        except (json.JSONDecodeError, KeyError):
            print(f"Warning: Corrupted standings cache metadata. Forcing refresh.")
    
    # Load or fetch standings
    if use_cache:
        standings_metrics = pd.read_parquet(cache_file)
        print(f"Loaded {len(standings_metrics)} standings records from cache: {cache_file}")
        print(f"First 5-6 rows of standings data:\n{standings_metrics.head(6).to_string(index=False)}")
    else:
        standings_metrics = pd.DataFrame()  # Initialize as DataFrame
        try:
            print(f"Fetching standings for season {season} using pybaseball.")
            standings_data = standings(season)
            all_standings = []
            for division_df in standings_data:
                for _, row in division_df.iterrows():
                    team_name = standardize_team_short_name(row['Tm'])
                    if team_name and team_name in TEAM_ID_MAPPING:
                        win_percentage = row.get('W-L%', 0.500)
                        games_back = row.get('GB', 0.0)
                        if isinstance(games_back, str) and games_back == '--':
                            games_back = 0.0
                        all_standings.append({
                            'Team': team_name,
                            'Win_Percentage': float(win_percentage),
                            'Games_Back': float(games_back)
                        })
            standings_metrics = pd.DataFrame(all_standings)
        except Exception as e:
            print(f"Error fetching standings for season {season}: {e}\n{traceback.format_exc()}")
            standings_metrics = pd.DataFrame()
        
        standings_metrics = standings_metrics[standings_metrics['Team'].notna()]
        
        if standings_metrics.empty:
            print(f"Warning: No standings data retrieved for {season}. Using default standings.")
            teams = list(TEAM_ID_MAPPING.keys())
            standings_metrics = pd.DataFrame({
                'Team': teams,
                'Win_Percentage': [0.500] * len(teams),
                'Games_Back': [0.0] * len(teams)
            })
        
        print(f"Retrieved {len(standings_metrics)} standings records for {season}. First 5-6 rows:\n{standings_metrics.head(6).to_string(index=False)}")
        
        # Cache the data
        standings_metrics.to_parquet(cache_file)
        with open(metadata_file, 'w') as f:
            json.dump({'last_refresh': current_date}, f)
    
    return standings_metrics