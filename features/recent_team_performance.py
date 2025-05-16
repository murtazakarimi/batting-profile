import pandas as pd
import os
import json
import traceback
from datetime import datetime
from pybaseball import team_game_logs
from utils import standardize_team_short_name, TEAM_ID_MAPPING

# Silence pandas warnings
pd.options.mode.chained_assignment = None

# Mapping for pybaseball team codes (aligned with Baseball-Reference abbreviations)
PYBASEBALL_TEAM_MAPPING = {
    'ATL': 'ATL',
    'WSH': 'WSN',
    'BAL': 'BAL',
    'MIN': 'MIN',
    'CWS': 'CHW',
    'CIN': 'CIN',
    'TB': 'TBR',
    'TOR': 'TOR',
    'HOU': 'HOU',
    'TEX': 'TEX',
    'OAK': 'ATH',  # Confirmed correct for 2025
    'LAD': 'LAD',
    'NYY': 'NYY',
    'BOS': 'BOS',
    'DET': 'DET',
    'CLE': 'CLE',
    'KC': 'KCR',
    'SEA': 'SEA',
    'LAA': 'LAA',
    'NYM': 'NYM',
    'PHI': 'PHI',
    'MIA': 'MIA',
    'CHC': 'CHC',
    'STL': 'STL',
    'MIL': 'MIL',
    'PIT': 'PIT',
    'SD': 'SDP',
    'SF': 'SFG',
    'ARI': 'ARI',
    'COL': 'COL'
}

def get_recent_team_performance(date_str, teams, season=2025, num_games=10):
    """Calculate recent team performance (win rate, run differential) using pybaseball."""
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/game_logs_{season}.parquet"
    metadata_file = f"{cache_dir}/game_logs_metadata.json"
    
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
            print(f"Warning: Corrupted game logs cache metadata. Forcing refresh.")
    
    # Load or fetch game logs
    if use_cache:
        game_logs = pd.read_parquet(cache_file)
        print(f"Loaded {len(game_logs)} game logs from cache: {cache_file}")
        print(f"First 5-6 rows of game logs data:\n{game_logs.head(6).to_string(index=False)}")
    else:
        game_logs = pd.DataFrame(columns=['Team', 'Date', 'Opp', 'W/L', 'R', 'RA', 'Home_Away'])
        start_date = pd.to_datetime('2025-03-27')
        end_date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
        if pd.isna(end_date):
            print(f"Error: Invalid date_str format: {date_str}. Using default end date.")
            end_date = pd.to_datetime('2025-05-14')
        
        failed_teams = []
        skipped_rows = []
        for team in teams:
            original_team = team
            team = standardize_team_short_name(team)
            if not team or team not in TEAM_ID_MAPPING:
                print(f"Warning: Team {original_team} standardized to {team} not found in TEAM_ID_MAPPING. Skipping.")
                failed_teams.append(team or original_team)
                continue
            pybaseball_team = PYBASEBALL_TEAM_MAPPING.get(team, team)
            url = f"https://www.baseball-reference.com/teams/{pybaseball_team}/{season}-schedule-scores.shtml"
            print(f"Fetching game logs for {team} (original: {original_team}, pybaseball: {pybaseball_team}, season={season}, url: {url}) using pybaseball.")
            try:
                team_logs = team_game_logs(season, pybaseball_team, "batting")
                print(f"Retrieved team_logs for {team}:\n{team_logs.head(6).to_string(index=False)}")
                if team_logs.empty:
                    print(f"Warning: No game logs retrieved for {team} (pybaseball: {pybaseball_team}).")
                    failed_teams.append(team)
                    continue
                # Clean and parse dates
                team_logs['Date'] = team_logs['Date'].str.replace(r'\s*\(\d+\)', '', regex=True)
                team_logs['Date'] = pd.to_datetime(team_logs['Date'] + f' {season}', format='%b %d %Y', errors='coerce')
                invalid_dates = team_logs[team_logs['Date'].isna()]
                if not invalid_dates.empty:
                    print(f"Warning: Invalid dates for {team}: {invalid_dates['Date'].tolist()}")
                    skipped_rows.extend([(team, row['Date'], 'Invalid date') for _, row in invalid_dates.iterrows()])
                team_logs = team_logs[
                    (team_logs['Date'] >= start_date) &
                    (team_logs['Date'] <= end_date) &
                    (team_logs['Date'].notna())
                ]
                for _, game in team_logs.iterrows():
                    opp = standardize_team_short_name(game.get('Opp', ''))
                    if not opp or opp not in TEAM_ID_MAPPING:
                        print(f"Warning: Invalid or unmapped opponent '{game.get('Opp', '')}' for {team} on {game['Date']}")
                        skipped_rows.append((team, game['Date'], game.get('Opp', '')))
                        continue
                    result = game.get('Rslt', '')
                    wl = 'W' if result.startswith('W') else 'L'
                    home_away = 'Home' if game.get('Home', False) else 'Away'
                    runs = game.get('R', 0)
                    # Parse RA from Rslt (e.g., "W,7-4" or "L,4-7" -> 4 or 7)
                    try:
                        ra = int(result.split(',')[1].split('-')[1 if wl == 'W' else 0])
                        print(f"Parsed RA={ra} for {team} on {game['Date']} from Rslt='{result}'")
                    except (IndexError, ValueError):
                        print(f"Warning: Invalid Rslt format '{result}' for {team} on {game['Date']}. Setting RA to 0.")
                        ra = 0
                    game_logs = pd.concat([game_logs, pd.DataFrame([{
                        'Team': team,
                        'Date': game['Date'],
                        'Opp': opp,
                        'W/L': wl,
                        'R': runs,
                        'RA': ra,
                        'Home_Away': home_away
                    }])], ignore_index=True)
            except Exception as e:
                print(f"Error fetching game log for {team} (original: {original_team}, pybaseball: {pybaseball_team}): {e}\n{traceback.format_exc()}")
                failed_teams.append(team)
                continue
        
        if skipped_rows:
            print(f"Skipped rows due to invalid data: {skipped_rows}")
        if game_logs.empty:
            raise RuntimeError(f"No game logs retrieved for {season}. Failed teams: {', '.join(failed_teams) if failed_teams else 'None'}. Skipped rows: {skipped_rows}. Ensure date parsing and opponent standardization are correct.")
        else:
            print(f"Retrieved {len(game_logs)} game logs for {season}.")
        
        print(f"First 5-6 rows of game logs data:\n{game_logs.head(6).to_string(index=False)}")
        
        # Cache the data
        game_logs.to_parquet(cache_file)
        with open(metadata_file, 'w') as f:
            json.dump({'last_refresh': current_date}, f)
    
    # Standardize date format to date-only and clean data
    if not game_logs.empty and 'Date' in game_logs.columns:
        game_logs['Date'] = pd.to_datetime(game_logs['Date'], format='%Y-%m-%d', errors='coerce').dt.date
        game_logs['Team'] = game_logs['Team'].apply(standardize_team_short_name)
        game_logs = game_logs[game_logs['Team'].notna() & game_logs['W/L'].notna()]
    
    team_performance = []
    for team in teams:
        team = standardize_team_short_name(team)
        if not team:
            continue
        
        print(f"Processing recent games for {team} with date_str: {date_str}")
        try:
            date_filter = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce').date()
            if pd.isna(date_filter):
                print(f"Error: Invalid date_str format for {team}: {date_str}. Skipping.")
                continue
            # Filter games
            recent_games = game_logs[
                (game_logs['Team'] == team) &
                (game_logs['Date'] <= date_filter) &
                (game_logs['W/L'].notna())
            ].sort_values('Date', ascending=False).head(num_games)
            print(f"Found {len(recent_games)} recent games for {team} on or before {date_str}")
            
            # Early season: Allow fewer games or no games
            if recent_games.empty and date_filter <= pd.to_datetime('2025-04-10').date():
                recent_games = game_logs[
                    (game_logs['Team'] == team) &
                    (game_logs['Date'] <= date_filter) &
                    (game_logs['W/L'].notna())
                ].sort_values('Date', ascending=False)
                print(f"Early season: Found {len(recent_games)} games for {team} on or before {date_str}")
                if recent_games.empty:
                    print(f"Early season: No games found for {team} on or before {date_str}. Using default metrics.")
                    team_performance.append({
                        'Team': team,
                        'Win_Rate': 0.500,
                        'Run_Differential': 0.0
                    })
                    continue
            
            if recent_games.empty:
                raise RuntimeError(f"No recent games found for {team} on or before {date_str}. Ensure game logs are retrieved successfully.")
            
            wins = sum(1 for wl in recent_games['W/L'] if wl.startswith('W'))
            win_rate = wins / len(recent_games) if len(recent_games) > 0 else 0.500
            run_diff = (recent_games['R'] - recent_games['RA']).mean() if len(recent_games) > 0 else 0.0
            
            team_performance.append({
                'Team': team,
                'Win_Rate': win_rate,
                'Run_Differential': run_diff
            })
        except Exception as e:
            print(f"Error filtering recent games for {team}: {e}")
            raise
    
    return pd.DataFrame(team_performance)