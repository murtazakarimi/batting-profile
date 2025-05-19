import pandas as pd
import statsapi
from fuzzywuzzy import process
from datetime import datetime
import re
import numpy as np
import os

# Load players.csv for ID and team mappings
PLAYERS_CSV_PATH = "data/players.csv"
if os.path.exists(PLAYERS_CSV_PATH):
    PLAYERS_DF = pd.read_csv(PLAYERS_CSV_PATH)
    PLAYERS_DF['MLBAMID'] = PLAYERS_DF['MLBAMID'].astype(str)
    PLAYERS_DF['FanGraphsID'] = PLAYERS_DF['FanGraphsID'].astype(str)
else:
    print(f"Error: {PLAYERS_CSV_PATH} not found. Falling back to empty mapping.")
    PLAYERS_DF = pd.DataFrame(columns=['MLBAMID', 'FanGraphsID', 'Name', 'Team'])

def standardize_name(name, is_statcast_format=False):
    """Standardize player names for consistency."""
    if not isinstance(name, str):
        return ""
    name = name.strip().replace("'", "").replace(".", "")
    if is_statcast_format:
        parts = name.split(", ")
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}".strip()
    return name

def standardize_team_short_name(team_name):
    """Convert team names to standardized short names."""
    if not isinstance(team_name, str) or not team_name.strip() or team_name.strip() == "- - -":
        print(f"Warning: Invalid team name {team_name}. Returning 'UNKNOWN'.")
        return "UNKNOWN"
    team_name = team_name.strip()
    team_mapping = {
        'Arizona Diamondbacks': 'ARI', 'Arizona': 'ARI', 'Diamondbacks': 'ARI', 'D-backs': 'ARI', 'AZ': 'ARI', 'ARI': 'ARI',
        'Atlanta Braves': 'ATL', 'Atlanta': 'ATL', 'Braves': 'ATL', 'ATL': 'ATL',
        'Baltimore Orioles': 'BAL', 'Baltimore': 'BAL', 'Orioles': 'BAL', 'BAL': 'BAL',
        'Boston Red Sox': 'BOS', 'Boston': 'BOS', 'Red Sox': 'BOS', 'BOS': 'BOS',
        'Chicago Cubs': 'CHC', 'Cubs': 'CHC', 'CHC': 'CHC',
        'Chicago White Sox': 'CWS', 'White Sox': 'CWS', 'ChiSox': 'CWS', 'CHW': 'CWS', 'CWS': 'CWS',
        'Cincinnati Reds': 'CIN', 'Cincinnati': 'CIN', 'Reds': 'CIN', 'CIN': 'CIN',
        'Cleveland Guardians': 'CLE', 'Cleveland': 'CLE', 'Guardians': 'CLE', 'CLE': 'CLE',
        'Colorado Rockies': 'COL', 'Colorado': 'COL', 'Rockies': 'COL', 'COL': 'COL',
        'Detroit Tigers': 'DET', 'Detroit': 'DET', 'Tigers': 'DET', 'DET': 'DET',
        'Houston Astros': 'HOU', 'Houston': 'HOU', 'Astros': 'HOU', 'HOU': 'HOU',
        'Kansas City Royals': 'KC', 'Kansas City': 'KC', 'Royals': 'KC', 'KCR': 'KC', 'KC': 'KC',
        'Los Angeles Angels': 'LAA', 'Angels': 'LAA', 'Anaheim': 'LAA', 'LAA': 'LAA',
        'Los Angeles Dodgers': 'LAD', 'Dodgers': 'LAD', 'LA Dodgers': 'LAD', 'LAD': 'LAD',
        'Miami Marlins': 'MIA', 'Miami': 'MIA', 'Marlins': 'MIA', 'MIA': 'MIA',
        'Milwaukee Brewers': 'MIL', 'Milwaukee': 'MIL', 'Brewers': 'MIL', 'MIL': 'MIL',
        'Minnesota Twins': 'MIN', 'Minnesota': 'MIN', 'Twins': 'MIN', 'MIN': 'MIN',
        'New York Mets': 'NYM', 'Mets': 'NYM', 'NY Mets': 'NYM', 'NYM': 'NYM',
        'New York Yankees': 'NYY', 'Yankees': 'NYY', 'NY Yankees': 'NYY', 'NYY': 'NYY',
        'Oakland Athletics': 'OAK', 'Oakland': 'OAK', 'Athletics': 'OAK', 'A\'s': 'OAK', 'ATH': 'OAK', 'OAK': 'OAK',
        'Philadelphia Phillies': 'PHI', 'Philadelphia': 'PHI', 'Phillies': 'PHI', 'PHI': 'PHI',
        'Pittsburgh Pirates': 'PIT', 'Pittsburgh': 'PIT', 'Pirates': 'PIT', 'PIT': 'PIT',
        'San Diego Padres': 'SD', 'San Diego': 'SD', 'Padres': 'SD', 'SDP': 'SD', 'SD': 'SD',
        'San Francisco Giants': 'SF', 'San Francisco': 'SF', 'Giants': 'SF', 'SFG': 'SF', 'SF': 'SF',
        'Seattle Mariners': 'SEA', 'Seattle': 'SEA', 'Mariners': 'SEA', 'SEA': 'SEA',
        'St. Louis Cardinals': 'STL', 'St. Louis': 'STL', 'Cardinals': 'STL', 'STL': 'STL',
        'Tampa Bay Rays': 'TB', 'Tampa Bay': 'TB', 'Rays': 'TB', 'TBR': 'TB', 'TB': 'TB',
        'Texas Rangers': 'TEX', 'Texas': 'TEX', 'Rangers': 'TEX', 'TEX': 'TEX',
        'Toronto Blue Jays': 'TOR', 'Toronto': 'TOR', 'Blue Jays': 'TOR', 'TOR': 'TOR',
        'Washington Nationals': 'WSH', 'Washington': 'WSH', 'Nationals': 'WSH', 'Nats': 'WSH', 'Wash': 'WSH', 'WSN': 'WSH',
        'FA': 'FA', 'Free Agent': 'FA'
    }
    team_name_lower = team_name.lower()
    for short_name in set(team_mapping.values()):
        if team_name_lower == short_name.lower():
            return short_name
    for full_name, short_name in team_mapping.items():
        if full_name.lower() in team_name_lower:
            return short_name
    # print(f"Warning: No team mapping for {team_name}. Trying fuzzy matching.")
    matches = process.extractOne(team_name_lower, [k.lower() for k in team_mapping.keys()])
    if matches and matches[1] > 80:
        matched_name = matches[0]
        for full_name, short_name in team_mapping.items():
            if full_name.lower() == matched_name.lower():
                return short_name
    print(f"Warning: No alternative mapping found for {team_name}. Returning 'UNKNOWN'.")
    return "UNKNOWN"

def create_name_mapping(mlbam_ids, fangraphs_ids=None, roster=None, manual_name_mapping=None):
    """Map MLBAM IDs to standardized names using players.csv, checking FanGraphs IDs explicitly."""
    mlbam_ids = list(mlbam_ids) if isinstance(mlbam_ids, (list, np.ndarray)) else []
    fangraphs_ids = list(fangraphs_ids) if isinstance(fangraphs_ids, (list, np.ndarray)) else [] if fangraphs_ids else []
    
    if not mlbam_ids and not fangraphs_ids:
        print("Warning: No MLBAM or FanGraphs IDs provided for name mapping.")
        return pd.DataFrame(columns=['mlbam_id', 'name'])
    
    try:
        # Initialize player lookup DataFrame
        player_lookup = pd.DataFrame(columns=['mlbam_id', 'name'])
        mapped_fg_ids = set()  # Track FanGraphs IDs that have been mapped
        
        # Map MLBAM IDs first
        if mlbam_ids:
            # print(f"Mapping {len(mlbam_ids)} MLBAM IDs: {mlbam_ids}")
            mlbam_lookup = PLAYERS_DF[PLAYERS_DF['MLBAMID'].isin([str(id) for id in mlbam_ids])][['MLBAMID', 'Name']]
            mlbam_lookup = mlbam_lookup.rename(columns={'MLBAMID': 'mlbam_id', 'Name': 'name'})
            mlbam_lookup['name'] = mlbam_lookup['name'].apply(standardize_name)
            player_lookup = pd.concat([player_lookup, mlbam_lookup], ignore_index=True)
            # print(f"Found {len(mlbam_lookup)} MLBAM IDs in players.csv")
        
        # Handle FanGraphs IDs (including any misclassified mlbam_ids)
        all_fg_ids = list(set(fangraphs_ids + mlbam_ids))  # Treat mlbam_ids as potential FanGraphs IDs
        if all_fg_ids:
            # print(f"Mapping {len(all_fg_ids)} FanGraphs IDs: {all_fg_ids}")
            fg_lookup = PLAYERS_DF[PLAYERS_DF['FanGraphsID'].isin([str(id) for id in all_fg_ids])][['MLBAMID', 'Name', 'FanGraphsID']]
            fg_lookup = fg_lookup.rename(columns={'MLBAMID': 'mlbam_id', 'Name': 'name'})
            fg_lookup['name'] = fg_lookup['name'].apply(standardize_name)
            if not fg_lookup.empty:
                # print(f"Found {len(fg_lookup)} FanGraphs IDs in players.csv: {fg_lookup[['mlbam_id', 'name', 'FanGraphsID']].to_dict('records')}")
                mapped_fg_ids.update(fg_lookup['FanGraphsID'].astype(str).tolist())
                player_lookup = pd.concat([player_lookup, fg_lookup[['mlbam_id', 'name']]], ignore_index=True)
                player_lookup = player_lookup.drop_duplicates(subset=['mlbam_id'], keep='first')
            # else:
            #     print("No FanGraphs IDs found in players.csv")
        
        # Mark unmapped MLBAM IDs as missing, excluding those mapped via FanGraphs IDs
        still_missing = [id for id in mlbam_ids if str(id) not in player_lookup['mlbam_id'].values and str(id) not in mapped_fg_ids]
        if still_missing:
            # print(f"Warning: {len(still_missing)} MLBAM IDs not found in players.csv: {still_missing}")
            missing_df = pd.DataFrame([
                {'mlbam_id': str(id), 'name': f"Missing_{id}"}
                for id in still_missing
            ])
            player_lookup = pd.concat([player_lookup, missing_df], ignore_index=True)
        
        # Use roster names as fallback for missing names
        if roster:
            roster_name_map = {str(player['mlbam_id']): player['name'] for player in roster}
            player_lookup['name'] = player_lookup.apply(
                lambda row: roster_name_map.get(row['mlbam_id'], row['name'])
                if row['name'].startswith("Missing_") else row['name'],
                axis=1
            )
        
        # Apply manual name mapping if provided
        if manual_name_mapping:
            manual_df = pd.DataFrame(manual_name_mapping.items(), columns=['mlbam_id', 'name'])
            manual_df['mlbam_id'] = manual_df['mlbam_id'].astype(str)
            player_lookup = pd.concat([player_lookup, manual_df], ignore_index=True)
            player_lookup = player_lookup.drop_duplicates(subset=['mlbam_id'], keep='last')
        
        # print(f"Created name mapping for {len(player_lookup)} players.")
        return player_lookup
    except Exception as e:
        print(f"Error in create_name_mapping: {e}")
        return pd.DataFrame(columns=['mlbam_id', 'name'])

def get_team_roster(team_name, batter_df, min_pa=10):
    """Fetch team roster using players.csv and extract IDs."""
    try:
        team_name = standardize_team_short_name(team_name)
        team_id = TEAM_ID_MAPPING.get(team_name)
        if not team_id:
            print(f"Warning: No team ID for {team_name}. Returning empty roster.")
            return []
        
        roster = statsapi.roster(team_id, rosterType='active', season=datetime.now().year)
        players = []
        for line in roster.split('\n'):
            if line.strip() and '#' in line:
                match = re.match(r'#(\d+)\s+(\w{1,2})\s+([^,]+)(?:,.*|$)', line.strip())
                if match:
                    number, position, name = match.groups()
                    name = name.strip()
                    if name:
                        players.append({'name': standardize_name(name), 'position': position})
                else:
                    print(f"Warning: Failed to parse roster line: {line}")
        
        if not players:
            print(f"Warning: No players found in roster for {team_name}.")
            return []
        
        # Map roster players to players.csv
        id_mappings = []
        for player in players:
            player_name = player['name']
            csv_match = PLAYERS_DF[PLAYERS_DF['Name'].str.lower() == player_name.lower()]
            if not csv_match.empty:
                player_ids = {
                    'mlbam_id': str(csv_match.iloc[0]['MLBAMID']),
                    'key_fangraphs': str(csv_match.iloc[0]['FanGraphsID']),
                    'key_bbref': '',
                    'name': player_name,
                    'position': player['position']
                }
                id_mappings.append(player_ids)
                # print(f"Successfully mapped {player_name} to MLBAM ID {player_ids['mlbam_id']}")
            # else:
            #     print(f"Warning: No ID mapping for {player_name} in players.csv.")
        
        if not id_mappings:
            print(f"Warning: No ID mappings retrieved for {team_name} roster.")
            return []
        
        id_df = pd.DataFrame(id_mappings)
        id_df['mlbam_id'] = id_df['mlbam_id'].astype(str)
        id_df['key_fangraphs'] = id_df['key_fangraphs'].astype(str)
        
        if 'IDfg' in batter_df.columns and 'PA' in batter_df.columns:
            batter_df = batter_df.copy()
            batter_df['key_fangraphs'] = batter_df['IDfg'].astype(str)
            roster_with_pa = id_df.merge(
                batter_df[['key_fangraphs', 'PA', 'Name']],
                on='key_fangraphs',
                how='left'
            )
            filtered_roster = roster_with_pa[roster_with_pa['PA'].fillna(0) >= min_pa]
            if filtered_roster.empty:
                print(f"Warning: No players with >= {min_pa} PAs for {team_name}.")
                return []
            
            missing_names = filtered_roster[filtered_roster['Name'].isna()]['mlbam_id'].tolist()
            if missing_names:
                print(f"Warning: {len(missing_names)} roster players not found in batter_df: {missing_names}")
            
            return filtered_roster[['mlbam_id', 'key_fangraphs', 'key_bbref', 'name', 'position']].to_dict('records')
        else:
            print(f"Warning: batter_df missing 'IDfg' or 'PA' columns. Returning unfiltered roster for {team_name}.")
            return id_df[['mlbam_id', 'key_fangraphs', 'key_bbref', 'name', 'position']].to_dict('records')
    
    except Exception as e:
        print(f"Error fetching roster for {team_name}: {e}")
        return []

TEAM_ID_MAPPING = {
    'ARI': 109, 'ATL': 144, 'BAL': 110, 'BOS': 111, 'CHC': 112, 'CWS': 145,
    'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116, 'HOU': 117, 'KC': 118,
    'LAA': 108, 'LAD': 119, 'MIA': 146, 'MIL': 158, 'MIN': 142, 'NYM': 121,
    'NYY': 147, 'OAK': 133, 'PHI': 143, 'PIT': 134, 'SD': 135, 'SEA': 136,
    'SF': 137, 'STL': 138, 'TB': 139, 'TEX': 140, 'TOR': 141, 'WSH': 120
}