import pandas as pd
from pybaseball import chadwick_register, playerid_reverse_lookup, playerid_lookup

try:
    chadwick_register = chadwick_register
except ImportError:
    chadwick_register = None

# Predefined team ID mapping for statsapi
TEAM_ID_MAPPING = {
    'ARI': 109, 'ATL': 144, 'BAL': 110, 'BOS': 111, 'CHC': 112, 'CIN': 113, 'CLE': 114, 'COL': 115,
    'CWS': 145, 'DET': 116, 'HOU': 117, 'KC': 118, 'LAA': 108, 'LAD': 119, 'MIA': 146, 'MIL': 158,
    'MIN': 142, 'NYM': 121, 'NYY': 147, 'OAK': 133, 'PHI': 143, 'PIT': 134, 'SD': 135, 'SEA': 136,
    'SF': 137, 'STL': 138, 'TB': 139, 'TEX': 140, 'TOR': 141, 'WSH': 120
}

def standardize_name(name, is_statcast_format=False):
    try:
        if is_statcast_format:
            last, first = name.strip().split(', ', 1)
            return f"{first.strip().split()[0]} {last.strip()}"
        return name.strip()
    except:
        return name.strip()

def standardize_team_short_name(team):
    if not team or team.strip() == "- - -" or team.strip() == "":
        return None
    team = team.strip().upper()  # Case-insensitive matching
    team_short_mapping = {
        # Full team names
        'NEW YORK YANKEES': 'NYY',
        'LOS ANGELES DODGERS': 'LAD',
        'CHICAGO CUBS': 'CHC',
        'CHICAGO WHITE SOX': 'CWS',
        'NEW YORK METS': 'NYM',
        'OAKLAND ATHLETICS': 'OAK',
        'SAN FRANCISCO GIANTS': 'SF',
        'BOSTON RED SOX': 'BOS',
        'TORONTO BLUE JAYS': 'TOR',
        'TAMPA BAY RAYS': 'TB',
        'BALTIMORE ORIOLES': 'BAL',
        'CLEVELAND GUARDIANS': 'CLE',
        'DETROIT TIGERS': 'DET',
        'KANSAS CITY ROYALS': 'KC',
        'MINNESOTA TWINS': 'MIN',
        'HOUSTON ASTROS': 'HOU',
        'LOS ANGELES ANGELS': 'LAA',
        'SEATTLE MARINERS': 'SEA',
        'TEXAS RANGERS': 'TEX',
        'ATLANTA BRAVES': 'ATL',
        'MIAMI MARLINS': 'MIA',
        'PHILADELPHIA PHILLIES': 'PHI',
        'WASHINGTON NATIONALS': 'WSH',
        'MILWAUKEE BREWERS': 'MIL',
        'ST. LOUIS CARDINALS': 'STL',
        'PITTSBURGH PIRATES': 'PIT',
        'CINCINNATI REDS': 'CIN',
        'COLORADO ROCKIES': 'COL',
        'ARIZONA DIAMONDBACKS': 'ARI',
        'SAN DIEGO PADRES': 'SD',
        # Short names
        'YANKEES': 'NYY',
        'DODGERS': 'LAD',
        'CUBS': 'CHC',
        'WHITE SOX': 'CWS',
        'METS': 'NYM',
        'ATHLETICS': 'OAK',
        'GIANTS': 'SF',
        'RED SOX': 'BOS',
        'BLUE JAYS': 'TOR',
        'RAYS': 'TB',
        'ORIOLES': 'BAL',
        'GUARDIANS': 'CLE',
        'TIGERS': 'DET',
        'ROYALS': 'KC',
        'TWINS': 'MIN',
        'DESK': 'HOU',
        'ANGELS': 'LAA',
        'MARINERS': 'SEA',
        'RANGERS': 'TEX',
        'BRAVES': 'ATL',
        'MARLINS': 'MIA',
        'PHILLIES': 'PHI',
        'NATIONALS': 'WSH',
        'BREWERS': 'MIL',
        'CARDINALS': 'STL',
        'PIRATES': 'PIT',
        'REDS': 'CIN',
        'ROCKIES': 'COL',
        'DIAMONDBACKS': 'ARI',
        'PADRES': 'SD',
        # Abbreviations
        'NYY': 'NYY',
        'LAD': 'LAD',
        'CHC': 'CHC',
        'CWS': 'CWS',
        'CHW': 'CWS',
        'NYM': 'NYM',
        'OAK': 'OAK',
        'ATH': 'OAK',
        'SF': 'SF',
        'SFG': 'SF',
        'BOS': 'BOS',
        'TOR': 'TOR',
        'TB': 'TB',
        'TBR': 'TB',
        'BAL': 'BAL',
        'CLE': 'CLE',
        'DET': 'DET',
        'KC': 'KC',
        'KCR': 'KC',
        'MIN': 'MIN',
        'HOU': 'HOU',
        'LAA': 'LAA',
        'SEA': 'SEA',
        'TEX': 'TEX',
        'ATL': 'ATL',
        'MIA': 'MIA',
        'PHI': 'PHI',
        'WSH': 'WSH',
        'WSN': 'WSH',
        'MIL': 'MIL',
        'STL': 'STL',
        'PIT': 'PIT',
        'CIN': 'CIN',
        'COL': 'COL',
        'ARI': 'ARI',
        'SD': 'SD',
        'SDP': 'SD',
    }
    result = team_short_mapping.get(team, None)
    if result is None:
        print(f"Warning: Team '{team}' not found in mapping. Returning None.")
        return None
    return result

def create_name_mapping(batter_ids, manual_mapping=None):
    if manual_mapping is not None:
        return pd.DataFrame(manual_mapping)
    
    if chadwick_register is None:
        return pd.DataFrame()
    
    try:
        player_ids = chadwick_register()
        mapping = player_ids[player_ids['key_mlbam'].isin(batter_ids)][['key_mlbam', 'name_first', 'name_last']]
        mapping['name'] = mapping.apply(
            lambda x: f"{x['name_first'].capitalize()} {x['name_last'].capitalize()}", axis=1
        )
        mapping = mapping[['key_mlbam', 'name']].rename(columns={'key_mlbam': 'mlbam_id'})
        mapping = mapping.dropna(subset=['name'])
        return mapping
    except:
        return pd.DataFrame()

def get_park_factors():
    park_factors = {
        "ARI": 104,
        "ATL": 99,
        "BAL": 101,
        "BOS": 102,
        "CHC": 103,
        "CIN": 108,
        "CLE": 97,
        "COL": 115,
        "CWS": 105,
        "DET": 94,
        "HOU": 102,
        "KC": 98,
        "LAA": 100,
        "LAD": 96,
        "MIA": 95,
        "MIL": 103,
        "MIN": 99,
        "NYM": 96,
        "NYY": 104,
        "OAK": 92,
        "PHI": 106,
        "PIT": 97,
        "SD": 93,
        "SEA": 94,
        "SF": 91,
        "STL": 98,
        "TB": 100,
        "TEX": 101,
        "TOR": 102,
        "WSH": 99,
    }
    return park_factors