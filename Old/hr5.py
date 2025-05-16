import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pybaseball
from pybaseball import statcast
from pybaseball import batting_stats
from pybaseball import pitching_stats
from pybaseball import playerid_lookup
from pybaseball import playerid_reverse_lookup
import statsapi
from datetime import datetime
import pytz
from timezonefinder import TimezoneFinder
import requests

pybaseball.cache.enable()

# Load data for 2025 season
batter_df = batting_stats(2025, end_season=None, league='all', qual=1, ind=1)
pitcher_df = pitching_stats(2025, end_season=None, league='all', qual=1, ind=1)
statcast_df = statcast(start_dt='2025-03-10', end_dt='2025-05-14', team=None)

try:
    from pybaseball import chadwick_register
except ImportError:
    chadwick_register = None

class BallparkWeather:
    def __init__(self, api_key):
        self.api_key = api_key
        self.team_ballparks = {
            "ARI": {"stadium": "Chase Field", "lat": 33.4455, "lon": -112.0667},
            "ATL": {"stadium": "Truist Park", "lat": 33.8908, "lon": -84.4678},
            "BAL": {"stadium": "Camden Yards", "lat": 39.2839, "lon": -76.6218},
            "BOS": {"stadium": "Fenway Park", "lat": 42.3467, "lon": -71.0972},
            "CHC": {"stadium": "Wrigley Field", "lat": 41.9484, "lon": -87.6553},
            "CIN": {"stadium": "Great American Ball Park", "lat": 39.0972, "lon": -84.5072},
            "CLE": {"stadium": "Progressive Field", "lat": 41.4962, "lon": -81.6852},
            "COL": {"stadium": "Coors Field", "lat": 39.7559, "lon": -104.9942},
            "CWS": {"stadium": "Guaranteed Rate Field", "lat": 41.8299, "lon": -87.6339},
            "DET": {"stadium": "Comerica Park", "lat": 42.3390, "lon": -83.0490},
            "HOU": {"stadium": "Minute Maid Park", "lat": 29.7572, "lon": -95.3556},
            "KC": {"stadium": "Kauffman Stadium", "lat": 39.0516, "lon": -94.4804},
            "LAA": {"stadium": "Angel Stadium", "lat": 33.8003, "lon": -117.8827},
            "LAD": {"stadium": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400},
            "MIA": {"stadium": "LoanDepot Park", "lat": 25.7780, "lon": -80.2195},
            "MIL": {"stadium": "American Family Field", "lat": 43.0280, "lon": -87.9711},
            "MIN": {"stadium": "Target Field", "lat": 44.9817, "lon": -93.2789},
            "NYM": {"stadium": "Citi Field", "lat": 40.7571, "lon": -73.8458},
            "NYY": {"stadium": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262},
            "OAK": {"stadium": "Oakland Coliseum", "lat": 37.7516, "lon": -122.2005},
            "PHI": {"stadium": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665},
            "PIT": {"stadium": "PNC Park", "lat": 40.4469, "lon": -80.0057},
            "SD": {"stadium": "Petco Park", "lat": 32.7073, "lon": -117.1573},
            "SEA": {"stadium": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325},
            "SF": {"stadium": "Oracle Park", "lat": 37.7786, "lon": -122.3893},
            "STL": {"stadium": "Busch Stadium", "lat": 38.6226, "lon": -90.1928},
            "TB": {"stadium": "Tropicana Field", "lat": 27.7683, "lon": -82.6534},
            "TEX": {"stadium": "Globe Life Field", "lat": 32.7473, "lon": -97.0847},
            "TOR": {"stadium": "Rogers Centre", "lat": 43.6414, "lon": -79.3894},
            "WSH": {"stadium": "Nationals Park", "lat": 38.8728, "lon": -77.0075},
        }

    def get_home_team(self, matchup_str):
        parts = matchup_str.strip().upper().split("VS")
        if len(parts) != 2:
            raise ValueError("Matchup format must be like 'NYY vs SEA'")
        return parts[1].strip()

    def get_forecast(self, matchup_str, game_hour=19, game_date=None):
        home_team = self.get_home_team(matchup_str)
        if home_team not in self.team_ballparks:
            raise ValueError(f"Unknown team code: {home_team}")

        park = self.team_ballparks[home_team]
        lat, lon = park["lat"], park["lon"]
        stadium = park["stadium"]

        if not game_date:
            game_date = datetime.now().strftime('%Y-%m-%d')

        url = f"http://api.weatherapi.com/v1/forecast.json?key={self.api_key}&q={lat},{lon}&dt={game_date}"
        response = requests.get(url)
        data = response.json()

        if "forecast" not in data:
            raise Exception(f"Weather API error: {data.get('error', {}).get('message')}")

        hour_data = data["forecast"]["forecastday"][0]["hour"][game_hour]

        return {
            "stadium": stadium,
            "location": f"{lat},{lon}",
            "datetime": f"{game_date} {game_hour}:00",
            "condition": hour_data['condition']['text'],
            "temp_f": hour_data['temp_f'],
            "feelslike_f": hour_data['feelslike_f'],
            "humidity": hour_data['humidity'],
            "wind_mph": hour_data['wind_mph'],
            "wind_dir": hour_data['wind_dir'],
            "gust_mph": hour_data['gust_mph'],
            "precip_in": hour_data['precip_in'],
            "cloud": hour_data['cloud'],
            "uv": hour_data['uv']
        }

    def calculate_hr_multiplier(self, weather):
        multiplier = 1.00
        temp = weather["temp_f"]
        humidity = weather["humidity"]
        wind = weather["wind_mph"]
        wind_dir = weather["wind_dir"].upper()

        if temp > 70:
            multiplier += 0.01 * ((temp - 70) // 5)
        if humidity > 50:
            multiplier += 0.01 * ((humidity - 50) // 10)
        if "OUT" in wind_dir or wind_dir in ("W", "WSW", "SW", "WNW"):
            multiplier += min(0.15, wind * 0.01)
        elif wind_dir in ("IN", "N", "NNE", "NNW"):
            multiplier -= min(0.10, wind * 0.01)

        return round(multiplier, 3)

    def pretty_print(self, weather, hr_mult=None, park_factor=None):
        print(f"\n🌤️ Weather Forecast for {weather['stadium']} ({weather['datetime']}):")
        print(f"Condition: {weather['condition']}")
        print(f"Temperature: {weather['temp_f']}°F (Feels like: {weather['feelslike_f']}°F)")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Wind: {weather['wind_mph']} mph {weather['wind_dir']}")
        print(f"Gusts: {weather['gust_mph']} mph")
        print(f"Precipitation: {weather['precip_in']} in")
        print(f"Cloud Cover: {weather['cloud']}%")
        print(f"UV Index: {weather['uv']}")
        if hr_mult:
            print(f"💣 Weather Home Run Multiplier: x{hr_mult}")
        if park_factor:
            print(f"🏟️ Park Factor (HR): {park_factor}")

def get_park_factors():
    """
    Fetch park factors from FanGraphs or use static values for 2025.
    Returns a dictionary mapping team abbreviations to HR park factors.
    Note: FanGraphs park factors are normalized around 100 (100 = league average).
    """
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

def standardize_name(name, is_statcast_format=False):
    try:
        if is_statcast_format:
            last, first = name.strip().split(', ', 1)
            return f"{first.strip().split()[0]} {last.strip()}"
        return name.strip()
    except:
        return name.strip()

def standardize_team_short_name(team):
    if not team:
        return None
    team = team.strip()
    team_short_mapping = {
        'New York Yankees': 'NYY',
        'Los Angeles Dodgers': 'LAD',
        'Chicago Cubs': 'CHC',
        'Chicago White Sox': 'CWS',
        'New York Mets': 'NYM',
        'Oakland Athletics': 'OAK',
        'San Francisco Giants': 'SF',
        'Boston Red Sox': 'BOS',
        'Toronto Blue Jays': 'TOR',
        'Tampa Bay Rays': 'TB',
        'Baltimore Orioles': 'BAL',
        'Cleveland Guardians': 'CLE',
        'Detroit Tigers': 'DET',
        'Kansas City Royals': 'KC',
        'Minnesota Twins': 'MIN',
        'Houston Astros': 'HOU',
        'Los Angeles Angels': 'LAA',
        'Seattle Mariners': 'SEA',
        'Texas Rangers': 'TEX',
        'Atlanta Braves': 'ATL',
        'Miami Marlins': 'MIA',
        'Philadelphia Phillies': 'PHI',
        'Washington Nationals': 'WSH',
        'Milwaukee Brewers': 'MIL',
        'St. Louis Cardinals': 'STL',
        'Pittsburgh Pirates': 'PIT',
        'Cincinnati Reds': 'CIN',
        'Colorado Rockies': 'COL',
        'Arizona Diamondbacks': 'ARI',
        'San Diego Padres': 'SD',
    }
    return team_short_mapping.get(team, team)

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

def get_hr_prospects(pitcher_name, statcast_df, batter_df, pitcher_df, weather=None, min_pitches=5, team=None, manual_name_mapping=None, home_team=None, park_factors=None):
    statcast_df = statcast_df.copy()
    statcast_df = statcast_df.reset_index(drop=True)
    statcast_df['player_name_standard'] = statcast_df['player_name'].apply(
        lambda x: standardize_name(x, is_statcast_format=True)
    )
    
    # Define barrel criteria
    statcast_df['is_barrel'] = (
        (statcast_df['launch_speed'] >= 95) & 
        (statcast_df['launch_angle'].between(10, 40))
    ).fillna(False).astype(int)
    
    pitcher_statcast = statcast_df[statcast_df['player_name_standard'] == pitcher_name].copy()
    if pitcher_statcast.empty:
        print(f"No data for pitcher {pitcher_name} in statcast_df.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Pull%', 'K%', 'BB%', 'HR/FB', 'Pitcher_HR/9', 'Recent_Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand', 'Weather_Adjustment', 'Park_Factor_Adjustment'])
    
    # Get pitcher handedness
    if 'p_throws' in pitcher_statcast.columns and not pitcher_statcast['p_throws'].isna().all():
        pitcher_hand = pitcher_statcast['p_throws'].mode().iloc[0] if not pitcher_statcast['p_throws'].empty else 'R'
    else:
        pitcher_row = pitcher_df[pitcher_df['Name'] == pitcher_name]
        if not pitcher_row.empty and 'IDfg' in pitcher_row.columns:
            fangraphs_id = pitcher_row['IDfg'].iloc[0]
            id_mapping = playerid_reverse_lookup([fangraphs_id], key_type='fangraphs')
            if not id_mapping.empty:
                mlbam_id = id_mapping['key_mlbam'].iloc[0]
                pitcher_info = playerid_lookup(mlbam_id, key_type='mlbam')
                pitcher_hand = pitcher_info['p_throws'].iloc[0] if not pitcher_info.empty else 'R'
            else:
                pitcher_hand = 'R'
        else:
            pitcher_hand = 'R'
            if not pitcher_row.empty:
                print(f"Warning: 'IDfg' column not found in pitcher_df. Columns available: {pitcher_df.columns}")
    print(f"Pitcher {pitcher_name} handedness: {pitcher_hand}")

    # Add batter handedness
    if 'stand' in statcast_df.columns:
        statcast_df['batter_hand'] = statcast_df['stand'].map({'L': 'L', 'R': 'R'}).fillna('R')
        # print(f"Batter hand distribution in statcast_df: {statcast_df['batter_hand'].value_counts().to_dict()}")
    else:
        print("Warning: 'stand' column not found in statcast_df. Defaulting batter_hand to 'R'.")
        statcast_df['batter_hand'] = 'R'

    pitcher_statcast = pitcher_statcast.merge(
        statcast_df[['batter', 'game_pk', 'at_bat_number', 'batter_hand']].drop_duplicates(),
        on=['batter', 'game_pk', 'at_bat_number'],
        how='left'
    )
    pitcher_statcast['batter_hand'] = pitcher_statcast['batter_hand'].fillna('R')
    # print(f"Batter hand distribution in pitcher_statcast: {pitcher_statcast['batter_hand'].value_counts().to_dict()}")

    # Calculate HR rates by handedness
    hr_by_hand = pitcher_statcast[pitcher_statcast['events'] == 'home_run'].groupby('batter_hand').size()
    total_pa_by_hand = pitcher_statcast.groupby('batter_hand').size()
    hr_rate_by_hand = (hr_by_hand / total_pa_by_hand).fillna(0).to_dict()
    if 'L' not in hr_rate_by_hand:
        hr_rate_by_hand['L'] = 0.035
    if 'R' not in hr_rate_by_hand:
        hr_rate_by_hand['R'] = 0.035
    print(f"HR rates by batter hand vs. {pitcher_name}: {hr_rate_by_hand}")

    # Calculate vulnerable pitch types
    total_pitches_hand = pitcher_statcast.groupby(['pitch_type', 'batter_hand']).size()
    hr_by_pitch_hand = pitcher_statcast[pitcher_statcast['events'] == 'home_run'].groupby(['pitch_type', 'batter_hand']).size()
    barrel_by_pitch_hand = pitcher_statcast[pitcher_statcast['is_barrel'] == 1].groupby(['pitch_type', 'batter_hand']).size()
    hr_rate_pitch_hand = (hr_by_pitch_hand / total_pitches_hand).fillna(0).reset_index(name='hr_rate')
    barrel_rate_pitch_hand = (barrel_by_pitch_hand / total_pitches_hand).fillna(0).reset_index(name='barrel_rate')
    
    pitch_vulnerability = hr_rate_pitch_hand.merge(barrel_rate_pitch_hand, on=['pitch_type', 'batter_hand'], how='outer').fillna(0)
    vulnerable_pitches = pitch_vulnerability[
        (pitch_vulnerability['hr_rate'] > 0) | (pitch_vulnerability['barrel_rate'] > 0.05)
    ][['pitch_type', 'batter_hand']].drop_duplicates()
    
    all_pitches = pitcher_statcast['pitch_type'].unique()
    for hand in ['L', 'R']:
        for pitch in all_pitches:
            if not ((vulnerable_pitches['pitch_type'] == pitch) & (vulnerable_pitches['batter_hand'] == hand)).any():
                vulnerable_pitches = pd.concat([
                    vulnerable_pitches,
                    pd.DataFrame({'pitch_type': [pitch], 'batter_hand': [hand]})
                ], ignore_index=True)
    # print(f"Vulnerable pitch types by batter hand: {vulnerable_pitches.to_dict()}")

    if vulnerable_pitches.empty:
        print(f"No vulnerable pitches identified for {pitcher_name}. Using all pitches.")
        vulnerable_pitches = pd.DataFrame({
            'pitch_type': all_pitches,
            'batter_hand': ['R'] * len(all_pitches)
        })

    # Calculate recent Flyball% (past week: May 7 to May 14, 2025)
    recent_statcast = statcast_df[
        (statcast_df['game_date'] >= '2025-05-07') & 
        (statcast_df['game_date'] <= '2025-05-14')
    ].copy()
    recent_statcast['is_flyball'] = (
        (recent_statcast['bb_type'] == 'fly_ball') | 
        (recent_statcast['launch_angle'] >= 25)
    ).fillna(False).astype(int)
    recent_flyball = recent_statcast.groupby('batter').agg({
        'is_flyball': 'mean',
        'events': 'count'
    }).rename(columns={'is_flyball': 'Recent_Flyball%'}).reset_index()
    recent_flyball = recent_flyball[recent_flyball['events'] >= 5]

    batter_stats = []
    for _, row in vulnerable_pitches.iterrows():
        pitch_type, batter_hand = row['pitch_type'], row['batter_hand']
        pitch_data = statcast_df[(statcast_df['pitch_type'] == pitch_type) & (statcast_df['batter_hand'] == batter_hand)].groupby('batter').agg({
            'events': [
                lambda x: (x == 'home_run').sum(),
                lambda x: (x.notna()).sum()
            ],
            'is_barrel': 'mean',
            'launch_speed': 'mean',
            'launch_angle': 'mean',
            'woba_value': 'mean',
            'batter_hand': 'first'
        })
        
        pitch_data.columns = ['hr_count', 'pitches_faced', 'barrel%', 'exit_velo', 'launch_angle', 'woba', 'batter_hand']
        pitch_data['PA'] = pitch_data['pitches_faced'] / 4
        pitch_data['hr_rate'] = pitch_data['hr_count'] / pitch_data['PA'].replace(0, np.nan)
        pitch_data['hr_rate'] = pitch_data['hr_rate'].fillna(0.01)
        pitch_data['hardhit%'] = pitch_data['exit_velo'].apply(lambda x: 1 if pd.notna(x) and x >= 95 else 0)
        pitch_data['pitch_type'] = pitch_type
        batter_stats.append(pitch_data.reset_index())
    
    if batter_stats:
        batter_performance = pd.concat(batter_stats, ignore_index=True)
        batter_performance = batter_performance[batter_performance['pitches_faced'] >= min_pitches]
        # print(f"Batter hand distribution in batter_performance: {batter_performance['batter_hand'].value_counts().to_dict()}")
    else:
        print("No batter data for vulnerable pitch types.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Pull%', 'K%', 'BB%', 'HR/FB', 'Pitcher_HR/9', 'Recent_Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand', 'Weather_Adjustment', 'Park_Factor_Adjustment'])
    
    unique_batter_ids = batter_performance['batter'].unique()
    name_mapping = create_name_mapping(unique_batter_ids, manual_name_mapping)
    if name_mapping.empty:
        print("Name mapping failed.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Pull%', 'K%', 'BB%', 'HR/FB', 'Pitcher_HR/9', 'Recent_Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand', 'Weather_Adjustment', 'Park_Factor_Adjustment'])
    
    batter_performance = batter_performance.merge(name_mapping, left_on='batter', right_on='mlbam_id', how='left')
    
    # Add batter metrics
    batter_metrics = batter_df[['Name', 'Team', 'HR', 'ISO', 'Barrel%', 'HardHit%', 'HR/FB', 'wRC+', 'AVG', 'FB%', 'Pull%', 'K%', 'BB%']].copy()
    batter_metrics['Team'] = batter_metrics['Team'].apply(standardize_team_short_name)
    batter_metrics = batter_metrics.rename(columns={'FB%': 'Flyball%'})
    
    batter_performance = batter_performance.merge(batter_metrics, left_on='name', right_on='Name', how='left')
    
    # Add recent Flyball%
    batter_performance = batter_performance.merge(recent_flyball[['batter', 'Recent_Flyball%']], on='batter', how='left')
    
    # Add pitcher metrics (HR/9 for the specific pitcher)
    pitcher_metrics = pitcher_df[pitcher_df['Name'] == pitcher_name][['Name', 'HR/9']].copy()
    if not pitcher_metrics.empty:
        pitcher_hr9 = pitcher_metrics['HR/9'].iloc[0]
    else:
        pitcher_hr9 = pitcher_df['HR/9'].mean()
    batter_performance['Pitcher_HR/9'] = pitcher_hr9
    
    # Fill missing metrics with league averages
    for col in ['Flyball%', 'Barrel%', 'wRC+', 'HardHit%', 'Pull%', 'K%', 'BB%', 'HR/FB', 'Recent_Flyball%']:
        batter_performance[col] = batter_performance[col].fillna(batter_metrics[col].mean() if col in batter_metrics else recent_flyball['Recent_Flyball%'].mean())
    
    if batter_performance['Name'].isna().all():
        print("Merge with batter_df failed.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Pull%', 'K%', 'BB%', 'HR/FB', 'Pitcher_HR/9', 'Recent_Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand', 'Weather_Adjustment', 'Park_Factor_Adjustment'])
    
    batter_performance = batter_performance.rename(columns={'Name': 'batter_name'}).drop(columns=['name'], errors='ignore')
    
    if team:
        team = standardize_team_short_name(team)
        batter_performance = batter_performance[batter_performance['Team'] == team]
        if batter_performance.empty:
            print(f"No batters found for team {team}.")
            return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Pull%', 'K%', 'BB%', 'HR/FB', 'Pitcher_HR/9', 'Recent_Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand', 'Weather_Adjustment', 'Park_Factor_Adjustment'])
    
    # Check team batter handedness
    stl_batters = batter_df[batter_df['Team'] == team][['Name']]
    stl_batter_ids = batter_performance[batter_performance['Team'] == team]['batter'].unique()
    stl_hand_mapping = statcast_df[statcast_df['batter'].isin(stl_batter_ids)][['batter', 'batter_hand']].drop_duplicates()
    print(f"{team} batter handedness: {stl_hand_mapping.to_dict()}")

    # Prepare data for logistic regression
    features = ['barrel%', 'Flyball%', 'wRC+', 'HardHit%', 'Pull%', 'K%', 'BB%', 'HR/FB', 'Pitcher_HR/9', 'Recent_Flyball%']
    X = batter_performance[features].fillna(0)
    y = (batter_performance['hr_count'] > 0).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression model
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    try:
        model.fit(X_scaled, y)
        batter_performance['hr_probability'] = model.predict_proba(X_scaled)[:, 1]
    except:
        batter_performance['hr_probability'] = batter_performance['hr_rate']
    
    # Adjust probabilities based on handedness
    batter_performance['hand_adjust'] = batter_performance['batter_hand'].map(hr_rate_by_hand).fillna(0.035)
    batter_performance['hr_probability'] = batter_performance['hr_probability'] * (1 + batter_performance['hand_adjust'])
    
    # Apply weather adjustment
    weather_adjustment = 1.0
    if weather:
        weather_obj = BallparkWeather(api_key=None)
        weather_adjustment = weather_obj.calculate_hr_multiplier(weather)
    
    batter_performance['hr_probability'] = batter_performance['hr_probability'] * weather_adjustment
    batter_performance['weather_adjust'] = (weather_adjustment - 1) * 100
    
    # Apply park factor adjustment
    park_factor_adjustment = 1.0
    if park_factors and home_team:
        park_factor = park_factors.get(home_team, 100)
        park_factor_adjustment = park_factor / 100
    batter_performance['hr_probability'] = batter_performance['hr_probability'] * park_factor_adjustment
    batter_performance['park_factor_adjust'] = (park_factor_adjustment - 1) * 100
    
    # Scale probabilities
    league_avg_hr_rate = 0.035
    scaling_factor = 3
    batter_performance['hr_probability'] = np.clip(
        batter_performance['hr_probability'] * scaling_factor,
        league_avg_hr_rate,
        0.15
    )
    
    # Pitch usage by handedness
    pitch_counts = pitcher_statcast.groupby(['pitch_type', 'batter_hand']).size()
    total_pitches = pitch_counts.groupby('batter_hand').sum()
    pitch_usage = (pitch_counts / total_pitches).reset_index(name='usage')
    
    batter_performance = batter_performance.merge(
        pitch_usage,
        on=['pitch_type', 'batter_hand'],
        how='left'
    )
    batter_performance['usage'] = batter_performance['usage'].fillna(0.1)
    batter_performance['weighted_hr_prob'] = batter_performance['hr_probability'] * batter_performance['usage']
    
    # Aggregate by batter
    batter_summary = batter_performance.groupby(['batter_name', 'Team', 'batter_hand']).agg({
        'PA': 'sum',
        'weighted_hr_prob': 'sum',
        'Barrel%': 'mean',
        'HardHit%': 'mean',
        'wRC+': 'mean',
        'AVG': 'mean',
        'Flyball%': 'mean',
        'Pull%': 'mean',
        'K%': 'mean',
        'BB%': 'mean',
        'HR/FB': 'mean',
        'Pitcher_HR/9': 'mean',
        'Recent_Flyball%': 'mean',
        'pitch_type': lambda x: ', '.join(sorted(set(x))),
        'weather_adjust': 'mean',
        'park_factor_adjust': 'mean'
    }).reset_index()
    
    # Update matchup score with new features
    batter_summary['matchup_score'] = (
        batter_summary['weighted_hr_prob'] * 0.3 +
        batter_summary['Barrel%'] * 0.2 +
        batter_summary['HardHit%'] * 0.15 +
        batter_summary['wRC+'] / 100 * 0.1 +
        batter_summary['Pull%'] * 0.1 +
        batter_summary['HR/FB'] * 0.1 +
        batter_summary['Recent_Flyball%'] * 0.05
    )
    
    batter_summary = batter_summary[batter_summary['PA'] >= 5]
    print(f"Batter hand distribution in batter_summary: {batter_summary['batter_hand'].value_counts().to_dict()}")
    
    prospects = batter_summary.sort_values('matchup_score', ascending=False)[
        ['batter_name', 'Team', 'PA', 'weighted_hr_prob', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Pull%', 'K%', 'BB%', 'HR/FB', 'Pitcher_HR/9', 'Recent_Flyball%', 'pitch_type', 'batter_hand', 'weather_adjust', 'park_factor_adjust']
    ].rename(columns={
        'batter_name': 'Name',
        'weighted_hr_prob': 'hr_probability',
        'pitch_type': 'Vulnerable_Pitch_Types',
        'batter_hand': 'Batter_Hand',
        'weather_adjust': 'Weather_Adjustment',
        'park_factor_adjust': 'Park_Factor_Adjustment'
    })
    
    return prospects.head(10)

def process_today_schedule(api_key):
    TODAY_SLASHED = "05/14/2025"
    tf = TimezoneFinder()
    
    # Fetch park factors
    park_factors = get_park_factors()
    
    # Fetch schedule
    schedule = statsapi.schedule(date=TODAY_SLASHED, sportId=1, include_series_status=True)
    
    # Initialize BallparkWeather
    weather_obj = BallparkWeather(api_key)
    
    all_prospects = []
    
    for game in schedule:
        # Get game details
        home_team = game['home_name']
        away_team = game['away_name']
        home_pitcher = game['home_probable_pitcher']
        away_pitcher = game['away_probable_pitcher']
        game_datetime = datetime.strptime(game['game_datetime'], '%Y-%m-%dT%H:%M:%SZ')
        game_datetime = pytz.utc.localize(game_datetime)
        
        # Convert team names to abbreviations
        home_team_abbr = standardize_team_short_name(home_team)
        away_team_abbr = standardize_team_short_name(away_team)
        
        if not home_team_abbr or not away_team_abbr:
            print(f"Skipping game {away_team} vs {home_team}: Invalid team abbreviation")
            continue
        
        # Get ballpark timezone
        park = weather_obj.team_ballparks.get(home_team_abbr)
        if not park:
            print(f"Skipping game {away_team} vs {home_team}: No ballpark data for {home_team_abbr}")
            continue
        
        lat, lon = park['lat'], park['lon']
        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        if not timezone_str:
            print(f"Skipping game {away_team} vs {home_team}: Could not determine timezone")
            continue
        
        timezone = pytz.timezone(timezone_str)
        local_game_time = game_datetime.astimezone(timezone)
        game_hour = local_game_time.hour
        game_date = local_game_time.strftime('%Y-%m-%d')
        
        # Get weather forecast
        matchup = f"{away_team_abbr} vs {home_team_abbr}"
        try:
            weather = weather_obj.get_forecast(matchup, game_hour=game_hour, game_date=game_date)
            park_factor = park_factors.get(home_team_abbr, 100)
            # weather_obj.pretty_print(weather, weather_obj.calculate_hr_multiplier(weather), park_factor)
        except Exception as e:
            print(f"Failed to get weather for {matchup}: {e}")
            weather = None
        
        # Process home pitcher vs away team
        print(f"\nProcessing {home_pitcher} vs {away_team_abbr}")
        home_prospects = get_hr_prospects(
            pitcher_name=home_pitcher,
            statcast_df=statcast_df,
            batter_df=batter_df,
            pitcher_df=pitcher_df,
            weather=weather,
            min_pitches=5,
            team=away_team_abbr,
            manual_name_mapping=None,
            home_team=home_team_abbr,
            park_factors=park_factors
        )
        if not home_prospects.empty:
            home_prospects['Game'] = matchup
            home_prospects['Pitcher'] = home_pitcher
            all_prospects.append(home_prospects)
        
        # Process away pitcher vs home team
        print(f"\nProcessing {away_pitcher} vs {home_team_abbr}")
        away_prospects = get_hr_prospects(
            pitcher_name=away_pitcher,
            statcast_df=statcast_df,
            batter_df=batter_df,
            pitcher_df=pitcher_df,
            weather=weather,
            min_pitches=5,
            team=home_team_abbr,
            manual_name_mapping=None,
            home_team=home_team_abbr,
            park_factors=park_factors
        )
        if not away_prospects.empty:
            away_prospects['Game'] = matchup
            away_prospects['Pitcher'] = away_pitcher
            all_prospects.append(away_prospects)
    
    if all_prospects:
        final_results = pd.concat(all_prospects, ignore_index=True)
        # Sort by hr_probability in descending order
        final_results = final_results.sort_values('hr_probability', ascending=False)
        print("\nFinal Home Run Prospects for All Games (Sorted by HR Probability):")
        print(final_results)
        return final_results
    else:
        print("No prospects found for today's games.")
        return pd.DataFrame()

# Run the process
api_key = ""  # Replace with your WeatherAPI key
results = process_today_schedule(api_key)