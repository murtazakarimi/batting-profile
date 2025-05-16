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

pybaseball.cache.enable()

# Load data for 2025 season
batter_df = batting_stats(2025, end_season=None, league='all', qual=1, ind=1)
pitcher_df = pitching_stats(2025, end_season=None, league='all', qual=1, ind=1)
statcast_df = statcast(start_dt='2025-03-10', end_dt='2025-05-13', team=None)

try:
    from pybaseball import chadwick_register
except ImportError:
    chadwick_register = None

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
    team = team.strip().upper()
    team_short_mapping = {
        'NYY': 'NYY', 'Yankees': 'NYY', 'New York Yankees': 'NYY',
        'LAD': 'LAD', 'Dodgers': 'LAD', 'Los Angeles Dodgers': 'LAD',
        'CHC': 'CHC', 'Cubs': 'CHC', 'Chicago Cubs': 'CHC',
        'CHW': 'CHW', 'White Sox': 'CHW', 'Chicago White Sox': 'CHW',
        'NYM': 'NYM', 'Mets': 'NYM', 'New York Mets': 'NYM',
        'OAK': 'OAK', 'Athletics': 'OAK', 'Oakland Athletics': 'OAK',
        'SFG': 'SFG', 'Giants': 'SFG', 'San Francisco Giants': 'SFG',
        'BOS': 'BOS', 'Red Sox': 'BOS', 'Boston Red Sox': 'BOS',
        'TOR': 'TOR', 'Blue Jays': 'TOR', 'Toronto Blue Jays': 'TOR',
        'TBR': 'TBR', 'Rays': 'TBR', 'Tampa Bay Rays': 'TBR',
        'BAL': 'BAL', 'Orioles': 'BAL', 'Baltimore Orioles': 'BAL',
        'CLE': 'CLE', 'Guardians': 'CLE', 'Cleveland Guardians': 'CLE',
        'DET': 'DET', 'Tigers': 'DET', 'Detroit Tigers': 'DET',
        'KCR': 'KCR', 'Royals': 'KCR', 'Kansas City Royals': 'KCR',
        'MIN': 'MIN', 'Twins': 'MIN', 'Minnesota Twins': 'MIN',
        'HOU': 'HOU', 'Astros': 'HOU', 'Houston Astros': 'HOU',
        'LAA': 'LAA', 'Angels': 'LAA', 'Los Angeles Angels': 'LAA',
        'SEA': 'SEA', 'Mariners': 'SEA', 'Seattle Mariners': 'SEA',
        'TEX': 'TEX', 'Rangers': 'TEX', 'Texas Rangers': 'TEX',
        'ATL': 'ATL', 'Braves': 'ATL', 'Atlanta Braves': 'ATL',
        'MIA': 'MIA', 'Marlins': 'MIA', 'Miami Marlins': 'MIA',
        'PHI': 'PHI', 'Phillies': 'PHI', 'Philadelphia Phillies': 'PHI',
        'WSN': 'WSN', 'Nationals': 'WSN', 'Washington Nationals': 'WSN',
        'MIL': 'MIL', 'Brewers': 'MIL', 'Milwaukee Brewers': 'MIL',
        'STL': 'STL', 'Cardinals': 'STL', 'St. Louis Cardinals': 'STL',
        'PIT': 'PIT', 'Pirates': 'PIT', 'Pittsburgh Pirates': 'PIT',
        'CIN': 'CIN', 'Reds': 'CIN', 'Cincinnati Reds': 'CIN',
        'COL': 'COL', 'Rockies': 'COL', 'Colorado Rockies': 'COL',
        'ARI': 'ARI', 'Diamondbacks': 'ARI', 'Arizona Diamondbacks': 'ARI',
        'SDP': 'SDP', 'Padres': 'SDP', 'San Diego Padres': 'SDP',
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

def get_hr_prospects(pitcher_name, statcast_df, batter_df, pitcher_df, min_pitches=5, team=None, manual_name_mapping=None):
    statcast_df = statcast_df.copy()
    # Reset index to ensure unique indices
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
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand'])
    
    # Get pitcher handedness from statcast_df['p_throws']
    if 'p_throws' in pitcher_statcast.columns and not pitcher_statcast['p_throws'].isna().all():
        pitcher_hand = pitcher_statcast['p_throws'].mode().iloc[0] if not pitcher_statcast['p_throws'].empty else 'R'
    else:
        # Fallback to playerid_reverse_lookup
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

    # Add batter handedness from statcast_df['stand']
    if 'stand' in statcast_df.columns:
        statcast_df['batter_hand'] = statcast_df['stand'].map({'L': 'L', 'R': 'R'}).fillna('R')
        print(f"Batter hand distribution in statcast_df: {statcast_df['batter_hand'].value_counts().to_dict()}")
    else:
        print("Warning: 'stand' column not found in statcast_df. Defaulting batter_hand to 'R'.")
        statcast_df['batter_hand'] = 'R'

    # Merge batter_hand into pitcher_statcast using batter and game identifiers
    pitcher_statcast = pitcher_statcast.merge(
        statcast_df[['batter', 'game_pk', 'at_bat_number', 'batter_hand']].drop_duplicates(),
        on=['batter', 'game_pk', 'at_bat_number'],
        how='left'
    )
    pitcher_statcast['batter_hand'] = pitcher_statcast['batter_hand'].fillna('R')
    print(f"Batter hand distribution in pitcher_statcast: {pitcher_statcast['batter_hand'].value_counts().to_dict()}")

    # Calculate HR rates by pitcher hand vs. batter hand
    hr_by_hand = pitcher_statcast[pitcher_statcast['events'] == 'home_run'].groupby('batter_hand').size()
    total_pa_by_hand = pitcher_statcast.groupby('batter_hand').size()
    hr_rate_by_hand = (hr_by_hand / total_pa_by_hand).fillna(0).to_dict()
    # Add league average HR rate for missing handedness
    if 'L' not in hr_rate_by_hand:
        hr_rate_by_hand['L'] = 0.035  # League average
    if 'R' not in hr_rate_by_hand:
        hr_rate_by_hand['R'] = 0.035
    print(f"HR rates by batter hand vs. {pitcher_name}: {hr_rate_by_hand}")

    # Calculate vulnerable pitch types by handedness, including all pitches thrown
    total_pitches_hand = pitcher_statcast.groupby(['pitch_type', 'batter_hand']).size()
    hr_by_pitch_hand = pitcher_statcast[pitcher_statcast['events'] == 'home_run'].groupby(['pitch_type', 'batter_hand']).size()
    barrel_by_pitch_hand = pitcher_statcast[pitcher_statcast['is_barrel'] == 1].groupby(['pitch_type', 'batter_hand']).size()
    hr_rate_pitch_hand = (hr_by_pitch_hand / total_pitches_hand).fillna(0).reset_index(name='hr_rate')
    barrel_rate_pitch_hand = (barrel_by_pitch_hand / total_pitches_hand).fillna(0).reset_index(name='barrel_rate')
    
    # Combine HR and barrel rates to identify vulnerable pitches
    pitch_vulnerability = hr_rate_pitch_hand.merge(barrel_rate_pitch_hand, on=['pitch_type', 'batter_hand'], how='outer').fillna(0)
    vulnerable_pitches = pitch_vulnerability[
        (pitch_vulnerability['hr_rate'] > 0) | (pitch_vulnerability['barrel_rate'] > 0.05)
    ][['pitch_type', 'batter_hand']].drop_duplicates()
    
    # Add all pitch types thrown to both handedness types
    all_pitches = pitcher_statcast['pitch_type'].unique()
    for hand in ['L', 'R']:
        for pitch in all_pitches:
            if not ((vulnerable_pitches['pitch_type'] == pitch) & (vulnerable_pitches['batter_hand'] == hand)).any():
                vulnerable_pitches = pd.concat([
                    vulnerable_pitches,
                    pd.DataFrame({'pitch_type': [pitch], 'batter_hand': [hand]})
                ], ignore_index=True)
    print(f"Vulnerable pitch types by batter hand: {vulnerable_pitches.to_dict()}")

    if vulnerable_pitches.empty:
        print(f"No vulnerable pitches identified for {pitcher_name}. Using all pitches.")
        vulnerable_pitches = pd.DataFrame({
            'pitch_type': all_pitches,
            'batter_hand': ['R'] * len(all_pitches)
        })

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
        print(f"Batter hand distribution in batter_performance: {batter_performance['batter_hand'].value_counts().to_dict()}")
    else:
        print("No batter data for vulnerable pitch types.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand'])
    
    unique_batter_ids = batter_performance['batter'].unique()
    name_mapping = create_name_mapping(unique_batter_ids, manual_name_mapping)
    if name_mapping.empty:
        print("Name mapping failed.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand'])
    
    batter_performance = batter_performance.merge(name_mapping, left_on='batter', right_on='mlbam_id', how='left')
    
    batter_metrics = batter_df[['Name', 'Team', 'HR', 'ISO', 'Barrel%', 'HardHit%', 'HR/FB', 'wRC+', 'AVG', 'FB%']].copy()
    batter_metrics['Team'] = batter_metrics['Team'].apply(standardize_team_short_name)
    batter_metrics = batter_metrics.rename(columns={'FB%': 'Flyball%'})
    
    batter_performance = batter_performance.merge(batter_metrics, left_on='name', right_on='Name', how='left')
    
    # Fill missing metrics with league averages
    batter_performance['Flyball%'] = batter_performance['Flyball%'].fillna(batter_metrics['Flyball%'].mean())
    batter_performance['Barrel%'] = batter_performance['Barrel%'].fillna(batter_metrics['Barrel%'].mean())
    batter_performance['wRC+'] = batter_performance['wRC+'].fillna(batter_metrics['wRC+'].mean())
    batter_performance['HardHit%'] = batter_performance['HardHit%'].fillna(batter_metrics['HardHit%'].mean())
    
    if batter_performance['Name'].isna().all():
        print("Merge with batter_df failed.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand'])
    
    batter_performance = batter_performance.rename(columns={'Name': 'batter_name'}).drop(columns=['name'], errors='ignore')
    
    if team:
        team = standardize_team_short_name(team)
        batter_performance = batter_performance[batter_performance['Team'] == team]
        if batter_performance.empty:
            print(f"No batters found for team {team}.")
            return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand'])
    
    # Check STL batter handedness
    stl_batters = batter_df[batter_df['Team'] == team][['Name']]
    stl_batter_ids = batter_performance[batter_performance['Team'] == team]['batter'].unique()
    stl_hand_mapping = statcast_df[statcast_df['batter'].isin(stl_batter_ids)][['batter', 'batter_hand']].drop_duplicates()
    print(f"STL batter handedness: {stl_hand_mapping.to_dict()}")

    # Prepare data for logistic regression
    features = ['barrel%', 'Flyball%', 'wRC+', 'HardHit%']
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
    
    # Scale probabilities to realistic range (8-12% for top hitters)
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
        'pitch_type': lambda x: ', '.join(sorted(set(x)))
    }).reset_index()
    
    batter_summary['matchup_score'] = (
        batter_summary['weighted_hr_prob'] * 0.4 +
        batter_summary['Barrel%'] * 0.3 +
        batter_summary['HardHit%'] * 0.2 +
        batter_summary['wRC+'] / 100 * 0.1
    )
    
    batter_summary = batter_summary[batter_summary['PA'] >= 5]
    print(f"Batter hand distribution in batter_summary: {batter_summary['batter_hand'].value_counts().to_dict()}")
    
    prospects = batter_summary.sort_values('matchup_score', ascending=False)[
        ['batter_name', 'Team', 'PA', 'weighted_hr_prob', 'matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'pitch_type', 'batter_hand']
    ].rename(columns={
        'batter_name': 'Name',
        'weighted_hr_prob': 'hr_probability',
        'pitch_type': 'Vulnerable_Pitch_Types',
        'batter_hand': 'Batter_Hand'
    })
    
    return prospects.head(10)

# Example usage
pitcher_name = "Aaron Nola"
team = "STL"
hr_prospects = get_hr_prospects(
    pitcher_name, statcast_df, batter_df, pitcher_df, 
    min_pitches=5, team=team, manual_name_mapping=None
)
print("Top Home Run Prospects vs.", pitcher_name)
print(hr_prospects)