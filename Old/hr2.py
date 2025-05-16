import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import pybaseball
from pybaseball import statcast
from pybaseball import batting_stats
from pybaseball import pitching_stats
from pybaseball import playerid_reverse_lookup

pybaseball.cache.enable()

batter_df = batting_stats(2025, end_season=None, league='all', qual=1, ind=1)
pitcher_df = pitching_stats(2025, end_season=None, league='all', qual=1, ind=1)
statcast_df = statcast(start_dt='2025-03-10', end_dt='2025-05-13', team=None)

def standardize_name(name, is_statcast_format=False):
    try:
        if is_statcast_format:
            last, first = name.strip().split(', ', 1)
            name = f"{first.strip().split()[0]} {last.strip()}"
        name = ' '.join(name.strip().split()).title()
        return name
    except:
        return name.strip().title()

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
        'SFG': 'SFG', 'SF': 'SFG', 'Giants': 'SFG', 'San Francisco Giants': 'SFG',
        'BOS': 'BOS', 'Red Sox': 'BOS', 'Boston Red Sox': 'BOS',
        'TOR': 'TOR', 'Blue Jays': 'TOR', 'Toronto Blue Jays': 'TOR',
        'TBR': 'TBR', 'TB': 'TBR', 'Rays': 'TBR', 'Tampa Bay Rays': 'TBR',
        'BAL': 'BAL', 'Orioles': 'BAL', 'Baltimore Orioles': 'BAL',
        'CLE': 'CLE', 'Guardians': 'CLE', 'Cleveland Guardians': 'CLE',
        'DET': 'DET', 'Tigers': 'DET', 'Detroit Tigers': 'DET',
        'KCR': 'KCR', 'Royals': 'KCR', 'Kansas City Royals': 'KCR', 'KC': 'KCR',
        'MIN': 'MIN', 'Twins': 'MIN', 'Minnesota Twins': 'MIN',
        'HOU': 'HOU', 'Astros': 'HOU', 'Houston Astros': 'HOU',
        'LAA': 'LAA', 'Angels': 'LAA', 'Los Angeles Angels': 'LAA', 'ANA': 'LAA',
        'SEA': 'SEA', 'Mariners': 'SEA', 'Seattle Mariners': 'SEA',
        'TEX': 'TEX', 'Rangers': 'TEX', 'Texas Rangers': 'TEX',
        'ATL': 'ATL', 'Braves': 'ATL', 'Atlanta Braves': 'ATL',
        'MIA': 'MIA', 'Marlins': 'MIA', 'Miami Marlins': 'MIA', 'FLA': 'MIA',
        'PHI': 'PHI', 'Phillies': 'PHI', 'Philadelphia Phillies': 'PHI',
        'WSN': 'WSN', 'Nationals': 'WSN', 'Washington Nationals': 'WSN', 'WSH': 'WSN',
        'MIL': 'MIL', 'Brewers': 'MIL', 'Milwaukee Brewers': 'MIL',
        'STL': 'STL', 'Cardinals': 'STL', 'St. Louis Cardinals': 'STL',
        'PIT': 'PIT', 'Pirates': 'PIT', 'Pittsburgh Pirates': 'PIT',
        'CIN': 'CIN', 'Reds': 'CIN', 'Cincinnati Reds': 'CIN',
        'COL': 'COL', 'Rockies': 'COL', 'Colorado Rockies': 'COL',
        'ARI': 'ARI', 'Diamondbacks': 'ARI', 'Arizona Diamondbacks': 'ARI', 'AZ': 'ARI',
        'SDP': 'SDP', 'Padres': 'SDP', 'San Diego Padres': 'SDP', 'SD': 'SDP',
    }
    return team_short_mapping.get(team, team)

def create_name_mapping(batter_ids, statcast_df):
    batter_ids = [int(id) for id in batter_ids]
    lookup = playerid_reverse_lookup(batter_ids, key_type='mlbam')
    print(f"playerid_reverse_lookup returned {len(lookup)} records")
    
    mapping = pd.DataFrame({
        'mlbam_id': lookup['key_mlbam'].astype(str),
        'name': (lookup['name_first'] + ' ' + lookup['name_last']).apply(standardize_name),
        'Bats': lookup['bat_side'].map({'Right': 'R', 'Left': 'L', 'Switch': 'B'}).fillna('R') if 'bat_side' in lookup.columns else 'R'
    })
    
    missing_ids = set(batter_ids) - set(int(id) for id in lookup['key_mlbam'])
    if missing_ids:
        print(f"Missing IDs in lookup: {missing_ids}")
        missing_df = pd.DataFrame({
            'mlbam_id': [str(id) for id in missing_ids],
            'name': [f"Unknown_{id}" for id in missing_ids],
            'Bats': 'R'
        })
        mapping = pd.concat([mapping, missing_df], ignore_index=True)
    
    print(f"Name mapping created for {len(mapping)} batters")
    return mapping

def get_hr_prospects(pitcher_name, statcast_df, batter_df, pitcher_df, min_pitches=5, team=None):
    statcast_df = statcast_df.copy()
    
    statcast_df['home_team'] = statcast_df['home_team'].apply(standardize_team_short_name)
    statcast_df['away_team'] = statcast_df['away_team'].apply(standardize_team_short_name)
    
    statcast_df['player_name_standard'] = statcast_df['player_name'].apply(
        lambda x: standardize_name(x, is_statcast_format=True)
    )
    
    statcast_df['is_barrel'] = (
        (statcast_df['launch_speed'] >= 95) & 
        (statcast_df['launch_angle'].between(10, 40))
    ).fillna(False).astype(int)
    
    # Determine batter_team using inning_top_bot
    if 'inning_top_bot' not in statcast_df.columns:
        print("Error: 'inning_top_bot' column missing in statcast_df.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'xwOBA', 'AVG', 'Flyball%', 'Vulnerable_Pitch_Types', 'Expected_HR'])
    statcast_df['batter_team'] = statcast_df.apply(
        lambda x: x['away_team'] if x['inning_top_bot'] == 'top' else x['home_team'], axis=1
    )
    
    pitcher_statcast = statcast_df[statcast_df['player_name_standard'] == pitcher_name].copy()
    if pitcher_statcast.empty:
        print(f"No data for pitcher {pitcher_name} in statcast_df.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'xwOBA', 'AVG', 'Flyball%', 'Vulnerable_Pitch_Types', 'Expected_HR'])
    
    hr_by_pitch = pitcher_statcast[pitcher_statcast['events'] == 'home_run'].groupby('pitch_type').size()
    total_pitches = pitcher_statcast.groupby('pitch_type').size()
    hr_rate = (hr_by_pitch / total_pitches).fillna(0).reset_index(name='hr_rate')
    vulnerable_pitches = hr_rate[hr_rate['hr_rate'] > 0]['pitch_type'].tolist()
    
    if not vulnerable_pitches:
        print(f"No HRs allowed by {pitcher_name} on any pitch type.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'xwOBA', 'AVG', 'Flyball%', 'Vulnerable_Pitch_Types', 'Expected_HR'])
    
    print(f"Pitcher {pitcher_name} vulnerable pitch types: {vulnerable_pitches}")
    print(f"Pitcher HR events: {len(pitcher_statcast[pitcher_statcast['events'] == 'home_run'])}")
    
    batter_stats = []
    for pitch_type in vulnerable_pitches:
        pitch_data = statcast_df[statcast_df['pitch_type'] == pitch_type].groupby('batter').agg({
            'events': [
                lambda x: (x == 'home_run').sum(),
                lambda x: (x.notna()).sum()
            ],
            'is_barrel': 'mean',
            'launch_speed': 'mean',
            'launch_angle': 'mean',
            'woba_value': 'mean',
            'batter_team': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
        })
        
        pitch_data.columns = ['hr_count', 'pitches_faced', 'barrel%', 'exit_velo', 'launch_angle', 'woba', 'Team']
        pitch_data['PA'] = pitch_data['pitches_faced'] / 4
        pitch_data['hr_rate'] = pitch_data['hr_count'] / pitch_data['PA'].replace(0, np.nan)
        pitch_data['hr_rate'] = pitch_data['hr_rate'].fillna(0.01)
        pitch_data['hardhit%'] = pitch_data['exit_velo'].apply(lambda x: 1 if pd.notna(x) and x >= 95 else 0)
        pitch_data['pitch_type'] = pitch_type
        batter_stats.append(pitch_data.reset_index())
    
    if batter_stats:
        batter_performance = pd.concat(batter_stats, ignore_index=True)
        batter_performance = batter_performance[batter_performance['pitches_faced'] >= min_pitches]
    else:
        print("No batter data for vulnerable pitch types.")
        return pd.DataFrame(columns=['Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%', 'HardHit%', 'xwOBA', 'AVG', 'Flyball%', 'Vulnerable_Pitch_Types', 'Expected_HR'])
    
    unique_batter_ids = batter_performance['batter'].unique()
    
    batter_performance['batter'] = batter_performance['batter'].astype(str)
    
    name_mapping = create_name_mapping(unique_batter_ids, statcast_df)
    batter_performance = batter_performance.merge(name_mapping, left_on='batter', right_on='mlbam_id', how='left')
    batter_performance['name'] = batter_performance['name'].fillna('Unknown_' + batter_performance['batter'])
    
    batter_metrics = batter_df[['Name', 'HR', 'ISO', 'Barrel%', 'HardHit%', 'HR/FB', 'AVG', 'FB%', 'wOBA']].copy()
    batter_metrics = batter_metrics.rename(columns={'FB%': 'Flyball%'})
    
    batter_performance = batter_performance.merge(batter_metrics, left_on='name', right_on='Name', how='left')
    print(f"Matched {len(batter_performance[batter_performance['Name'].notna()])} batters in batter_metrics merge")
    
    batter_performance['Flyball%'] = batter_performance['Flyball%'].fillna(batter_metrics['Flyball%'].mean())
    batter_performance['Barrel%'] = batter_performance['Barrel%'].fillna(batter_metrics['Barrel%'].mean())
    batter_performance['HardHit%'] = batter_performance['HardHit%'].fillna(batter_metrics['HardHit%'].mean())
    batter_performance['AVG'] = batter_performance['AVG'].fillna(batter_metrics['AVG'].mean())
    batter_performance['woba'] = batter_performance['woba'].fillna(batter_metrics['wOBA'].mean() if 'wOBA' in batter_metrics.columns else 0.320)
    
    if team:
        team = standardize_team_short_name(team)
        pre_filter_count = len(batter_performance)
        batter_performance = batter_performance[batter_performance['Team'] == team]
        if batter_performance.empty:
            print(f"No batters found for team {team} after filtering. {pre_filter_count} batters were present before filter.")
        else:
            print(f"Found {len(batter_performance)} batters for team {team}")
            print(f"NYY batter names: {batter_performance['name'].unique().tolist()}")
    
    print(f"Batters before grouping: {len(batter_performance['batter_name'].unique())}")
    print(f"Team assignments: {batter_performance[['batter_name', 'Team']].drop_duplicates().to_dict()}")
    
    league_avg_hr_rate = 0.035
    batter_performance['hr_probability'] = batter_performance['hr_rate']
    batter_performance['hr_probability'] = batter_performance['hr_probability'].fillna(
        league_avg_hr_rate * (batter_performance['Barrel%'] / 0.1) * (batter_performance['Flyball%'] / 0.3)
    )
    
    park_factors = {
        'NYY': {'R': 1.15, 'L': 1.05, 'S': 1.10},
        'TBR': {'R': 0.95, 'L': 0.90, 'S': 0.92},
        'BOS': {'R': 1.02, 'L': 1.08, 'S': 1.05},
        'TOR': {'R': 1.04, 'L': 1.03, 'S': 1.04},
        'BAL': {'R': 0.98, 'L': 1.00, 'S': 0.99},
        'CLE': {'R': 1.01, 'L': 1.00, 'S': 1.01},
        'DET': {'R': 0.97, 'L': 0.98, 'S': 0.98},
        'KCR': {'R': 0.99, 'L': 0.99, 'S': 0.99},
        'MIN': {'R': 1.00, 'L': 1.01, 'S': 1.01},
        'CHW': {'R': 1.10, 'L': 1.12, 'S': 1.11},
        'HOU': {'R': 1.03, 'L': 1.05, 'S': 1.04},
        'LAA': {'R': 1.00, 'L': 1.02, 'S': 1.01},
        'OAK': {'R': 0.94, 'L': 0.93, 'S': 0.94},
        'SEA': {'R': 0.96, 'L': 0.95, 'S': 0.96},
        'TEX': {'R': 1.02, 'L': 1.03, 'S': 1.03},
        'ATL': {'R': 1.01, 'L': 1.00, 'S': 1.01},
        'MIA': {'R': 0.93, 'L': 0.94, 'S': 0.94},
        'NYM': {'R': 0.98, 'L': 0.97, 'S': 0.98},
        'PHI': {'R': 1.06, 'L': 1.07, 'S': 1.07},
        'WSN': {'R': 0.99, 'L': 1.00, 'S': 1.00},
        'CHC': {'R': 1.01, 'L': 1.02, 'S': 1.02},
        'CIN': {'R': 1.13, 'L': 1.14, 'S': 1.14},
        'MIL': {'R': 1.05, 'L': 1.04, 'S': 1.05},
        'PIT': {'R': 0.97, 'L': 0.98, 'S': 0.98},
        'STL': {'R': 0.96, 'L': 0.97, 'S': 0.97},
        'ARI': {'R': 1.03, 'L': 1.04, 'S': 1.04},
        'COL': {'R': 1.20, 'L': 1.22, 'S': 1.21},
        'LAD': {'R': 1.02, 'L': 1.01, 'S': 1.02},
        'SDP': {'R': 0.95, 'L': 0.94, 'S': 0.95},
        'SFG': {'R': 0.93, 'L': 0.92, 'S': 0.93},
    }
    batter_performance['handedness'] = batter_performance['Bats'].map({'R': 'R', 'L': 'L', 'B': 'S'}).fillna('R')
    batter_performance['park_factor'] = batter_performance.apply(
        lambda x: park_factors.get(team, {'R': 1.0, 'L': 1.0, 'S': 1.0}).get(x['handedness'], 1.0), axis=1
    )
    batter_performance['hr_probability'] = batter_performance['hr_probability'] * batter_performance['park_factor']
    
    matchup_data = pitcher_statcast[pitcher_statcast['batter'].isin(batter_performance['batter'].astype(int))].groupby(['batter', 'pitch_type']).agg(
        events=('events', lambda x: (x == 'home_run').sum()),
        pitches_faced=('events', 'size')
    ).reset_index()
    matchup_data['batter'] = matchup_data['batter'].astype(str)
    matchup_data['matchup_hr_rate'] = matchup_data['events'] / (matchup_data['pitches_faced'] / 4)
    batter_performance = batter_performance.merge(matchup_data[['batter', 'pitch_type', 'matchup_hr_rate']], on=['batter', 'pitch_type'], how='left')
    batter_performance['matchup_hr_rate'] = batter_performance['matchup_hr_rate'].fillna(0)
    batter_performance['hr_probability'] = (
        batter_performance['hr_probability'] * 0.98 + batter_performance['matchup_hr_rate'] * 0.02
    )
    
    print(f"Total batter HRs: {batter_performance['hr_count'].sum()}")
    print(f"Sample hr inputs: {batter_performance[['batter_name', 'hr_count', 'PA', 'hr_rate', 'Bats']].head().to_dict()}")
    
    pitch_counts = pitcher_statcast.groupby('pitch_type').size()
    total_pitches = pitch_counts.sum()
    pitch_usage = pd.DataFrame({
        'pitch_type': pitch_counts.index,
        'usage': pitch_counts / total_pitches
    }).reset_index(drop=True)
    
    batter_performance = batter_performance.merge(pitch_usage, on='pitch_type', how='left')
    batter_performance['usage'] = batter_performance['usage'].fillna(0.1)
    batter_performance['weighted_hr_prob'] = batter_performance['hr_probability'] * batter_performance['usage']
    
    batter_performance = batter_performance[batter_performance['pitches_faced'] >= min_pitches]
    
    print(f"NaN counts before grouping: Barrel% {batter_performance['Barrel%'].isna().sum()}, HardHit% {batter_performance['HardHit%'].isna().sum()}, woba {batter_performance['woba'].isna().sum()}")
    print(f"Batters before grouping: {len(batter_performance['batter_name'].unique())}")
    
    batter_summary = batter_performance.groupby(['batter_name', 'Team']).agg({
        'PA': 'sum',
        'weighted_hr_prob': 'sum',
        'Barrel%': 'mean',
        'HardHit%': 'mean',
        'woba': 'mean',
        'AVG': 'mean',
        'Flyball%': 'mean',
        'pitch_type': lambda x: ', '.join(sorted(set(x)))
    }).reset_index()
    
    batter_summary['xwOBA'] = batter_summary['woba'].fillna(batter_metrics['wOBA'].mean() if 'wOBA' in batter_metrics.columns else 0.320) * 100
    batter_summary['matchup_score'] = (
        batter_summary['weighted_hr_prob'].fillna(0) * 0.3 +
        batter_summary['Barrel%'].fillna(batter_metrics['Barrel%'].mean()) * 0.3 +
        batter_summary['HardHit%'].fillna(batter_metrics['HardHit%'].mean()) * 0.2 +
        batter_summary['xwOBA'].fillna(batter_summary['xwOBA'].mean()) * 0.2
    )
    
    batter_summary['Expected_HR'] = batter_summary['weighted_hr_prob'] * batter_summary['PA']
    
    prospects = batter_summary.sort_values('matchup_score', ascending=False)[
        ['batter_name', 'Team', 'PA', 'weighted_hr_prob', 'matchup_score', 'Barrel%', 'HardHit%', 'xwOBA', 'AVG', 'Flyball%', 'pitch_type', 'Expected_HR']
    ].rename(columns={'batter_name': 'Name', 'weighted_hr_prob': 'hr_probability', 'pitch_type': 'Vulnerable_Pitch_Types'})
    
    return prospects.head(10)

# Example usage
pitcher_name = "Jack Flaherty"
team = "NYY"
hr_prospects = get_hr_prospects(
    pitcher_name, statcast_df, batter_df, pitcher_df, 
    min_pitches=5, team=team
)
print("Top Home Run Prospects vs.", pitcher_name)
print(hr_prospects)