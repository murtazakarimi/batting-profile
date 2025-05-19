import pandas as pd
import numpy as np
from datetime import datetime
from utils import standardize_name, standardize_team_short_name, create_name_mapping, PLAYERS_DF
from features import (
    get_batter_handedness_data, get_matchup_history, get_pitch_specific_metrics,
    get_weighted_hr_rate, get_long_flyout_count, get_swing_discipline_score,
    get_vulnerable_pitches, get_recent_flyball_rate, train_and_predict,
    apply_weather_and_park_adjustments, get_recent_hr_streak, get_pitch_location_vulnerability,
    get_pitch_batter_similarity, get_bullpen_matchup
)

class HomeRunProjectionTool:
    """A tool to predict home run prospects for batters against a pitcher."""
    
    def __init__(self, feature_config=None):
        """Initialize with feature configuration."""
        self.default_config = {
            'batter_handedness': True,
            'matchup_history': True,
            'pitch_metrics': True,
            'weighted_hr_rate': True,
            'long_flyouts': True,
            'swing_discipline': True,
            'vulnerable_pitches': True,
            'recent_flyball': True,
            'model_prediction': True,
            'adjustments': True
        }
        self.feature_config = feature_config or self.default_config
        self.prediction_date = datetime.now().date()
    
    def get_prospects(self, pitcher_name, statcast_df, batter_df, pitcher_df, weather=None,
                      min_pitches=3, team=None, manual_name_mapping=None, home_team=None,
                      park_factors=None, player_ids=None):
        """Predict home run prospects for batters against a pitcher."""
        statcast_df = statcast_df.copy().reset_index(drop=True)
        statcast_df['player_name_standard'] = statcast_df['player_name'].apply(
            lambda x: standardize_name(x, is_statcast_format=True)
        )
        
        # Ensure batter column in statcast_df is string
        statcast_df['batter'] = statcast_df['batter'].astype(str)
        print(f"statcast_df['batter'] dtype: {statcast_df['batter'].dtype}")
        
        # Handle missing Statcast data
        statcast_df['pitch_type'] = statcast_df['pitch_type'].fillna('Unknown')
        statcast_df['events'] = statcast_df['events'].fillna('Unknown')
        
        # Add batter_hand column to statcast_df
        if 'stand' in statcast_df.columns:
            statcast_df['batter_hand'] = statcast_df['stand'].map({'L': 'L', 'R': 'R'}).fillna('R')
        else:
            print("Warning: 'stand' column not found in statcast_df. Defaulting batter_hand to 'R'.")
            statcast_df['batter_hand'] = 'R'
        
        # Define barrel criteria
        statcast_df['is_barrel'] = (
            (statcast_df['launch_speed'] >= 95) & 
            (statcast_df['launch_angle'].between(10, 40))
        ).fillna(False).astype(int)
        
        # Initialize empty DataFrame for no-data case
        empty_cols = [
            'Name', 'Team', 'PA', 'hr_probability', 'bp_hr_probability', 'matchup_score',
            'bp_matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%',
            'Pull%', 'K%', 'BB%', 'HR/FB', 'Pitcher_HR/9', 'Recent_Flyball%',
            'Vulnerable_Pitch_Types', 'Batter_Hand', 'Weather_Adjustment',
            'Park_Factor_Adjustment', 'Matchup_HR_Rate', 'Matchup_PA', 'Matchup_wOBA',
            'Weighted_HR_Rate', 'Long_Flyout_Count', 'Swing_Discipline_Score',
            'Recent_HR_Score', 'Similarity_HR_Rate'
        ]
        
        # Get batter handedness data
        pitcher_hand, pitcher_statcast, hr_rate_by_hand = (None, None, None)
        if self.feature_config['batter_handedness']:
            pitcher_hand, pitcher_statcast, hr_rate_by_hand = get_batter_handedness_data(
                pitcher_name, statcast_df, pitcher_df
            )
            if pitcher_statcast is None:
                print(f"No data for pitcher {pitcher_name} in statcast_df.")
                return pd.DataFrame(columns=empty_cols)
            # Ensure pitcher_statcast['batter'] is string
            pitcher_statcast['batter'] = pitcher_statcast['batter'].astype(str)
            print(f"Pitcher {pitcher_name} handedness: {pitcher_hand}")
            print(f"HR rates by batter hand vs. {pitcher_name}: {hr_rate_by_hand}")
            if hr_rate_by_hand.get('L', 0.0) == 0.0 and hr_rate_by_hand.get('R', 0.0) == 0.0:
                print(f"Warning: Skipping {pitcher_name} due to zero HR rates for both L and R batters.")
                return pd.DataFrame(columns=empty_cols)
        
        # Get vulnerable pitches
        vulnerable_pitches = pd.DataFrame()
        if self.feature_config['vulnerable_pitches']:
            vulnerable_pitches = get_vulnerable_pitches(pitcher_statcast, self.prediction_date)
            if vulnerable_pitches.empty:
                print(f"No vulnerable pitches identified for {pitcher_name}.")
        
        # Calculate batter performance per pitch type
        batter_stats = []
        for _, row in vulnerable_pitches.iterrows():
            pitch_type, batter_hand = row['pitch_type'], row['batter_hand']
            pitch_data = statcast_df[
                (statcast_df['pitch_type'] == pitch_type) & 
                (statcast_df['batter_hand'] == batter_hand) &
                (statcast_df['game_date'] >= '2025-01-01')
            ]
            if player_ids:
                mlbam_ids = player_ids.get('mlbam', [])
                pitch_data = pitch_data[pitch_data['batter'].isin([str(id) for id in mlbam_ids])]
            
            if pitch_data.empty:
                continue
            
            pitch_agg = pitch_data.groupby('batter').agg({
                'events': [
                    lambda x: (x == 'home_run').sum(),
                    lambda x: (x.notna() & (x != 'Unknown')).sum()
                ],
                'is_barrel': 'mean',
                'launch_speed': 'mean',
                'launch_angle': 'mean',
                'woba_value': 'mean',
                'batter_hand': 'first'
            })
            pitch_agg.columns = ['hr_count', 'pitches_faced', 'barrel%', 'exit_velo', 'launch_angle', 'woba', 'batter_hand']
            pitch_agg['PA'] = pitch_agg['pitches_faced'] / 4
            pitch_agg['hr_rate'] = pitch_agg['hr_count'] / pitch_agg['PA'].replace(0, np.nan)
            pitch_agg['hr_rate'] = pitch_agg['hr_rate'].fillna(0.01)
            pitch_agg['hardhit%'] = pitch_agg['exit_velo'].apply(lambda x: 1 if pd.notna(x) and x >= 95 else 0)
            pitch_agg['pitch_type'] = pitch_type
            batter_stats.append(pitch_agg.reset_index())
        
        # Add rostered players with insufficient matchup data
        if player_ids and batter_df is not None:
            fangraphs_ids = player_ids.get('fangraphs', [])
            roster = player_ids.get('roster', [])
            roster_missing = [pid for pid in player_ids.get('mlbam', []) if str(pid) not in [bs['batter'].iloc[0] for bs in batter_stats]]
            if roster_missing:
                print(f"Adding {len(roster_missing)} rostered players with season-level stats: {roster_missing}")
                # Preprocess batter_df to align teams with players.csv
                batter_df = batter_df.copy()
                batter_df['IDfg'] = batter_df['IDfg'].astype(str)
                batter_df = batter_df.merge(
                    PLAYERS_DF[['FanGraphsID', 'Team', 'Name']].rename(columns={'FanGraphsID': 'IDfg'}),
                    on='IDfg',
                    how='left',
                    suffixes=('', '_csv')
                )
                batter_df['Team'] = batter_df['Team_csv'].combine_first(batter_df['Team']).apply(standardize_team_short_name)
                batter_df['Name'] = batter_df['Name_csv'].combine_first(batter_df['Name'])
                batter_df = batter_df.drop(columns=['Team_csv', 'Name_csv'], errors='ignore')
                print(f"Unique teams in batter_df after standardization: {batter_df['Team'].unique().tolist()}")
                
                season_stats = batter_df[batter_df['IDfg'].isin([str(fg_id) for fg_id in fangraphs_ids])][[
                    'IDfg', 'Name', 'Team', 'wRC+', 'AVG', 'FB%', 'Pull%', 'K%', 'BB%', 'HR/FB'
                ]].rename(columns={'IDfg': 'batter', 'FB%': 'Flyball%'})
                season_stats['batter'] = season_stats['batter'].astype(str)
                if not season_stats.empty:
                    # Map FanGraphs IDs to MLBAM IDs
                    fg_to_mlbid = PLAYERS_DF.set_index('FanGraphsID')['MLBAMID'].to_dict()
                    season_stats['batter'] = season_stats['batter'].map(lambda x: str(fg_to_mlbid.get(str(x), x)))
                    season_stats['batter_hand'] = 'R'
                    season_stats['hr_count'] = 0
                    season_stats['pitches_faced'] = 10
                    season_stats['PA'] = season_stats['pitches_faced'] / 4
                    season_stats['hr_rate'] = 0.01
                    season_stats['barrel%'] = season_stats.get('Barrel%', 0.05)
                    season_stats['exit_velo'] = 85.0
                    season_stats['launch_angle'] = 10.0
                    season_stats['woba'] = 0.3
                    season_stats['hardhit%'] = 0.3
                    season_stats['pitch_type'] = 'Unknown'
                    batter_stats.append(season_stats)
                else:
                    # Fallback: Use players.csv and roster data
                    print(f"Warning: No season stats found for FanGraphs IDs: {fangraphs_ids}. Using players.csv and roster data.")
                    roster_stats = pd.DataFrame([
                        {
                            'batter': str(player['mlbam_id']),
                            'Name': PLAYERS_DF[PLAYERS_DF['MLBAMID'] == str(player['mlbam_id'])]['Name'].iloc[0]
                                    if str(player['mlbam_id']) in PLAYERS_DF['MLBAMID'].values else player['name'],
                            'Team': PLAYERS_DF[PLAYERS_DF['MLBAMID'] == str(player['mlbam_id'])]['Team'].iloc[0]
                                    if str(player['mlbam_id']) in PLAYERS_DF['MLBAMID'].values else team if team else 'UNKNOWN',
                            'wRC+': 100.0,
                            'AVG': 0.250,
                            'Flyball%': 0.35,
                            'Pull%': 0.40,
                            'K%': 0.20,
                            'BB%': 0.08,
                            'HR/FB': 0.15,
                            'batter_hand': 'R',
                            'hr_count': 0,
                            'pitches_faced': 10,
                            'PA': 2.5,
                            'hr_rate': 0.01,
                            'barrel%': 0.05,
                            'exit_velo': 85.0,
                            'launch_angle': 10.0,
                            'woba': 0.3,
                            'hardhit%': 0.3,
                            'pitch_type': 'Unknown'
                        } for player in roster if str(player['mlbam_id']) in [str(pid) for pid in roster_missing]
                    ])
                    if not roster_stats.empty:
                        batter_stats.append(roster_stats)
                    else:
                        print(f"Warning: No roster stats could be generated for missing players: {roster_missing}")
        
        if not batter_stats:
            print("No batter data for vulnerable pitch types or rostered players.")
            return pd.DataFrame(columns=empty_cols)
        
        batter_performance = pd.concat(batter_stats, ignore_index=True)
        batter_performance = batter_performance[batter_performance['pitches_faced'] >= min_pitches]
        
        # Ensure batter column is string
        batter_performance['batter'] = batter_performance['batter'].astype(str)
        print(f"batter_performance['batter'] dtype: {batter_performance['batter'].dtype}")
        print(f"Batter IDs after mapping: {batter_performance['batter'].unique().tolist()}")
        
        # Map batter IDs to names and FanGraphs IDs
        unique_batter_ids = batter_performance['batter'].unique()
        name_mapping = create_name_mapping(
            mlbam_ids=unique_batter_ids,
            fangraphs_ids=player_ids.get('fangraphs', []) if player_ids else [],
            roster=player_ids.get('roster', []) if player_ids else [],
            manual_name_mapping=manual_name_mapping
        )
        print(f"Name mapping contents: {name_mapping[['mlbam_id', 'name']].to_dict('records')}")
        if name_mapping.empty:
            print("Warning: Name mapping failed for all batters.")
            name_mapping = statcast_df[['batter', 'player_name_standard']].drop_duplicates()
            name_mapping = name_mapping.rename(columns={'batter': 'mlbam_id', 'player_name_standard': 'name'})
        
        name_mapping['mlbam_id'] = name_mapping['mlbam_id'].astype(str)
        batter_performance = batter_performance.merge(
            name_mapping, left_on='batter', right_on='mlbam_id', how='left'
        )
        
        if batter_performance['name'].isna().any():
            missing_names = batter_performance[batter_performance['name'].isna()]['batter'].unique()
            print(f"Warning: {len(missing_names)} batters failed name mapping: {missing_names.tolist()}")
            # Fallback: Use players.csv names
            csv_name_map = PLAYERS_DF.set_index('MLBAMID')['Name'].to_dict()
            batter_performance['name'] = batter_performance.apply(
                lambda row: csv_name_map.get(str(row['batter']), row['name']) if pd.isna(row['name']) else row['name'],
                axis=1
            )
        
        # Create a mapping from MLBAM IDs to FanGraphs IDs using players.csv
        fg_id_mapping = PLAYERS_DF[['MLBAMID', 'FanGraphsID', 'Name']].rename(
            columns={'MLBAMID': 'mlbam_id', 'FanGraphsID': 'key_fangraphs', 'Name': 'name'}
        )
        fg_id_mapping['mlbam_id'] = fg_id_mapping['mlbam_id'].astype(str)
        fg_id_mapping['key_fangraphs'] = fg_id_mapping['key_fangraphs'].astype(str)
        
        batter_performance = batter_performance.merge(
            fg_id_mapping, on=['mlbam_id', 'name'], how='left'
        )
        
        # Add batter metrics using FanGraphs IDs
        batter_metrics = batter_df[[
            'IDfg', 'Name', 'Team', 'HR', 'ISO', 'Barrel%', 'HardHit%', 'HR/FB', 'wRC+',
            'AVG', 'FB%', 'Pull%', 'K%', 'BB%'
        ]].copy()
        batter_metrics['Team'] = batter_metrics['Team'].apply(standardize_team_short_name)
        batter_metrics['IDfg'] = batter_metrics['IDfg'].astype(str)
        batter_metrics = batter_metrics.merge(
            PLAYERS_DF[['FanGraphsID', 'Team', 'Name']].rename(columns={'FanGraphsID': 'IDfg'}),
            on='IDfg',
            how='left',
            suffixes=('', '_csv')
        )
        batter_metrics['Team'] = batter_metrics['Team_csv'].combine_first(batter_metrics['Team']).apply(standardize_team_short_name)
        batter_metrics['Name'] = batter_metrics['Name_csv'].combine_first(batter_metrics['Name'])
        batter_metrics = batter_metrics.drop(columns=['Team_csv', 'Name_csv'], errors='ignore')
        batter_metrics = batter_metrics.rename(columns={'FB%': 'Flyball%'})
        print(f"Available FanGraphs IDs in batter_df: {batter_metrics['IDfg'].tolist()}")
        print(f"Available batter names in batter_df: {batter_metrics['Name'].tolist()}")
        
        batter_performance = batter_performance.merge(
            batter_metrics, left_on='key_fangraphs', right_on='IDfg', how='left', suffixes=('', '_metrics')
        )
        
        missing_batters = batter_performance[batter_performance['Name'].isna()]['batter'].unique()
        if len(missing_batters) > 0:
            print(f"Warning: {len(missing_batters)} batters excluded due to missing metrics: {missing_batters.tolist()}")
            # Fallback: Use default metrics and players.csv data
            batter_performance.loc[batter_performance['Name'].isna(), 'Name'] = batter_performance['name']
            batter_performance.loc[batter_performance['Team'].isna(), 'Team'] = batter_performance['batter'].map(
                lambda x: PLAYERS_DF[PLAYERS_DF['MLBAMID'] == str(x)]['Team'].iloc[0]
                if str(x) in PLAYERS_DF['MLBAMID'].values else team if team else 'UNKNOWN'
            )
            for col in ['HR', 'ISO', 'Barrel%', 'HardHit%', 'HR/FB', 'wRC+', 'AVG', 'Flyball%', 'Pull%', 'K%', 'BB%']:
                if col not in batter_performance.columns:
                    batter_performance[col] = 0
                batter_performance[col] = batter_performance[col].fillna(
                    batter_metrics[col].mean() if not batter_metrics[col].isna().all() else 0
                )
        
        # Add pitcher HR/9
        pitcher_metrics = pitcher_df[pitcher_df['Name'] == pitcher_name][['Name', 'HR/9']].copy()
        batter_performance['Pitcher_HR/9'] = (
            pitcher_metrics['HR/9'].iloc[0] if not pitcher_metrics.empty else pitcher_df['HR/9'].mean()
        )
        
        # Add feature data
        if self.feature_config['matchup_history']:
            try:
                matchup_history = get_matchup_history(pitcher_statcast, self.prediction_date, include_minor_league=True)
                matchup_history['batter'] = matchup_history['batter'].astype(str)
                print(f"matchup_history['batter'] dtype: {matchup_history['batter'].dtype}")
                batter_performance = batter_performance.merge(
                    matchup_history[['batter', 'matchup_hr_rate', 'matchup_pa', 'matchup_woba']],
                    on='batter', how='left'
                )
            except Exception as e:
                print(f"Error merging matchup_history: {e}. Using default values.")
                batter_performance['matchup_hr_rate'] = 0.035
                batter_performance['matchup_pa'] = 0
                batter_performance['matchup_woba'] = 0.3
            batter_performance['matchup_hr_rate'] = batter_performance['matchup_hr_rate'].fillna(0.035)
            batter_performance['matchup_pa'] = batter_performance['matchup_pa'].fillna(0)
            batter_performance['matchup_woba'] = batter_performance['matchup_woba'].fillna(0.3)
        
        if self.feature_config['pitch_metrics']:
            pitch_metrics = get_pitch_specific_metrics(pitcher_statcast, min_pitches)
            pitch_metrics['batter'] = pitch_metrics['batter'].astype(str)
            print(f"pitch_metrics['batter'] dtype: {pitch_metrics['batter'].dtype}")
            batter_performance = batter_performance.merge(
                pitch_metrics[['batter', 'pitch_type', 'pitch_exit_velo', 'pitch_launch_angle']],
                on=['batter', 'pitch_type'], how='left'
            )
            batter_performance['pitch_exit_velo'] = batter_performance['pitch_exit_velo'].fillna(
                batter_performance['exit_velo'].mean() if not batter_performance['exit_velo'].isna().all() else 85.0
            )
            batter_performance['pitch_launch_angle'] = batter_performance['pitch_launch_angle'].fillna(
                batter_performance['launch_angle'].mean() if not batter_performance['launch_angle'].isna().all() else 10.0
            )
            zone_vulnerability = get_pitch_location_vulnerability(pitcher_statcast, self.prediction_date)
            batter_performance = batter_performance.merge(
                zone_vulnerability[['zone_region', 'batter_hand']],
                on='batter_hand', how='left'
            )
            batter_performance['Zone_HR_Adjust'] = batter_performance['zone_region'].apply(
                lambda x: 1.2 if x in ['Middle-In', 'Middle-Middle', 'High-Middle'] else 1.0
            )
        
        if self.feature_config['weighted_hr_rate']:
            weighted_hr_data = get_weighted_hr_rate(statcast_df, self.prediction_date)
            weighted_hr_data['batter'] = weighted_hr_data['batter'].astype(str)
            print(f"weighted_hr_data['batter'] dtype: {weighted_hr_data['batter'].dtype}")
            batter_performance = batter_performance.merge(
                weighted_hr_data[['batter', 'weighted_hr_rate']],
                on='batter', how='left'
            )
            batter_performance['weighted_hr_rate'] = batter_performance['weighted_hr_rate'].fillna(0.035)
        
        if self.feature_config['long_flyouts']:
            long_flyouts = get_long_flyout_count(statcast_df, prediction_date=self.prediction_date)
            long_flyouts['batter'] = long_flyouts['batter'].astype(str)
            print(f"long_flyouts['batter'] dtype: {long_flyouts['batter'].dtype}")
            batter_performance = batter_performance.merge(
                long_flyouts[['batter', 'Long_Flyout_Count']],
                on='batter', how='left'
            )
            batter_performance['Long_Flyout_Count'] = batter_performance['Long_Flyout_Count'].fillna(0)
        
        if self.feature_config['swing_discipline']:
            swing_discipline = get_swing_discipline_score(statcast_df, self.prediction_date)
            swing_discipline['batter'] = swing_discipline['batter'].astype(str)
            print(f"swing_discipline['batter'] dtype: {swing_discipline['batter'].dtype}")
            batter_performance = batter_performance.merge(
                swing_discipline[['batter', 'Swing_Discipline_Score']],
                on='batter', how='left'
            )
            batter_performance['Swing_Discipline_Score'] = batter_performance['Swing_Discipline_Score'].fillna(0)
        
        if self.feature_config['recent_flyball']:
            recent_flyball = get_recent_flyball_rate(statcast_df)
            recent_flyball['batter'] = recent_flyball['batter'].astype(str)
            print(f"recent_flyball['batter'] dtype: {recent_flyball['batter'].dtype}")
            batter_performance = batter_performance.merge(
                recent_flyball[['batter', 'Recent_Flyball%']],
                on='batter', how='left'
            )
            batter_performance['Recent_Flyball%'] = batter_performance['Recent_Flyball%'].fillna(
                batter_metrics['Flyball%'].mean() if not batter_metrics['Flyball%'].isna().all() else 0
            )
            recent_hr = get_recent_hr_streak(statcast_df, self.prediction_date)
            recent_hr['batter'] = recent_hr['batter'].astype(str)
            print(f"recent_hr['batter'] dtype: {recent_hr['batter'].dtype}")
            batter_performance = batter_performance.merge(
                recent_hr[['batter', 'Recent_HR_Score']],
                on='batter', how='left'
            )
            batter_performance['Recent_HR_Score'] = batter_performance['Recent_HR_Score'].fillna(0)
        
        # Add pitch and batter similarity
        similarity_scores = get_pitch_batter_similarity(statcast_df, pitcher_name, self.prediction_date)
        similarity_scores['batter'] = similarity_scores['batter'].astype(str)
        print(f"similarity_scores['batter'] dtype: {similarity_scores['batter'].dtype}")
        batter_performance = batter_performance.merge(
            similarity_scores[['batter', 'Similarity_HR_Rate']],
            on='batter', how='left'
        )
        batter_performance['Similarity_HR_Rate'] = batter_performance['Similarity_HR_Rate'].fillna(0.035)
        
        # Add bullpen matchup score
        bullpen_scores = get_bullpen_matchup(statcast_df, home_team, self.prediction_date)
        bullpen_scores['batter'] = bullpen_scores['batter'].astype(str)
        print(f"bullpen_scores['batter'] dtype: {bullpen_scores['batter'].dtype}")
        batter_performance = batter_performance.merge(
            bullpen_scores[['batter', 'bp_hr_probability', 'bp_matchup_score']],
            on='batter', how='left'
        )
        batter_performance['bp_hr_probability'] = batter_performance['bp_hr_probability'].fillna(0.035)
        batter_performance['bp_matchup_score'] = batter_performance['bp_matchup_score'].fillna(0.035)
        
        # Fill missing metrics
        for col in ['Flyball%', 'Barrel%', 'wRC+', 'HardHit%', 'Pull%', 'K%', 'BB%', 'HR/FB']:
            if col not in batter_performance.columns:
                batter_performance[col] = 0
            batter_performance[col] = batter_performance[col].fillna(
                batter_metrics[col].mean() if not batter_metrics[col].isna().all() else 0
            )
        
        if batter_performance['Name'].isna().all():
            print("Merge with batter_df failed completely.")
            return pd.DataFrame(columns=empty_cols)
        
        batter_performance = batter_performance.rename(columns={'Name': 'batter_name'}).drop(
            columns=['name'], errors='ignore'
        )
        
        if team:
            team = standardize_team_short_name(team)
            batter_performance = batter_performance[batter_performance['Team'] == team]
            if batter_performance.empty:
                print(f"No batters found for team {team}.")
                return pd.DataFrame(columns=empty_cols)
        
        # Apply model prediction
        features = [
            'barrel%', 'Flyball%', 'wRC+', 'HardHit%', 'Pull%', 'K%', 'BB%', 'HR/FB',
            'Pitcher_HR/9', 'Recent_Flyball%', 'matchup_hr_rate', 'matchup_woba',
            'pitch_exit_velo', 'pitch_launch_angle', 'weighted_hr_rate', 'Long_Flyout_Count',
            'Swing_Discipline_Score', 'Recent_HR_Score', 'Similarity_HR_Rate',
            'bp_hr_probability', 'bp_matchup_score'
        ]
        if self.feature_config['model_prediction']:
            # Log batter_performance state
            print(f"batter_performance shape: {batter_performance.shape}")
            print(f"batter_performance columns: {batter_performance.columns.tolist()}")
            print(f"batter_performance['hr_rate'] summary: {batter_performance['hr_rate'].describe()}")
            if batter_performance[features].isna().any().any():
                print("Warning: Missing values in features. Filling with median.")
                batter_performance[features] = batter_performance[features].fillna(batter_performance[features].median())
            
            batter_performance['hr_probability'] = train_and_predict(batter_performance, features)
            # Check for missing players
            expected_mlbid = ['514888', '572233', '701305', '701358']
            missing_mlbid = [mid for mid in expected_mlbid if mid not in batter_performance['batter'].values]
            if missing_mlbid:
                print(f"Missing MLBAM IDs in batter_performance: {missing_mlbid}")
            
            if self.feature_config['batter_handedness']:
                batter_performance['hand_adjust'] = batter_performance['batter_hand'].map(
                    hr_rate_by_hand
                ).fillna(0.035)
                batter_performance['hr_probability'] *= (1 + batter_performance['hand_adjust'])
            batter_performance['hr_probability'] *= batter_performance['Zone_HR_Adjust']
        else:
            batter_performance['hr_probability'] = batter_performance['hr_rate']
        
        # Apply weather and park adjustments
        if self.feature_config['adjustments']:
            batter_performance = apply_weather_and_park_adjustments(
                batter_performance, weather, home_team, park_factors
            )
        
        # Scale probabilities dynamically
        league_avg_hr_rate = 0.035
        pitcher_hr9 = batter_performance['Pitcher_HR/9'].iloc[0] if not batter_performance['Pitcher_HR/9'].empty else pitcher_df['HR/9'].mean()
        pitcher_factor = min(2.0, max(0.5, pitcher_hr9 / pitcher_df['HR/9'].mean()))
        batter_factor = batter_performance['weighted_hr_rate'].apply(lambda x: min(2.0, max(0.5, x / league_avg_hr_rate)))
        scaling_factor = 4.0 * pitcher_factor * batter_factor
        batter_performance['hr_probability'] = np.clip(
            batter_performance['hr_probability'] * scaling_factor,
            league_avg_hr_rate,
            0.30
        )
        batter_performance['bp_hr_probability'] = np.clip(
            batter_performance['bp_hr_probability'] * scaling_factor,
            league_avg_hr_rate,
            0.30
        )
        
        # Calculate pitch usage
        if self.feature_config['batter_handedness']:
            pitcher_statcast['time_weight'] = np.exp(-(pd.to_datetime(self.prediction_date) - pitcher_statcast['game_date']).dt.days / 30)
            pitch_counts = pitcher_statcast.groupby(['pitch_type', 'batter_hand']).agg({'time_weight': 'sum'})
            total_pitches = pitch_counts.groupby('batter_hand').sum()
            pitch_usage = (pitch_counts / total_pitches).reset_index().rename(columns={'time_weight': 'usage'})
            batter_performance = batter_performance.merge(
                pitch_usage, on=['pitch_type', 'batter_hand'], how='left'
            )
            batter_performance['usage'] = batter_performance['usage'].fillna(0.1)
            batter_performance['weighted_hr_prob'] = (
                batter_performance['hr_probability'] * batter_performance['usage']
            )
        else:
            batter_performance['weighted_hr_prob'] = batter_performance['hr_probability']
        
        # Aggregate by batter
        batter_summary = batter_performance.groupby(['batter_name', 'Team', 'batter_hand']).agg({
            'PA': 'sum',
            'weighted_hr_prob': 'sum',
            'hr_probability': 'mean',
            'bp_hr_probability': 'mean',
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
            'park_factor_adjust': 'mean',
            'matchup_hr_rate': 'mean',
            'matchup_pa': 'sum',
            'matchup_woba': 'mean',
            'weighted_hr_rate': 'mean',
            'Long_Flyout_Count': 'mean',
            'Swing_Discipline_Score': 'mean',
            'Recent_HR_Score': 'mean',
            'Similarity_HR_Rate': 'mean',
            'bp_matchup_score': 'mean'
        }).reset_index()
        
        # Calculate matchup score
        batter_summary['matchup_score'] = (
            batter_summary['weighted_hr_prob'] * 0.2 +
            batter_summary['Barrel%'] * 0.2 +
            batter_summary['HardHit%'] * 0.15 +
            batter_summary['wRC+'] / 100 * 0.1 +
            batter_summary['Pull%'] * 0.1 +
            batter_summary['HR/FB'] * 0.1 +
            batter_summary['Recent_Flyball%'] * 0.05 +
            batter_summary['matchup_hr_rate'] * 0.05 +
            batter_summary['weighted_hr_rate'] * 0.05 +
            batter_summary['Long_Flyout_Count'] * 0.05 +
            batter_summary['Swing_Discipline_Score'] * 0.05 +
            batter_summary['Recent_HR_Score'] * 0.1 +
            batter_summary['Similarity_HR_Rate'] * 0.05 +
            batter_summary['bp_matchup_score'] * 0.05
        )
        
        batter_summary = batter_summary[batter_summary['PA'] >= 3]
        print(f"Batter hand distribution: {batter_summary['batter_hand'].value_counts().to_dict()}")
        
        prospects = batter_summary.sort_values('matchup_score', ascending=False)[[
            'batter_name', 'Team', 'PA', 'hr_probability', 'bp_hr_probability', 'matchup_score',
            'bp_matchup_score', 'Barrel%', 'HardHit%', 'wRC+', 'AVG', 'Flyball%',
            'Pull%', 'K%', 'BB%', 'HR/FB', 'Pitcher_HR/9', 'Recent_Flyball%',
            'pitch_type', 'batter_hand', 'weather_adjust', 'park_factor_adjust',
            'matchup_hr_rate', 'matchup_pa', 'matchup_woba', 'weighted_hr_rate',
            'Long_Flyout_Count', 'Swing_Discipline_Score', 'Recent_HR_Score',
            'Similarity_HR_Rate'
        ]].rename(columns={
            'batter_name': 'Name',
            'weighted_hr_prob': 'hr_probability',
            'pitch_type': 'Vulnerable_Pitch_Types',
            'batter_hand': 'Batter_Hand',
            'weather_adjust': 'Weather_Adjustment',
            'park_factor_adjust': 'Park_Factor_Adjustment',
            'matchup_hr_rate': 'Matchup_HR_Rate',
            'matchup_pa': 'Matchup_PA',
            'matchup_woba': 'Matchup_wOBA',
            'weighted_hr_rate': 'Weighted_HR_Rate',
            'Long_Flyout_Count': 'Long_Flyout_Count',
            'Swing_Discipline_Score': 'Swing_Discipline_Score',
            'Recent_HR_Score': 'Recent_HR_Score',
            'Similarity_HR_Rate': 'Similarity_HR_Rate'
        })
        
        return prospects.head(100)