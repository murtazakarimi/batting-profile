import pandas as pd
import numpy as np
from utils import standardize_name, standardize_team_short_name, create_name_mapping
from features import (
    get_batter_handedness_data, get_matchup_history, get_pitch_specific_metrics,
    get_weighted_hr_rate, get_long_flyout_count, get_swing_discipline_score,
    get_vulnerable_pitches, get_recent_flyball_rate, train_and_predict,
    apply_weather_and_park_adjustments, get_recent_hr_streak, get_pitch_location_vulnerability,
    get_pitch_batter_similarity
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
        self.prediction_date = '2025-05-15'
    
    def get_prospects(self, pitcher_name, statcast_df, batter_df, pitcher_df, weather=None,
                      min_pitches=5, team=None, manual_name_mapping=None, home_team=None,
                      park_factors=None):
        """Predict home run prospects for batters against a pitcher."""
        statcast_df = statcast_df.copy().reset_index(drop=True)
        statcast_df['player_name_standard'] = statcast_df['player_name'].apply(
            lambda x: standardize_name(x, is_statcast_format=True)
        )
        
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
            'Name', 'Team', 'PA', 'hr_probability', 'matchup_score', 'Barrel%',
            'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Pull%', 'K%', 'BB%', 'HR/FB',
            'Pitcher_HR/9', 'Recent_Flyball%', 'Vulnerable_Pitch_Types', 'Batter_Hand',
            'Weather_Adjustment', 'Park_Factor_Adjustment', 'Matchup_HR_Rate',
            'Matchup_PA', 'Matchup_wOBA', 'Weighted_HR_Rate', 'Long_Flyout_Count',
            'Swing_Discipline_Score', 'Recent_HR_Score', 'Similarity_HR_Rate'
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
            print(f"Pitcher {pitcher_name} handedness: {pitcher_hand}")
            print(f"HR rates by batter hand vs. {pitcher_name}: {hr_rate_by_hand}")
        
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
                (statcast_df['game_date'] >= '2025-01-01')  # Focus on 2025 for batter stats
            ].groupby('batter').agg({
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
            pitch_data.columns = ['hr_count', 'pitches_faced', 'barrel%', 'exit_velo', 'launch_angle', 'woba', 'batter_hand']
            pitch_data['PA'] = pitch_data['pitches_faced'] / 4
            pitch_data['hr_rate'] = pitch_data['hr_count'] / pitch_data['PA'].replace(0, np.nan)
            pitch_data['hr_rate'] = pitch_data['hr_rate'].fillna(0.01)
            pitch_data['hardhit%'] = pitch_data['exit_velo'].apply(lambda x: 1 if pd.notna(x) and x >= 95 else 0)
            pitch_data['pitch_type'] = pitch_type
            batter_stats.append(pitch_data.reset_index())
        
        if not batter_stats:
            print("No batter data for vulnerable pitch types.")
            return pd.DataFrame(columns=empty_cols)
        
        batter_performance = pd.concat(batter_stats, ignore_index=True)
        batter_performance = batter_performance[batter_performance['pitches_faced'] >= min_pitches]
        
        # Map batter IDs to names
        unique_batter_ids = batter_performance['batter'].unique()
        name_mapping = create_name_mapping(unique_batter_ids, manual_name_mapping)
        if name_mapping.empty:
            print("Warning: Name mapping failed for all batters.")
            # Fallback: Use player_name_standard from statcast_df
            name_mapping = statcast_df[['batter', 'player_name_standard']].drop_duplicates()
            name_mapping = name_mapping.rename(columns={'batter': 'mlbam_id', 'player_name_standard': 'name'})
        
        batter_performance = batter_performance.merge(
            name_mapping, left_on='batter', right_on='mlbam_id', how='left'
        )
        
        # Log merge issues
        if batter_performance['name'].isna().any():
            missing_names = batter_performance[batter_performance['name'].isna()]['batter'].unique()
            # print(f"Warning: {len(missing_names)} batters failed name mapping: {missing_names.tolist()}")
        
        # Add batter metrics
        batter_metrics = batter_df[[
            'Name', 'Team', 'HR', 'ISO', 'Barrel%', 'HardHit%', 'HR/FB', 'wRC+',
            'AVG', 'FB%', 'Pull%', 'K%', 'BB%'
        ]].copy()
        batter_metrics['Team'] = batter_metrics['Team'].apply(standardize_team_short_name)
        batter_metrics = batter_metrics.rename(columns={'FB%': 'Flyball%'})
        batter_performance = batter_performance.merge(
            batter_metrics, left_on='name', right_on='Name', how='left'
        )
        
        # Validate batter inclusion
        missing_batters = batter_performance[batter_performance['Name'].isna()]['batter'].unique()
        # if len(missing_batters) > 0:
        #     print(f"Warning: {len(missing_batters)} batters excluded due to missing metrics: {missing_batters.tolist()}")
        
        # Add pitcher HR/9
        pitcher_metrics = pitcher_df[pitcher_df['Name'] == pitcher_name][['Name', 'HR/9']].copy()
        batter_performance['Pitcher_HR/9'] = (
            pitcher_metrics['HR/9'].iloc[0] if not pitcher_metrics.empty else pitcher_df['HR/9'].mean()
        )
        
        # Add feature data
        if self.feature_config['matchup_history']:
            matchup_history = get_matchup_history(pitcher_statcast, self.prediction_date, include_minor_league=True)
            batter_performance = batter_performance.merge(
                matchup_history[['batter', 'matchup_hr_rate', 'matchup_pa', 'matchup_woba']],
                on='batter', how='left'
            )
            batter_performance['matchup_hr_rate'] = batter_performance['matchup_hr_rate'].fillna(0.035)
            batter_performance['matchup_pa'] = batter_performance['matchup_pa'].fillna(0)
            batter_performance['matchup_woba'] = batter_performance['matchup_woba'].fillna(0.3)
        
        if self.feature_config['pitch_metrics']:
            pitch_metrics = get_pitch_specific_metrics(pitcher_statcast, min_pitches)
            batter_performance = batter_performance.merge(
                pitch_metrics[['batter', 'pitch_type', 'pitch_exit_velo', 'pitch_launch_angle']],
                on=['batter', 'pitch_type'], how='left'
            )
            batter_performance['pitch_exit_velo'] = batter_performance['pitch_exit_velo'].fillna(
                batter_performance['exit_velo'].mean()
            )
            batter_performance['pitch_launch_angle'] = batter_performance['pitch_launch_angle'].fillna(
                batter_performance['launch_angle'].mean()
            )
            # Add pitch location vulnerability
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
            batter_performance = batter_performance.merge(
                weighted_hr_data[['batter', 'weighted_hr_rate']],
                on='batter', how='left'
            )
            batter_performance['weighted_hr_rate'] = batter_performance['weighted_hr_rate'].fillna(0.035)
        
        if self.feature_config['long_flyouts']:
            long_flyouts = get_long_flyout_count(statcast_df)
            batter_performance = batter_performance.merge(
                long_flyouts[['batter', 'Long_Flyout_Count']],
                on='batter', how='left'
            )
            batter_performance['Long_Flyout_Count'] = batter_performance['Long_Flyout_Count'].fillna(0)
        
        if self.feature_config['swing_discipline']:
            swing_discipline = get_swing_discipline_score(statcast_df, self.prediction_date)
            batter_performance = batter_performance.merge(
                swing_discipline[['batter', 'Swing_Discipline_Score']],
                on='batter', how='left'
            )
            batter_performance['Swing_Discipline_Score'] = batter_performance['Swing_Discipline_Score'].fillna(0)
        
        if self.feature_config['recent_flyball']:
            recent_flyball = get_recent_flyball_rate(statcast_df)
            batter_performance = batter_performance.merge(
                recent_flyball[['batter', 'Recent_Flyball%']],
                on='batter', how='left'
            )
            batter_performance['Recent_Flyball%'] = batter_performance['Recent_Flyball%'].fillna(
                batter_metrics['Flyball%'].mean() if not batter_metrics['Flyball%'].isna().all() else 0
            )
            # Add recent HR streak
            recent_hr = get_recent_hr_streak(statcast_df, self.prediction_date)
            batter_performance = batter_performance.merge(
                recent_hr[['batter', 'Recent_HR_Score']],
                on='batter', how='left'
            )
            batter_performance['Recent_HR_Score'] = batter_performance['Recent_HR_Score'].fillna(0)
        
        # Add pitch and batter similarity
        similarity_scores = get_pitch_batter_similarity(statcast_df, pitcher_name, self.prediction_date)
        batter_performance = batter_performance.merge(
            similarity_scores[['batter', 'Similarity_HR_Rate']],
            on='batter', how='left'
        )
        batter_performance['Similarity_HR_Rate'] = batter_performance['Similarity_HR_Rate'].fillna(0.035)
        
        # Fill missing metrics
        for col in ['Flyball%', 'Barrel%', 'wRC+', 'HardHit%', 'Pull%', 'K%', 'BB%', 'HR/FB']:
            batter_performance[col] = batter_performance[col].fillna(
                batter_metrics[col].mean() if not batter_metrics[col].isna().all() else 0
            )
        
        if batter_performance['Name'].isna().all():
            print("Merge with batter_df failed.")
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
            'Swing_Discipline_Score', 'Recent_HR_Score', 'Similarity_HR_Rate'
        ]
        if self.feature_config['model_prediction']:
            batter_performance['hr_probability'] = train_and_predict(batter_performance, features)
            if self.feature_config['batter_handedness']:
                batter_performance['hand_adjust'] = batter_performance['batter_hand'].map(
                    hr_rate_by_hand
                ).fillna(0.035)
                batter_performance['hr_probability'] *= (1 + batter_performance['hand_adjust'])
            # Apply zone adjustment
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
            'Similarity_HR_Rate': 'mean'
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
            batter_summary['Similarity_HR_Rate'] * 0.1  # Add similarity score
        )
        
        batter_summary = batter_summary[batter_summary['PA'] >= 5]
        print(f"Batter hand distribution: {batter_summary['batter_hand'].value_counts().to_dict()}")
        
        prospects = batter_summary.sort_values('matchup_score', ascending=False)[[
            'batter_name', 'Team', 'PA', 'weighted_hr_prob', 'matchup_score', 'Barrel%',
            'HardHit%', 'wRC+', 'AVG', 'Flyball%', 'Pull%', 'K%', 'BB%', 'HR/FB',
            'Pitcher_HR/9', 'Recent_Flyball%', 'pitch_type', 'batter_hand',
            'weather_adjust', 'park_factor_adjust', 'matchup_hr_rate', 'matchup_pa',
            'matchup_woba', 'weighted_hr_rate', 'Long_Flyout_Count', 'Swing_Discipline_Score',
            'Recent_HR_Score', 'Similarity_HR_Rate'
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
        
        return prospects.head(50)