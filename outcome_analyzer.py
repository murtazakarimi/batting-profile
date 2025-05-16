import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from utils import standardize_team_short_name, TEAM_ID_MAPPING
from features.recent_team_performance import PYBASEBALL_TEAM_MAPPING
from weather import BallparkWeather
from features import (
    get_team_offensive_metrics,
    get_team_pitching_metrics,
    get_recent_team_performance,
    get_team_standings,
    apply_weather_and_park_adjustments
)

# Silence pandas warnings
pd.options.mode.chained_assignment = None

class OutcomeProjectionTool:
    """A tool to predict matchup outcomes (home team win probability)."""
    
    def __init__(self, feature_config=None):
        """Initialize with feature configuration."""
        self.default_config = {
            'model_prediction': True,
            'team_offense': True,
            'team_pitching': True,
            'pitcher_metrics': True,
            'recent_performance': True,
            'standings': True,
            'weather_adjustments': True,
            'park_factors': True
        }
        self.feature_config = feature_config or self.default_config
        self.prediction_date = '2025-05-14'
        self.model = LogisticRegression(random_state=42, max_iter=2000)
        self.scaler = StandardScaler()

    def get_pitcher_metrics(self, pitcher_name, pitcher_df):
        """Get metrics for a specific pitcher."""
        pitcher_data = pitcher_df[pitcher_df['Name'] == pitcher_name]
        if pitcher_data.empty:
            return {'pitcher_ERA': 4.00, 'pitcher_HR/9': 1.20, 'pitcher_K/9': 8.00}
        return {
            'pitcher_ERA': pitcher_data['ERA'].iloc[0],
            'pitcher_HR/9': pitcher_data['HR/9'].iloc[0],
            'pitcher_K/9': pitcher_data['K/9'].iloc[0]
        }

    def get_historical_training_data(self, batter_df, pitcher_df, park_factors, season=2025, end_date='2025-05-14'):
        """Fetch historical game outcomes and features for training with caching."""
        cache_dir = "data_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/training_data_{season}.parquet"
        metadata_file = f"{cache_dir}/training_data_metadata.json"
        
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
                print(f"Warning: Corrupted training data cache metadata. Forcing refresh.")
        
        if use_cache:
            return pd.read_parquet(cache_file)
        
        # Get team-level metrics
        team_offense = get_team_offensive_metrics(batter_df) if self.feature_config['team_offense'] else pd.DataFrame()
        team_pitching = get_team_pitching_metrics(pitcher_df) if self.feature_config['team_pitching'] else pd.DataFrame()
        standings = get_team_standings(season, date=end_date.replace('-', '/')) if self.feature_config['standings'] else pd.DataFrame()

        training_data = []
        teams = team_offense['Team'].unique() if not team_offense.empty else list(TEAM_ID_MAPPING.keys())
        
        game_logs = []
        start_date = pd.to_datetime('2025-03-27')
        end_date = pd.to_datetime(end_date, format='%Y-%m-%d', errors='coerce')
        if pd.isna(end_date):
            print(f"Error: Invalid end_date format: {end_date}. Using default end date.")
            end_date = pd.to_datetime('2025-05-14')
        
        from pybaseball import team_game_logs
        failed_teams = []
        skipped_rows = []
        for team in teams:
            if team not in TEAM_ID_MAPPING:
                print(f"Warning: Team {team} not found in TEAM_ID_MAPPING. Skipping.")
                failed_teams.append(team)
                continue
            pybaseball_team = PYBASEBALL_TEAM_MAPPING.get(team, team)
            try:
                print(f"Fetching game logs for {team} (pybaseball: {pybaseball_team}, season={season}) using pybaseball.")
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
                    runs_allowed = game.get('RA', 0)
                    game_logs.append({
                        'Team': team,
                        'Date': game['Date'],
                        'Opp': opp,
                        'W/L': wl,
                        'R': runs,
                        'RA': runs_allowed,
                        'Home_Away': home_away
                    })
            except Exception as e:
                print(f"Error fetching game log for {team} (pybaseball: {pybaseball_team}): {e}")
                failed_teams.append(team)
        
        game_logs = pd.DataFrame(game_logs)
        if skipped_rows:
            print(f"Skipped rows due to invalid data: {skipped_rows}")
        if game_logs.empty:
            raise RuntimeError(f"No game logs retrieved for {season}. Failed teams: {', '.join(failed_teams) if failed_teams else 'None'}. Skipped rows: {skipped_rows}. Ensure team mappings and data processing are correct.")
        else:
            print(f"Retrieved {len(game_logs)} game logs for {season}. Sample: {game_logs.head().to_dict()}")
        
        game_logs['Date'] = pd.to_datetime(game_logs['Date'], format='%Y-%m-%d', errors='coerce')
        
        # Limit iterations by unique dates and teams
        unique_dates = game_logs['Date'].dt.date.unique()
        max_iterations = 1000  # Prevent infinite loops
        iteration_count = 0
        
        for game_date in unique_dates:
            if iteration_count >= max_iterations:
                print(f"Warning: Reached maximum iterations ({max_iterations}). Stopping processing.")
                break
            game_date_str = game_date.strftime('%Y-%m-%d')
            games_on_date = game_logs[game_logs['Date'].dt.date == game_date]
            for _, game in games_on_date.iterrows():
                if iteration_count >= max_iterations:
                    print(f"Warning: Reached maximum iterations ({max_iterations}). Stopping processing.")
                    break
                iteration_count += 1
                
                if pd.isna(game['Date']) or game['Date'].date() > end_date.date():
                    continue
                
                home_team = standardize_team_short_name(game['Home_Away'] == 'Home' and game['Team'] or game['Opp'])
                away_team = standardize_team_short_name(game['Home_Away'] == 'Away' and game['Team'] or game['Opp'])
                if not (home_team and away_team):
                    continue
                
                # Label: 1 if home team won, 0 otherwise
                is_home = game['Home_Away'] == 'Home'
                wl = game['W/L']
                home_win = 1 if (is_home and wl.startswith('W')) or (not is_home and wl.startswith('L')) else 0
                
                # Features
                home_offense = pd.DataFrame() if team_offense.empty else team_offense[team_offense['Team'] == home_team]
                away_offense = pd.DataFrame() if team_offense.empty else team_offense[team_offense['Team'] == away_team]
                home_pitching = pd.DataFrame() if team_pitching.empty else team_pitching[team_pitching['Team'] == home_team]
                away_pitching = pd.DataFrame() if team_pitching.empty else team_pitching[team_pitching['Team'] == away_team]
                home_standings = pd.DataFrame() if standings.empty else standings[standings['Team'] == home_team]
                away_standings = pd.DataFrame() if standings.empty else standings[standings['Team'] == away_team]
                
                # Recent performance up to game date
                try:
                    recent_perf = get_recent_team_performance(game_date_str, [home_team, away_team], season, num_games=10)
                except Exception as e:
                    print(f"Error getting recent performance for {home_team} vs {away_team} on {game_date_str}: {e}")
                    continue
                home_recent = pd.DataFrame() if recent_perf.empty else recent_perf[recent_perf['Team'] == home_team]
                away_recent = pd.DataFrame() if recent_perf.empty else recent_perf[recent_perf['Team'] == away_team]
                
                features = {
                    'home_wRC+': home_offense['wRC+'].iloc[0] if not home_offense.empty else 100,
                    'away_wRC+': away_offense['wRC+'].iloc[0] if not away_offense.empty else 100,
                    'home_ERA': home_pitching['ERA'].iloc[0] if not home_pitching.empty else 4.00,
                    'away_ERA': away_pitching['ERA'].iloc[0] if not away_pitching.empty else 4.00,
                    'home_HR_Rate': home_offense['HR_Rate'].iloc[0] if not home_offense.empty else 0.035,
                    'away_HR_Rate': away_offense['HR_Rate'].iloc[0] if not away_offense.empty else 0.035,
                    'home_Win_Rate': home_recent['Win_Rate'].iloc[0] if not home_recent.empty else 0.500,
                    'away_Win_Rate': away_recent['Win_Rate'].iloc[0] if not away_recent.empty else 0.500,
                    'home_Run_Differential': home_recent['Run_Differential'].iloc[0] if not home_recent.empty else 0.0,
                    'away_Run_Differential': away_recent['Run_Differential'].iloc[0] if not away_recent.empty else 0.0,
                    'home_Win_Percentage': home_standings['Win_Percentage'].iloc[0] if not home_standings.empty else 0.500,
                    'away_Win_Percentage': away_standings['Win_Percentage'].iloc[0] if not away_standings.empty else 0.500,
                    'home_Games_Back': home_standings['Games_Back'].iloc[0] if not home_standings.empty else 0.0,
                    'away_Games_Back': away_standings['Games_Back'].iloc[0] if not away_standings.empty else 0.0,
                    'weather_adjustment_factor': 1.0,  # Simplified: no historical weather
                    'park_factor_adjustment_factor': park_factors.get(home_team, 100) / 100.0 if self.feature_config['park_factors'] else 1.0,
                    'home_win': home_win
                }
                
                training_data.append(features)
        
        training_df = pd.DataFrame(training_data)
        if not training_df.empty:
            print(f"Training data size: {len(training_df)} games. Sample: {training_df.head().to_dict()}")
            training_df.to_parquet(cache_file)
            with open(metadata_file, 'w') as f:
                json.dump({'last_refresh': current_date}, f)
        else:
            print(f"Warning: No training data generated for {season}.")
        
        return training_df

    def train_and_predict(self, features_df, training_data):
        """Train a logistic regression model and predict win probabilities."""
        if features_df.empty:
            return np.zeros(0)

        feature_columns = [
            'home_wRC+', 'away_wRC+', 'home_ERA', 'away_ERA',
            'home_HR_Rate', 'away_HR_Rate',
            'home_Win_Rate', 'away_Win_Rate', 'home_Run_Differential', 'away_Run_Differential',
            'home_Win_Percentage', 'away_Win_Percentage', 'home_Games_Back', 'away_Games_Back',
            'weather_adjustment_factor', 'park_factor_adjustment_factor'
        ]
        
        X_pred = features_df[feature_columns].fillna(0)
        
        if training_data.empty:
            print("No training data available. Falling back to baseline probability.")
            return np.full(len(X_pred), 0.5)
        
        X_train = training_data[feature_columns].fillna(0)
        y_train = training_data['home_win']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_pred_scaled = self.scaler.transform(X_pred)
        
        try:
            # Cross-validation
            scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            print(f"Cross-validation ROC-AUC: {scores.mean():.3f} (±{scores.std():.3f})")
            
            self.model.fit(X_train_scaled, y_train)
            probabilities = self.model.predict_proba(X_pred_scaled)[:, 1]  # Probability of home team win
        except Exception as e:
            print(f"Model training failed: {e}. Falling back to baseline probability.")
            probabilities = np.full(len(X_pred), 0.5)
        
        return probabilities

    def get_outcome_predictions(self, schedule_df, statcast_df, batter_df, pitcher_df,
                              weather_obj, park_factors, date_str):
        """Predict matchup outcomes for the given schedule."""
        if schedule_df.empty:
            return pd.DataFrame()

        # Get team-level metrics
        team_offense = get_team_offensive_metrics(batter_df) if self.feature_config['team_offense'] else pd.DataFrame()
        team_pitching = get_team_pitching_metrics(pitcher_df) if self.feature_config['team_pitching'] else pd.DataFrame()
        standings = get_team_standings(season=2025, date=datetime.strptime(date_str, '%Y-%m-%d').strftime('%m/%d/%Y')) if self.feature_config['standings'] else pd.DataFrame()
        teams = schedule_df['home_name'].unique().tolist() + schedule_df['away_name'].unique().tolist()
        teams = [t for t in teams if t]
        recent_perf = get_recent_team_performance(date_str, teams, season=2025) if self.feature_config['recent_performance'] else pd.DataFrame()

        # Get training data
        training_data = self.get_historical_training_data(batter_df, pitcher_df, park_factors, season=2025) if self.feature_config['model_prediction'] else pd.DataFrame()

        # Train model once
        feature_columns = [
            'home_wRC+', 'away_wRC+', 'home_ERA', 'away_ERA',
            'home_HR_Rate', 'away_HR_Rate',
            'home_Win_Rate', 'away_Win_Rate', 'home_Run_Differential', 'away_Run_Differential',
            'home_Win_Percentage', 'away_Win_Percentage', 'home_Games_Back', 'away_Games_Back',
            'weather_adjustment_factor', 'park_factor_adjustment_factor'
        ]
        if not training_data.empty and self.feature_config['model_prediction']:
            X_train = training_data[feature_columns].fillna(0)
            y_train = training_data['home_win']
            X_train_scaled = self.scaler.fit_transform(X_train)
            try:
                scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
                print(f"Cross-validation ROC-AUC: {scores.mean():.3f} (±{scores.std():.3f})")
                self.model.fit(X_train_scaled, y_train)
            except Exception as e:
                print(f"Model training failed: {e}. Falling back to baseline probability.")

        predictions = []
        for _, row in schedule_df.iterrows():
            home_team = row['home_name']
            away_team = row['away_name']
            game_date = row['game_date'].strftime('%Y-%m-%d')
            game_hour = row['game_hour']
            home_pitcher = row['home_pitcher']
            away_pitcher = row['away_pitcher']

            if not (home_team and away_team):
                continue

            matchup_str = f"{away_team} vs {home_team}"
            print(f"Processing matchup: {matchup_str} on {game_date} at {game_hour}:00")

            # Get weather data
            weather_data = None
            if self.feature_config['weather_adjustments']:
                try:
                    weather_data = weather_obj.get_forecast(matchup_str, game_hour, game_date)
                except Exception as e:
                    print(f"Error fetching weather for {matchup_str}: {e}")

            # Collect features
            home_offense = pd.DataFrame() if team_offense.empty else team_offense[team_offense['Team'] == home_team]
            away_offense = pd.DataFrame() if team_offense.empty else team_offense[team_offense['Team'] == away_team]
            home_pitching = pd.DataFrame() if team_pitching.empty else team_pitching[team_pitching['Team'] == home_team]
            away_pitching = pd.DataFrame() if team_pitching.empty else team_pitching[team_pitching['Team'] == away_team]
            home_standings = pd.DataFrame() if standings.empty else standings[standings['Team'] == home_team]
            away_standings = pd.DataFrame() if standings.empty else standings[standings['Team'] == away_team]
            home_recent = pd.DataFrame() if recent_perf.empty else recent_perf[recent_perf['Team'] == home_team]
            away_recent = pd.DataFrame() if recent_perf.empty else recent_perf[recent_perf['Team'] == away_team]
            home_pitcher_metrics = self.get_pitcher_metrics(home_pitcher, pitcher_df)
            away_pitcher_metrics = self.get_pitcher_metrics(away_pitcher, pitcher_df)

            # Default values
            features = {
                'home_wRC+': home_offense['wRC+'].iloc[0] if not home_offense.empty else 100,
                'away_wRC+': away_offense['wRC+'].iloc[0] if not away_offense.empty else 100,
                'home_ERA': home_pitching['ERA'].iloc[0] if not home_pitching.empty else 4.00,
                'away_ERA': away_pitching['ERA'].iloc[0] if not away_pitching.empty else 4.00,
                'home_HR_Rate': home_offense['HR_Rate'].iloc[0] if not home_offense.empty else 0.035,
                'away_HR_Rate': away_offense['HR_Rate'].iloc[0] if not away_offense.empty else 0.035,
                'home_Win_Rate': home_recent['Win_Rate'].iloc[0] if not home_recent.empty else 0.500,
                'away_Win_Rate': away_recent['Win_Rate'].iloc[0] if not away_recent.empty else 0.500,
                'home_Run_Differential': home_recent['Run_Differential'].iloc[0] if not home_recent.empty else 0.0,
                'away_Run_Differential': away_recent['Run_Differential'].iloc[0] if not away_recent.empty else 0.0,
                'home_Win_Percentage': home_standings['Win_Percentage'].iloc[0] if not home_standings.empty else 0.500,
                'away_Win_Percentage': away_standings['Win_Percentage'].iloc[0] if not away_standings.empty else 0.500,
                'home_Games_Back': home_standings['Games_Back'].iloc[0] if not home_standings.empty else 0.0,
                'away_Games_Back': away_standings['Games_Back'].iloc[0] if not away_standings.empty else 0.0,
                'weather_adjustment_factor': 1.0,
                'park_factor_adjustment_factor': park_factors.get(home_team, 100) / 100.0 if self.feature_config['park_factors'] else 1.0
            }

            # Apply weather and park adjustments
            if self.feature_config['weather_adjustments'] or self.feature_config['park_factors']:
                temp_df = pd.DataFrame([features])
                temp_df = apply_weather_and_park_adjustments(temp_df, weather_data, home_team, park_factors)
                features['weather_adjustment_factor'] = temp_df['weather_adjustment_factor'].iloc[0]
                features['park_factor_adjustment_factor'] = temp_df['park_factor_adjustment_factor'].iloc[0]

            features_df = pd.DataFrame([features])
            if self.feature_config['model_prediction'] and hasattr(self.model, 'classes_'):
                X_pred = features_df[feature_columns].fillna(0)
                X_pred_scaled = self.scaler.transform(X_pred)
                win_probability = self.model.predict_proba(X_pred_scaled)[:, 1][0]
            else:
                win_probability = 0.5

            predictions.append({
                'Game': matchup_str,
                'Game_Date': game_date,
                'Home_Team': home_team,
                'Away_Team': away_team,
                'Home_Pitcher': home_pitcher,
                'Away_Pitcher': away_pitcher,
                'Home_Win_Probability': win_probability,
                'Home_wRC+': features['home_wRC+'],
                'Away_wRC+': features['away_wRC+'],
                'Home_ERA': features['home_ERA'],
                'Away_ERA': features['away_ERA'],
                'Home_Win_Rate': features['home_Win_Rate'],
                'Away_Win_Rate': features['away_Win_Rate'],
                'Home_Run_Differential': features['home_Run_Differential'],
                'Away_Run_Differential': features['away_Run_Differential'],
                'Home_Win_Percentage': features['home_Win_Percentage'],
                'Away_Win_Percentage': features['away_Win_Percentage'],
                'Home_Games_Back': features['home_Games_Back'],
                'Away_Games_Back': features['away_Games_Back'],
                'Weather_Adjustment': (features['weather_adjustment_factor'] - 1) * 100,
                'Park_Factor_Adjustment': (features['park_factor_adjustment_factor'] - 1) * 100
            })

        if not predictions:
            print("No valid matchups to predict.")
            return pd.DataFrame()

        return pd.DataFrame(predictions)