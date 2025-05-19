import pandas as pd
import numpy as np
import requests
from datetime import datetime
import statsapi
from dateutil import parser, tz
from data_loader import load_baseball_data
from prospect_analyzer import HomeRunProjectionTool
from outcome_analyzer import OutcomeProjectionTool
from weather import BallparkWeather
from utils import standardize_team_short_name, TEAM_ID_MAPPING, get_team_roster
import os

def get_schedule(date_str='2025-05-18'):
    """Fetch the MLB schedule with robust retries, relying on statsapi for 2025 data."""
    try:
        for attempt in range(3):
            try:
                schedule = statsapi.schedule(date=date_str)
                if schedule:
                    break
                print(f"No games found for {date_str} on attempt {attempt + 1}")
            except Exception as e:
                print(f"Error fetching schedule on attempt {attempt + 1}: {e}")
        else:
            print(f"Failed to fetch schedule for {date_str} after 3 attempts")
            return pd.DataFrame()
        
        schedule_data = []
        processed_teams = set()
        for game in schedule:
            away_name = game.get('away_name', '')
            home_name = game.get('home_name', '')
            away_pitcher = game.get('away_probable_pitcher', 'Unknown')
            home_pitcher = game.get('home_probable_pitcher', 'Unknown')
            game_date = game.get('game_date', date_str)
            game_datetime = game.get('game_datetime', '')
            try:
                dt = parser.isoparse(game_datetime)
                dt_edt = dt.astimezone(tz.gettz('America/New_York'))
                game_hour = dt_edt.hour
            except (ValueError, TypeError) as e:
                print(f"Warning: Failed to parse game time: {e}. Defaulting to 19.")
                game_hour = 19
            
            if not (away_name and home_name):
                print(f"Skipping game with missing teams: away={away_name}, home={home_name}")
                continue
            
            schedule_data.append({
                'away_name': away_name,
                'home_name': home_name,
                'game_date': game_date,
                'game_hour': game_hour,
                'away_pitcher': away_pitcher,
                'home_pitcher': home_pitcher
            })
            processed_teams.add(standardize_team_short_name(away_name))
            processed_teams.add(standardize_team_short_name(home_name))
        
        if not schedule_data:
            print(f"No valid games with team data for {date_str}")
            return pd.DataFrame()
        
        schedule_df = pd.DataFrame(schedule_data)
        schedule_df['game_date'] = pd.to_datetime(schedule_df['game_date'], format='%Y-%m-%d')
        schedule_df['away_name'] = schedule_df['away_name'].apply(standardize_team_short_name)
        schedule_df['home_name'] = schedule_df['home_name'].apply(standardize_team_short_name)
        
        print(f"Processed {len(schedule_data)} games for {date_str}: {', '.join(sorted(processed_teams))}")
        if len(schedule_data) < 10:
            print(f"Warning: Only {len(schedule_data)} games returned; a full MLB slate typically includes 10-15 games.")
        
        return schedule_df
    except Exception as e:
        print(f"Critical error fetching schedule for {date_str}: {e}")
        return pd.DataFrame()

def process_today_schedule(api_key, date_str='2025-05-18', model_type='hr', export_each=False):
    """Process the MLB schedule for a given date to predict home run prospects or matchup outcomes."""
    batter_df, pitcher_df, statcast_df = load_baseball_data()
    weather_obj = BallparkWeather(api_key=api_key)
    park_factors = {
        'ARI': 105, 'ATL': 102, 'BAL': 98, 'BOS': 100, 'CHC': 99, 'CIN': 103, 'CLE': 97,
        'COL': 115, 'CWS': 101, 'DET': 96, 'HOU': 100, 'KC': 98, 'LAA': 99, 'LAD': 97,
        'MIA': 95, 'MIL': 102, 'MIN': 100, 'NYM': 98, 'NYY': 103, 'OAK': 94, 'PHI': 101,
        'PIT': 99, 'SD': 96, 'SEA': 94, 'SF': 93, 'STL': 98, 'TB': 97, 'TEX': 101, 'TOR': 99, 'WSH': 100
    }
    
    schedule_df = get_schedule(date_str)
    if schedule_df.empty:
        print("No schedule data available.")
        return pd.DataFrame()
    
    print("Schedule loaded:")
    print(schedule_df[['away_name', 'home_name', 'game_date', 'game_hour', 'away_pitcher', 'home_pitcher']])
    print('')
    
    if model_type.lower() == 'hr':
        feature_config = {
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
        hr_tool = HomeRunProjectionTool(feature_config=feature_config)
        
        all_prospects = []
        if export_each:
            output_base_dir = f"output/{date_str.replace('-', '')}/homeruns"
            os.makedirs(output_base_dir, exist_ok=True)
        
        for _, row in schedule_df.iterrows():
            away_team = row['away_name']
            home_team = row['home_name']
            game_date = row['game_date'].strftime('%Y-%m-%d')
            game_hour = row['game_hour']
            away_pitcher = row['away_pitcher']
            home_pitcher = row['home_pitcher']
            
            if not (away_team and home_team):
                print(f"Skipping game with invalid teams: away={away_team}, home={home_team}")
                continue
            
            matchup_str = f"{away_team} vs {home_team}"
            csv_matchup_str = f"{away_team}-{home_team}"
            print('')
            print(f"Processing game: {matchup_str} on {game_date} at {game_hour}:00")
            
            # Fetch team rosters
            away_roster = get_team_roster(away_team, batter_df, min_pa=10)
            home_roster = get_team_roster(home_team, batter_df, min_pa=10)
            away_player_ids = {
                'mlbam': [player['mlbam_id'] for player in away_roster],
                'fangraphs': [player['key_fangraphs'] for player in away_roster if player['key_fangraphs']],
                'roster': away_roster  # Pass full roster for debugging
            }
            home_player_ids = {
                'mlbam': [player['mlbam_id'] for player in home_roster],
                'fangraphs': [player['key_fangraphs'] for player in home_roster if player['key_fangraphs']],
                'roster': home_roster  # Pass full roster for debugging
            }
            # print(f"Away team {away_team} roster: {[player['name'] for player in away_roster]}")
            # print(f"Home team {home_team} roster: {[player['name'] for player in home_roster]}")
            
            try:
                weather_data = weather_obj.get_forecast(matchup_str, game_hour=game_hour, game_date=game_date)
            except (ValueError, Exception) as e:
                print(f"Error fetching weather for {matchup_str}: {e}")
                weather_data = None
            
            # Process home team prospects (away team batters vs. home pitcher)
            home_prospects = hr_tool.get_prospects(
                pitcher_name=home_pitcher,
                statcast_df=statcast_df,
                batter_df=batter_df,
                pitcher_df=pitcher_df,
                weather=weather_data,
                team=away_team,
                home_team=home_team,
                park_factors=park_factors,
                player_ids=away_player_ids
            )
            
            # Process away team prospects (home team batters vs. away pitcher)
            away_prospects = hr_tool.get_prospects(
                pitcher_name=away_pitcher,
                statcast_df=statcast_df,
                batter_df=batter_df,
                pitcher_df=pitcher_df,
                weather=weather_data,
                team=home_team,
                home_team=home_team,
                park_factors=park_factors,
                player_ids=home_player_ids
            )
            
            # Calculate total expected home runs
            total_hr = 0.0
            if not home_prospects.empty and not away_prospects.empty:
                starter_hr = (home_prospects['hr_probability'].sum() + away_prospects['hr_probability'].sum())
                bullpen_hr = (home_prospects['bp_hr_probability'].sum() + away_prospects['bp_hr_probability'].sum())
                total_hr = (2/3 * starter_hr + 1/3 * bullpen_hr)
                print(f"Expected total home runs for {matchup_str}: {total_hr:.2f}")
            else:
                print(f"Warning: Insufficient data for {matchup_str}. Expected total home runs: 0.00")
            
            # Combine prospects for both pitchers
            matchup_prospects = []
            if not home_prospects.empty:
                home_prospects['Game'] = matchup_str
                home_prospects['Pitcher'] = home_pitcher
                home_prospects['Game_Date'] = game_date
                matchup_prospects.append(home_prospects)
            
            if not away_prospects.empty:
                away_prospects['Game'] = matchup_str
                away_prospects['Pitcher'] = away_pitcher
                away_prospects['Game_Date'] = game_date
                matchup_prospects.append(away_prospects)
            
            if matchup_prospects:
                combined_prospects = pd.concat(matchup_prospects, ignore_index=True)
                if combined_prospects['Name'].duplicated().any():
                    print(f"Warning: Found {combined_prospects['Name'].duplicated().sum()} duplicate batters in {csv_matchup_str}. Keeping highest matchup_score.")
                    combined_prospects = combined_prospects.sort_values('matchup_score', ascending=False)
                    combined_prospects = combined_prospects.drop_duplicates(subset=['Name'], keep='first')
                all_prospects.append(combined_prospects)
                if export_each:
                    csv_path = f"{output_base_dir}/{csv_matchup_str}.csv"
                    combined_prospects.to_csv(csv_path, index=False)
                    print(f"Saved matchup prospects to {csv_path}")
        
        if not all_prospects:
            print("No prospects found for the given schedule.")
            return pd.DataFrame()
        
        all_prospects = pd.concat(all_prospects, ignore_index=True)
        if all_prospects['Name'].duplicated().any():
            print(f"Warning: Found {all_prospects['Name'].duplicated().sum()} duplicate batters in final output. Keeping highest matchup_score.")
            all_prospects = all_prospects.sort_values('matchup_score', ascending=False)
            all_prospects = all_prospects.drop_duplicates(subset=['Name'], keep='first')
        return all_prospects
    
    elif model_type.lower() == 'outcome':
        feature_config = {
            'model_prediction': True,
            'team_offense': True,
            'team_pitching': True,
            'pitcher_metrics': True,
            'recent_performance': True,
            'standings': True,
            'weather_adjustments': True,
            'park_factors': True
        }
        outcome_tool = OutcomeProjectionTool(feature_config=feature_config)
        
        predictions = outcome_tool.get_outcome_predictions(
            schedule_df=schedule_df,
            statcast_df=statcast_df,
            batter_df=batter_df,
            pitcher_df=pitcher_df,
            weather_obj=weather_obj,
            park_factors=park_factors,
            date_str=date_str
        )
        return predictions.sort_values('Home_Win_Probability', ascending=False)
    
    else:
        print(f"Invalid model_type: {model_type}. Choose 'hr' or 'outcome'.")
        return pd.DataFrame()

if __name__ == "__main__":
    api_key = "714a571b971e414d9b6193548251405"
    date_str = '2025-05-18'
    model_type = 'hr'
    results = process_today_schedule(api_key, date_str, model_type)
    
    if not results.empty:
        if model_type.lower() == 'hr':
            print("\nTop Home Run Prospects for", date_str)
            print(results[[
                'Game', 'Pitcher', 'Name', 'Team', 'hr_probability', 'bp_hr_probability',
                'matchup_score', 'bp_matchup_score', 'Barrel%', 'HardHit%', 'wRC+',
                'Recent_Flyball%', 'Long_Flyout_Count', 'Swing_Discipline_Score',
                'Recent_HR_Score'
            ]])
            output_base_dir = "output"
            os.makedirs(output_base_dir, exist_ok=True)
            export_path = f"{output_base_dir}/hr_predictions_{date_str.replace('-', '')}.csv"
            results.to_csv(export_path, index=False)
            print(f"Saved complete prospects to {export_path}")
        else:
            print("\nMatchup Outcome Predictions for", date_str)
            print(results[[
                'Game', 'Home_Team', 'Away_Team', 'Home_Pitcher', 'Away_Pitcher',
                'Home_Win_Probability', 'Home_wRC+', 'Away_wRC+', 'Home_ERA', 'Away_ERA',
                'Home_Win_Rate', 'Away_Win_Rate', 'Home_Run_Differential', 'Away_Run_Differential',
                'Home_Win_Percentage', 'Away_Win_Percentage', 'Home_Games_Back', 'Away_Games_Back'
            ]])
    else:
        print("No results found.")