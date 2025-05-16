import pandas as pd
import statsapi
import zoneinfo
from commons import *
from helper import *

df = pd.read_csv('data/lineup.csv')
df.columns = df.columns.str.strip()

matchups = []
teams_list = []
schedule_dropdown = []
schedule = statsapi.schedule(date=TODAY_SLASHED, sportId=1, include_series_status=True)

for game in schedule:
    game_time_in_utc = datetime.fromisoformat(game['game_datetime'].replace("Z", "+00:00"))
    est_timezone = zoneinfo.ZoneInfo('America/New_York')
    game_time_in_est = game_time_in_utc.astimezone(est_timezone)

    schedule_dropdown.append(f"{get_key_from_value(MLB_TEAMS, game['away_name'])} vs. {get_key_from_value(MLB_TEAMS, game['home_name'])} - {convert_24_to_12(game_time_in_est.strftime("%H:%M"))}")

# Check if 'weather' column exists
if 'weather' not in df.columns:
    print("Error: 'weather' column not found in the CSV. Available columns:", df.columns.tolist())
    print("Please verify the column name in your CSV file.")
    exit(1)

weather_groups = df.groupby('weather')

# for weather, group in weather_groups:
#     # Get unique team codes for this weather condition
#     teams = group['team code'].unique()

#     if len(teams) == 2:
#         # Assume two teams with the same weather are playing each other
#         teams_list.append([teams[0], teams[1]])

#     else:
#         # Handle cases where weather condition doesn't clearly indicate a matchup
#         print(f"Warning: Weather condition '{weather}' has {len(teams)} teams: {teams}")

# # Function to get players for a team in a specific weather condition
# def get_team_players(team_code, weather):
#     team_data = df[(df['team code'] == team_code) & (df['weather'] == weather)]
    
#     # Sort by batting order
#     team_data = team_data.sort_values('batting order')
#     players = team_data[['batting order', 'player name']].values.tolist()
    
#     # Get starting pitcher (SP)
#     sp = team_data[team_data['batting order'] == 'SP']['player name'].values
#     sp = sp[0] if len(sp) > 0 else None
#     return players, sp