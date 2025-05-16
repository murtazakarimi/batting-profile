import statsapi
import streamlit as st
import pandas as pd
import numpy as np
# import matchup
# import helper
# import commons
from pybaseball import statcast
from pybaseball import standings
from pybaseball import team_game_logs
from pybaseball import statcast_batter
from pybaseball import playerid_lookup
from pybaseball import batting_stats_range
from pybaseball import batting_stats
from pybaseball import pitching_stats
from weather import BallparkWeather

data = team_game_logs(2025, "ATH", "batting")
print(data)
# Create sidebar and load match ups
# with st.sidebar:
#     selected_game = st.selectbox(
#     "Select a Match Up",
#     matchup.schedule_dropdown
#     )

# Selected match up
# selected_matchup = matchup.schedule[matchup.schedule_dropdown.index(selected_game)]

games = statsapi.schedule(start_date='03/10/2025', end_date='2025/05/15', team=None)

st.write(games)

# # Title of the match between opponents and venue
# st.header(f"{selected_matchup['away_name']} vs. {selected_matchup['home_name']}")
# st.write(f"{selected_matchup['venue_name']}")

# Columns
# col1, col2 = st.columns(2)

# home_team_abr = helper.get_key_from_value(matchup.MLB_TEAMS, selected_matchup['home_name'])
# away_team_abr = helper.get_key_from_value(matchup.MLB_TEAMS, selected_matchup['away_name'])

# get_lineups = statsapi.get("game", {"gamePk": selected_matchup['game_id']})

# # Away confirmed line up
# away_lineup = get_lineups['liveData']['boxscore']['teams']['away']['batters']

# # Home confirmed line up
# home_lineup = get_lineups['liveData']['boxscore']['teams']['home']['batters']

# def is_barrel(ev, la):
#     if pd.isna(ev) or pd.isna(la):
#         return False
#     return ev >= 98 and 26 <= la <= 30

# def is_hard_hit(ev):
#     if pd.isna(ev):
#         return False
#     return ev >= 95

# def is_fly_ball(la):
#     if pd.isna(la):
#         return False
#     return 25 <= la <= 50


# # API KEY: 714a571b971e414d9b6193548251405

# api_key = "714a571b971e414d9b6193548251405"
# matchup = "NYY vs SEA"
# game_hour = 21  # 9 PM ET

# # weather_tool = BallparkWeather(api_key)
# # forecast = weather_tool.get_forecast(matchup, game_hour)
# # hr_mult = weather_tool.calculate_hr_multiplier(forecast)
# # weather_tool.pretty_print(forecast, hr_mult)

# # all_batting_stats = batting_stats_range("2025-05-01", "2025-05-05")
# all_batting_stats2 = batting_stats(2025, end_season=None, league='all', qual=1, ind=1)
# all_pitching_stats2 = pitching_stats(2025, end_season=None, league='all', qual=1, ind=1)
# statcast_df = statcast(start_dt='2025-04-10', end_dt='2025-05-10', team=None)
# pitcher_name = 'deGrom, Jacob'
# # pitcher_statcast = statcast_df[statcast_df['player_name'] == pitcher_name].copy()

# # st.dataframe(statcast_df)
# # st.write(all_batting_stats2)
# # st.write('#######')
# # st.write(all_pitching_stats2)
# # st.write('#######')
# # st.write(statcast_df.columns)
# # If line ups are not confirmed
# # [680757, 682177, 608070, 700932, 467793, 666310, 672356, 686823, 682657, 671106] #

# if not away_lineup or home_lineup:
#     away_roster = statsapi.roster(selected_matchup['away_id'], rosterType=None, season=commons.today.year, date=None)
#     home_roster = statsapi.roster(selected_matchup['home_id'], rosterType=None, season=commons.today.year, date=None)
    
#     home_df = pd.DataFrame(columns=all_batting_stats.columns)
#     sp = selected_matchup['home_probable_pitcher']

#     st.write(sp)
#     # pitcher_data = statcast_pitcher(start_dt=[yesterday's date], end_dt=None, player_id)

#     for idx, player in helper.parse_roster(home_roster).items():
#         player_data = statsapi.lookup_player(player['name'])
#         player_id = player_data[0]['id']
        
#         home_df = pd.concat([home_df, all_batting_stats[all_batting_stats['mlbID'] == player_id]], ignore_index=True)
#         home_df_clean = home_df.drop(columns=['Age', '#days', 'Lev', 'Tm', 'PA', 'R', '2B', '3B', 'BB', 'IBB', 'SO', 'HBP', 'SH', 'SF', 'GDP', 'CS', 'mlbID']).sort_values(by='BA', ascending=False)
#     # st.dataframe(home_df_clean)





