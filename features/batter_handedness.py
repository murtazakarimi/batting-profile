import pandas as pd
from pybaseball import playerid_reverse_lookup, playerid_lookup
from utils import standardize_name

def get_batter_handedness_data(pitcher_name, statcast_df, pitcher_df):
    """Calculate pitcher and batter handedness and HR rates by handedness."""
    statcast_df = statcast_df.copy()
    statcast_df['player_name_standard'] = statcast_df['player_name'].apply(
        lambda x: standardize_name(x, is_statcast_format=True)
    )
    
    pitcher_statcast = statcast_df[statcast_df['player_name_standard'] == pitcher_name].copy()
    if pitcher_statcast.empty:
        return None, None, None

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

    # Debug merge keys
    # print(f"Debug: pitcher_statcast shape before merge: {pitcher_statcast.shape}")
    # print(f"Debug: statcast_df unique merge keys: {statcast_df[['batter', 'game_pk', 'at_bat_number']].drop_duplicates().shape}")
    # print(f"Debug: pitcher_statcast unique merge keys: {pitcher_statcast[['batter', 'game_pk', 'at_bat_number']].drop_duplicates().shape}")
    
    # Ensure batter_hand is in pitcher_statcast
    pitcher_statcast = pitcher_statcast.merge(
        statcast_df[['batter', 'game_pk', 'at_bat_number', 'batter_hand']].drop_duplicates(),
        on=['batter', 'game_pk', 'at_bat_number'],
        how='left'
    )
    
    # Debug merge result
    # print(f"Debug: pitcher_statcast columns after merge: {pitcher_statcast.columns.tolist()}")
    
    # Handle batter_hand (check for suffixed columns)
    if 'batter_hand' not in pitcher_statcast.columns:
        if 'batter_hand_y' in pitcher_statcast.columns:
            pitcher_statcast['batter_hand'] = pitcher_statcast['batter_hand_y'].fillna('R')
        elif 'batter_hand_x' in pitcher_statcast.columns:
            pitcher_statcast['batter_hand'] = pitcher_statcast['batter_hand_x'].fillna('R')
        else:
            print(f"Warning: 'batter_hand' not found in pitcher_statcast after merge for {pitcher_name}. Defaulting to 'R'.")
            pitcher_statcast['batter_hand'] = 'R'
    else:
        pitcher_statcast['batter_hand'] = pitcher_statcast['batter_hand'].fillna('R')

    # Clean up suffixed columns
    for col in ['batter_hand_x', 'batter_hand_y']:
        if col in pitcher_statcast.columns:
            pitcher_statcast = pitcher_statcast.drop(columns=col)

    # Calculate HR rates by handedness
    hr_by_hand = pitcher_statcast[pitcher_statcast['events'] == 'home_run'].groupby('batter_hand').size()
    total_pa_by_hand = pitcher_statcast.groupby('batter_hand').size()
    hr_rate_by_hand = (hr_by_hand / total_pa_by_hand).fillna(0).to_dict()
    hr_rate_by_hand['L'] = hr_rate_by_hand.get('L', 0.035)
    hr_rate_by_hand['R'] = hr_rate_by_hand.get('R', 0.035)

    return pitcher_hand, pitcher_statcast, hr_rate_by_hand