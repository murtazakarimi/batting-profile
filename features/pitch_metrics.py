import pandas as pd

def get_pitch_specific_metrics(pitcher_statcast, min_pitches=5):
    """Calculate pitch-specific exit velocity and launch angle."""
    pitch_metrics = pitcher_statcast.groupby(['batter', 'pitch_type']).agg({
        'launch_speed': 'mean',
        'launch_angle': 'mean',
        'events': 'count'
    }).rename(columns={
        'launch_speed': 'pitch_exit_velo',
        'launch_angle': 'pitch_launch_angle',
        'events': 'pitch_count'
    }).reset_index()
    pitch_metrics = pitch_metrics[pitch_metrics['pitch_count'] >= min_pitches]
    return pitch_metrics