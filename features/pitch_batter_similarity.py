import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def get_pitch_batter_similarity(statcast_df, pitcher_name, prediction_date='2025-05-15', n_clusters=10):
    """Calculate similarity scores for batters based on pitcher and batter clustering."""
    statcast_df = statcast_df.copy()
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date'])
    prediction_date = pd.to_datetime(prediction_date)
    
    # Filter to 2023-2025 data
    statcast_df = statcast_df[statcast_df['game_date'] >= '2023-01-01']
    
    # --- Pitcher Clustering ---
    # Aggregate pitcher metrics: speed, handedness, pitch type frequency, spin rate
    try:
        pitcher_metrics = statcast_df.groupby('pitcher').agg({
            'release_speed': 'mean',
            'p_throws': 'first',
            'release_spin_rate': 'mean',
            'pitch_type': lambda x: x.value_counts(normalize=True).to_dict()
        }).reset_index()
    except KeyError as e:
        print(f"Error: Missing columns in statcast_df: {e}. Using fallback metrics.")
        try:
            pitcher_metrics = statcast_df.groupby('pitcher').agg({
                'p_throws': 'first',
                'pitch_type': lambda x: x.value_counts(normalize=True).to_dict()
            }).reset_index()
            pitcher_metrics['release_speed'] = 90.0  # Default average speed
            pitcher_metrics['release_spin_rate'] = 2200.0  # Default average spin rate
        except KeyError as e2:
            print(f"Error: Fallback failed: {e2}. Returning default similarity scores.")
            return pd.DataFrame({'batter': statcast_df['batter'].unique(), 'Similarity_HR_Rate': 0.035})
    
    # Extract pitch type frequencies (e.g., FF, SL, CH)
    pitch_types = ['FF', 'SL', 'CH', 'CU', 'SI', 'FC', 'KN', 'Unknown']
    for pt in pitch_types:
        pitcher_metrics[f'pitch_freq_{pt}'] = pitcher_metrics['pitch_type'].apply(
            lambda x: x.get(pt, 0)
        )
    pitcher_metrics = pitcher_metrics.drop(columns=['pitch_type'])
    
    # Convert handedness to binary (R=1, L=0)
    pitcher_metrics['p_throws'] = pitcher_metrics['p_throws'].map({'R': 1, 'L': 0}).fillna(1)
    
    # Handle missing values
    pitcher_metrics = pitcher_metrics.fillna({
        'release_speed': pitcher_metrics['release_speed'].mean() if 'release_speed' in pitcher_metrics else 90.0,
        'release_spin_rate': pitcher_metrics['release_spin_rate'].mean() if 'release_spin_rate' in pitcher_metrics else 2200.0
    })
    
    # Standardize features
    scaler = StandardScaler()
    pitcher_features = ['release_speed', 'p_throws', 'release_spin_rate'] + [f'pitch_freq_{pt}' for pt in pitch_types]
    pitcher_data = scaler.fit_transform(pitcher_metrics[pitcher_features])
    
    # Apply K-means clustering
    try:
        kmeans_p = KMeans(n_clusters=n_clusters, random_state=42)
        pitcher_metrics['cluster'] = kmeans_p.fit_predict(pitcher_data)
    except Exception as e:
        print(f"Error in pitcher clustering: {e}. Returning default similarity scores.")
        return pd.DataFrame({'batter': statcast_df['batter'].unique(), 'Similarity_HR_Rate': 0.035})
    
    # Find the target pitcher's cluster
    target_pitcher = pitcher_metrics[pitcher_metrics['pitcher'] == statcast_df[statcast_df['player_name_standard'] == pitcher_name]['pitcher'].iloc[0]]
    if target_pitcher.empty:
        print(f"Warning: Pitcher {pitcher_name} not found in statcast_df. Using default cluster.")
        target_cluster = 0
    else:
        target_cluster = target_pitcher['cluster'].iloc[0]
    
    # Get similar pitchers
    similar_pitchers = pitcher_metrics[pitcher_metrics['cluster'] == target_cluster]['pitcher'].tolist()
    
    # --- Batter Clustering ---
    # Aggregate batter metrics: handedness, barrel rate, exit velocity, pull percentage
    try:
        batter_metrics = statcast_df.groupby('batter').agg({
            'batter_hand': 'first',
            'is_barrel': 'mean',
            'launch_speed': 'mean',
            'hit_location': lambda x: (x.isin([3, 4])).mean()  # Pull side (left field)
        }).rename(columns={
            'is_barrel': 'barrel_rate',
            'launch_speed': 'exit_velocity',
            'hit_location': 'pull_percentage'
        }).reset_index()
    except KeyError as e:
        print(f"Error: Missing batter columns in statcast_df: {e}. Returning default similarity scores.")
        return pd.DataFrame({'batter': statcast_df['batter'].unique(), 'Similarity_HR_Rate': 0.035})
    
    # Convert handedness to binary
    batter_metrics['batter_hand'] = batter_metrics['batter_hand'].map({'R': 1, 'L': 0}).fillna(1)
    
    # Handle missing values
    batter_metrics = batter_metrics.fillna({
        'barrel_rate': batter_metrics['barrel_rate'].mean(),
        'exit_velocity': batter_metrics['exit_velocity'].mean(),
        'pull_percentage': batter_metrics['pull_percentage'].mean()
    })
    
    # Standardize features
    batter_features = ['batter_hand', 'barrel_rate', 'exit_velocity', 'pull_percentage']
    batter_data = scaler.fit_transform(batter_metrics[batter_features])
    
    # Apply K-means clustering
    try:
        kmeans_b = KMeans(n_clusters=n_clusters, random_state=42)
        batter_metrics['cluster'] = kmeans_b.fit_predict(batter_data)
    except Exception as e:
        print(f"Error in batter clustering: {e}. Returning default similarity scores.")
        return pd.DataFrame({'batter': statcast_df['batter'].unique(), 'Similarity_HR_Rate': 0.035})
    
    # --- Compute Similarity Score ---
    # For each batter, calculate HR rate against similar pitchers
    similar_pitcher_data = statcast_df[statcast_df['pitcher'].isin(similar_pitchers)]
    hr_performance = similar_pitcher_data[similar_pitcher_data['events'] == 'home_run'].groupby('batter').size()
    pa_performance = similar_pitcher_data.groupby('batter').size()
    similarity_scores = (hr_performance / pa_performance).reset_index(name='Similarity_HR_Rate')
    similarity_scores['Similarity_HR_Rate'] = similarity_scores['Similarity_HR_Rate'].fillna(0.035)
    
    # Merge with batter clusters
    similarity_scores = similarity_scores.merge(
        batter_metrics[['batter', 'cluster']], on='batter', how='left'
    )
    
    # For each batter in the game, assign the average Similarity_HR_Rate of their cluster
    batter_similarity = batter_metrics[['batter', 'cluster']].merge(
        similarity_scores.groupby('cluster')['Similarity_HR_Rate'].mean().reset_index(),
        on='cluster', how='left'
    )
    batter_similarity['Similarity_HR_Rate'] = batter_similarity['Similarity_HR_Rate'].fillna(0.035)
    
    return batter_similarity[['batter', 'Similarity_HR_Rate']]