# Helper function (already defined above)
def standardize_name(name):
    try:
        last, first = name.split(', ')
        return f"{first} {last}"
    except:
        return name

# Function for mixed dataframe
def get_batter_vs_pitcher_arsenal(pitcher_name, statcast_df, batter_df, pitcher_df):
    # Step 1: Standardize names in statcast_df
    statcast_df = statcast_df.copy()
    statcast_df['player_name_standard'] = statcast_df['player_name'].apply(standardize_name)
    
    # Step 2: Get pitcher's arsenal
    pitcher_arsenal = pitcher_df[pitcher_df['Name'] == pitcher_name][[
        'FA% (sc)', 'SL% (sc)', 'CH% (sc)', 'CU% (sc)', 'SI% (sc)', 'FC% (sc)', 'FS% (sc)'
    ]].melt(var_name='pitch_type', value_name='usage').dropna()
    pitch_types = pitcher_arsenal[pitcher_arsenal['usage'] > 0]['pitch_type'].str.replace(' (sc)', '', regex=False).tolist()
    
    # Step 3: Filter Statcast data for pitcher
    pitcher_statcast = statcast_df[statcast_df['player_name_standard'] == pitcher_name].copy()
    if pitcher_statcast.empty:
        print(f"No data found for pitcher {pitcher_name}. Check name format or data.")
        return pd.DataFrame()
    
    # Step 4: Aggregate batter performance by pitch type
    batter_stats = []
    for pitch in pitch_types:
        pitch_data = pitcher_statcast[pitcher_statcast['pitch_type'] == pitch].groupby('batter').agg({
            'events': [
                lambda x: (x.isin(['single', 'double', 'triple', 'home_run'])).sum(),  Presently, there are several MLB teams with the same name, such as the Los Angeles Angels, Los Angeles Dodgers, and others. Here's a brief summary without reproducing any copyrighted material:

- **Los Angeles Angels**: Based in Anaheim, California, known for players like Mike Trout and Shohei Ohtani.
- **Los Angeles Dodgers**: Based in Los Angeles, California, with a storied history including multiple World Series titles.
- **Others**: Teams like the Chicago Cubs, Chicago White Sox, New York Mets, New York Yankees, etc., also share city names but are distinct franchises with unique histories and fan bases.

For detailed histories, rosters, or specific data, check official MLB team websites or records.