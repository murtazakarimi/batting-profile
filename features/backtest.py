import pandas as pd
from sklearn.metrics import precision_score, recall_score
import importlib

def backtest_model(start_date='2025-04-01', end_date='2025-05-14', model_type='hr', api_key="714a571b971e414d9b6193548251405", statcast_df=None):
    """Backtest the HR model and compute precision/recall."""
    if statcast_df is None:
        raise ValueError("statcast_df must be provided")
    
    # Dynamically import process_today_schedule to avoid circular import
    main_module = importlib.import_module('main')
    process_today_schedule = getattr(main_module, 'process_today_schedule')
    
    dates = pd.date_range(start_date, end_date)
    results = []
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        predictions = process_today_schedule(api_key, date_str, model_type)
        actual_hrs = statcast_df[
            (statcast_df['game_date'] == date_str) & 
            (statcast_df['events'] == 'home_run')
        ]['batter'].unique()
        predicted_names = predictions['Name'].unique() if not predictions.empty else []
        hits = len(set(actual_hrs).intersection(predicted_names))
        precision = precision_score(
            [1 if name in actual_hrs else 0 for name in predicted_names],
            [1] * len(predicted_names), zero_division=0
        ) if predicted_names else 0
        recall = hits / len(actual_hrs) if len(actual_hrs) > 0 else 0
        print(f"Date {date_str}: {hits}/{len(actual_hrs)} HRs, Precision: {precision:.3f}, Recall: {recall:.3f}")
        results.append({
            'Date': date_str,
            'Hits': hits,
            'Total_HRs': len(actual_hrs),
            'Precision': precision,
            'Recall': recall
        })
    return pd.DataFrame(results)