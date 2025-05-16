import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def train_and_predict(data, features, use_ensemble=True):
    """Train an ensemble model to predict HR probability."""
    if data.empty:
        return np.zeros(len(data))
    
    X = data[features].fillna(0)
    # Placeholder: Use historical HR outcomes for training
    y = data['hr_count'].div(data['PA'].replace(0, 1)).apply(lambda x: 1 if x > 0.035 else 0)
    
    if use_ensemble:
        xgb_model = XGBClassifier(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1)
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
        
        # Train both models
        xgb_model.fit(X, y)
        rf_model.fit(X, y)
        
        # Cross-validate for model weights
        xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')
        rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='roc_auc')
        xgb_weight = xgb_scores.mean() / (xgb_scores.mean() + rf_scores.mean())
        rf_weight = rf_scores.mean() / (xgb_scores.mean() + rf_scores.mean())
        print(f"Ensemble weights: XGBoost={xgb_weight:.3f}, RandomForest={rf_weight:.3f}")
        
        # Combine predictions
        xgb_probs = xgb_model.predict_proba(X)[:, 1]
        rf_probs = rf_model.predict_proba(X)[:, 1]
        final_probs = xgb_weight * xgb_probs + rf_weight * rf_probs
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=2000)
        model.fit(X, y)
        final_probs = model.predict_proba(X)[:, 1]
    
    return final_probs