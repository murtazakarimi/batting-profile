import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

def train_and_predict(data, features, target='hr_probability'):
    """Train an XGBoost model to predict home run probabilities and return predictions."""
    try:
        # Validate input data
        if data.empty or not all(f in data.columns for f in features):
            print("Warning: Empty data or missing features. Returning default probabilities.")
            return np.full(len(data), 0.035)  # Default league-average HR probability
        
        # Check for missing values
        if data[features].isna().any().any():
            print("Warning: Missing values in features. Filling with median.")
            data[features] = data[features].fillna(data[features].median())
        
        # Prepare features and target
        X = data[features]
        y = data[target] if target in data.columns else (data['hr_rate'] > 0).astype(int)
        
        # Split data (use small test size due to potentially small dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=(y > 0) if (y > 0).any() else None
        )
        
        # Configure XGBoost model
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 4,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'base_score': 0.5,  # Set valid base_score for logistic loss
            'random_state': 42
        }
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train model
        xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Predict probabilities
        ddata = xgb.DMatrix(X)
        predictions = xgb_model.predict(ddata)
        
        # Evaluate model
        if len(X_test) > 0:
            test_auc = roc_auc_score(y_test, xgb_model.predict(dtest))
        
        return predictions
    
    except Exception as e:
        print(f"Error in train_and_predict: {e}")
        return np.full(len(data), 0.035)  # Fallback to default probability