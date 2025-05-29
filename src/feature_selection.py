# src/feature_selection.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List

# Add back the missing functions
def remove_collinear(df: pd.DataFrame, threshold: float = 0.9) -> Tuple[pd.DataFrame, List[str]]:
    """Remove highly correlated numeric features"""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr().abs()
    mask = np.tril(np.ones(corr.shape), k=-1).astype(bool)
    upper = corr.where(mask)
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return df.drop(columns=to_drop), to_drop

def remove_high_missing(df: pd.DataFrame, missing_pct: float = 0.5) -> Tuple[pd.DataFrame, List[str]]:
    """Remove columns with excessive missing values"""
    pct_missing = df.isnull().mean()
    to_drop = pct_missing[pct_missing > missing_pct].index.tolist()
    return df.drop(columns=to_drop), to_drop

def select_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 20,
    random_state: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    
    # Convert all possible numeric dtypes explicitly
    X = X.convert_dtypes()
    
    # Safeguard checks
    if X.shape[1] == 0:
        raise ValueError("Empty feature matrix passed to feature selector")
        
    numeric_cols = X.select_dtypes(include=[np.number, 'Int64', 'Float64']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric features available for importance scoring")
    
    X_num = X[numeric_cols]
    
    # Handle potential NaN values
    X_num = X_num.fillna(X_num.median())
    
    # Verify label alignment
    if len(X_num) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X_num)} vs {len(y)}")
    
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf.fit(X_num, y)
    
    imp = pd.Series(rf.feature_importances_, index=X_num.columns)
    top = imp.nlargest(top_k).index.tolist()
    
    return X_num[top], top