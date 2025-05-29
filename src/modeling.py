# src/modeling.py

from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.utils import setup_logging
from collections import Counter


logger = setup_logging(__name__)

# Configurable constants
TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_SAMPLE = True  # toggle SMOTE on/off

def train_test_data(
    df: pd.DataFrame, 
    features: List[str], 
    target: str = "Churn"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[features]
    y = df[target].map({"No":0, "Yes":1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )
    if SMOTE_SAMPLE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logger.info(f"Applied SMOTE: {Counter(y_train)}")
    return X_train, X_test, y_train, y_test

def build_models() -> Dict[str, object]:
    return {
        "Logistic": LogisticRegression(
            solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE
        )
    }

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }

def train_and_evaluate(
    df: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_data(df, features)
    models = build_models()
    results = []
    for name, mdl in models.items():
        logger.info(f"Training {name}…")
        mdl.fit(X_train, y_train)
        metrics = evaluate(mdl, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
        logger.info(f"{name} metrics: {metrics}")
    return pd.DataFrame(results).set_index("model")
from sklearn.model_selection import GridSearchCV

def tune_random_forest(
    X_train, y_train,
    param_grid: dict = None,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1
):
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
    rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    grid = GridSearchCV(
        rf, param_grid, cv=cv, scoring=scoring,
        n_jobs=n_jobs, verbose=2
    )
    grid.fit(X_train, y_train)
    return grid

