# src/data_preprocessing.py

import pandas as pd
from pathlib import Path
from typing import Tuple

def load_raw_data(raw_dir: Path = Path("data/raw")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(raw_dir / r"C:\Users\aserm\OneDrive\Bureau\telecomchurn\data\raw\cell2celltrain.csv")  # bdel lpath hna
    df_holdout = pd.read_csv(raw_dir /r"C:\Users\aserm\OneDrive\Bureau\telecomchurn\data\raw\cell2cellholdout.csv") # hna
    return df_train, df_holdout

def clean_data(
    df_train: pd.DataFrame, 
    df_holdout: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 1) Drop no-columns >50% missing (we saw none)
    # 2) Impute numeric & categorical from train stats
    num_cols = [c for c in df_train.select_dtypes(include=["number"]).columns if c != "Churn"]
    cat_cols = [c for c in df_train.select_dtypes(include=["object","category"]).columns if c != "Churn"]
    
    # Numeric → median
    medians = df_train[num_cols].median()
    df_train[num_cols] = df_train[num_cols].fillna(medians)
    df_holdout[num_cols] = df_holdout[num_cols].fillna(medians)
    
    # Categorical → mode
    modes = df_train[cat_cols].mode().iloc[0]
    df_train[cat_cols] = df_train[cat_cols].fillna(modes)
    df_holdout[cat_cols] = df_holdout[cat_cols].fillna(modes)
    
    return df_train, df_holdout

def save_clean_data(
    df_train: pd.DataFrame, 
    df_holdout: pd.DataFrame, 
    proc_dir: Path = Path("data/processed")
) -> None:
    proc_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(proc_dir / r"C:\Users\aserm\OneDrive\Bureau\telecomchurn\data\raw\cell2celltrain.csv", index=False)  #bdel path
    df_holdout.to_csv(proc_dir / r"C:\Users\aserm\OneDrive\Bureau\telecomchurn\data\raw\cell2cellholdout.csv", index=False) #same
