# src/eda_visualization.py

from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import setup_logging, FIGURES_PATH

logger = setup_logging(__name__)

# Absolute paths
EDA_PATH = FIGURES_PATH / "eda"
EDA_PATH.mkdir(parents=True, exist_ok=True)

# Constants
CHURN_COL      = "Churn"
TENURE_COL     = "MonthsInService"
TENURE_BUCKETS = [0, 3, 12, 24, 60]

def plot_churn_distribution(df: pd.DataFrame) -> None:
    counts = df[CHURN_COL].map({"No":0,"Yes":1}).value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot.bar(ax=ax)
    ax.set_title("Churn Distribution (0=No,1=Yes)")
    ax.set_xlabel(CHURN_COL)
    ax.set_ylabel("Count")
    fig.savefig(EDA_PATH / "churn_distribution.png")
    plt.close(fig)
    logger.info(f"Saved EDA plot → {EDA_PATH / 'churn_distribution.png'}")

def plot_numeric_feature(df: pd.DataFrame, feat: str) -> None:
    if feat not in df.columns:
        logger.warning(f"{feat} not in df, skipping")
        return
    sample = df.sample(20000, random_state=1) if len(df)>20000 else df
    fig, ax = plt.subplots()
    sample[feat].hist(bins=20, ax=ax)
    ax.set_title(f"{feat} Distribution")
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    out = EDA_PATH / f"{feat.lower()}_dist.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved EDA plot → {out}")

def plot_tenure_buckets(df: pd.DataFrame) -> None:
    if TENURE_COL not in df.columns:
        logger.warning("Tenure column missing, skipping")
        return
    df2 = df.copy()
    df2["_churn"]    = df2[CHURN_COL].map({"No":0,"Yes":1})
    df2["bucket"]   = pd.cut(df2[TENURE_COL], TENURE_BUCKETS)
    rates = df2.groupby("bucket")["_churn"].mean()
    fig, ax = plt.subplots()
    rates.plot.bar(ax=ax)
    ax.set_title("Churn Rate by Tenure Bucket")
    ax.set_xlabel("MonthsInService bucket")
    ax.set_ylabel("Churn Rate")
    out = EDA_PATH / "churn_by_tenure.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"Saved EDA plot → {out}")

def run_eda(df: pd.DataFrame, features: List[str]) -> None:
    plot_churn_distribution(df)
    for f in features:
        plot_numeric_feature(df, f)
    plot_tenure_buckets(df)
