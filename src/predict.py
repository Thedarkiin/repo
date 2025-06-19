# src/predict.py
import sys
from pathlib import Path
import pandas as pd
import joblib

SELECTED = [
    "CurrentEquipmentDays", "PercChangeMinutes", "MonthlyMinutes",
    "MonthsInService", "MonthlyRevenue", "PercChangeRevenues",
    "PeakCallsInOut", "OffPeakCallsInOut", "ReceivedCalls", "UnansweredCalls"
]

def predict(input_csv: Path, model_pkl: Path, output_csv: Path):
    # Convert to absolute paths to avoid relative path issues
    input_csv = Path(r"C:\Users\aserm\OneDrive\Bureau\telecomchurn\data\processed\holdout_clean.csv")  #bdel path
    model_pkl = Path(r"C:\Users\aserm\OneDrive\Bureau\telecomchurn\models\rf_best.pkl")  # here hadi hiya fin kaytsjl lmodel ila bghina ndiru deployement b snowflake or smg
    output_csv = Path(r"C:\Users\aserm\OneDrive\Bureau\telecomchurn\data\predictions\holdout_preds.csv") #.

    # Check if files exist
    print(f"Absolute path checked: {input_csv.absolute()}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not model_pkl.exists():
        raise FileNotFoundError(f"Model PKL not found: {model_pkl}")

    # Load data and validate features
    df = pd.read_csv(input_csv)
    missing = [col for col in SELECTED if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features in CSV: {missing}")

    # Generate predictions
    model = joblib.load(model_pkl)
    X = df[SELECTED]
    df["Churn_Prob"] = model.predict_proba(X)[:, 1]
    df["Churn_Pred"] = model.predict(X)

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python src/predict.py <input.csv> <model.pkl> <output.csv>")
    
    # Resolve paths from command-line arguments
    inp = Path(sys.argv[1]).resolve()
    mdl = Path(sys.argv[2]).resolve()
    out = Path(sys.argv[3]).resolve()
    
    predict(inp, mdl, out)