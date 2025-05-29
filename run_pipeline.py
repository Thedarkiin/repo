# run_pipeline.py

import sys
from pathlib import Path
import logging

# 1️⃣ Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_pipeline")

# 2️⃣ Ensure src/ is on the path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

# 3️⃣ Imports
from data_preprocessing import load_raw_data, clean_data, save_clean_data
from eda_visualization  import run_eda
from feature_selection import remove_collinear, remove_high_missing, select_by_importance
from modeling         import train_test_data, build_models, tune_random_forest
from model_evaluation import save_confusion_matrix, save_roc_comparison
from predict           import predict

def main():
    # Paths
    raw_dir    = ROOT / "data" / "raw"
    proc_dir   = ROOT / "data" / "processed"
    pred_dir   = ROOT / "data" / "predictions"
    models_dir = ROOT / "models"

    # 1) Load & clean data
    logger.info("Loading raw data…")
    df_train, df_holdout = load_raw_data(raw_dir)
    logger.info("Cleaning data…")
    df_train, df_holdout = clean_data(df_train, df_holdout)
    save_clean_data(df_train, df_holdout, proc_dir)

    # 2) EDA
    logger.info("Running EDA…")
    key_feats = [
        "MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge",
        "MonthsInService", "PercChangeMinutes"
    ]
    run_eda(df_train, key_feats)

    # 3) Feature selection
    logger.info("Selecting features…")
    y = df_train["Churn"].map({"No": 0, "Yes": 1})
    X = df_train.drop(columns=["Churn", "CustomerID"])
    X1, drop1 = remove_collinear(X)
    X2, drop2 = remove_high_missing(X1)
    X_sel, top_feats = select_by_importance(X2, y, top_k=10)
    logger.info(f"Top features: {top_feats}")

    # 4) Baseline modeling & evaluation
    logger.info("Training & evaluating baseline models…")
    X_train, X_test, y_train, y_test = train_test_data(df_train, top_feats)
    results = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        save_confusion_matrix(name, y_test, y_pred)
        results[name] = (y_test, y_proba)
    save_roc_comparison(results)

    # 5) Hyperparameter tuning (skip if model exists)
    model_file = models_dir / "rf_best.pkl"
    if model_file.exists():
        logger.info(f"Found existing model at {model_file}, loading instead of tuning.")
        import joblib
        best_rf = joblib.load(model_file)
    else:
        logger.info("Tuning Random Forest (this may take a while)…")
        grid = tune_random_forest(X_train, y_train, cv=3, scoring="roc_auc")
        best_rf = grid.best_estimator_
        models_dir.mkdir(exist_ok=True)
        import joblib
        joblib.dump(best_rf, model_file)
        logger.info(f"Saved tuned RF → {model_file}")

    # 6) Holdout predictions
    logger.info("Generating holdout predictions…")
    pred_file = pred_dir / "holdout_preds.csv"
    predict(
        input_csv = proc_dir / "holdout_clean.csv",
        model_pkl = model_file,
        output_csv= pred_file
    )

    logger.info("Pipeline complete! All artifacts are in data/, models/, and reports/figures/")

if __name__ == "__main__":
    main()
