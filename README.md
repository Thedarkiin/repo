# 📊 Telecom Churn Prediction – Data Science Project

An end-to-end machine learning pipeline for predicting customer churn in the telecom industry.  
This project was developed as a capstone for a final-year project, majoring in Data Science Engineering (if that’s a thing, lol).

---

## 🎯 Objective

Churn prediction is a critical task for telecom companies.  
Acquiring a new customer can cost up to 7× more than retaining an existing one.  
This project aims to identify customers at high risk of churn using machine learning techniques,  
enabling the business to take proactive retention actions.

---

## 📁 Project Structure

```telecomchurn/
│
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Cleaned & transformed data
│
├── data/predictions/             # Final predictions (CSV with churn probabilities)
├── models/                       # Trained models (.pkl) — removed from GitHub (too large)
├── reports/figures/              # EDA and model evaluation plots
│
├── notebooks/                    # EDA & experimentation notebooks — removed (experimentation)
│
├── src/                          # Source code modules
│   ├── data_preprocessing.py
│   ├── eda_visualization.py
│   ├── feature_selection.py
│   ├── modeling.py
│   ├── model_evaluation.py
│   └── predict.py
│
├── run_pipeline.py               # Main script to run the full pipeline
├── environment.yml               # Conda environment file (all dependencies here)
└── README.md                     # Project overview
```
---

## ⚙️ How to Set Up

### 1. Clone the Repository

```git clone https://github.com/Thedarkiin/repo.git
cd telecomchurn
```

### 2. Create the Conda Environment
```
conda env create -f environment.yml
conda activate telecomchurn
```
### 3. (Optional) Install Any Missing Dependencies
```
pip install -r requirements.txt
```
---

## 🚀 How to Run the Pipeline
```
python run_pipeline.py
```
This pipeline performs the following steps:

- Load raw data  
- Clean and preprocess data  
- Generate and save EDA plots  
- Select relevant features  
- Train and evaluate several models  
- Tune a Random Forest model  
- Generate predictions on holdout data  

All outputs are saved in the data/, models/, and reports/figures/ folders.

---

## 🔍 Features Used

- Monthly revenue and usage statistics  
- Call types (e.g., inbound/outbound, dropped)  
- Customer demographics (e.g., age, household)  
- Contract and service details (tenure, handset type, etc.)

---

## 📊 Key Outputs

- Confusion matrices for each model  
- ROC curve comparisons  
- Top features by importance  
- Churn probability scores on holdout data

---

## 📈 Performance Overview

The best model (Random Forest) was selected based on ROC AUC and precision/recall trade-offs.  
All evaluations were performed on a held-out test set to ensure unbiased performance metrics.

---

## 🧠 What You’ll Learn

- How to structure a complete machine learning pipeline  
- How to handle class imbalance using SMOTE  
- How to evaluate classification models effectively  
- How to deploy and store ML models  
- Best practices for modular, readable Python code

---

## 📌 Notes

- All data used is synthetic or anonymized (Kaggle, lol).  
- All scripts follow clean, reproducible engineering standards.
