# Loan Default Prediction

**Group:** MachineStillLearning  
**Course Project** | December 2025

Predicting whether a borrower will become **90+ days past due** within two years using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset from Kaggle.

---

## Problem Statement

Binary classification on `SeriousDlqin2yrs` — a highly imbalanced target (~6.7% default rate). The goal is to build a reliable ML pipeline that minimizes false positives while still catching true defaulters.

---

## Dataset

| Split    | Records |
|----------|---------|
| Training | 150,000 |
| Test     | 101,503 |

**Key features:** revolving utilization, delinquency counts (30–59, 60–89, 90+ days), debt ratio, monthly income, number of open credit lines, dependents, and real estate loans.

Download from Kaggle and place in `data/raw/`:
```
data/raw/cs-training.csv
data/raw/cs-test.csv
```

---

## Pipeline Overview

```
Raw Data
   │
   ▼
Memory Optimization (downcast dtypes → ~70% size reduction)
   │
   ▼
Exploratory Data Analysis (correlation heatmap, distributions)
   │
   ▼
Missing Value Imputation (median — MonthlyIncome ~20%, Dependents ~3%)
   │
   ▼
Outlier Detection & Removal (Isolation Forest → ~1,170 rows removed)
   │
   ▼
Feature Engineering (outlier-clipped variants of skewed features)
   │
   ▼
Scaling (RobustScaler)
   │
   ▼
Train/Validation Split (stratified 80/20)
   │
   ▼
Class Imbalance Handling (Random Undersampling + SMOTE)
   │
   ▼
Model Training & Evaluation
```

---

## Models & Results

| Model                        | Accuracy | ROC-AUC |
|------------------------------|----------|---------|
| Logistic Regression (baseline) | 0.803  | 0.854   |
| Random Forest                | 0.816    | 0.858   |
| **Histogram Gradient Boosting** | **0.938** | **0.865** |

Gradient Boosting reduced false positives by **>90%** compared to Logistic Regression.

---

## Key Findings

- **Top predictors:** 30–59 day delinquency count, revolving utilization, 90-day late payments
- **Income** was not a dominant predictor — behavioral history mattered far more
- No strong multicollinearity found (all pairwise correlations < 0.7)

---

## Project Structure

```
loan-default-prediction/
├── data/
│   └── raw/               ← place Kaggle CSVs here (not tracked by git)
├── notebooks/
│   └── loan_default_prediction.ipynb
├── models/                ← saved model files (not tracked by git)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/loan_default_prediction.ipynb
```

---

## Team

| Member       | Contribution                              |
|--------------|-------------------------------------------|
| Srinivasa    | Data preprocessing & memory optimization  |
| Riyan        | Model implementation & tuning             |
| Sarthak      | Visualizations & evaluation metrics       |
| All members  | Documentation, interpretation, presentation |

---

## References

- Kaggle "Give Me Some Credit" dataset
- scikit-learn documentation
- Seaborn & Matplotlib libraries
- Google Colab (training environment)
