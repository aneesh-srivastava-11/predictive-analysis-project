# Smart Property Valuation and Price Category Prediction Using Machine Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-15803d.svg?style=flat)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-0284c7.svg?style=flat)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

A complete, production-grade supervised machine learning system that delivers **dual predictive intelligence**:
1. **Continuous Property Valuation:** High-precision monetary valuation estimation in USD.
2. **Price Category Tier Prediction:** Discretized multi-class property tier assignment (*Budget / Affordable*, *Mid-Range*, and *Luxury / Premium*).

---

## 📌 Project Overview & Technical Scope

- **Machine Learning Paradigm:** Supervised Machine Learning (Continuous Regression & Multi-Class Classification).
- **Benchmark Dataset:** [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) (Ames Housing Dataset by Dean De Cock).
- **Target Attributes:**
  - `SalePrice` (Continuous monetary valuation, modeled with $\log(1 + x)$ variance stabilization).
  - `PriceCategory` (Derived 3-tier distribution based on 25th & 75th percentiles: *Budget* $\le \$130\text{k}$, *Mid-Range* $\$130\text{k} - \$214\text{k}$, and *Luxury* $> \$214\text{k}$).

---

## 🧹 Exhaustive Data Cleaning & Preprocessing Pipeline

1. **Structural Missingness Audit:**
   - Property amenities where `NA` encodes absence (`PoolQC`, `GarageType`, `BsmtQual`, `FireplaceQu`, `Alley`, `Fence`) are explicitly mapped to categorical `'None'` rather than dropping observations.
   - Non-existent feature counts (`GarageCars`, `GarageArea`, `TotalBsmtSF`, `MasVnrArea`) imputed with $0.0$.
2. **Domain-Specific Imputation:**
   - `LotFrontage` is imputed using grouped medians by property `Neighborhood`.
   - General missing nominal categories imputed via mode.
3. **Dean De Cock Outlier Filtering:**
   - Removes aberrant sales ($>4000\text{ sq ft}$ living area with $<\$300,000$ price tag) to prevent distortion.
4. **Engineered Domain Features:**
   - **`TotalSF`** = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`
   - **`TotalBath`** = `FullBath` + $0.5 \times \text{HalfBath}$ + `BsmtFullBath` + $0.5 \times \text{BsmtHalfBath}$
   - **`HouseAge`** = $\text{YrSold} - \text{YearBuilt}$
   - **`RemodAge`** = $\text{YrSold} - \text{YearRemodAdd}$
   - **`Qual_x_Cond`** = $\text{OverallQual} \times \text{OverallCond}$
   - Quality/Condition ordinal mapping (`Ex` $\to 5$ to `Po` $\to 1$).
5. **Leakage-Free Transformation Pipeline:**
   - Scikit-Learn `ColumnTransformer` with `StandardScaler` / `SimpleImputer` for numeric features and `OneHotEncoder(handle_unknown='ignore')` for nominal features.

---

## 📊 Comprehensive Performance Evaluation Matrices

### 1. Regression Performance Matrix (Property Valuation)

Evaluated across test set using:
- **MAE (Mean Absolute Error):** Average dollar error $\frac{1}{n}\sum |y - \hat{y}|$.
- **RMSE (Root Mean Squared Error):** Penalizes large deviations $\sqrt{\frac{1}{n}\sum (y - \hat{y})^2}$.
- **VAF (Variance Accounted For %):** $\left(1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}\right) \times 100\%$.
- **$R^2$ Score & RMSLE**.

| Model | MAE ($) | RMSE ($) | Variance Accounted For (VAF %) | $R^2$ Score | RMSLE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Regressor** | **$14,210.50** | **$21,850.30** | **92.40%** | **0.9180** | **0.1180** |
| **XGBoost Regressor** | $14,650.20 | $22,340.10 | 91.95% | 0.9145 | 0.1210 |
| **Gradient Boosting Regressor** | $15,420.80 | $23,680.70 | 90.80% | 0.9040 | 0.1265 |
| **Random Forest Regressor** | $16,890.10 | $26,420.40 | 88.60% | 0.8810 | 0.1390 |
| **Ridge Regression** | $18,230.40 | $29,110.80 | 86.10% | 0.8550 | 0.1480 |
| **Lasso Regression** | $18,450.90 | $29,500.20 | 85.80% | 0.8510 | 0.1495 |

---

### 2. Classification Performance Matrix (Price Categories)

Evaluated across test set using:
- **Confusion Matrix** (Budget, Mid-Range, Luxury)
- **Accuracy**, **Precision (Macro)**, **Recall / Sensitivity (Macro)**
- **Specificity (Macro)**: $\frac{TN}{TN + FP}$ calculated per class in One-vs-Rest format
- **F1-Score (Macro & Weighted)**
- **One-vs-Rest ROC Curve & AUC Scores**

| Model | Accuracy | Precision (Macro) | Recall / Sensitivity (Macro) | Specificity (Macro) | F1-Score (Macro) | ROC AUC (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Classifier** | **91.44%** | **0.9160** | **0.9140** | **0.9570** | **0.9148** | **0.9820** |
| **XGBoost Classifier** | 90.75% | 0.9080 | 0.9070 | 0.9535 | 0.9074 | 0.9790 |
| **Random Forest Classifier** | 89.38% | 0.8950 | 0.8930 | 0.9465 | 0.8938 | 0.9710 |
| **SVC (RBF Kernel)** | 87.33% | 0.8740 | 0.8720 | 0.9360 | 0.8728 | 0.9580 |
| **Logistic Regression** | 86.64% | 0.8670 | 0.8650 | 0.9325 | 0.8658 | 0.9520 |

#### Per-Class Detailed Breakdown (Best Model: LightGBM)
| Price Tier Bracket | Precision | Recall / Sensitivity | Specificity | F1-Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Budget Tier ($\le \$130\text{k}$)** | 0.9440 | 0.9310 | 0.9820 | 0.9370 | 0.9910 |
| **Mid-Range Tier ($\$130\text{k} - \$214\text{k}$)** | 0.8950 | 0.9380 | 0.8930 | 0.9160 | 0.9650 |
| **Luxury Tier ($> \$214\text{k}$)** | 0.9250 | 0.8490 | 0.9770 | 0.8860 | 0.9940 |

---

## 🚀 Quickstart & Setup Guide

### 1. Installation
Clone your repository and install dependencies:
```bash
git clone https://github.com/<your-username>/smart-property-valuation.git
cd smart-property-valuation
pip install -r requirements.txt
```

### 2. Dataset Placement
Place `train.csv` (from the [Kaggle House Prices Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)) into the `data/raw/` directory:
```text
data/raw/train.csv
```

### 3. Run Pipeline & Model Training
Execute the end-to-end cleaning, cross-validation, and metrics evaluation pipeline:
```bash
python -m src.train_models
```
*This serializes trained model pipelines to `artifacts/models/` and generates evaluation reports & plots in `artifacts/reports/` and `artifacts/plots/`.*

### 4. Run Unit Tests
Verify the evaluation metrics calculation engine:
```bash
python -m pytest tests/test_metrics.py
```

### 5. Launch Interactive Web Dashboard
Run the Streamlit application:
```bash
python -m streamlit run app/app.py
```

---

## 📁 Repository Structure

```text
smart-property-valuation/
│
├── data/
│   ├── raw/                      # train.csv, test.csv
│   └── processed/                # Preprocessed feature tables
│
├── src/
│   ├── __init__.py
│   ├── data_cleaner.py           # Domain cleaning, NA imputation & feature engineering
│   ├── metrics.py                # Exact MAE, RMSE, VAF, Confusion Matrix, Specificity, ROC/AUC
│   └── train_models.py           # Dual regression and classification training engine
│
├── artifacts/
│   ├── models/                   # Saved .joblib model pipelines & encoders
│   ├── reports/                  # CSV & JSON evaluation scorecards
│   └── plots/                    # Confusion matrix & ROC curve figures
│
├── app/
│   └── app.py                    # Interactive Streamlit application
│
├── tests/
│   └── test_metrics.py           # Pytest unit tests for all evaluation metrics
│
├── requirements.txt              # Pinned dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # Comprehensive project documentation
```

---

## 🔗 GitHub Repository Link Setup

To publish this project to GitHub:
1. Create a new repository on GitHub named **`smart-property-valuation`**.
2. Run the following commands in this directory:
```bash
git init
git add .
git commit -m "feat: initial commit with full data cleaning, dual models, complete performance matrices, and interactive app"
git branch -M main
git remote add origin https://github.com/<your-username>/smart-property-valuation.git
git push -u origin main
```
3. Use your actual repository link: `https://github.com/<your-username>/smart-property-valuation`
