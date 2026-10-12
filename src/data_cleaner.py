"""
Data Cleaning and Preprocessing Module
for Smart Property Valuation & Price Category Prediction.

Features:
- Identifies and handles structural missingness (meaningful NA: 'None' for amenities)
- Median imputation for numerical features (e.g. LotFrontage grouped by Neighborhood)
- Outlier filtering (Dean De Cock recommended threshold: GrLivArea > 4000 with SalePrice < 300,000)
- Domain feature engineering (TotalSF, TotalBath, HouseAge, RemodAge, Quality scores)
- Dual target preparation:
    - Regression Target: SalePrice & LogSalePrice
    - Classification Target: PriceCategory (Budget, Mid-Range, Luxury)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(data_path="data/raw/train.csv"):
    """
    Loads raw CSV data. Falls back to finding CSV in current directory or data/raw/
    """
    if not os.path.exists(data_path):
        candidates = [
            "train.csv",
            "data/train.csv",
            "data/raw/train.csv",
            "AmesHousing.csv",
            "data/raw/AmesHousing.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please place 'train.csv' inside 'data/raw/'."
        )

    df = pd.read_csv(data_path)
    print(f"[DataLoader] Loaded {df.shape[0]} rows and {df.shape[1]} columns from {data_path}")
    return df


def clean_and_audit_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Performs comprehensive data auditing and cleaning.
    """
    df = df.copy()

    # Drop ID column if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    elif "Order" in df.columns and "PID" in df.columns:
        df = df.drop(columns=["Order", "PID"], errors="ignore")

    # 1. Outlier Filtering (recommended for Ames dataset: abnormal sales with GrLivArea > 4000)
    if is_train and "SalePrice" in df.columns and "GrLivArea" in df.columns:
        outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
        num_outliers = outlier_mask.sum()
        if num_outliers > 0:
            print(f"[DataCleaning] Removing {num_outliers} extreme outliers (GrLivArea > 4000 & SalePrice < 300k)")
            df = df[~outlier_mask].reset_index(drop=True)

    # 2. Structural Missing Categoricals (NA encodes feature absence, e.g. No Pool, No Garage)
    structural_na_cols = [
        "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "PoolQC", "Fence", "MiscFeature"
    ]
    for col in structural_na_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # 3. Numeric amenities (if amenity is absent, area/count is 0)
    zero_fill_num_cols = [
        "GarageYrBlt", "GarageCars", "GarageArea",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
    ]
    for col in zero_fill_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 4. Grouped Median Imputation: LotFrontage by Neighborhood
    if "LotFrontage" in df.columns:
        if "Neighborhood" in df.columns:
            df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(x.median())
            )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median() if not df["LotFrontage"].empty else 0.0)

    # 5. Categorical Mode Imputation for standard missing values
    mode_cols = [
        "MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd",
        "SaleType", "Utilities", "Functional"
    ]
    for col in mode_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)

    # 6. Remaining missing values fallback
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col != "SalePrice" and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Missing")

    # 7. Type conversions for non-ordinal numbers
    type_cast_cols = ["MSSubClass", "YrSold", "MoSold"]
    for col in type_cast_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    print(f"[DataCleaning] Data cleaned successfully. Remaining missing values: {df.isnull().sum().sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs high-value domain features for valuation and price tiers.
    """
    df = df.copy()

    # Total Square Footage
    bsmt = df["TotalBsmtSF"] if "TotalBsmtSF" in df.columns else 0
    flr1 = df["1stFlrSF"] if "1stFlrSF" in df.columns else (df["FirstFlrSF"] if "FirstFlrSF" in df.columns else 0)
    flr2 = df["2ndFlrSF"] if "2ndFlrSF" in df.columns else (df["SecondFlrSF"] if "SecondFlrSF" in df.columns else 0)
    df["TotalSF"] = bsmt + flr1 + flr2

    # Comprehensive Bathroom Count
    fb = df["FullBath"] if "FullBath" in df.columns else 0
    hb = df["HalfBath"] if "HalfBath" in df.columns else 0
    bfb = df["BsmtFullBath"] if "BsmtFullBath" in df.columns else 0
    bhb = df["BsmtHalfBath"] if "BsmtHalfBath" in df.columns else 0
    df["TotalBath"] = fb + 0.5 * hb + bfb + 0.5 * bhb

    # House Age & Remodeling Age
    yr_sold = pd.to_numeric(df["YrSold"], errors="coerce") if "YrSold" in df.columns else 2010
    yr_built = pd.to_numeric(df["YearBuilt"], errors="coerce") if "YearBuilt" in df.columns else 1970
    yr_remod = pd.to_numeric(df["YearRemodAdd"], errors="coerce") if "YearRemodAdd" in df.columns else yr_built

    df["HouseAge"] = (yr_sold - yr_built).clip(lower=0)
    df["RemodAge"] = (yr_sold - yr_remod).clip(lower=0)
    df["IsRemodeled"] = (yr_built != yr_remod).astype(int)

    # Porch Square Footage
    open_p = df["OpenPorchSF"] if "OpenPorchSF" in df.columns else 0
    enc_p = df["EnclosedPorch"] if "EnclosedPorch" in df.columns else 0
    ssn_p = df["3SsnPorch"] if "3SsnPorch" in df.columns else 0
    scr_p = df["ScreenPorch"] if "ScreenPorch" in df.columns else 0
    df["TotalPorchSF"] = open_p + enc_p + ssn_p + scr_p

    # Quality & Condition interaction
    if "OverallQual" in df.columns and "OverallCond" in df.columns:
        df["Qual_x_Cond"] = df["OverallQual"] * df["OverallCond"]

    # Ordinal Quality mapping for standard attributes
    quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
    qual_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
    for qcol in qual_cols:
        if qcol in df.columns:
            df[qcol + "_Score"] = df[qcol].map(quality_map).fillna(0)

    print(f"[FeatureEngineering] Engineered features added. Final column count: {df.shape[1]}")
    return df


def create_price_categories(y_series: pd.Series):
    """
    Discretizes continuous SalePrice into 3 categories:
    - Budget: Lower 25th percentile
    - Mid-Range: 25th - 75th percentile
    - Luxury: Upper 75th percentile
    """
    q25 = y_series.quantile(0.25)
    q75 = y_series.quantile(0.75)

    def assign_category(val):
        if val <= q25:
            return "Budget"
        elif val <= q75:
            return "Mid-Range"
        else:
            return "Luxury"

    categories = y_series.apply(assign_category)
    thresholds = {"q25": float(q25), "q75": float(q75)}
    print(f"[CategoryCreation] Quantiles calculated: Budget <= ${q25:,.0f}, Luxury > ${q75:,.0f}")
    return categories, thresholds


def prepare_full_dataset(data_path="data/raw/train.csv", test_size=0.2, random_state=42):
    """
    End-to-end cleaning, engineering, and train/test splitting for dual regression & classification.
    """
    raw_df = load_raw_data(data_path)
    cleaned_df = clean_and_audit_data(raw_df, is_train=True)
    featured_df = engineer_features(cleaned_df)

    if "SalePrice" not in featured_df.columns:
        raise ValueError("Dataset must contain 'SalePrice' target column.")

    y_regression = featured_df["SalePrice"].copy()
    y_classification, thresholds = create_price_categories(y_regression)

    X = featured_df.drop(columns=["SalePrice"])

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_regression, y_classification,
        test_size=test_size,
        random_state=random_state,
        stratify=y_classification
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "y_clf_train": y_clf_train,
        "y_clf_test": y_clf_test,
        "thresholds": thresholds,
        "full_data": featured_df
    }


if __name__ == "__main__":
    print("Testing data cleaner standalone...")
    try:
        data_bundle = prepare_full_dataset()
        print(f"X_train shape: {data_bundle['X_train'].shape}, X_test shape: {data_bundle['X_test'].shape}")
    except Exception as e:
        print(f"Standalone execution note: {e}")
