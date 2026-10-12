"""
Model Training and Cross-Validation Pipeline
for Smart Property Valuation (Regression) & Price Category Prediction (Classification).

Executes:
1. Preprocessing Pipeline (Numeric Imputation/Scaling + One-Hot Categorical Encoding)
2. Regression Model Suite:
   - Ridge Regression
   - Lasso Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
   - LightGBM Regressor
3. Classification Model Suite:
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier
   - LightGBM Classifier
   - Support Vector Classifier (SVC)
4. Evaluates all models against the requested performance matrices:
   - Regression: MAE, RMSE, VAF (Variance Accounted For), R2, RMSLE
   - Classification: Confusion Matrix, Accuracy, Precision, Recall/Sensitivity, Specificity, F1 Score, ROC, AUC
5. Serializes trained models and generates evaluation reports & plots into `artifacts/`.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Regression models
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

# Local modules
from src.data_cleaner import prepare_full_dataset
from src.metrics import evaluate_regression, evaluate_classification


def build_preprocessor(X: pd.DataFrame):
    """
    Creates a scikit-learn ColumnTransformer for numerical and categorical features.
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features


def train_regression_suite(X_train, y_train, X_test, y_test, preprocessor):
    """
    Trains and evaluates the full regression model suite on SalePrice.
    Fits with log1p(SalePrice) for stability and evaluates on original dollar scale.
    """
    print("\n" + "="*60)
    print("TRAINING REGRESSION MODELS (PROPERTY VALUATION)")
    print("="*60)

    # Transform target to log scale for training
    y_train_log = np.log1p(y_train)

    regressors = {
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=5000),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1),
        "LightGBM Regressor": LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
    }

    results = []
    trained_pipelines = {}

    for name, model in regressors.items():
        print(f"--> Training {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train_log)
        y_pred_log = pipeline.predict(X_test)
        y_pred_dollar = np.expm1(y_pred_log)

        metrics = evaluate_regression(y_test, y_pred_dollar, model_name=name)
        results.append(metrics)
        trained_pipelines[name] = pipeline

    df_results = pd.DataFrame(results).sort_values(by="MAE", ascending=True).reset_index(drop=True)
    print("\n--- REGRESSION PERFORMANCE MATRIX ---")
    print(df_results.to_string(index=False))

    return df_results, trained_pipelines


def train_classification_suite(X_train, y_train, X_test, y_test, preprocessor):
    """
    Trains and evaluates the full classification model suite on PriceCategory.
    Calculates Confusion Matrix, Accuracy, Precision, Recall/Sensitivity, Specificity, F1, and ROC/AUC.
    """
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODELS (PRICE CATEGORY PREDICTION)")
    print("="*60)

    class_names = ["Budget", "Mid-Range", "Luxury"]
    le = LabelEncoder()
    le.fit(class_names)

    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        "Support Vector Classifier (SVC)": SVC(probability=True, kernel="rbf", C=2.0, random_state=42),
        "XGBoost Classifier": XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42, n_jobs=-1),
        "LightGBM Classifier": LGBMClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42, verbose=-1)
    }

    summary_list = []
    full_evaluations = {}
    trained_pipelines = {}

    for name, model in classifiers.items():
        print(f"--> Training {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train_enc)
        y_pred_enc = pipeline.predict(X_test)
        y_pred_labels = le.inverse_transform(y_pred_enc)

        y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

        eval_res = evaluate_classification(
            y_true=y_test,
            y_pred=y_pred_labels,
            y_prob=y_prob,
            class_names=list(le.classes_),
            model_name=name
        )

        summary_list.append(eval_res["summary"])
        full_evaluations[name] = eval_res
        trained_pipelines[name] = pipeline

    df_summary = pd.DataFrame(summary_list).sort_values(by="F1-Score (Macro)", ascending=False).reset_index(drop=True)
    print("\n--- CLASSIFICATION PERFORMANCE MATRIX ---")
    print(df_summary.to_string(index=False))

    return df_summary, full_evaluations, trained_pipelines, le


def save_reports_and_plots(reg_df, clf_df, clf_evals, artifacts_dir="artifacts"):
    """
    Saves metrics tables and plots (Confusion Matrices & ROC Curves).
    """
    os.makedirs(f"{artifacts_dir}/reports", exist_ok=True)
    os.makedirs(f"{artifacts_dir}/plots", exist_ok=True)

    # Save CSV and JSON summary tables
    reg_df.to_csv(f"{artifacts_dir}/reports/regression_metrics.csv", index=False)
    clf_df.to_csv(f"{artifacts_dir}/reports/classification_metrics.csv", index=False)

    with open(f"{artifacts_dir}/reports/classification_evaluations.json", "w") as f:
        json.dump(clf_evals, f, indent=2)

    # Plot Confusion Matrix for Best Classifier
    best_clf_name = clf_df.iloc[0]["Model"]
    best_cm = np.array(clf_evals[best_clf_name]["confusion_matrix"])
    class_names = clf_evals[best_clf_name]["class_names"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {best_clf_name}")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.tight_layout()
    plt.savefig(f"{artifacts_dir}/plots/best_confusion_matrix.png", dpi=200)
    plt.close()

    # Plot ROC Curves for Best Classifier
    plt.figure(figsize=(7, 5))
    roc_data = clf_evals[best_clf_name]["roc_curves"]
    for cname, curve_info in roc_data.items():
        plt.plot(curve_info["fpr"], curve_info["tpr"], label=f"{cname} (AUC = {curve_info['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.title(f"One-vs-Rest ROC Curves: {best_clf_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{artifacts_dir}/plots/best_roc_curve.png", dpi=200)
    plt.close()

    print(f"\n[Artifacts] Reports & plots successfully saved to '{artifacts_dir}/'")


def run_full_pipeline(data_path="data/raw/train.csv"):
    """
    Main orchestration routine.
    """
    os.makedirs("artifacts/models", exist_ok=True)

    # 1. Load, clean, and prepare datasets
    data_bundle = prepare_full_dataset(data_path=data_path)
    X_train = data_bundle["X_train"]
    X_test = data_bundle["X_test"]
    y_reg_train = data_bundle["y_reg_train"]
    y_reg_test = data_bundle["y_reg_test"]
    y_clf_train = data_bundle["y_clf_train"]
    y_clf_test = data_bundle["y_clf_test"]
    thresholds = data_bundle["thresholds"]

    # 2. Build ColumnTransformer
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    # 3. Train Regression Models
    reg_df, reg_pipelines = train_regression_suite(X_train, y_reg_train, X_test, y_reg_test, preprocessor)

    # 4. Train Classification Models
    clf_df, clf_evals, clf_pipelines, label_encoder = train_classification_suite(X_train, y_clf_train, X_test, y_clf_test, preprocessor)

    # 5. Save Model Artifacts
    best_reg_name = reg_df.iloc[0]["Model"]
    best_clf_name = clf_df.iloc[0]["Model"]

    joblib.dump(reg_pipelines[best_reg_name], "artifacts/models/best_regressor.joblib")
    joblib.dump(clf_pipelines[best_clf_name], "artifacts/models/best_classifier.joblib")
    joblib.dump(label_encoder, "artifacts/models/label_encoder.joblib")

    with open("artifacts/models/thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    with open("artifacts/models/feature_metadata.json", "w") as f:
        json.dump({
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "sample_row": X_train.iloc[0].to_dict()
        }, f, indent=2)

    # 6. Save Evaluation Tables & Plots
    save_reports_and_plots(reg_df, clf_df, clf_evals)

    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Top Regressor: {best_reg_name} (MAE: ${reg_df.iloc[0]['MAE']:,.0f}, VAF: {reg_df.iloc[0]['VAF (%)']}%)")
    print(f"Top Classifier: {best_clf_name} (Accuracy: {clf_df.iloc[0]['Accuracy']*100:.2f}%, F1: {clf_df.iloc[0]['F1-Score (Macro)']:.4f})")
    print("="*60)


if __name__ == "__main__":
    run_full_pipeline()
