"""
Custom Metrics and Comprehensive Performance Matrix Evaluator
for Smart Property Valuation & Price Category Prediction.

Includes:
- Regression: MAE, RMSE, VAF (Variance Accounted For), R2, RMSLE
- Classification: Confusion Matrix, Accuracy, Precision, Recall/Sensitivity,
                 Specificity, F1 Score, ROC Curve, AUC
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)


# =====================================================================
# 1. REGRESSION EVALUATION METRICS
# =====================================================================

def variance_accounted_for(y_true, y_pred):
    """
    Computes Variance Accounted For (VAF) percentage:
    VAF = (1 - Var(y_true - y_pred) / Var(y_true)) * 100%
    Ideal value is 100%.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    var_diff = np.var(y_true - y_pred)
    var_true = np.var(y_true)
    if var_true == 0:
        return 0.0
    vaf = (1.0 - (var_diff / var_true)) * 100.0
    return float(vaf)


def evaluate_regression(y_true, y_pred, model_name="Model"):
    """
    Evaluates regression model against:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - VAF (Variance Accounted For %)
    - R2 Score
    - RMSLE (Root Mean Squared Logarithmic Error)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Floor at 0 for financial values
    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
    y_true_clipped = np.clip(y_true, a_min=0, a_max=None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    vaf = variance_accounted_for(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    try:
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true_clipped), np.log1p(y_pred_clipped)))
    except Exception:
        rmsle = np.nan

    return {
        "Model": model_name,
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "VAF (%)": round(float(vaf), 2),
        "R2 Score": round(float(r2), 4),
        "RMSLE": round(float(rmsle), 4)
    }


# =====================================================================
# 2. CLASSIFICATION EVALUATION METRICS
# =====================================================================

def calculate_specificity_per_class(cm):
    """
    Calculates Specificity (True Negative Rate) for each class in a multi-class confusion matrix:
    Specificity_i = TN_i / (TN_i + FP_i)
    """
    cm = np.asarray(cm)
    n_classes = cm.shape[0]
    specificities = []

    total_samples = np.sum(cm)
    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fp + fn)

        denominator = tn + fp
        spec = (tn / denominator) if denominator > 0 else 0.0
        specificities.append(float(spec))

    return specificities


def evaluate_classification(y_true, y_pred, y_prob=None, class_names=None, model_name="Model"):
    """
    Evaluates multi-class classification model against:
    - Confusion Matrix
    - Accuracy
    - Precision (Macro & Per-Class)
    - Recall / Sensitivity (Macro & Per-Class)
    - Specificity (Macro & Per-Class)
    - F1-Score (Macro & Weighted)
    - ROC Curve & AUC Score (One-vs-Rest)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if class_names is None:
        class_names = sorted(list(np.unique(y_true)))

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)  # Sensitivity
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)  # Sensitivity
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    spec_per_class = calculate_specificity_per_class(cm)
    spec_macro = float(np.mean(spec_per_class))

    # ROC / AUC (One-vs-Rest)
    auc_macro = None
    auc_per_class = {}
    roc_curves = {}

    if y_prob is not None:
        try:
            # One-hot encode y_true using class_names
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=class_names)
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([1 - y_bin, y_bin])

            # Ensure valid probability distribution & calculate macro AUC
            auc_macro = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")

            for idx, cname in enumerate(class_names):
                fpr, tpr, thresholds = roc_curve(y_bin[:, idx], y_prob[:, idx])
                c_auc = auc(fpr, tpr)
                auc_per_class[cname] = round(float(c_auc), 4)
                roc_curves[cname] = {
                    "fpr": [round(float(v), 5) for v in fpr],
                    "tpr": [round(float(v), 5) for v in tpr],
                    "auc": round(float(c_auc), 4)
                }
        except Exception as e:
            auc_macro = None

    summary = {
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision (Macro)": round(float(prec_macro), 4),
        "Recall / Sensitivity (Macro)": round(float(rec_macro), 4),
        "Specificity (Macro)": round(float(spec_macro), 4),
        "F1-Score (Macro)": round(float(f1_macro), 4),
        "F1-Score (Weighted)": round(float(f1_weighted), 4),
        "ROC AUC (Macro)": round(float(auc_macro), 4) if auc_macro is not None else "N/A"
    }

    per_class_table = []
    for idx, cname in enumerate(class_names):
        per_class_table.append({
            "Class": cname,
            "Precision": round(float(prec_per_class[idx]), 4),
            "Recall / Sensitivity": round(float(rec_per_class[idx]), 4),
            "Specificity": round(float(spec_per_class[idx]), 4),
            "F1-Score": round(float(f1_per_class[idx]), 4),
            "AUC": auc_per_class.get(cname, "N/A")
        })

    return {
        "summary": summary,
        "per_class": per_class_table,  # clean serializable list of dicts
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "roc_curves": roc_curves
    }
