"""
evaluation.py
=============
Roll No : 3122247001061
Course  : ICS1512 — Machine Learning Laboratory, Semester 5

PURPOSE
-------
Provides four reusable functions:
    classification_metrics()       — evaluates a classifier's predictions
    regression_metrics()           — evaluates a regressor's predictions
    cross_validate_model()         — 5-fold cross validation
    compare_classification_models()— comparison table across multiple models

The notebook imports and calls these functions as needed per experiment.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# Faculty-required plot formatting
FONT_FAMILY = "Times New Roman"
FONT_SIZE   = 15
FIGURE_DPI  = 600
FIGURE_FMT  = "eps"


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def classification_metrics(y_test, y_pred, y_prob=None,
                            class_names=None, figures_path="../figures/",
                            filename="confusion_matrix"):
    """
    Compute and display all evaluation metrics for a classification model.

    Metrics computed
    ----------------
    Accuracy   : fraction of correct predictions
    Precision  : of all predicted positives, how many are truly positive
    Recall     : of all actual positives, how many did the model catch
    F1 Score   : harmonic mean of Precision and Recall
    ROC-AUC    : area under the ROC curve (binary only, requires y_prob)

    Also prints the full classification report and plots the confusion matrix.

    Parameters
    ----------
    y_test      : true labels (from train_test_split)
    y_pred      : predicted labels (from classification_model)
    y_prob      : predicted probabilities (from classification_model), or None
    class_names : list of human-readable class names, e.g. ["No", "Yes"]
    figures_path: directory to save the confusion matrix (None = don't save)
    filename    : base filename for the confusion matrix image, without extension
                  e.g. "confusion_matrix_gaussian_nb" → saves confusion_matrix_gaussian_nb.eps
                  default: "confusion_matrix"

    Returns
    -------
    metrics_df : pd.DataFrame with columns [Metric, Value]
    cm_df      : pd.DataFrame — labeled confusion matrix
    report     : str — full per-class classification report
    """

    # --- Scalar metrics -------------------------------------------------------
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    metrics = {
        "Accuracy" : round(accuracy,  4),
        "Precision": round(precision, 4),
        "Recall"   : round(recall,    4),
        "F1 Score" : round(f1,        4),
    }

    # ROC-AUC: only for binary classification when probabilities are provided
    n_classes = len(np.unique(y_test))
    if y_prob is not None and n_classes == 2:
        roc_auc = roc_auc_score(y_test, np.asarray(y_prob)[:, 1])
        metrics["ROC-AUC"] = round(roc_auc, 4)
    else:
        print("  Note: ROC-AUC skipped (requires binary classification + y_prob)")

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

    # --- Confusion matrix -----------------------------------------------------
    unique_classes = sorted(np.unique(y_test).tolist())
    labels         = class_names if class_names else unique_classes

    cm    = confusion_matrix(y_test, y_pred, labels=unique_classes)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name   = "Actual"
    cm_df.columns.name = "Predicted"

    # --- Classification report ------------------------------------------------
    report = classification_report(y_test, y_pred,
                                   target_names=class_names, zero_division=0)

    # --- Print results --------------------------------------------------------
    print(f"\n{'='*50}")
    print("  CLASSIFICATION EVALUATION RESULTS")
    print(f"{'='*50}")
    print(metrics_df.to_string(index=False))
    print(f"\n  Confusion Matrix:\n")
    for line in cm_df.to_string().splitlines():
        print(f"    {line}")
    print(f"\n  Classification Report:\n{report}")

    # --- Plot confusion matrix ------------------------------------------------
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size"  : FONT_SIZE,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, ax=ax,
                annot_kws={"size": FONT_SIZE - 2})
    ax.set_title("Confusion Matrix", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontweight="bold")
    ax.set_ylabel("Actual Class",    fontweight="bold")
    plt.tight_layout()

    if figures_path:
        os.makedirs(figures_path, exist_ok=True)
        path = os.path.join(figures_path, f"{filename}.{FIGURE_FMT}")
        fig.savefig(path, format=FIGURE_FMT, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.show()
    plt.close(fig)

    return metrics_df, cm_df, report


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def regression_metrics(y_test, y_pred, figures_path="../figures/"):
    """
    Compute and display all evaluation metrics for a regression model.

    Metrics computed
    ----------------
    MAE  : Mean Absolute Error — average of |actual - predicted|
    MSE  : Mean Squared Error  — average of (actual - predicted)²
    RMSE : Root MSE            — square root of MSE, same unit as target
    R²   : R-squared           — fraction of variance explained by the model
           R² = 1.0 is perfect; R² = 0 means the model is as good as mean

    Also plots Actual vs Predicted scatter plot.

    Parameters
    ----------
    y_test       : true target values (from train_test_split)
    y_pred       : predicted values (from regression_model)
    figures_path : directory to save actual_vs_predicted.eps (None = don't save)

    Returns
    -------
    metrics_df : pd.DataFrame with columns [Metric, Value]
    """

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    metrics = {
        "MAE" : round(mae,  4),
        "MSE" : round(mse,  4),
        "RMSE": round(rmse, 4),
        "R²"  : round(r2,   4),
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

    # --- Print results --------------------------------------------------------
    print(f"\n{'='*50}")
    print("  REGRESSION EVALUATION RESULTS")
    print(f"{'='*50}")
    print(metrics_df.to_string(index=False))
    print(f"{'='*50}\n")

    # --- Plot: Actual vs Predicted --------------------------------------------
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size"  : FONT_SIZE,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0", edgecolors="white", s=40)

    # Perfect prediction line (y = x)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val],
            color="red", linewidth=1.5, linestyle="--", label="Perfect Prediction")

    ax.set_xlabel("Actual Values",    fontweight="bold")
    ax.set_ylabel("Predicted Values", fontweight="bold")
    ax.set_title("Actual vs Predicted", fontsize=FONT_SIZE, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if figures_path:
        os.makedirs(figures_path, exist_ok=True)
        path = os.path.join(figures_path, f"actual_vs_predicted.{FIGURE_FMT}")
        fig.savefig(path, format=FIGURE_FMT, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.show()
    plt.close(fig)

    return metrics_df


# =============================================================================
# CROSS VALIDATION
# =============================================================================

def cross_validate_model(model, X, y, cv=5):
    """
    Perform k-Fold Cross Validation on a trained (or untrained) sklearn model.

    The model is cloned internally — your original fitted model is untouched.

    Metrics computed per fold
    -------------------------
    Accuracy, Precision (weighted), Recall (weighted), F1 (weighted)

    Parameters
    ----------
    model : any sklearn classifier (e.g. GaussianNB(), KNeighborsClassifier())
    X     : full feature matrix (before splitting — CV does its own splits)
    y     : full label array
    cv    : number of folds (default: 5)

    Returns
    -------
    scores_df  : pd.DataFrame — per-fold scores for each metric
    summary_df : pd.DataFrame — mean and std for each metric across folds
    """

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    labels  = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fold_scores = {}
    for metric, label in zip(metrics, labels):
        scores = cross_val_score(model, X, y, cv=skf,
                                 scoring=metric, n_jobs=-1)
        fold_scores[label] = scores

    scores_df = pd.DataFrame(fold_scores,
                             index=[f"Fold {i+1}" for i in range(cv)])

    summary_df = pd.DataFrame({
        "Metric": labels,
        "Mean"  : [round(scores_df[l].mean(), 4) for l in labels],
        "Std"   : [round(scores_df[l].std(), 4)  for l in labels],
    })

    print(f"\n{'='*50}")
    print(f"  {cv}-Fold Cross Validation — {type(model).__name__}")
    print(f"{'='*50}")
    print(scores_df.round(4).to_string())
    print(f"\n  Summary:")
    print(summary_df.to_string(index=False))
    print(f"{'='*50}\n")

    return scores_df, summary_df


# =============================================================================
# MODEL COMPARISON TABLE
# =============================================================================

def compare_classification_models(results, output_path="../output/"):
    """
    Build and save a comparison table for multiple classification models.

    Parameters
    ----------
    results     : list of dicts, one dict per model.
                  Each dict must have keys:
                      "Model"          : str  — model name
                      "Accuracy"       : float
                      "Precision"      : float
                      "Recall"         : float
                      "F1 Score"       : float
                      "ROC-AUC"        : float or "-"
                      "Train Time (s)" : float
                      "Predict Time (s)": float
    output_path : directory to save model_comparison.csv

    Returns
    -------
    comparison_df : pd.DataFrame — the full comparison table

    Example
    -------
    results = [
        {"Model": "GaussianNB",  "Accuracy": 0.82, ..., "Train Time (s)": 0.01},
        {"Model": "KNN",         "Accuracy": 0.91, ..., "Train Time (s)": 0.03},
    ]
    df = compare_classification_models(results, output_path="../output/")
    """

    comparison_df = pd.DataFrame(results, columns=[
        "Model", "Accuracy", "Precision", "Recall",
        "F1 Score", "ROC-AUC", "Train Time (s)", "Predict Time (s)",
    ])

    print(f"\n{'='*70}")
    print("  MODEL COMPARISON TABLE")
    print(f"{'='*70}")
    print(comparison_df.to_string(index=False))
    print(f"{'='*70}\n")

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, "model_comparison.csv")
        comparison_df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    return comparison_df
