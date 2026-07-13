"""
evaluation.py
=============

Classification Evaluation module for tabular classification datasets.

RESPONSIBILITY
--------------
This module is responsible for ONE thing: measuring, visualizing, and
reporting the quality of predictions produced by a trained model.

It does NOT:
    - Train models                    (done by models.py)
    - Generate predictions            (done by models.py)
    - Preprocess or transform data    (done by preprocessing.py)
    - Select features                 (done by feature_selection.py)
    - Perform EDA                     (done by eda.py)
    - Split data                      (done explicitly in the notebook)

PUBLIC INTERFACE
----------------
The notebook calls exactly ONE function:

    metrics_df, cm_df, report_str = classification_metrics(
        y_test, y_pred, y_prob,
        class_names=["Rejected", "Approved"]
    )

Everything else is a private internal helper (prefixed with _).

PIPELINE POSITION
-----------------
    classification_model()       -- models.py (train + predict)
    classification_metrics()     -- THIS MODULE (measure quality)

RETURNS
-------
    metrics_df   : pd.DataFrame [Metric, Value]
                   Accuracy, Precision, Recall, F1 Score, ROC-AUC.
    cm_df        : pd.DataFrame (confusion matrix, labeled)
                   Index = Actual classes. Columns = Predicted classes.
    report_str   : str (full sklearn classification report, per-class breakdown)

METRICS SUMMARY
---------------
    Accuracy        : Correct predictions / total predictions.
    Precision       : Of positive predictions, fraction truly positive (weighted).
    Recall          : Of actual positives, fraction correctly found (weighted).
    F1 Score        : Harmonic mean of Precision and Recall (weighted).
    ROC-AUC         : Discriminative ability. Binary + y_prob required.
                      Skipped (not raised) for multiclass or y_prob=None.
    Confusion Matrix: Table of actual vs predicted class counts.
    Classification Report: Full per-class breakdown with support.

PLOT OUTPUT
-----------
    confusion_matrix.eps  Saved to figures_path (unless figures_path=None).
    Format: Times New Roman, 15pt bold labels, 600 DPI, EPS.

DESIGN NOTES
------------
    - All averaging uses average='weighted' for class-imbalance robustness.
    - ROC-AUC skips gracefully (no exception) when not applicable.
    - Plot formatting constants are duplicated from eda.py pending
      introduction of plot_utils.py (YAGNI -- not yet justified).

VERSION 1 LIMITATIONS
---------------------
    - ROC-AUC supports binary classification only.
      Multiclass ROC-AUC (multi_class='ovr') is a Version 2 extension.

PARAMETER NAMING (Consistent across the full framework)
--------------------------------------------------------
    class_names     : Human-readable labels for each class.
    figures_path    : Directory for saving EPS output files.

AUTHOR
------
    Roll No : 3122247001061
    Course  : Machine Learning Laboratory -- Semester 5
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PLOT FORMATTING CONSTANTS
#
# Intentionally duplicated from eda.py.
# When a third module requires these settings, they will be extracted
# into plot_utils.py (YAGNI -- not yet justified with only two modules).
#
# IMPORTANT: If faculty requires a formatting change, update BOTH eda.py
# and evaluation.py until plot_utils.py is introduced.
# =============================================================================

_FONT_FAMILY   = "Times New Roman"
_FONT_SIZE     = 15
_FIGURE_DPI    = 600
_FIGURE_FORMAT = "eps"
_CM_FILENAME   = "confusion_matrix"


# =============================================================================
# PRIVATE FUNCTION 0: Validate Inputs
#
# PURPOSE  : Guard clause. Fail fast with clear, actionable error messages.
# PRINTS   : Nothing.
# RETURNS  : None (raises on failure).
#
# Design note: sklearn metric functions produce cryptic errors on shape
# mismatches. This function surfaces every invalid condition BEFORE any
# metric is computed, with messages that tell the student exactly what
# went wrong and how to fix it.
# =============================================================================

def _validate_inputs(
    y_test,
    y_pred,
    y_prob,
    class_names: Optional[List[str]],
) -> None:
    """
    Validate all inputs to classification_metrics() before any computation.

    Purpose
    -------
    Fail fast. Provide clear, actionable error messages before sklearn
    encounters invalid input and raises unhelpful errors.

    Parameters
    ----------
    y_test : array-like
        Ground truth class labels.
    y_pred : array-like
        Predicted class labels from the model.
    y_prob : array-like or None
        Class probability estimates (n_samples × n_classes). Optional.
    class_names : list of str or None
        Human-readable class labels. Optional.

    Returns
    -------
    None
        Silent on success. Raises descriptive exceptions on failure.

    Raises
    ------
    TypeError
        If y_test or y_pred cannot be converted to numpy arrays.
    ValueError
        If arrays are empty, shapes mismatch, y_prob is malformed,
        or class_names count does not match unique classes.
    """
    # -- Convert to numpy for shape inspection (non-destructive) --------------
    try:
        y_te = np.asarray(y_test)
        y_pr = np.asarray(y_pred)
    except Exception as exc:
        raise TypeError(
            f"y_test and y_pred must be array-like (numpy array, pandas Series, "
            f"or Python list). Conversion failed: {exc}"
        ) from exc

    # -- Dimensionality -------------------------------------------------------
    if y_te.ndim != 1:
        raise ValueError(
            f"y_test must be 1-dimensional, got shape {y_te.shape}.\n"
            f"Extract the target vector with: y_test = selected_df[target_column]"
        )

    if y_pr.ndim != 1:
        raise ValueError(
            f"y_pred must be 1-dimensional, got shape {y_pr.shape}.\n"
            f"y_pred is returned as a 1D array by classification_model()."
        )

    # -- Non-empty ------------------------------------------------------------
    if len(y_te) == 0:
        raise ValueError(
            "y_test is empty. "
            "Verify that train_test_split() produced a non-empty test set."
        )

    # -- Length match ---------------------------------------------------------
    if len(y_te) != len(y_pr):
        raise ValueError(
            f"y_test and y_pred must have the same number of samples.\n"
            f"  y_test : {len(y_te)} samples\n"
            f"  y_pred : {len(y_pr)} samples\n"
            f"Ensure y_pred was generated from the same X_test used to produce y_test."
        )

    # -- y_prob validation (optional) -----------------------------------------
    if y_prob is not None:
        try:
            y_pb = np.asarray(y_prob)
        except Exception as exc:
            raise TypeError(
                f"y_prob must be array-like (numpy array or pandas DataFrame). "
                f"It is returned by classification_model(). Error: {exc}"
            ) from exc

        if y_pb.ndim != 2:
            raise ValueError(
                f"y_prob must be 2-dimensional (n_samples × n_classes), "
                f"got shape {y_pb.shape}.\n"
                f"y_prob is returned as a 2D array by classification_model()."
            )

        if y_pb.shape[0] != len(y_te):
            raise ValueError(
                f"y_prob and y_test must have the same number of samples.\n"
                f"  y_test : {len(y_te)} samples\n"
                f"  y_prob : {y_pb.shape[0]} rows\n"
                f"Ensure y_pred and y_prob came from the same classification_model() call."
            )

    # -- class_names validation (optional) ------------------------------------
    if class_names is not None:
        unique_classes = np.unique(y_te)
        n_unique       = len(unique_classes)
        if len(class_names) != n_unique:
            raise ValueError(
                f"class_names has {len(class_names)} entries, but y_test "
                f"contains {n_unique} unique class"
                f"{'es' if n_unique != 1 else ''}.\n"
                f"  class_names    : {class_names}\n"
                f"  Unique classes : {unique_classes.tolist()}\n"
                f"Provide one name per unique class in sorted ascending order.\n"
                f"Example: class_names=['No', 'Yes'] maps class 0→'No', 1→'Yes'."
            )


# =============================================================================
# PRIVATE FUNCTION 1: Compute Scalar Metrics
#
# PURPOSE  : Compute all numeric evaluation metrics using sklearn.
# PRINTS   : Nothing.
# RETURNS  : (metrics_dict, roc_auc_note)
#
# Design note: Separating computation from data structuring follows SRP.
# _compute_scalar_metrics uses sklearn. _build_metrics_dataframe uses pandas.
# These are different dependencies and different change reasons.
#
# ROC-AUC policy: Skip (not raise) for multiclass or when y_prob=None.
# Return the skip reason as a string so the public function can print it.
# =============================================================================

def _compute_scalar_metrics(
    y_test,
    y_pred,
    y_prob,
) -> Tuple[Dict, Optional[str]]:
    """
    Compute all scalar evaluation metrics.

    Parameters
    ----------
    y_test : array-like
        Ground truth class labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : np.ndarray or None
        Class probability estimates, shape (n_samples, n_classes).
        Required for ROC-AUC. If None, ROC-AUC is skipped.

    Returns
    -------
    metrics_dict : dict
        {metric_name: float_value} for all computed metrics.
        Keys: "Accuracy", "Precision", "Recall", "F1 Score",
              and "ROC-AUC" if successfully computed.
    roc_auc_note : str or None
        If ROC-AUC was skipped, a human-readable explanation string.
        None if ROC-AUC was successfully computed.

    Notes
    -----
    All multi-class metrics use average='weighted': each class contributes
    proportionally to its support (sample count). This handles class
    imbalance more honestly than macro averaging.

    zero_division=0: Prevents warnings and errors when a class has no
    predicted samples (returns 0.0 instead of raising ZeroDivisionError).
    """
    metrics_dict   = {}
    roc_auc_note   = None

    # -- Accuracy -------------------------------------------------------------
    metrics_dict["Accuracy"] = float(accuracy_score(y_test, y_pred))

    # -- Precision (weighted) -------------------------------------------------
    metrics_dict["Precision"] = float(
        precision_score(y_test, y_pred, average="weighted", zero_division=0)
    )

    # -- Recall (weighted) ----------------------------------------------------
    metrics_dict["Recall"] = float(
        recall_score(y_test, y_pred, average="weighted", zero_division=0)
    )

    # -- F1 Score (weighted) --------------------------------------------------
    metrics_dict["F1 Score"] = float(
        f1_score(y_test, y_pred, average="weighted", zero_division=0)
    )

    # -- ROC-AUC (binary classification only) ---------------------------------
    unique_classes = np.unique(y_test)
    n_classes      = len(unique_classes)

    if n_classes != 2:
        roc_auc_note = (
            f"ROC-AUC skipped: requires binary classification. "
            f"Found {n_classes} classes {unique_classes.tolist()}. "
            f"Multiclass ROC-AUC is a Version 2 extension."
        )
    elif y_prob is None:
        roc_auc_note = (
            "ROC-AUC skipped: y_prob=None. "
            "Probability estimates are required for ROC-AUC computation. "
            "Pass y_prob from classification_model() to enable this metric."
        )
    else:
        try:
            # Binary: column 1 holds the positive class probability
            metrics_dict["ROC-AUC"] = float(
                roc_auc_score(y_test, np.asarray(y_prob)[:, 1])
            )
        except (IndexError, ValueError) as exc:
            roc_auc_note = f"ROC-AUC computation failed: {exc}"

    return metrics_dict, roc_auc_note


# =============================================================================
# PRIVATE FUNCTION 2: Build Metrics DataFrame
#
# PURPOSE  : Convert the raw metrics dict to a [Metric, Value] DataFrame.
# PRINTS   : Nothing.
# RETURNS  : pd.DataFrame
#
# Design note: Returning a DataFrame (not a dict) is intentional.
# Experiment 2 requires comparing metrics across models:
#     pd.concat([metrics_knn, metrics_dt]).
# This is only ergonomic when metrics are already DataFrames.
# Consistent with feature_ranking_df from feature_selection.py.
# =============================================================================

def _build_metrics_dataframe(metrics_dict: Dict) -> pd.DataFrame:
    """
    Convert the raw metrics dictionary to a structured DataFrame.

    Parameters
    ----------
    metrics_dict : dict
        {metric_name: float_value} from _compute_scalar_metrics().

    Returns
    -------
    pd.DataFrame
        Columns: [Metric, Value]
        Metric order preserved from dict insertion order.
        Values rounded to 4 decimal places.

    Example
    -------
        Metric    Value
        Accuracy  0.8230
        Precision 0.8140
        Recall    0.8020
        F1 Score  0.8079
        ROC-AUC   0.8870
    """
    metrics_df          = pd.DataFrame(
        list(metrics_dict.items()),
        columns=["Metric", "Value"],
    )
    metrics_df["Value"] = metrics_df["Value"].round(4)

    return metrics_df.reset_index(drop=True)


# =============================================================================
# PRIVATE FUNCTION 3: Build Confusion Matrix DataFrame
#
# PURPOSE  : Build a labeled confusion matrix as a pandas DataFrame.
# PRINTS   : Nothing.
# RETURNS  : pd.DataFrame (rows=actual, columns=predicted, labeled)
#
# Design note: Returning a labeled DataFrame (not a raw numpy array) is
# critical for readability. A student reading cm_df["Approved"]["Rejected"]
# immediately understands "actual=Approved, predicted=Rejected". Without
# labels, they must cross-reference the integer encoding mapping.
# =============================================================================

def _build_confusion_matrix(
    y_test,
    y_pred,
    class_names: Optional[List[str]],
) -> pd.DataFrame:
    """
    Build the confusion matrix as a labeled pandas DataFrame.

    Parameters
    ----------
    y_test : array-like
        Ground truth class labels.
    y_pred : array-like
        Predicted class labels.
    class_names : list of str or None
        If provided, used as row and column labels.
        Must be in ascending sorted order matching the integer class codes.
        If None, sorted integer class codes are used.

    Returns
    -------
    pd.DataFrame
        Confusion matrix.
            Index        : Actual class labels   (rows)
            Columns      : Predicted class labels (columns)
            Index.name   : "Actual"
            Columns.name : "Predicted"

    Notes
    -----
    sklearn's confusion_matrix() sorts labels in ascending order.
    If class_names=["No", "Yes"], "No" maps to the smallest class code
    and "Yes" maps to the largest. Ensure the order is consistent with
    your label encoding from preprocessing.py.
    """
    unique_classes = sorted(np.unique(y_test).tolist())
    labels         = class_names if class_names is not None else unique_classes

    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

    cm_df = pd.DataFrame(
        cm,
        index   = labels,
        columns = labels,
    )
    cm_df.index.name   = "Actual"
    cm_df.columns.name = "Predicted"

    return cm_df


# =============================================================================
# PRIVATE FUNCTION 4: Plot Confusion Matrix
#
# PURPOSE  : Render the confusion matrix as a heatmap. Save EPS if requested.
# PRINTS   : Nothing.
# RETURNS  : None (side effect: creates a file and/or displays a plot).
#
# Design note: Plotting is a side effect — intentionally isolated here.
# If the student wants the cm_df numbers without a plot, they can ignore
# this function's output and read cm_df from the return tuple directly.
#
# figures_path=None:  Plot is shown but not saved to disk.
# figures_path="...": Plot is shown AND saved as EPS at 600 DPI.
#
# Format constants are duplicated from eda.py.
# Technical debt: extract to plot_utils.py when a third module needs them.
# =============================================================================

def _plot_confusion_matrix(
    cm_df: pd.DataFrame,
    figures_path: Optional[str],
) -> None:
    """
    Render the confusion matrix as a seaborn heatmap and optionally save EPS.

    Parameters
    ----------
    cm_df : pd.DataFrame
        Labeled confusion matrix from _build_confusion_matrix().
    figures_path : str or None
        Directory to save confusion_matrix.eps.
        Created automatically if it does not exist.
        If None, plot is displayed but not saved.

    Returns
    -------
    None
        Pure side effect. No data is returned.

    Notes
    -----
    Font: Times New Roman, 15pt, bold labels -- consistent with eda.py.
    Format: EPS at 600 DPI -- consistent with faculty requirements.
    plt.close() is called after saving to prevent memory leaks from
    accumulating figure objects across multiple classification_metrics() calls.
    """
    font_props = {
        "family" : _FONT_FAMILY,
        "size"   : _FONT_SIZE,
        "weight" : "bold",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm_df,
        annot      = True,
        fmt        = "d",
        cmap       = "Blues",
        linewidths = 0.5,
        linecolor  = "gray",
        ax         = ax,
        annot_kws  = {
            "size"   : _FONT_SIZE - 2,
            "family" : _FONT_FAMILY,
        },
    )

    ax.set_title("Confusion Matrix", fontdict=font_props, pad=14)
    ax.set_xlabel("Predicted Class",  fontdict=font_props, labelpad=10)
    ax.set_ylabel("Actual Class",     fontdict=font_props, labelpad=10)
    ax.tick_params(labelsize=_FONT_SIZE - 3)

    # Apply font family to tick labels explicitly
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(_FONT_FAMILY)

    plt.tight_layout()

    if figures_path is not None:
        os.makedirs(figures_path, exist_ok=True)
        save_path = os.path.join(
            figures_path, f"{_CM_FILENAME}.{_FIGURE_FORMAT}"
        )
        plt.savefig(
            save_path,
            dpi         = _FIGURE_DPI,
            format      = _FIGURE_FORMAT,
            bbox_inches = "tight",
        )

    plt.show()
    plt.close(fig)


# =============================================================================
# PRIVATE FUNCTION 5: Generate Classification Report
#
# PURPOSE  : Wrap sklearn's classification_report with framework conventions.
# PRINTS   : Nothing.
# RETURNS  : str (full per-class breakdown)
#
# Design note: This thin wrapper exists for three reasons:
#   1. Maps framework parameter 'class_names' → sklearn parameter 'target_names'.
#   2. Applies zero_division=0 consistently.
#   3. Keeps the parameter mapping detail out of the public function.
# =============================================================================

def _generate_classification_report(
    y_test,
    y_pred,
    class_names: Optional[List[str]],
) -> str:
    """
    Generate the full sklearn classification report.

    Parameters
    ----------
    y_test : array-like
        Ground truth class labels.
    y_pred : array-like
        Predicted class labels.
    class_names : list of str or None
        Passed as target_names to sklearn's classification_report().
        If None, integer class codes are used as row labels.

    Returns
    -------
    str
        Full text classification report. Per-class columns:
        Precision, Recall, F1-Score, Support.
        Plus: macro average, weighted average.
        Print in the notebook with: print(report_str)

    Notes
    -----
    zero_division=0 prevents ZeroDivisionError when a class has no
    predicted samples (returns 0.0 for that class's Precision/Recall).
    """
    return classification_report(
        y_test,
        y_pred,
        target_names  = class_names,
        zero_division = 0,
    )


# =============================================================================
# PUBLIC FUNCTION -- The ONLY function the notebook should call.
#
# DESIGN PATTERN : Facade Pattern (consistent with all previous modules).
# WHY            : Evaluation logic does not belong in the notebook.
#                  One clean function call per pipeline step.
# =============================================================================

def classification_metrics(
    y_test,
    y_pred,
    y_prob=None,
    class_names=None,
    figures_path: Optional[str] = "../figures/",
) -> Tuple:
    """
    Compute, display, and return all evaluation metrics for a classifier.

    This is the ONLY function this module exposes to the notebook.
    It orchestrates all metric computation, visualization, and reporting
    in one call, keeping the notebook at one function call for evaluation.

    Pipeline
    --------
    Step 1 : Validate all inputs                (_validate_inputs)
    Step 2 : Compute scalar metrics             (_compute_scalar_metrics)
    Step 3 : Build metrics DataFrame            (_build_metrics_dataframe)
    Step 4 : Build confusion matrix DataFrame   (_build_confusion_matrix)
    Step 5 : Plot and optionally save heatmap   (_plot_confusion_matrix)
    Step 6 : Generate classification report     (_generate_classification_report)
    Step 7 : Print formatted summary
    Step 8 : Return 3-tuple

    Parameters
    ----------
    y_test : array-like of shape (n_test_samples,)
        Ground truth class labels from train_test_split().
        Must match the labels in y_pred.
    y_pred : array-like of shape (n_test_samples,)
        Predicted class labels from classification_model().
        Must have the same length as y_test.
    y_prob : np.ndarray of shape (n_test_samples, n_classes) or None, optional
        Class probability estimates from classification_model().
        Required for ROC-AUC. If None or multiclass, ROC-AUC is skipped
        (no exception is raised -- a note is printed instead).
        Default: None
    class_names : list of str or None, optional
        Human-readable names for each class.
        Must be in the same sorted order as the unique integer class labels.
        Example: y_test has classes 0 and 1.
                 class_names=["Rejected", "Approved"]
                 maps 0 → "Rejected", 1 → "Approved".
        If None, integer class codes are used as labels in all outputs.
        Default: None
    figures_path : str or None, optional
        Directory where confusion_matrix.eps is saved.
        Created automatically if it does not exist.
        If None, confusion matrix plot is displayed but not saved.
        Default: "../figures/"

    Returns
    -------
    metrics_df : pd.DataFrame
        Columns: [Metric, Value]. Values rounded to 4 decimal places.
        Contains: Accuracy, Precision, Recall, F1 Score, ROC-AUC (if computed).
        Use for Experiment 2 model comparison:
            metrics_knn = classification_metrics(y_test, y_pred_knn, ...)[0]
            metrics_dt  = classification_metrics(y_test, y_pred_dt, ...)[0]
            pd.concat([metrics_knn, metrics_dt])
    cm_df : pd.DataFrame
        Labeled confusion matrix.
            Index        = Actual class labels
            Columns      = Predicted class labels
            Index.name   = "Actual"
            Columns.name = "Predicted"
    report_str : str
        Full text of sklearn's classification_report().
        Per-class: Precision, Recall, F1, Support. Plus averages.
        Print in notebook: print(report_str)

    Raises
    ------
    TypeError
        If y_test or y_pred are not array-like.
    ValueError
        If array lengths mismatch, y_test is empty, y_prob is malformed,
        or class_names count does not match unique classes in y_test.

    Example
    -------
    # Standard usage (Loan Prediction)
    >>> metrics_df, cm_df, report_str = classification_metrics(
    ...     y_test,
    ...     y_pred,
    ...     y_prob,
    ...     class_names=["Rejected", "Approved"],
    ... )
    >>> print(metrics_df)
    >>> print(report_str)

    # Without probability estimates (ROC-AUC skipped)
    >>> metrics_df, cm_df, report_str = classification_metrics(
    ...     y_test, y_pred,
    ...     class_names=["Class A", "Class B"],
    ... )

    # Without class names (integer labels used in all outputs)
    >>> metrics_df, cm_df, report_str = classification_metrics(y_test, y_pred)

    # Disable file saving (plot shown interactively only)
    >>> metrics_df, cm_df, report_str = classification_metrics(
    ...     y_test, y_pred, y_prob, figures_path=None
    ... )

    Notes
    -----
    - ROC-AUC is computed only for binary problems when y_prob is provided.
    - All averaging uses average='weighted' to handle class imbalance.
    - confusion_matrix.eps is saved at 600 DPI in EPS format.
    - Version 1 Limitation: Multiclass ROC-AUC is a future extension.
    """
    print()
    print("=" * 60)
    print("  CLASSIFICATION EVALUATION")
    print("=" * 60)
    print()

    # -- Step 1: Validate all inputs ------------------------------------------
    _validate_inputs(y_test, y_pred, y_prob, class_names)

    y_te           = np.asarray(y_test)
    unique_classes = sorted(np.unique(y_te).tolist())
    n_classes      = len(unique_classes)
    n_test         = len(y_te)

    print(f"  Test samples        : {n_test}")
    print(f"  Classes detected    : {n_classes}  {unique_classes}")
    if class_names:
        print(f"  Class names         : {class_names}")
    print(f"  y_prob provided     : {'Yes' if y_prob is not None else 'No'}")
    if figures_path:
        print(f"  Figures output      : {figures_path}")
    print()

    # -- Step 2: Compute scalar metrics ---------------------------------------
    metrics_dict, roc_auc_note = _compute_scalar_metrics(y_test, y_pred, y_prob)

    # -- Step 3: Build metrics DataFrame --------------------------------------
    metrics_df = _build_metrics_dataframe(metrics_dict)

    # -- Step 4: Build confusion matrix DataFrame -----------------------------
    cm_df = _build_confusion_matrix(y_test, y_pred, class_names)

    # -- Step 5: Plot and optionally save confusion matrix --------------------
    print("  Generating confusion matrix...")
    _plot_confusion_matrix(cm_df, figures_path)

    if figures_path is not None:
        eps_path = os.path.join(figures_path, f"{_CM_FILENAME}.{_FIGURE_FORMAT}")
        print(f"  Saved               : {eps_path}")
    print()

    # -- Step 6: Generate classification report --------------------------------
    report_str = _generate_classification_report(y_test, y_pred, class_names)

    # -- Step 7: Print formatted summary --------------------------------------
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print()
    print(metrics_df.to_string(index=False))
    print()

    if roc_auc_note:
        print(f"  Note  : {roc_auc_note}")
        print()

    print("  Confusion Matrix (raw counts):")
    print()
    # Indent each line of the DataFrame string for visual alignment
    for line in cm_df.to_string().splitlines():
        print(f"    {line}")
    print()

    print("  Classification Report:")
    print("  " + "-" * 54)
    for line in report_str.splitlines():
        print(f"    {line}")

    print()
    print("=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print()

    return metrics_df, cm_df, report_str
