"""
models.py
=========

Classification Model Training module for tabular classification datasets.

RESPONSIBILITY
--------------
This module is responsible for ONE thing: instantiating a classification
algorithm, training it on labelled data, and producing predictions on
unseen test data.

It does NOT:
    - Perform EDA or generate plots
    - Preprocess or transform data values
    - Select features
    - Split data into train/test sets  (done explicitly in the notebook)
    - Evaluate prediction quality      (done by evaluation.py)

PUBLIC INTERFACE
----------------
The notebook calls exactly ONE function:

    trained_model, y_pred, y_prob, train_time, pred_time = (
        classification_model(X_train, y_train, X_test, model_name="knn")
    )

Everything else is a private internal helper (prefixed with _).

PIPELINE POSITION
-----------------
    train_test_split()           -- In notebook (explicit and visible)
    classification_model()       -- THIS MODULE (train + predict)
    classification_metrics()     -- evaluation.py (measure quality)

SUPPORTED MODELS
----------------
    "knn"                : K-Nearest Neighbours    (KNeighborsClassifier)
    "decision_tree"      : Decision Tree           (DecisionTreeClassifier)
    "naive_bayes"        : Gaussian Naive Bayes    (GaussianNB)
    "logistic_regression": Logistic Regression     (LogisticRegression)
    "svm"                : Support Vector Machine  (SVC, probability=True)
    "random_forest"      : Random Forest           (RandomForestClassifier)

FACTORY PATTERN
---------------
Models are registered in _MODEL_REGISTRY, a dictionary mapping name to class.
Adding a new model in a future version requires adding ONE line to the registry.
No control flow changes. Open for extension, closed for modification.

TIMING CONVENTION
-----------------
    training_time   : Wall-clock seconds for model.fit(X_train, y_train)
    prediction_time : Wall-clock seconds for model.predict(X_test) only.

    predict_proba() is NOT included in prediction_time because:
        - It is optional (not all pipelines need probabilities).
        - Convention: "inference time" refers to class-label prediction.
        - Including it would make timing inconsistent across models.

    Both times use time.perf_counter(): monotonic, high-resolution.
    Both are returned for Experiment 2 model comparison.

PARAMETER NAMING (Consistent across the full framework)
--------------------------------------------------------
    random_state  : Seed for reproducibility. Applied only to models
                    that support it. Default: 42.

ASSUMPTIONS (Version 1)
-----------------------
    - X_train and X_test are clean, fully numeric (no NaN, no strings).
    - preprocessing.py has already been applied.
    - feature_selection.py has already been applied.
    - train_test_split() has already been called in the notebook.
    - Target column is label-encoded (integer class labels).

AUTHOR
------
    Roll No : 3122247001061
    Course  : Machine Learning Laboratory -- Semester 5
"""

# =============================================================================
# IMPORTS
# =============================================================================

import time
import warnings
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# MODULE CONSTANTS
#
# _MODEL_REGISTRY  : Maps model_name string -> sklearn class.
#                    Single source of truth for supported algorithms.
#                    To add a new model: add ONE line to this dict.
#
# _RANDOM_STATE_MODELS : Models whose constructors accept random_state.
#                    KNeighborsClassifier and GaussianNB are deterministic;
#                    passing random_state to them raises TypeError.
#
# Both are module-level constants (not inside functions) so they are
# computed once at import time and shared across all calls.
# =============================================================================

_MODEL_REGISTRY: dict = {
    "knn"                : KNeighborsClassifier,
    "decision_tree"      : DecisionTreeClassifier,
    "naive_bayes"        : GaussianNB,
    "logistic_regression": LogisticRegression,
    "svm"                : SVC,
    "random_forest"      : RandomForestClassifier,
}

# frozenset: immutable, O(1) membership lookup, communicates "this is a fixed set"
_RANDOM_STATE_MODELS: frozenset = frozenset({
    "decision_tree",
    "logistic_regression",
    "svm",
    "random_forest",
})


# =============================================================================
# PRIVATE FUNCTION 0: Validate Inputs
#
# PURPOSE  : Guard clause. Fail fast with clear, actionable error messages.
# PRINTS   : Nothing.
# RETURNS  : None (raises on failure).
#
# Design note: sklearn's own shape-mismatch errors are cryptic
# ("Found input variables with inconsistent numbers of samples").
# This function surfaces every invalid condition BEFORE sklearn runs,
# with a message that tells the student exactly what went wrong and why.
# =============================================================================

def _validate_inputs(X_train, y_train, X_test, model_name: str) -> None:
    """
    Validate all inputs to classification_model() before any computation.

    Purpose
    -------
    Fail fast. Provide clear, actionable messages instead of cryptic
    sklearn errors when shapes mismatch, data is invalid, or the
    model name is unsupported.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training label vector.
    X_test : array-like
        Test feature matrix.
    model_name : str
        Algorithm identifier. Must be a key in _MODEL_REGISTRY.

    Returns
    -------
    None
        Silent on success. Raises descriptive exceptions on any failure.

    Raises
    ------
    TypeError
        If X_train, y_train, or X_test cannot be converted to numpy arrays.
    ValueError
        If shapes are invalid, NaN values present, or model_name unsupported.
    """
    # -- Convert to numpy for shape inspection (non-destructive) --------------
    try:
        X_tr = np.asarray(X_train, dtype=float)
        y_tr = np.asarray(y_train)
        X_te = np.asarray(X_test, dtype=float)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"X_train, y_train, and X_test must be numeric array-like "
            f"(numpy array or pandas DataFrame/Series with numeric dtypes).\n"
            f"Ensure classification_preprocessing() and "
            f"classification_feature_selection() have been applied.\n"
            f"Conversion error: {exc}"
        ) from exc

    # -- Shape checks ---------------------------------------------------------
    if X_tr.ndim != 2:
        raise ValueError(
            f"X_train must be 2-dimensional (n_samples × n_features), "
            f"got shape {X_tr.shape}.\n"
            f"Build X with: X = selected_df[selected_features]"
        )

    if X_te.ndim != 2:
        raise ValueError(
            f"X_test must be 2-dimensional (n_samples × n_features), "
            f"got shape {X_te.shape}."
        )

    if X_tr.shape[0] == 0:
        raise ValueError(
            "X_train contains 0 samples. "
            "Verify that train_test_split() produced a non-empty training set."
        )

    if X_te.shape[0] == 0:
        raise ValueError(
            "X_test contains 0 samples. "
            "Verify that train_test_split() produced a non-empty test set."
        )

    if y_tr.shape[0] != X_tr.shape[0]:
        raise ValueError(
            f"X_train and y_train must have the same number of samples.\n"
            f"  X_train: {X_tr.shape[0]} samples\n"
            f"  y_train: {y_tr.shape[0]} samples\n"
            f"Verify that train_test_split() was called correctly."
        )

    if X_te.shape[1] != X_tr.shape[1]:
        raise ValueError(
            f"X_train and X_test must have the same number of features.\n"
            f"  X_train: {X_tr.shape[1]} features\n"
            f"  X_test : {X_te.shape[1]} features\n"
            f"Ensure both X_train and X_test were indexed with the same "
            f"selected_features list."
        )

    # -- NaN checks -----------------------------------------------------------
    if np.isnan(X_tr).any():
        raise ValueError(
            "X_train contains NaN values. "
            "Run classification_preprocessing() before model training."
        )

    if np.isnan(X_te).any():
        raise ValueError(
            "X_test contains NaN values. "
            "Run classification_preprocessing() before model training."
        )

    # -- Model name check -----------------------------------------------------
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_name: '{model_name}'.\n"
            f"Supported model names:\n"
            + "\n".join(f"  '{k}'" for k in _MODEL_REGISTRY)
        )


# =============================================================================
# PRIVATE FUNCTION 1: Build Model  (Factory Function)
#
# PURPOSE  : Instantiate the correct sklearn estimator from _MODEL_REGISTRY.
# PRINTS   : Nothing.
# RETURNS  : An unfitted sklearn estimator object.
#
# Design note: This is the Factory Pattern. The public function says
# model_name="knn" -- this function decides that means KNeighborsClassifier
# and instantiates it correctly. The caller never imports sklearn classes.
#
# Two special injections:
#   1. random_state: Only passed to models in _RANDOM_STATE_MODELS.
#      KNN and GaussianNB are deterministic -- passing random_state raises TypeError.
#   2. SVM probability=True: SVC does not expose predict_proba() by default.
#      We always inject this so evaluation.py can compute ROC-AUC.
# =============================================================================

def _build_model(model_name: str, random_state: int, **model_params):
    """
    Instantiate the sklearn estimator for the given model_name.

    Parameters
    ----------
    model_name : str
        Key from _MODEL_REGISTRY identifying the algorithm.
    random_state : int
        Reproducibility seed. Injected only for models in _RANDOM_STATE_MODELS.
    **model_params : dict
        Additional sklearn constructor parameters supplied by the caller.
        e.g., n_neighbors=5, max_depth=3, C=1.0

    Returns
    -------
    model : sklearn estimator
        An instantiated, unfitted estimator ready for .fit().

    Raises
    ------
    ValueError
        If model_params contains an invalid parameter for the chosen model.
        Re-raised from TypeError with a descriptive message and fix hint.

    Notes
    -----
    setdefault() is used for random_state: the caller's explicit value in
    model_params takes precedence over the random_state parameter.
    This allows overriding: classification_model(..., random_state=0,
    decision_tree random_state=99) is not valid, but this is fine for V1.
    """
    model_class = _MODEL_REGISTRY[model_name]

    # Start with caller-supplied params
    kwargs = dict(model_params)

    # Inject random_state only for models that accept it.
    # setdefault: does NOT overwrite if the caller explicitly set random_state.
    if model_name in _RANDOM_STATE_MODELS:
        kwargs.setdefault("random_state", random_state)

    # SVM: always enable probability=True for predict_proba() support.
    # This uses Platt scaling (internal cross-validation), adding slight
    # overhead. Documented and intentional -- not a silent override.
    if model_name == "svm":
        kwargs["probability"] = True  # Required for ROC-AUC in evaluation.py

    try:
        model = model_class(**kwargs)
    except TypeError as exc:
        raise ValueError(
            f"Invalid parameter for model_name='{model_name}' "
            f"({model_class.__name__}).\n"
            f"Supplied parameters: {kwargs}\n"
            f"Original error      : {exc}\n"
            f"Reference           : https://scikit-learn.org/stable/"
            f"supervised_learning.html"
        ) from exc

    return model


# =============================================================================
# PRIVATE FUNCTION 2: Fit Model
#
# PURPOSE  : Fit the model on training data and measure training time.
# PRINTS   : Nothing.
# RETURNS  : (fitted_model, training_time_seconds)
#
# Design note: Isolating the timing here means the timing method can be
# changed in one place (e.g., switching to GPU event timing for neural
# networks in a future version) without touching the public function.
# =============================================================================

def _fit_model(model, X_train, y_train) -> Tuple:
    """
    Fit the model on training data and measure wall-clock training time.

    Parameters
    ----------
    model : sklearn estimator
        An instantiated (unfitted) estimator from _build_model().
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training labels.

    Returns
    -------
    fitted_model : sklearn estimator
        The same estimator after .fit() has been called.
        (sklearn's .fit() modifies in-place and also returns self.)
    training_time : float
        Wall-clock seconds elapsed during model.fit().
        Measured with time.perf_counter() -- monotonic, high-resolution.

    Notes
    -----
    time.perf_counter():
        - Monotonic: never goes backward (unlike time.time() on some systems).
        - High resolution: sub-microsecond on most platforms.
        - Correct choice for benchmarking short operations.
        - Returns absolute seconds since an undefined reference point;
          only the DIFFERENCE between two calls is meaningful.
    """
    t_start       = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t_start

    return model, training_time


# =============================================================================
# PRIVATE FUNCTION 3: Get Predictions
#
# PURPOSE  : Generate class predictions and probability estimates. Time prediction.
# PRINTS   : Nothing.
# RETURNS  : (y_pred, y_prob, prediction_time_seconds)
#
# Design note: prediction_time measures model.predict() ONLY.
# predict_proba() is NOT timed or included because:
#   (a) it is optional -- not all workflows need probabilities.
#   (b) including it would make timing inconsistent across models that
#       may or may not support it.
#   (c) by convention, "prediction time" refers to class-label inference.
# =============================================================================

def _get_predictions(model, X_test) -> Tuple:
    """
    Generate class predictions and probability estimates on the test set.

    Parameters
    ----------
    model : sklearn estimator
        A fitted estimator (after _fit_model has been called).
    X_test : array-like
        Test feature matrix.

    Returns
    -------
    y_pred : np.ndarray of shape (n_test_samples,)
        Predicted class labels.
    y_prob : np.ndarray of shape (n_test_samples, n_classes) or None
        Predicted class probabilities for each sample.
        None if the model does not support predict_proba().
    prediction_time : float
        Wall-clock seconds for model.predict(X_test) only.
        Does not include predict_proba() time.

    Notes
    -----
    predict_proba() is called AFTER timing to keep prediction_time
    a clean, consistent benchmark across all models.
    If predict_proba() raises (e.g., an edge case), y_prob silently
    returns None so the pipeline does not break. evaluation.py handles
    None by skipping ROC-AUC computation.
    """
    # -- Timed prediction: class labels only ----------------------------------
    t_start         = time.perf_counter()
    y_pred          = model.predict(X_test)
    prediction_time = time.perf_counter() - t_start

    # -- Untimed: probability estimates (optional, for ROC-AUC) --------------
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            # predict_proba is declared but fails at runtime.
            # Return None; evaluation.py will skip ROC-AUC gracefully.
            y_prob = None

    return y_pred, y_prob, prediction_time


# =============================================================================
# PUBLIC FUNCTION -- The ONLY function the notebook should call.
#
# DESIGN PATTERN : Facade Pattern (consistent with all other modules).
# WHY            : The notebook orchestrates the ML workflow.
#                  Model training logic does not belong in the notebook.
#                  One clean function call per pipeline step.
# =============================================================================

def classification_model(
    X_train,
    y_train,
    X_test,
    model_name: str = "knn",
    random_state: int = 42,
    **model_params,
) -> Tuple:
    """
    Train a classification model and generate predictions on the test set.

    This is the ONLY function this module exposes to the notebook.
    It instantiates, trains, predicts, and times the model in one call,
    keeping the notebook at exactly one function call for model training.

    Pipeline
    --------
    Step 1 : Validate all inputs            (_validate_inputs)
    Step 2 : Instantiate model via registry  (_build_model)
    Step 3 : Fit model, measure train time   (_fit_model)
    Step 4 : Predict, measure predict time   (_get_predictions)
    Step 5 : Print structured summary
    Step 6 : Return 5-tuple

    Parameters
    ----------
    X_train : array-like of shape (n_train_samples, n_features)
        Training feature matrix. Must be numeric, no NaN values.
        Build with: X_train, X_test, y_train, y_test = train_test_split(...)
    y_train : array-like of shape (n_train_samples,)
        Training labels. Must be numeric (integer class codes).
    X_test : array-like of shape (n_test_samples, n_features)
        Test feature matrix.
        Must have the same number of features as X_train.
    model_name : str, optional
        The classification algorithm to use.
        Supported values:
            "knn"                 -- K-Nearest Neighbours
            "decision_tree"       -- Decision Tree
            "naive_bayes"         -- Gaussian Naive Bayes
            "logistic_regression" -- Logistic Regression
            "svm"                 -- Support Vector Machine
            "random_forest"       -- Random Forest
        Default: "knn"
    random_state : int, optional
        Reproducibility seed. Applied only to models that support it
        (decision_tree, logistic_regression, svm, random_forest).
        KNN and GaussianNB are deterministic and do not receive this.
        Default: 42
    **model_params : dict
        Additional keyword arguments forwarded to the sklearn constructor.
        Examples by model:
            "knn"                 : n_neighbors=5, metric="euclidean"
            "decision_tree"       : max_depth=5, min_samples_split=2
            "logistic_regression" : C=1.0, max_iter=1000
            "svm"                 : C=1.0, kernel="rbf"
            "random_forest"       : n_estimators=100, max_depth=5

    Returns
    -------
    trained_model : sklearn estimator
        The fitted model object. Retain for inspection, saving, or
        predicting on new data: trained_model.predict(X_new)
    y_pred : np.ndarray of shape (n_test_samples,)
        Predicted class labels for X_test.
        Pass as second argument to classification_metrics().
    y_prob : np.ndarray of shape (n_test_samples, n_classes) or None
        Predicted class probabilities for X_test.
        Pass as third argument to classification_metrics() for ROC-AUC.
        None if the model does not support predict_proba().
    training_time : float
        Wall-clock seconds to fit the model on X_train.
        Use for Experiment 2 model comparison.
    prediction_time : float
        Wall-clock seconds for model.predict(X_test) (class labels only).
        Does NOT include predict_proba() time.
        Use for Experiment 2 model comparison.

    Raises
    ------
    TypeError
        If X_train, y_train, or X_test cannot be converted to numeric arrays.
    ValueError
        If model_name is unsupported, shapes mismatch, NaN values exist,
        or invalid model_params are supplied.

    Example
    -------
    # K-Nearest Neighbours
    >>> trained_model, y_pred, y_prob, train_t, pred_t = classification_model(
    ...     X_train, y_train, X_test,
    ...     model_name="knn",
    ...     n_neighbors=5,
    ... )

    # Decision Tree with controlled depth
    >>> trained_model, y_pred, y_prob, train_t, pred_t = classification_model(
    ...     X_train, y_train, X_test,
    ...     model_name="decision_tree",
    ...     max_depth=5,
    ... )

    # Pass predictions to evaluation
    >>> metrics_df, cm_df, report = classification_metrics(y_test, y_pred, y_prob)

    Notes
    -----
    - SVM (model_name="svm") always receives probability=True internally.
      This enables predict_proba() for ROC-AUC, at the cost of slightly
      longer training (Platt scaling via internal 5-fold cross-validation).
    - training_time and prediction_time use time.perf_counter():
      monotonic, high-resolution, correct for benchmarking.
    - prediction_time measures model.predict() only (class-label inference).
      This is the conventional definition for Experiment 2 comparisons.
    """
    print()
    print("=" * 60)
    print("  CLASSIFICATION MODEL TRAINING")
    print("=" * 60)

    # -- Step 1: Validate all inputs ------------------------------------------
    _validate_inputs(X_train, y_train, X_test, model_name)

    # Work with numpy for shape reporting (original inputs passed to sklearn)
    X_tr_arr = np.asarray(X_train, dtype=float)
    X_te_arr = np.asarray(X_test,  dtype=float)

    # -- Step 2: Instantiate model via registry --------------------------------
    model      = _build_model(model_name, random_state, **model_params)
    class_name = type(model).__name__

    print(f"  Algorithm          : {class_name}")
    print(f"  model_name key     : '{model_name}'")

    if model_params:
        print(f"  User parameters    : {model_params}")
    else:
        print(f"  User parameters    : (defaults)")

    if model_name in _RANDOM_STATE_MODELS:
        effective_rs = model_params.get("random_state", random_state)
        print(f"  random_state       : {effective_rs}")

    if model_name == "svm":
        print(f"  probability        : True  (auto-injected for predict_proba support)")

    print(f"  Training samples   : {X_tr_arr.shape[0]}")
    print(f"  Test samples       : {X_te_arr.shape[0]}")
    print(f"  Features           : {X_tr_arr.shape[1]}")
    print()

    # -- Step 3: Fit the model ------------------------------------------------
    trained_model, training_time = _fit_model(model, X_train, y_train)
    print(f"  Training Time      : {training_time:.6f} seconds")

    # -- Step 4: Generate predictions -----------------------------------------
    y_pred, y_prob, prediction_time = _get_predictions(trained_model, X_test)
    print(f"  Prediction Time    : {prediction_time:.6f} seconds")
    print()

    # -- Step 5: Prediction summary -------------------------------------------
    unique_classes, class_counts = np.unique(y_pred, return_counts=True)
    n_test = len(y_pred)

    print("  Prediction Summary (test set):")
    for cls, cnt in zip(unique_classes, class_counts):
        pct = cnt / n_test * 100
        print(f"    Class {cls} : {cnt:>4} predictions  ({pct:5.1f}%)")

    if y_prob is None:
        print()
        print("  Note: predict_proba() not available for this configuration.")
        print("        ROC-AUC will be skipped in classification_metrics().")

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Training Time      : {training_time:.6f} s")
    print(f"  Prediction Time    : {prediction_time:.6f} s")
    print("=" * 60)
    print()
    print("  Next step:")
    print("    metrics_df, cm_df, report = classification_metrics(")
    print("        y_test, y_pred, y_prob")
    print("    )")
    print()

    return trained_model, y_pred, y_prob, training_time, prediction_time
