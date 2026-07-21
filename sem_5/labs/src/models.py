"""
models.py
=========
Roll No : 3122247001061
Course  : ICS1512 — Machine Learning Laboratory, Semester 5

PURPOSE
-------
Provides two reusable functions:
    classification_model() — trains any classification algorithm
    regression_model()     — trains any regression algorithm

The notebook imports and calls exactly ONE of these functions per experiment.
"""

import time
import warnings

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# CLASSIFICATION MODEL
# =============================================================================

def classification_model(X_train, y_train, X_test, model_name="knn"):
    """
    Train a classification model and return predictions.

    Supported model_name values
    ---------------------------
    "knn"                 → K-Nearest Neighbours
    "decision_tree"       → Decision Tree
    "naive_bayes"         → Gaussian Naive Bayes
    "logistic_regression" → Logistic Regression
    "svm"                 → Support Vector Machine
    "random_forest"       → Random Forest

    Parameters
    ----------
    X_train    : training features  (after preprocessing + feature selection)
    y_train    : training labels
    X_test     : test features
    model_name : algorithm to use (default: "knn")

    Returns
    -------
    model           : the trained sklearn model object
    y_pred          : predicted class labels for X_test
    y_prob          : predicted probabilities (n_samples × n_classes), or None
    training_time   : seconds taken to train the model
    prediction_time : seconds taken to predict on X_test
    """

    # --- Pick the algorithm ---------------------------------------------------
    if model_name == "knn":
        model = KNeighborsClassifier()

    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(random_state=42)

    elif model_name == "naive_bayes":
        model = GaussianNB()

    elif model_name == "logistic_regression":
        model = LogisticRegression(random_state=42, max_iter=1000)

    elif model_name == "svm":
        # probability=True enables predict_proba() for ROC-AUC
        model = SVC(probability=True, random_state=42)

    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)

    else:
        raise ValueError(
            f"Unknown model_name: '{model_name}'. "
            "Choose from: knn, decision_tree, naive_bayes, "
            "logistic_regression, svm, random_forest"
        )

    # --- Train ---------------------------------------------------------------
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t0

    # --- Predict -------------------------------------------------------------
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - t0

    # --- Probabilities (needed for ROC-AUC) ----------------------------------
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

    # --- Print summary -------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  CLASSIFICATION MODEL: {type(model).__name__}")
    print(f"{'='*50}")
    print(f"  Training samples  : {len(y_train)}")
    print(f"  Test samples      : {len(y_pred)}")
    print(f"  Training time     : {training_time:.4f} s")
    print(f"  Prediction time   : {prediction_time:.4f} s")
    print(f"{'='*50}\n")

    return model, y_pred, y_prob, training_time, prediction_time


# =============================================================================
# REGRESSION MODEL
# =============================================================================

def regression_model(X_train, y_train, X_test, model_name="linear"):
    """
    Train a regression model and return predictions.

    Supported model_name values
    ---------------------------
    "linear"        → Linear Regression
    "decision_tree" → Decision Tree Regressor
    "knn"           → K-Nearest Neighbours Regressor
    "svm"           → Support Vector Regressor
    "random_forest" → Random Forest Regressor

    Parameters
    ----------
    X_train    : training features  (after preprocessing + feature selection)
    y_train    : continuous target values
    X_test     : test features
    model_name : algorithm to use (default: "linear")

    Returns
    -------
    model           : the trained sklearn model object
    y_pred          : predicted values for X_test
    training_time   : seconds taken to train the model
    prediction_time : seconds taken to predict on X_test
    """

    # --- Pick the algorithm ---------------------------------------------------
    if model_name == "linear":
        model = LinearRegression()

    elif model_name == "decision_tree":
        model = DecisionTreeRegressor(random_state=42)

    elif model_name == "knn":
        model = KNeighborsRegressor()

    elif model_name == "svm":
        model = SVR()

    elif model_name == "random_forest":
        model = RandomForestRegressor(random_state=42)

    else:
        raise ValueError(
            f"Unknown model_name: '{model_name}'. "
            "Choose from: linear, decision_tree, knn, svm, random_forest"
        )

    # --- Train ---------------------------------------------------------------
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t0

    # --- Predict -------------------------------------------------------------
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - t0

    # --- Print summary -------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  REGRESSION MODEL: {type(model).__name__}")
    print(f"{'='*50}")
    print(f"  Training samples  : {len(y_train)}")
    print(f"  Test samples      : {len(y_pred)}")
    print(f"  Training time     : {training_time:.4f} s")
    print(f"  Prediction time   : {prediction_time:.4f} s")
    print(f"{'='*50}\n")

    return model, y_pred, training_time, prediction_time
