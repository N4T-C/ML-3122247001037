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

Also provides two KNN tuning functions:
    tune_knn_grid()   — GridSearchCV
    tune_knn_random() — RandomizedSearchCV

The notebook imports and calls exactly ONE of these functions per experiment.
"""

import time
import warnings

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# CLASSIFICATION MODEL
# =============================================================================

def classification_model(X_train, y_train, X_test, model_name="knn",
                          algorithm="auto"):
    """
    Train a classification model and return predictions.

    Supported model_name values
    ---------------------------
    "knn"                 → K-Nearest Neighbours
    "decision_tree"       → Decision Tree
    "naive_bayes"         → Gaussian Naive Bayes
    "multinomial_nb"      → Multinomial Naive Bayes
    "bernoulli_nb"        → Bernoulli Naive Bayes
    "logistic_regression" → Logistic Regression
    "svm"                 → Support Vector Machine
    "random_forest"       → Random Forest

    Parameters
    ----------
    X_train    : training features  (after preprocessing + feature selection)
    y_train    : training labels
    X_test     : test features
    model_name : algorithm to use (default: "knn")
    algorithm  : only used when model_name="knn"
                 "auto" | "kd_tree" | "ball_tree" | "brute"
                 (default: "auto")

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
        model = KNeighborsClassifier(algorithm=algorithm)

    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(random_state=42)

    elif model_name == "naive_bayes":
        model = GaussianNB()

    elif model_name == "multinomial_nb":
        model = MultinomialNB()

    elif model_name == "bernoulli_nb":
        model = BernoulliNB()

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
            "Choose from: knn, decision_tree, naive_bayes, multinomial_nb, "
            "bernoulli_nb, logistic_regression, svm, random_forest"
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
# KNN HYPERPARAMETER TUNING
# =============================================================================

def tune_knn_grid(X_train, y_train):
    """
    Tune KNN using GridSearchCV (exhaustive search).

    Searches over:
        algorithm   : ['kd_tree']
        metric      : ['euclidean']
        weights     : ['uniform', 'distance']
        n_neighbors : [1, 3, 5, 7, 9, 11]

    Parameters
    ----------
    X_train : training features
    y_train : training labels

    Returns
    -------
    best_model  : fitted KNeighborsClassifier with best parameters
    best_params : dict of best hyperparameters found
    cv_results  : pd.DataFrame of all parameter combinations and their scores
    """

    param_grid = {
        "algorithm"  : ["kd_tree"],
        "metric"     : ["euclidean"],
        "weights"    : ["uniform", "distance"],
        "n_neighbors": [1, 3, 5, 7, 9, 11],
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5,
                               scoring="accuracy", n_jobs=-1)

    t0 = time.perf_counter()
    grid_search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    import pandas as pd
    cv_results = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_neighbors", "param_weights", "param_algorithm", "param_metric",
         "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")

    print(f"\n{'='*50}")
    print("  GridSearchCV — KNN Tuning")
    print(f"{'='*50}")
    print(f"  Total combinations : {len(cv_results)}")
    print(f"  Search time        : {elapsed:.4f} s")
    print(f"  Best parameters    : {grid_search.best_params_}")
    print(f"  Best CV accuracy   : {grid_search.best_score_:.4f}")
    print(f"{'='*50}\n")

    return grid_search.best_estimator_, grid_search.best_params_, cv_results


def tune_knn_random(X_train, y_train, n_iter=20, random_state=42):
    """
    Tune KNN using RandomizedSearchCV (random sampling of parameter space).

    Samples from:
        algorithm   : ['kd_tree']
        metric      : ['euclidean']
        weights     : ['uniform', 'distance']
        n_neighbors : [1, 3, 5, 7, 9, 11]

    Parameters
    ----------
    X_train      : training features
    y_train      : training labels
    n_iter       : number of random parameter combinations to try (default: 20)
    random_state : random seed for reproducibility (default: 42)

    Returns
    -------
    best_model  : fitted KNeighborsClassifier with best parameters
    best_params : dict of best hyperparameters found
    cv_results  : pd.DataFrame of sampled combinations and their scores
    """

    param_dist = {
        "algorithm"  : ["kd_tree"],
        "metric"     : ["euclidean"],
        "weights"    : ["uniform", "distance"],
        "n_neighbors": [1, 3, 5, 7, 9, 11],
    }

    knn = KNeighborsClassifier()
    random_search = RandomizedSearchCV(knn, param_dist, n_iter=n_iter, cv=5,
                                       scoring="accuracy", n_jobs=-1,
                                       random_state=random_state)

    t0 = time.perf_counter()
    random_search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    import pandas as pd
    cv_results = pd.DataFrame(random_search.cv_results_)[
        ["param_n_neighbors", "param_weights", "param_algorithm", "param_metric",
         "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")

    print(f"\n{'='*50}")
    print("  RandomizedSearchCV — KNN Tuning")
    print(f"{'='*50}")
    print(f"  Combinations tried : {n_iter}")
    print(f"  Search time        : {elapsed:.4f} s")
    print(f"  Best parameters    : {random_search.best_params_}")
    print(f"  Best CV accuracy   : {random_search.best_score_:.4f}")
    print(f"{'='*50}\n")

    return random_search.best_estimator_, random_search.best_params_, cv_results


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
