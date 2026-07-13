"""
feature_selection.py
====================

Feature Selection module for tabular classification datasets.

RESPONSIBILITY
--------------
This module is responsible for ONE thing: selecting the most informative
feature columns from a preprocessed, clean DataFrame.

It does NOT:
    - Perform EDA or generate plots
    - Preprocess or transform data values
    - Split data into train/test sets
    - Train models or evaluate accuracy

PUBLIC INTERFACE
----------------
The notebook should call exactly ONE function from this module:

    >>> selected_df, selected_features, feature_ranking_df = (
    ...     classification_feature_selection(clean_df, target_column="Loan_Status")
    ... )

Everything else is a private internal implementation (prefixed with _).

PIPELINE POSITION
-----------------
    classification_eda()            -- Step 1: Observe raw data
    classification_preprocessing()  -- Step 2: Clean data
    classification_feature_selection() -- Step 3: THIS MODULE
    train_test_split()              -- Step 4: Explicit in notebook
    Model Training                  -- Step 5
    Evaluation                      -- Step 6

SUPPORTED SELECTION METHODS
----------------------------
    "chi2"        : Chi-Square test. Tests statistical independence between
                    a feature and the target class. Requires NON-NEGATIVE
                    feature values. Suitable for encoded categorical and
                    count-based features.

    "anova"       : ANOVA F-test (f_classif). Tests whether the mean of a
                    numerical feature differs significantly across target
                    classes. Works on any numerical feature. Most appropriate
                    for continuous features with a categorical target.

    "mutual_info" : Mutual Information. Measures statistical dependency
                    between a feature and the target. Detects non-linear
                    relationships that chi2 and ANOVA may miss. Works for
                    any feature type.

PARAMETER NAMING (Consistent across the full framework)
--------------------------------------------------------
    target_column   : The label column. Excluded from the feature matrix.
    exclude_columns : Identifier columns. Excluded from selection.

VERSION 1 KNOWN LIMITATION
----------------------------
    Scoring functions are computed on the FULL dataset (before train-test
    split). In a production pipeline, they should be fit on training data
    only to prevent information leakage from the test set. This is a
    standard, accepted simplification for introductory lab settings.

ASSUMPTIONS
-----------
    - Dataset is preprocessed and clean (no NaN values in feature columns).
    - All feature columns are numeric (preprocessing.py must run first).
    - Target column is numeric (label-encoded). String targets are not
      supported by the underlying sklearn scoring functions.
    - Problem type is classification (categorical target).

AUTHOR
------
    Roll No : 3122247001061
    Course  : Machine Learning Laboratory -- Semester 5
"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)

warnings.filterwarnings("ignore")

# =============================================================================
# MODULE CONSTANT: Supported methods
#
# Single source of truth. To add a new method in a future version:
#   1. Add the method name to this list.
#   2. Add a corresponding private helper function (_select_<method>).
#   3. Add an entry to the dispatch dictionary in classification_feature_selection.
# =============================================================================

_SUPPORTED_METHODS = ["chi2", "anova", "mutual_info"]


# =============================================================================
# PRIVATE FUNCTION 0: Validate Inputs
#
# PURPOSE  : Guard clause. Fail fast with descriptive errors before any
#            computation begins.
# PRINTS   : Nothing.
# RETURNS  : None (raises on failure).
#
# Design note: This function validates the API contract:
#   - df must be a non-empty DataFrame.
#   - target_column must exist in df.
#   - method must be one of the supported options.
#   - k must be a positive integer.
#   - All exclude_columns must exist in df.
# =============================================================================

def _validate_inputs(df, target_column, method, k, exclude_columns):
    """
    Validate all inputs to classification_feature_selection before any work.

    Purpose
    -------
    Fail fast. Raise clear, actionable errors before computation starts
    rather than allowing cryptic sklearn errors deep inside the pipeline.

    Parameters
    ----------
    df : object
        Must be a non-empty pd.DataFrame.
    target_column : str
        Must be a column name that exists in df.
    method : str
        Must be one of _SUPPORTED_METHODS.
    k : int
        Must be a positive integer (>= 1).
    exclude_columns : list of str or None
        If provided, every column name must exist in df.

    Returns
    -------
    None
        Returns silently on success. Raises on any failure.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If df is empty, target_column is missing, method is unsupported,
        k < 1, or any column in exclude_columns does not exist.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got '{type(df).__name__}'. "
            f"Run classification_preprocessing() before feature selection."
        )

    if df.empty:
        raise ValueError(
            "The provided DataFrame is empty. "
            "Verify that preprocessing completed correctly."
        )

    if target_column is None:
        raise ValueError(
            "target_column is required for classification_feature_selection(). "
            "Provide the name of the target (label) column."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' was not found in the DataFrame.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: '{method}'.\n"
            f"Supported methods: {_SUPPORTED_METHODS}\n"
            f"  'chi2'        -- Chi-Square test (non-negative features required)\n"
            f"  'anova'       -- ANOVA F-test (any numerical features)\n"
            f"  'mutual_info' -- Mutual Information (any feature type)"
        )

    if not isinstance(k, int) or k < 1:
        raise ValueError(
            f"k must be a positive integer (k >= 1), got k={k!r}."
        )

    if exclude_columns is not None:
        missing_cols = [col for col in exclude_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"exclude_columns {missing_cols} were not found in the DataFrame.\n"
                f"Available columns: {df.columns.tolist()}"
            )


# =============================================================================
# PRIVATE FUNCTION 1: Get Feature Matrix and Target Vector
#
# PURPOSE  : Separate the feature matrix (X) and target vector (y) from the
#            full DataFrame. Exclude target and identifier columns from X.
# PRINTS   : Nothing.
# RETURNS  : (X_df, y, feature_names)
#
# Design note: This is the standard supervised learning setup:
#   X = all columns EXCEPT target and identifiers.
#   y = target column only.
# The feature_names list preserves column-to-score mapping throughout the
# pipeline without depending on positional indices.
# =============================================================================

def _get_feature_target(df, target_column, exclude_columns):
    """
    Extract the feature matrix and target vector from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed DataFrame.
    target_column : str
        The target (label) column name.
    exclude_columns : list of str or None
        Identifier column names to exclude from the feature matrix.

    Returns
    -------
    X_df : pd.DataFrame
        DataFrame containing only feature columns (no target, no identifiers).
    y : pd.Series
        The target column as a Series.
    feature_names : list of str
        Column names of X_df. Used to map scores back to feature names.
    """
    protected_cols = [target_column]
    if exclude_columns:
        protected_cols.extend(exclude_columns)

    feature_names = [col for col in df.columns if col not in protected_cols]
    X_df          = df[feature_names]
    y             = df[target_column]

    return X_df, y, feature_names


# =============================================================================
# PRIVATE FUNCTION 2: Apply Variance Filter
#
# PURPOSE  : Remove features whose variance is at or below the threshold.
#            A constant feature (variance = 0) carries zero information
#            and will cause division-by-zero in some scoring functions.
# PRINTS   : Nothing.
# RETURNS  : (X_filtered_df, removed_features)
#
# Design note: The variance threshold is a pre-filter that runs before the
# main scoring method. With the default threshold=0.0, only perfectly
# constant features (all values identical) are removed. This is the
# safest setting and is always appropriate.
# =============================================================================

def _apply_variance_filter(X_df, threshold):
    """
    Remove features with variance at or below the specified threshold.

    Parameters
    ----------
    X_df : pd.DataFrame
        The feature matrix.
    threshold : float
        Minimum variance a feature must have to be kept.
        - 0.0 (default): removes only constant features (all values identical).
        - Higher values: removes near-constant features too.

    Returns
    -------
    X_filtered_df : pd.DataFrame
        Feature matrix with low-variance columns removed.
    removed_features : list of str
        Names of removed columns (empty list if none removed).
    """
    variances        = X_df.var()
    removed_features = variances[variances <= threshold].index.tolist()
    X_filtered_df    = X_df.drop(columns=removed_features)

    return X_filtered_df, removed_features


# =============================================================================
# PRIVATE FUNCTION 3: Chi-Square Selection
#
# PURPOSE  : Select the top-k features using the Chi-Square test.
# PRINTS   : Nothing.
# RETURNS  : (selected_feature_names, all_scores_dict)
#
# REQUIRES : All feature values must be NON-NEGATIVE.
#            Chi-Square is based on observed vs expected frequencies.
#            Negative values have no frequency interpretation.
#            If negative values are present, a clear ValueError is raised.
#            FIX: Apply MinMaxScaler in classification_preprocessing()
#            before calling this module (scaling="minmax").
#
# Design note: Returning scores for ALL features (not just selected ones)
# allows the feature_ranking_df to show WHY features were rejected --
# their scores were below the threshold. This is essential for observation
# note writing.
# =============================================================================

def _select_chi2(X_df, y, k):
    """
    Select top-k features using the Chi-Square statistical test.

    Chi-Square tests whether each feature is statistically independent
    of the target variable. A high chi-square score means the feature
    and target are strongly dependent -- the feature is informative.

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature matrix. ALL values must be non-negative.
    y : pd.Series
        Target vector (must be numeric / label-encoded).
    k : int
        Number of top features to select. Must be <= X_df.shape[1].

    Returns
    -------
    selected_features : list of str
        Names of the k selected features.
    all_scores : dict
        Mapping of {feature_name: chi2_score} for EVERY feature
        in X_df, not just the selected ones. NaN scores replaced with 0.

    Raises
    ------
    ValueError
        If any feature column contains negative values.

    Notes
    -----
    Chi-Square requires non-negative values because it models frequencies.
    If your data contains negative values (e.g., after StandardScaler),
    fix this at the preprocessing stage:
        classification_preprocessing(..., scaling="minmax")
    MinMaxScaler produces values in [0, 1], satisfying the constraint.
    """
    # Guard: Chi-Square requires non-negative values.
    # Do NOT silently fix this here -- fixing data is preprocessing's job.
    negative_cols = X_df.columns[(X_df < 0).any()].tolist()
    if negative_cols:
        raise ValueError(
            f"Chi-Square (method='chi2') requires all feature values to be "
            f"non-negative.\n"
            f"The following columns contain negative values: {negative_cols}\n\n"
            f"Fix this during preprocessing by applying MinMaxScaler:\n"
            f"    clean_df = classification_preprocessing(\n"
            f"        df, target_column=..., scaling='minmax'\n"
            f"    )\n"
            f"MinMaxScaler rescales all values to the range [0, 1], which "
            f"satisfies the Chi-Square requirement."
        )

    feature_names = X_df.columns.tolist()

    selector = SelectKBest(chi2, k=k)
    selector.fit(X_df.values, y.values)

    # Replace NaN scores (can occur for zero-variance features) with 0
    scores = np.nan_to_num(selector.scores_, nan=0.0)

    support_mask      = selector.get_support()
    selected_features = [feature_names[i] for i, s in enumerate(support_mask) if s]
    all_scores        = dict(zip(feature_names, scores.tolist()))

    return selected_features, all_scores


# =============================================================================
# PRIVATE FUNCTION 4: ANOVA F-Test Selection
#
# PURPOSE  : Select top-k features using the ANOVA F-test.
# PRINTS   : Nothing.
# RETURNS  : (selected_feature_names, all_scores_dict)
#
# Design note: ANOVA (Analysis of Variance) tests whether the class means
# of a numerical feature are significantly different across target classes.
# A high F-statistic means the feature strongly separates the classes.
# Unlike Chi-Square, ANOVA works on any numerical values (including negative).
# =============================================================================

def _select_anova(X_df, y, k):
    """
    Select top-k features using the ANOVA F-test (f_classif).

    ANOVA tests whether the mean value of each feature differs
    significantly across target classes. A high F-score indicates
    that the feature is a strong discriminator between classes.

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature matrix. Values may be any numeric type (including negative).
    y : pd.Series
        Target vector (must be numeric / label-encoded).
    k : int
        Number of top features to select. Must be <= X_df.shape[1].

    Returns
    -------
    selected_features : list of str
        Names of the k selected features.
    all_scores : dict
        Mapping of {feature_name: F_score} for EVERY feature in X_df.
        NaN scores (from zero-variance features) are replaced with 0.

    Notes
    -----
    ANOVA assumes:
        - Numerical features.
        - Approximately normal distribution within each class.
        - Approximately equal variance across classes (homoscedasticity).
    Violations of these assumptions reduce the reliability of F-scores,
    but ANOVA remains widely used as a practical heuristic.
    """
    feature_names = X_df.columns.tolist()

    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_df.values, y.values)

    scores = np.nan_to_num(selector.scores_, nan=0.0)

    support_mask      = selector.get_support()
    selected_features = [feature_names[i] for i, s in enumerate(support_mask) if s]
    all_scores        = dict(zip(feature_names, scores.tolist()))

    return selected_features, all_scores


# =============================================================================
# PRIVATE FUNCTION 5: Mutual Information Selection
#
# PURPOSE  : Select top-k features using Mutual Information.
# PRINTS   : Nothing.
# RETURNS  : (selected_feature_names, all_scores_dict)
#
# Design note: Mutual Information quantifies how much information a feature
# shares with the target. Unlike Chi-Square and ANOVA, it detects non-linear
# dependencies. It is the most general of the three methods.
#
# Reproducibility note: mutual_info_classif uses k-NN estimation internally,
# which involves randomness. We fix random_state=42 to ensure consistent
# results across notebook runs.
# =============================================================================

def _select_mutual_info(X_df, y, k):
    """
    Select top-k features using Mutual Information (mutual_info_classif).

    Mutual Information measures the reduction in uncertainty about the
    target variable given the value of a feature. A score of 0 means the
    feature is completely independent of the target. Higher scores indicate
    stronger dependency (linear or non-linear).

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature matrix. Values may be any numeric type.
    y : pd.Series
        Target vector (must be numeric / label-encoded).
    k : int
        Number of top features to select. Must be <= X_df.shape[1].

    Returns
    -------
    selected_features : list of str
        Names of the k selected features.
    all_scores : dict
        Mapping of {feature_name: MI_score} for EVERY feature in X_df.
        Scores are always non-negative.

    Notes
    -----
    - Mutual Information scores are always >= 0.
    - random_state=42 is fixed for reproducibility across runs.
    - MI estimation uses k-nearest neighbours, making it slower than
      chi2 or ANOVA for large datasets.
    - Unlike chi2 and ANOVA, MI makes no distributional assumptions.
    """
    feature_names = X_df.columns.tolist()

    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X_df.values, y.values)

    scores = selector.scores_

    support_mask      = selector.get_support()
    selected_features = [feature_names[i] for i, s in enumerate(support_mask) if s]
    all_scores        = dict(zip(feature_names, scores.tolist()))

    return selected_features, all_scores


# =============================================================================
# PRIVATE FUNCTION 6: Build Feature Ranking DataFrame
#
# PURPOSE  : Convert the raw scores dictionary into a structured, sorted
#            DataFrame suitable for reporting and observation writing.
# PRINTS   : Nothing.
# RETURNS  : pd.DataFrame with columns [Rank, Feature, Score, Selected].
#
# Design note: Returning a DataFrame (not a dict) makes it trivial for
# the student to:
#   - Print the full ranking in the notebook.
#   - Save it to CSV for the lab report.
#   - Filter to show only selected features.
#   - Reference specific scores in observation markdown cells.
# =============================================================================

def _build_feature_ranking(all_scores, selected_features):
    """
    Build a structured feature ranking DataFrame from raw scores.

    Parameters
    ----------
    all_scores : dict
        Mapping of {feature_name: score} for EVERY feature (selected or not).
    selected_features : list of str
        Names of the features that were selected.

    Returns
    -------
    pd.DataFrame
        Columns: Rank, Feature, Score, Selected
        Sorted by Score descending (Rank 1 = highest score = most informative).
        Score is rounded to 4 decimal places for readability.

    Example output
    --------------
        Rank  Feature          Score  Selected
           1  Glucose         35.234      Yes
           2  BMI             28.441      Yes
           3  Age             12.007      Yes
           4  BloodPressure    3.112       No
           5  SkinThickness    1.005       No
    """
    selected_set = set(selected_features)  # O(1) membership lookup

    ranking_df = pd.DataFrame({
        "Feature" : list(all_scores.keys()),
        "Score"   : list(all_scores.values()),
    })

    ranking_df = ranking_df.sort_values("Score", ascending=False).reset_index(drop=True)
    ranking_df["Rank"]     = ranking_df.index + 1
    ranking_df["Score"]    = ranking_df["Score"].round(4)
    ranking_df["Selected"] = ranking_df["Feature"].apply(
        lambda f: "Yes" if f in selected_set else "No"
    )

    # Reorder columns to match the documented schema
    ranking_df = ranking_df[["Rank", "Feature", "Score", "Selected"]]

    return ranking_df


# =============================================================================
# PUBLIC FUNCTION -- The ONLY function the notebook should call.
#
# DESIGN PATTERN : Facade Pattern (consistent with eda.py and preprocessing.py)
# WHY            : The notebook orchestrates the ML workflow. Feature selection
#                  logic does not belong in the notebook.
#                  One clean function call per pipeline step.
# =============================================================================

def classification_feature_selection(
    df,
    target_column,
    method="chi2",
    k=10,
    exclude_columns=None,
    variance_threshold=0.0,
):
    """
    Select the most informative features for a tabular classification dataset.

    This is the ONLY function this module exposes to the notebook.
    It orchestrates all feature selection steps and returns a structured result.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed DataFrame from classification_preprocessing().
        Must have no NaN values in feature columns.
        All feature columns must be numeric.
        Target column must be numeric (label-encoded).
    target_column : str
        The name of the target (label) column.
        Excluded from the feature matrix. Included in the returned selected_df.
    method : str, optional
        The feature scoring method.
        - "chi2"        : Chi-Square test. Non-negative feature values required.
        - "anova"       : ANOVA F-test. Works on any numeric features.
        - "mutual_info" : Mutual Information. Detects non-linear relationships.
        Default: "chi2"
    k : int, optional
        Number of top features to select.
        Automatically capped at the number of available features if k exceeds it.
        Default: 10
    exclude_columns : list of str, optional
        Identifier columns (e.g., ["Loan_ID", "PassengerId"]).
        Excluded from the feature matrix and from selection scoring.
        Kept in the returned selected_df for reference.
        Drop these in the notebook when building X for model training.
        Default: None
    variance_threshold : float, optional
        Pre-filter: remove features with variance at or below this value.
        Default 0.0 removes only constant features (all values identical).
        Increase to remove near-constant features.
        Default: 0.0

    Returns
    -------
    selected_df : pd.DataFrame
        DataFrame containing ONLY selected feature columns + target_column
        + exclude_columns. Ready for train_test_split() after extracting X and y.
    selected_features : list of str
        Names of the k selected feature columns.
        Use this to build X: X = selected_df[selected_features]
    feature_ranking_df : pd.DataFrame
        Full ranking of ALL features with columns:
            Rank     : Integer rank (1 = most informative).
            Feature  : Column name.
            Score    : Score from the chosen method (rounded to 4 dp).
            Selected : "Yes" if selected, "No" if rejected.
        Sorted by Score descending.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If target_column is missing, method is unsupported, k < 1,
        exclude_columns do not exist, target is non-numeric,
        method="chi2" and negative feature values exist,
        or variance filter removes all available features.

    Example
    -------
    # Loan Prediction dataset
    >>> selected_df, selected_features, feature_ranking_df = (
    ...     classification_feature_selection(
    ...         clean_df,
    ...         target_column="Loan_Status",
    ...         method="chi2",
    ...         k=5,
    ...         exclude_columns=["Loan_ID"],
    ...     )
    ... )
    >>> print(feature_ranking_df)
    >>> X = selected_df[selected_features]
    >>> y = selected_df["Loan_Status"]
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

    Notes
    -----
    - train_test_split() must be called AFTER this function, in the notebook.
    - The target column must be label-encoded (numeric) before calling this.
      Preprocessing excludes the target from encoding by design. If your target
      is still a string, encode it manually before feature selection.
    - Version 1 Limitation: Scoring is fit on the full dataset (minor leakage).
    """
    print()
    print("=" * 60)
    print("  CLASSIFICATION FEATURE SELECTION")
    print(f"  Method              : {method}")
    print(f"  k (features to keep): {k}")
    print(f"  Target Column       : '{target_column}'")
    print(f"  Variance Threshold  : {variance_threshold}")
    if exclude_columns:
        print(f"  Excluded Columns    : {exclude_columns}")
    print("=" * 60)
    print()

    # -- Step 0: Validate all inputs ------------------------------------------
    _validate_inputs(df, target_column, method, k, exclude_columns)

    # -- Step 1: Extract feature matrix (X) and target vector (y) -------------
    X_df, y, feature_names = _get_feature_target(df, target_column, exclude_columns)

    if X_df.empty or X_df.shape[1] == 0:
        raise ValueError(
            "No feature columns found after excluding target_column and "
            "exclude_columns. The DataFrame must contain at least one feature."
        )

    # Guard: target must be numeric for sklearn scoring functions.
    if y.dtype == object or str(y.dtype) == "category":
        raise ValueError(
            f"target_column '{target_column}' contains non-numeric values "
            f"(dtype: {y.dtype}).\n"
            f"Sklearn feature scoring functions require a numeric target.\n"
            f"Fix: Label-encode the target before feature selection.\n"
            f"Example:\n"
            f"    from sklearn.preprocessing import LabelEncoder\n"
            f"    le = LabelEncoder()\n"
            f"    clean_df['{target_column}'] = le.fit_transform("
            f"clean_df['{target_column}'])"
        )

    # Guard: check for NaN values in feature columns.
    nan_cols = X_df.columns[X_df.isnull().any()].tolist()
    if nan_cols:
        raise ValueError(
            f"Feature columns contain NaN values: {nan_cols}\n"
            f"Run classification_preprocessing() before feature selection."
        )

    print(f"  Feature columns detected ({len(feature_names)}): {feature_names}")
    print()

    # -- Step 2: Apply variance pre-filter ------------------------------------
    X_filtered, removed_by_variance = _apply_variance_filter(X_df, variance_threshold)

    if removed_by_variance:
        print("=" * 60)
        print("  VARIANCE PRE-FILTER")
        print("=" * 60)
        print(f"  Threshold         : {variance_threshold}")
        print(f"  Removed features  : {removed_by_variance}")
        print(f"  Reason            : Variance <= {variance_threshold} "
              f"(near-constant, carry no information)")
        print()

    if X_filtered.empty or X_filtered.shape[1] == 0:
        raise ValueError(
            "All feature columns were removed by the variance filter "
            f"(threshold={variance_threshold}). "
            "Lower the variance_threshold or check that preprocessing "
            "produced valid numerical features."
        )

    # -- Step 3: Auto-cap k at number of available features -------------------
    n_features  = X_filtered.shape[1]
    k_safe      = min(k, n_features)

    if k_safe < k:
        print(
            f"  Note: k={k} requested but only {n_features} features available "
            f"after variance filtering. Using k={k_safe}."
        )
        print()

    # -- Step 4: Dispatch to the selected scoring method ----------------------
    _selector_dispatch = {
        "chi2"        : _select_chi2,
        "anova"       : _select_anova,
        "mutual_info" : _select_mutual_info,
    }

    selector_fn                  = _selector_dispatch[method]
    selected_features, all_scores = selector_fn(X_filtered, y, k_safe)

    # -- Step 5: Build the feature ranking DataFrame --------------------------
    feature_ranking_df = _build_feature_ranking(all_scores, selected_features)

    # -- Step 6: Print the ranking table --------------------------------------
    print("=" * 60)
    print(f"  FEATURE RANKING  (method: {method} | k={k_safe})")
    print("=" * 60)
    print()
    print(feature_ranking_df.to_string(index=False))
    print()

    # -- Step 7: Build the output DataFrame -----------------------------------
    # selected_df contains: selected feature columns + target + identifiers.
    output_cols   = selected_features.copy()
    if target_column not in output_cols:
        output_cols.append(target_column)
    if exclude_columns:
        for col in exclude_columns:
            if col not in output_cols:
                output_cols.append(col)

    selected_df = df[output_cols].reset_index(drop=True)

    # -- Step 8: Summary ------------------------------------------------------
    print("=" * 60)
    print("  FEATURE SELECTION COMPLETE")
    print(f"  Method              : {method}")
    print(f"  Features before     : {len(feature_names)}")
    print(f"  Features removed    : {len(removed_by_variance)} (variance filter)")
    print(f"  Features selected   : {len(selected_features)}")
    print(f"  Selected features   : {selected_features}")
    print(f"  Output shape        : {selected_df.shape}")
    print("=" * 60)
    print()
    print("  Next step in notebook:")
    print(f"    X = selected_df[selected_features]")
    print(f"    y = selected_df['{target_column}']")
    print(f"    X_train, X_test, y_train, y_test = train_test_split(X, y, ...)")
    print()

    return selected_df, selected_features, feature_ranking_df
