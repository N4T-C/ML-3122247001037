"""
preprocessing.py
================

Data Preprocessing module for tabular classification datasets.

RESPONSIBILITY
--------------
This module is responsible for ONE thing: transforming raw tabular data
into a clean, model-ready DataFrame.

It does NOT:
    - Perform EDA or generate plots
    - Split data into train/test sets  (done explicitly in the notebook)
    - Select or drop features          (done in feature_selection.py)
    - Train or evaluate models

PUBLIC INTERFACE
----------------
The notebook should call exactly ONE function from this module:

    >>> clean_df = classification_preprocessing(df, target_column="Loan_Status")

All other functions are private implementation details (prefixed with _).
Do not call them directly from the notebook.

PIPELINE ORDER (and why this order matters)
-------------------------------------------
    Step 0 -- Validate Input
        Fail fast: catch bad input before any transformation starts.

    Step 1 -- Handle Duplicates
        Remove before imputation -- no point computing statistics
        on rows that will be deleted.

    Step 2 -- Handle Missing Values
        Impute before encoding -- sklearn encoders fail on NaN values.

    Step 3 -- Encode Categorical Features
        Encode before scaling -- encoded integers may also need scaling.

    Step 4 -- Scale Numerical Features
        Scale last -- operates on the final, fully numeric feature set.

PARAMETER NAMING (Consistent across the full framework)
--------------------------------------------------------
    target_column   : The label column. Excluded from all transformations.
    exclude_columns : Identifier columns (Loan_ID, PassengerId, etc.).
                      Kept in the DataFrame but excluded from all transformations.
                      Drop these in feature_selection.py before model training.

VERSION 1 KNOWN LIMITATION
----------------------------
    Scalers are currently fit on the FULL dataset (before train-test split).
    In a production pipeline, scalers should be fit on training data only
    and then applied (transform, no fit) to the test data.
    Fitting on all data causes minor data leakage from the test set.
    This will be resolved in a future version using sklearn Pipeline objects.
    It is acceptable and common practice in introductory lab settings.

ASSUMPTIONS
-----------
    - Dataset is tabular (CSV-based).
    - Problem type is classification (categorical target).
    - train_test_split() is called AFTER this function, in the notebook.

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
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PRIVATE FUNCTION 0: Validate Input
#
# PURPOSE  : Guard clause. Verify the input is correct before any work begins.
# PRINTS   : Nothing on success. Raises descriptive exceptions on failure.
# RETURNS  : None
#
# Design note: This is called FIRST in the pipeline (Step 0).
# "Fail fast" is a fundamental Software Engineering principle:
# the earlier you catch a bug, the cheaper it is to fix.
# A descriptive error here is far more helpful than a cryptic
# pandas AttributeError raised five steps later.
# =============================================================================

def _validate_dataframe(df, target_column=None, exclude_columns=None):
    """
    Validate the input DataFrame before any transformation begins.

    Purpose
    -------
    Raises descriptive errors immediately if the input is invalid.
    Prevents cryptic pandas errors from surfacing deep inside the pipeline.

    Parameters
    ----------
    df : object
        The input to validate. Must be a non-empty pd.DataFrame.
    target_column : str, optional
        If provided, validated to exist in df.columns.
    exclude_columns : list of str, optional
        If provided, each column is validated to exist in df.columns.

    Returns
    -------
    None
        Returns silently on success. Raises on any failure.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If df is empty, or if specified columns do not exist in df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got '{type(df).__name__}'. "
            f"Make sure you loaded your dataset with pd.read_csv() "
            f"before calling classification_preprocessing()."
        )

    if df.empty:
        raise ValueError(
            "The provided DataFrame is empty. "
            "Verify that your dataset file loaded correctly."
        )

    if target_column is not None and target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' was not found in the DataFrame.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if exclude_columns:
        missing_cols = [col for col in exclude_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"exclude_columns {missing_cols} were not found in the DataFrame.\n"
                f"Available columns: {df.columns.tolist()}"
            )


# =============================================================================
# PRIVATE FUNCTION 1: Handle Duplicates
#
# PURPOSE  : Detect and remove exact duplicate rows.
# PRINTS   : Count before and after, strategy applied.
# RETURNS  : df without duplicate rows.
#
# Design note: Duplicates are a separate concern from missing values.
# A row can be a perfect duplicate with zero missing values. Mixing
# duplicate handling with missing value imputation would violate
# the Single Responsibility Principle.
#
# Why remove duplicates before imputation?
# If a row is a duplicate, imputing it wastes computation -- the row
# will be dropped anyway. Always delete before computing.
# =============================================================================

def _handle_duplicates(df, strategy="drop"):
    """
    Detect and remove exact duplicate rows from the DataFrame.

    A duplicate row is one that is an exact copy of another row across
    every column. Duplicates can silently inflate model accuracy if the
    same row appears in both training and test sets after the split.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    strategy : str, optional
        How to handle duplicate rows.
        - "drop" : Remove all duplicates, keeping the first occurrence.
        Default: "drop"

    Returns
    -------
    pd.DataFrame
        A new DataFrame with duplicate rows removed.
        The index is reset to maintain continuous integer indexing.

    Notes
    -----
    Version 1 supports only "drop". Future versions may support:
        "keep_last" : Keep the last occurrence of each duplicate.
        "flag"      : Add a boolean column marking duplicates instead
                      of removing them.
    """
    print("=" * 60)
    print("1. DUPLICATE HANDLING")
    print("=" * 60)

    dup_count = int(df.duplicated().sum())

    if dup_count == 0:
        print("  No duplicate rows found.")
        print()
        return df

    if strategy == "drop":
        df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)
        print(f"  Strategy          : drop (keep first occurrence)")
        print(f"  Duplicate rows    : {dup_count}")
        print(f"  Rows before       : {len(df)}")
        print(f"  Rows after        : {len(df_clean)}")
        print()
        return df_clean

    raise ValueError(
        f"Unknown duplicate_strategy: '{strategy}'. "
        f"Valid options: 'drop'."
    )


# =============================================================================
# PRIVATE FUNCTION 2: Handle Missing Values
#
# PURPOSE  : Impute or remove missing values from feature columns.
# PRINTS   : Strategy, affected columns, fill values.
# RETURNS  : df with no NaN values in feature columns.
#
# Design note: Only FEATURE columns are imputed. Target and identifier
# columns are excluded (passed via exclude_cols). The target column
# must not be imputed -- a row with a missing label has no ground truth
# and cannot be used for supervised learning. Identifier columns have
# no statistical meaning; imputing an ID column is nonsensical.
#
# Why the specific order for "mean"/"median" strategy?
# Numerical columns: filled with mean or median (statistically appropriate).
# Categorical columns: ALWAYS filled with mode, regardless of whether
# strategy is "mean" or "median" -- mean and median do not apply to
# categorical data (what is the mean of ["cat", "dog", "bird"]?).
# =============================================================================

def _handle_missing_values(df, strategy="mean", exclude_cols=None):
    """
    Impute or remove missing values from feature columns.

    Only feature columns participate in imputation. The target column
    and identifier columns (passed via exclude_cols) are untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Duplicates should already be handled.
    strategy : str, optional
        The imputation method.
        - "mean"   : Numerical NaN --> column mean.
                     Categorical NaN --> column mode.
                     Use when the distribution is approximately symmetric.
        - "median" : Numerical NaN --> column median.
                     Categorical NaN --> column mode.
                     Use when the distribution is skewed or has outliers.
                     Median is more robust to extreme values than mean.
        - "mode"   : All columns (numerical + categorical) --> column mode.
                     Use when the dataset is mostly categorical.
        - "drop"   : Drop any row with at least one missing feature value.
                     Use when missing data is minimal (<5%) and likely random.
        Default: "mean"
    exclude_cols : list of str, optional
        Columns to skip during imputation (target + identifier columns).
        Default: None

    Returns
    -------
    pd.DataFrame
        A new DataFrame with no missing values in feature columns.
        The index is reset if rows were dropped.

    Notes
    -----
    Version 1 Limitation:
        Fill values (mean/median/mode) are computed from the FULL dataset,
        before train-test split. Strictly, they should be computed from the
        training set only to prevent data leakage. This is an acceptable
        simplification for learning purposes.
    """
    if exclude_cols is None:
        exclude_cols = []

    feature_cols  = [col for col in df.columns if col not in exclude_cols]
    total_missing = int(df[feature_cols].isnull().sum().sum())

    print("=" * 60)
    print("2. MISSING VALUE HANDLING")
    print("=" * 60)

    if total_missing == 0:
        print("  No missing values found in feature columns.")
        print()
        return df

    df_clean         = df.copy()
    numerical_cols   = (
        df_clean[feature_cols]
        .select_dtypes(include=["number"])
        .columns.tolist()
    )
    categorical_cols = (
        df_clean[feature_cols]
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )

    print(f"  Strategy          : {strategy}")
    print(f"  Total NaN values  : {total_missing}")
    print()

    if strategy in ("mean", "median"):
        # -- Numerical columns: fill with mean or median ----------------------
        for col in numerical_cols:
            n_missing = int(df_clean[col].isnull().sum())
            if n_missing > 0:
                fill_val = (
                    df_clean[col].mean()
                    if strategy == "mean"
                    else df_clean[col].median()
                )
                df_clean[col] = df_clean[col].fillna(fill_val)
                print(
                    f"  '{col}' ({n_missing} NaN)"
                    f" --> filled with {strategy} = {fill_val:.4f}"
                )

        # -- Categorical columns: mode is always used -------------------------
        # (Mean and median have no meaning for categorical data.)
        for col in categorical_cols:
            n_missing = int(df_clean[col].isnull().sum())
            if n_missing > 0:
                fill_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(fill_val)
                print(
                    f"  '{col}' ({n_missing} NaN)"
                    f" --> filled with mode = '{fill_val}'"
                    f"  (mode used for categorical regardless of strategy)"
                )

    elif strategy == "mode":
        for col in numerical_cols + categorical_cols:
            n_missing = int(df_clean[col].isnull().sum())
            if n_missing > 0:
                fill_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(fill_val)
                print(f"  '{col}' ({n_missing} NaN) --> filled with mode = '{fill_val}'")

    elif strategy == "drop":
        before   = len(df_clean)
        df_clean = df_clean.dropna(subset=feature_cols).reset_index(drop=True)
        after    = len(df_clean)
        print(f"  Rows dropped      : {before - after}")
        print(f"  Rows remaining    : {after}")

    else:
        raise ValueError(
            f"Unknown missing_strategy: '{strategy}'. "
            f"Valid options: 'mean', 'median', 'mode', 'drop'."
        )

    print()
    return df_clean


# =============================================================================
# PRIVATE FUNCTION 3: Encode Categorical Features
#
# PURPOSE  : Convert string/object columns to integers for model compatibility.
# PRINTS   : Method, columns encoded, label-to-integer mappings.
# RETURNS  : df with all object/category columns replaced by numbers.
#
# Design note: ML algorithms work with numbers, not strings. This function
# bridges that gap. It is deliberately separate from scaling because:
#   - Encoding changes the DATA TYPE (object --> int64)
#   - Scaling changes the DATA MAGNITUDE (0-100k --> 0-1)
# These are different transformations with different purposes.
#
# Target and identifier columns are excluded -- they are either already
# numeric or should not be encoded as features.
# =============================================================================

def _encode_categorical_features(df, method="label", exclude_cols=None):
    """
    Convert categorical (string/object) columns to numerical format.

    ML models require numerical input. This function converts string-valued
    columns to integers using the specified encoding method.

    Target and identifier columns are excluded from encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Missing values should already be handled.
    method : str or None, optional
        The encoding method.
        - "label"  : Assign an integer to each unique category value.
                     Example: ["Male", "Female"] --> [0, 1]  (alphabetical)
                     Note: Integers here are arbitrary labels -- they do NOT
                     imply that one category is "greater than" another.
                     Appropriate for: binary categories, or when the model
                     internally handles nominal vs ordinal features.
        - "onehot" : Create one new binary column per unique category value.
                     Example: ["Male", "Female"] --> [Gender_Female, Gender_Male]
                     Each row has exactly one 1 and the rest 0.
                     Appropriate for: nominal (unordered) categories.
                     Warning: significantly increases the number of columns.
        - None     : No encoding applied. Use when all features are already
                     numerical.
        Default: "label"
    exclude_cols : list of str, optional
        Columns to skip (target + identifier columns).
        Default: None

    Returns
    -------
    pd.DataFrame
        A new DataFrame where object/category columns are replaced by integers
        (label) or new binary columns (onehot).

    Notes
    -----
    Label Encoding: Categories are sorted alphabetically before assigning
    integers. The resulting integers carry no ordinal meaning unless the
    original categories are naturally ordered (e.g., Low=0, Medium=1, High=2).

    One-Hot Encoding: drop_first=False is used to retain all categories.
    Dropping the first category (dummy variable trap avoidance) is left
    as a deliberate choice for the notebook/student, not automated here.
    """
    if exclude_cols is None:
        exclude_cols = []

    feature_cols     = [col for col in df.columns if col not in exclude_cols]
    categorical_cols = (
        df[feature_cols]
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )

    print("=" * 60)
    print("3. CATEGORICAL ENCODING")
    print("=" * 60)

    if not categorical_cols:
        print("  No categorical feature columns found. Encoding skipped.")
        print()
        return df

    if method is None:
        print("  Encoding set to None. No encoding applied.")
        print()
        return df

    df_encoded = df.copy()

    if method == "label":
        print(f"  Method            : Label Encoding")
        print(f"  Columns           : {categorical_cols}")
        print()
        for col in categorical_cols:
            le              = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            # Print the mapping so the student can record it in observations
            mapping         = dict(
                zip(le.classes_, le.transform(le.classes_).tolist())
            )
            print(f"  '{col}' --> {mapping}")

    elif method == "onehot":
        before_shape = df_encoded.shape
        df_encoded   = pd.get_dummies(
            df_encoded,
            columns=categorical_cols,
            drop_first=False,   # Keep all categories; student decides later
        )
        after_shape  = df_encoded.shape
        print(f"  Method            : One-Hot Encoding")
        print(f"  Columns encoded   : {categorical_cols}")
        print(f"  Shape before      : {before_shape}")
        print(
            f"  Shape after       : {after_shape}"
            f"  ({after_shape[1] - before_shape[1]} new columns added)"
        )

    else:
        raise ValueError(
            f"Unknown encoding: '{method}'. "
            f"Valid options: 'label', 'onehot', None."
        )

    print()
    return df_encoded


# =============================================================================
# PRIVATE FUNCTION 4: Scale Features
#
# PURPOSE  : Normalize or standardize numerical feature columns.
# PRINTS   : Method used, columns scaled.
# RETURNS  : df with numerical feature columns scaled.
#
# Design note: Why is a single _scale_features() better than separate
# _standardize_features() and _normalize_features() functions?
# -> All three scalers (standard, minmax, robust) share identical logic:
#    (1) choose a scaler object, (2) fit, (3) transform.
# -> Separate functions would repeat this logic and violate DRY.
# -> Adding a new scaler (e.g., MaxAbsScaler) requires zero structural change.
# -> method=None provides a clean "opt-out" without an if-block in the caller.
#
# Why do tree-based models NOT need scaling?
# Decision trees split on threshold values (e.g., "is Age > 35?").
# The scale of features does not affect where the threshold is, only the
# split VALUE changes -- and the model relearns this regardless of scale.
# For distance-based models (KNN, SVM), scale matters enormously.
# =============================================================================

def _scale_features(df, method, exclude_cols=None):
    """
    Scale numerical feature columns to a common range or distribution.

    Distance-based and gradient-based ML algorithms (KNN, SVM, Logistic
    Regression) are sensitive to feature scale. A column with values 0-100,000
    will dominate a column with values 0-1, even if both carry equal information.
    Scaling eliminates this artificial dominance.

    Tree-based models (Decision Tree, Random Forest) are scale-invariant.
    Use scaling=None for those.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Categorical encoding should already be applied.
    method : str or None
        The scaling method.
        - "standard" : Standardization. Transforms to mean=0, std=1.
                       Formula: z = (x - mean) / std
                       Appropriate when data is approximately normally distributed.
        - "minmax"   : Min-Max Normalization. Transforms to range [0, 1].
                       Formula: x_norm = (x - min) / (max - min)
                       Appropriate when bounded values are needed.
                       Sensitive to outliers (outliers compress the rest).
        - "robust"   : Robust Scaling. Uses the IQR (interquartile range).
                       Formula: x_robust = (x - median) / IQR
                       Appropriate when the dataset contains significant outliers.
                       Less sensitive to outliers than standard or minmax.
        - None       : No scaling applied.
                       Use for tree-based models (Decision Tree, Random Forest).
    exclude_cols : list of str, optional
        Columns to skip (target + identifier columns).
        Default: None

    Returns
    -------
    pd.DataFrame
        A new DataFrame with numerical feature columns scaled in-place.
        Target and identifier columns are unchanged.

    Notes
    -----
    Version 1 Limitation -- Data Leakage:
        The scaler is currently fit on the FULL dataset (before train-test split).
        In a production ML pipeline, the correct approach is:
            1. Fit the scaler on X_train only.
            2. Use scaler.transform() (NOT fit_transform()) on X_test.
        Fitting on all data allows test-set statistics to influence the scaler,
        which is a form of data leakage. This is a known and accepted
        simplification for introductory lab settings. It will be addressed
        in a future version using sklearn Pipeline objects.
        For small, clean datasets, the practical impact is minimal.
    """
    print("=" * 60)
    print("4. FEATURE SCALING")
    print("=" * 60)

    if method is None:
        print("  Scaling set to None. No scaling applied.")
        print("  (Appropriate for tree-based models: Decision Tree, Random Forest)")
        print()
        return df

    if exclude_cols is None:
        exclude_cols = []

    feature_cols   = [col for col in df.columns if col not in exclude_cols]
    numerical_cols = (
        df[feature_cols]
        .select_dtypes(include=["number"])
        .columns.tolist()
    )

    if not numerical_cols:
        print("  No numerical feature columns found. Scaling skipped.")
        print()
        return df

    # Map method name --> (scaler object, readable label)
    scaler_map = {
        "standard" : (StandardScaler(), "StandardScaler (mean=0, std=1)"),
        "minmax"   : (MinMaxScaler(),    "MinMaxScaler (range [0, 1])"),
        "robust"   : (RobustScaler(),    "RobustScaler (IQR-based, outlier-robust)"),
    }

    if method not in scaler_map:
        raise ValueError(
            f"Unknown scaling: '{method}'. "
            f"Valid options: 'standard', 'minmax', 'robust', None."
        )

    scaler, label      = scaler_map[method]
    df_scaled          = df.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

    print(f"  Method            : {label}")
    print(f"  Columns scaled    : {numerical_cols}")
    print()

    return df_scaled


# =============================================================================
# PUBLIC FUNCTION -- The ONLY function the notebook should call.
#
# DESIGN PATTERN : Facade Pattern (same as classification_eda in eda.py)
# WHY            : The notebook orchestrates the ML workflow. It should not
#                  contain preprocessing logic. One function call per step
#                  keeps the notebook readable as a workflow document.
# =============================================================================

def classification_preprocessing(
    df,
    target_column=None,
    exclude_columns=None,
    missing_strategy="mean",
    duplicate_strategy="drop",
    encoding="label",
    scaling=None,
):
    """
    Perform complete preprocessing for a tabular classification dataset.

    This is the ONLY function this module exposes to the notebook.
    It orchestrates all 4 preprocessing steps in the correct pipeline order.

    The original DataFrame is NEVER modified -- all work is done on a copy.

    Pipeline
    --------
    Step 0  : Validate input (raises on invalid input)
    Step 1  : Handle duplicates
    Step 2  : Handle missing values
    Step 3  : Encode categorical features
    Step 4  : Scale numerical features

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame returned by EDA (classification_eda).
        Pass the original -- this function works on an internal copy.
    target_column : str, optional
        Name of the target (label) column.
        Excluded from ALL transformations (imputation, encoding, scaling).
        Kept unchanged in the returned DataFrame.
        Default: None
    exclude_columns : list of str, optional
        Names of identifier columns to protect from all transformations.
        Examples: ["Loan_ID"], ["PassengerId", "Name"], ["Customer_ID"]
        These columns are KEPT in the output DataFrame but are never
        imputed, encoded, or scaled. They should be dropped in
        feature_selection.py before model training.
        Default: None
    missing_strategy : str, optional
        Imputation method for missing values.
        Options: "mean", "median", "mode", "drop"
        Default: "mean"
    duplicate_strategy : str, optional
        How to handle duplicate rows.
        Options: "drop"
        Default: "drop"
    encoding : str or None, optional
        Encoding method for categorical columns.
        Options: "label", "onehot", None
        Default: "label"
    scaling : str or None, optional
        Scaling method for numerical columns.
        Options: "standard", "minmax", "robust", None
        Use None for tree-based models (Decision Tree, Random Forest).
        Default: None

    Returns
    -------
    pd.DataFrame
        A fully preprocessed DataFrame. Ready for feature_selection.py
        and then train_test_split() in the notebook.

    Example
    -------
    # Loan Prediction dataset
    >>> clean_df = classification_preprocessing(
    ...     df,
    ...     target_column="Loan_Status",
    ...     exclude_columns=["Loan_ID"],
    ...     missing_strategy="mean",
    ...     encoding="label",
    ... )

    # Diabetes dataset (no categoricals, has outliers, KNN model)
    >>> clean_df = classification_preprocessing(
    ...     df,
    ...     target_column="Outcome",
    ...     missing_strategy="median",
    ...     encoding=None,
    ...     scaling="robust",
    ... )

    # Iris dataset (clean, needs scaling for distance-based models)
    >>> clean_df = classification_preprocessing(
    ...     df,
    ...     target_column="Species",
    ...     encoding="label",
    ...     scaling="standard",
    ... )

    Notes
    -----
    - train_test_split() must be called AFTER this function, in the notebook.
    - Identifier columns in exclude_columns are kept in the output DataFrame.
      Drop them in feature_selection.py before extracting X and y.
    - See module docstring for the Version 1 scaler data leakage limitation.
    """
    print()
    print("=" * 60)
    print("  CLASSIFICATION PREPROCESSING")
    print(f"  Missing Strategy   : {missing_strategy}")
    print(f"  Duplicate Strategy : {duplicate_strategy}")
    print(f"  Encoding           : {encoding}")
    print(f"  Scaling            : {scaling}")
    if target_column:
        print(f"  Target Column      : '{target_column}'  (excluded from all transformations)")
    if exclude_columns:
        print(f"  Excluded Columns   : {exclude_columns}  (identifier columns, excluded from all transformations)")
    print("=" * 60)
    print()

    # -- Step 0: Validate input -----------------------------------------------
    # Fail fast: verify the DataFrame and all specified columns are valid
    # before any transformation begins.
    _validate_dataframe(df, target_column=target_column, exclude_columns=exclude_columns)

    # Build the protected columns list: target + identifiers.
    # These are passed to every helper as exclude_cols.
    # One concept (columns to skip) --> one variable --> passed once.
    protected_cols = []
    if target_column:
        protected_cols.append(target_column)
    if exclude_columns:
        protected_cols.extend(exclude_columns)

    # IMPORTANT: Work on a copy. Never modify the caller's DataFrame.
    df_work = df.copy()

    # -- Step 1: Handle Duplicates --------------------------------------------
    df_work = _handle_duplicates(df_work, strategy=duplicate_strategy)

    # -- Step 2: Handle Missing Values ----------------------------------------
    df_work = _handle_missing_values(
        df_work,
        strategy=missing_strategy,
        exclude_cols=protected_cols,
    )

    # -- Step 3: Encode Categorical Features ----------------------------------
    df_work = _encode_categorical_features(
        df_work,
        method=encoding,
        exclude_cols=protected_cols,
    )

    # -- Step 4: Scale Numerical Features -------------------------------------
    df_work = _scale_features(
        df_work,
        method=scaling,
        exclude_cols=protected_cols,
    )

    # -- Summary --------------------------------------------------------------
    print("=" * 60)
    print("  PREPROCESSING COMPLETE")
    print(f"  Input shape        : {df.shape}")
    print(f"  Output shape       : {df_work.shape}")
    print(f"  Remaining NaN      : {df_work.isnull().sum().sum()}")
    print("=" * 60)
    print()

    return df_work
