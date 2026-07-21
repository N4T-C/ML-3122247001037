"""
preprocessing.py
================
Roll No : 3122247001061
Course  : ICS1512 — Machine Learning Laboratory, Semester 5

PURPOSE
-------
Provides ONE reusable preprocessing function:
    preprocess(df, target_column, ...)

The notebook imports and calls this single function on the raw dataset.
It returns a clean, model-ready DataFrame.

PIPELINE ORDER (why this order matters)
----------------------------------------
Step 1 — Drop Duplicates      : remove before imputing (no point imputing rows we'll delete)
Step 2 — Handle Missing Values: impute before encoding (encoders fail on NaN)
Step 3 — Encode Categoricals  : convert strings to numbers (models need numbers)
Step 4 — Scale Numerical Cols : normalize scale (important for KNN, SVM, Logistic)
"""

import warnings

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")


def preprocess(df, target_column,
               exclude_columns=None,
               missing_strategy="mean",
               encoding="label",
               scaling=None):
    """
    Clean and prepare a raw DataFrame for model training.

    This is the ONLY function the notebook calls from this module.
    The original DataFrame is never modified — all work is done on a copy.

    Parameters
    ----------
    df               : raw pd.DataFrame from pd.read_csv()
    target_column    : name of the label column — excluded from all transformations
    exclude_columns  : list of ID columns to exclude (e.g. ["Loan_ID"])
                       kept in the output but never encoded or scaled
    missing_strategy : how to fill NaN values
                       "mean"   → fill numerical NaN with column mean
                       "median" → fill numerical NaN with column median
                       "mode"   → fill all NaN with most frequent value
                       "drop"   → drop rows that have any missing value
    encoding         : how to convert categorical columns to numbers
                       "label"  → LabelEncoder (each category → integer)
                       "onehot" → pd.get_dummies (each category → binary column)
                       None     → skip encoding (if all columns are numeric)
    scaling          : how to scale numerical feature columns
                       "standard" → StandardScaler (mean=0, std=1)
                       "minmax"   → MinMaxScaler (range [0, 1])
                       None       → skip scaling (use for tree-based models)

    Returns
    -------
    clean_df : pd.DataFrame — preprocessed, model-ready DataFrame

    Note: train_test_split() must be called AFTER this function, in the notebook.
    """

    if exclude_columns is None:
        exclude_columns = []

    # Columns to protect from ALL transformations (target + ID columns)
    protected = [target_column] + exclude_columns

    # Always work on a copy — never modify the caller's DataFrame
    df = df.copy()

    # -------------------------------------------------------------------------
    # STEP 1 — DROP DUPLICATES
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("STEP 1: Drop Duplicates")
    print("=" * 50)

    n_before = len(df)
    df       = df.drop_duplicates().reset_index(drop=True)
    n_after  = len(df)

    print(f"  Rows before : {n_before}")
    print(f"  Rows after  : {n_after}  ({n_before - n_after} duplicates removed)")
    print()

    # -------------------------------------------------------------------------
    # STEP 2 — HANDLE MISSING VALUES
    # -------------------------------------------------------------------------
    print("=" * 50)
    print(f"STEP 2: Handle Missing Values  (strategy: '{missing_strategy}')")
    print("=" * 50)

    # Only operate on feature columns (not target, not ID columns)
    feature_cols     = [col for col in df.columns if col not in protected]
    numerical_cols   = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include="object").columns.tolist()

    total_missing = df[feature_cols].isnull().sum().sum()

    if total_missing == 0:
        print("  No missing values found.\n")

    elif missing_strategy == "mean":
        # Numerical → mean, Categorical → mode
        for col in numerical_cols:
            if df[col].isnull().any():
                fill = df[col].mean()
                df[col].fillna(fill, inplace=True)
                print(f"  '{col}' → filled with mean = {fill:.4f}")
        for col in categorical_cols:
            if df[col].isnull().any():
                fill = df[col].mode()[0]
                df[col].fillna(fill, inplace=True)
                print(f"  '{col}' → filled with mode = '{fill}'")

    elif missing_strategy == "median":
        # Numerical → median, Categorical → mode
        for col in numerical_cols:
            if df[col].isnull().any():
                fill = df[col].median()
                df[col].fillna(fill, inplace=True)
                print(f"  '{col}' → filled with median = {fill:.4f}")
        for col in categorical_cols:
            if df[col].isnull().any():
                fill = df[col].mode()[0]
                df[col].fillna(fill, inplace=True)
                print(f"  '{col}' → filled with mode = '{fill}'")

    elif missing_strategy == "mode":
        # All columns → mode
        for col in feature_cols:
            if df[col].isnull().any():
                fill = df[col].mode()[0]
                df[col].fillna(fill, inplace=True)
                print(f"  '{col}' → filled with mode = '{fill}'")

    elif missing_strategy == "drop":
        n_before = len(df)
        df       = df.dropna(subset=feature_cols).reset_index(drop=True)
        print(f"  Dropped {n_before - len(df)} rows with missing values.")

    else:
        raise ValueError(f"Unknown missing_strategy: '{missing_strategy}'. "
                         "Choose: 'mean', 'median', 'mode', 'drop'")
    print()

    # -------------------------------------------------------------------------
    # STEP 3 — ENCODE CATEGORICAL FEATURES
    # -------------------------------------------------------------------------
    print("=" * 50)
    print(f"STEP 3: Encode Categorical Features  (method: '{encoding}')")
    print("=" * 50)

    # Recalculate after possible row drops
    feature_cols     = [col for col in df.columns if col not in protected]
    categorical_cols = df[feature_cols].select_dtypes(include="object").columns.tolist()

    if not categorical_cols or encoding is None:
        print("  Skipped (no categorical columns or encoding=None).\n")

    elif encoding == "label":
        # Assign an integer to each unique category value (alphabetical order)
        for col in categorical_cols:
            le          = LabelEncoder()
            df[col]     = le.fit_transform(df[col].astype(str))
            mapping     = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
            print(f"  '{col}' → {mapping}")
        print()

    elif encoding == "onehot":
        # Create one binary column per category value
        before = df.shape
        df     = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        print(f"  One-Hot Encoding applied.")
        print(f"  Columns: {before[1]} → {df.shape[1]} (+{df.shape[1] - before[1]} new columns)\n")

    else:
        raise ValueError(f"Unknown encoding: '{encoding}'. Choose: 'label', 'onehot', None")

    # -------------------------------------------------------------------------
    # STEP 4 — SCALE NUMERICAL FEATURES
    # -------------------------------------------------------------------------
    print("=" * 50)
    print(f"STEP 4: Scale Numerical Features  (method: '{scaling}')")
    print("=" * 50)

    if scaling is None:
        print("  Skipped (scaling=None).")
        print("  Tip: Tree-based models (Decision Tree, Random Forest) don't need scaling.\n")

    else:
        # Only scale feature columns, not target or ID columns
        feature_cols   = [col for col in df.columns if col not in protected]
        numerical_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()

        if not numerical_cols:
            print("  No numerical columns to scale.\n")

        else:
            if scaling == "standard":
                scaler = StandardScaler()   # mean=0, std=1
            elif scaling == "minmax":
                scaler = MinMaxScaler()     # range [0, 1]
            else:
                raise ValueError(f"Unknown scaling: '{scaling}'. Choose: 'standard', 'minmax', None")

            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            print(f"  Scaler       : {type(scaler).__name__}")
            print(f"  Columns      : {numerical_cols}\n")

    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"  Output shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  NaN remaining: {df.isnull().sum().sum()}")
    print()

    return df
