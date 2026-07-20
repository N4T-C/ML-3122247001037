"""
feature_selection.py
====================
Roll No : 3122247001061
Course  : ICS1512 — Machine Learning Laboratory, Semester 5

PURPOSE
-------
Provides ONE reusable feature selection function:
    select_features(df, target_column, method, k, exclude_columns)

The notebook imports and calls this single function after preprocessing.
It returns the top-k most informative features using a statistical test.

SUPPORTED METHODS
-----------------
"chi2"        : Chi-Square test — measures statistical independence between
                each feature and the target. Requires NON-NEGATIVE values.
                Use after label encoding (encoded values are ≥ 0).

"anova"       : ANOVA F-test — tests whether the mean of a feature differs
                significantly across target classes. Works on any numbers.
                Best for continuous numerical features.

"mutual_info" : Mutual Information — measures how much knowing a feature's
                value reduces uncertainty about the target. Detects non-linear
                relationships that chi2 and ANOVA may miss.
"""

import warnings

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

warnings.filterwarnings("ignore")


def select_features(df, target_column, method="chi2", k=5, exclude_columns=None):
    """
    Select the top-k most informative features using a statistical scoring method.

    This is the ONLY function the notebook calls from this module.
    The DataFrame must be fully preprocessed (no NaN, all numeric features).

    Parameters
    ----------
    df             : preprocessed pd.DataFrame (from preprocess())
    target_column  : name of the label column
    method         : scoring method — "chi2", "anova", or "mutual_info"
    k              : number of top features to keep
    exclude_columns: list of ID columns to exclude from selection (e.g. ["Loan_ID"])

    Returns
    -------
    selected_features : list of the top-k feature column names
    feature_scores_df : pd.DataFrame with columns [Feature, Score, Rank]

    Usage in notebook
    -----------------
    selected_features, scores_df = select_features(clean_df, "Loan_Status",
                                                    method="chi2", k=8)
    X = clean_df[selected_features]
    y = clean_df["Loan_Status"]
    """

    if exclude_columns is None:
        exclude_columns = []

    # Build the feature matrix (exclude target and ID columns)
    protected    = [target_column] + exclude_columns
    feature_cols = [col for col in df.columns if col not in protected]

    X = df[feature_cols]
    y = df[target_column]

    # -------------------------------------------------------------------------
    # Pick the scoring function
    # -------------------------------------------------------------------------
    if method == "chi2":
        scorer = chi2
    elif method == "anova":
        scorer = f_classif
    elif method == "mutual_info":
        scorer = mutual_info_classif
    else:
        raise ValueError(
            f"Unknown method: '{method}'. Choose: 'chi2', 'anova', 'mutual_info'"
        )

    # -------------------------------------------------------------------------
    # Cap k if it exceeds the number of available features
    # -------------------------------------------------------------------------
    k = min(k, len(feature_cols))

    # -------------------------------------------------------------------------
    # Apply SelectKBest to score and rank all features
    # -------------------------------------------------------------------------
    selector = SelectKBest(scorer, k=k)
    selector.fit(X, y)

    # Build a ranking DataFrame for display in the notebook
    scores_df = pd.DataFrame({
        "Feature": feature_cols,
        "Score"  : selector.scores_,
    })
    scores_df = scores_df.sort_values("Score", ascending=False).reset_index(drop=True)
    scores_df["Rank"]  = scores_df.index + 1
    scores_df["Score"] = scores_df["Score"].round(4)
    scores_df = scores_df[["Rank", "Feature", "Score"]]  # reorder columns

    # The top-k features (by score)
    selected_features = scores_df.head(k)["Feature"].tolist()

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  FEATURE SELECTION")
    print(f"{'='*50}")
    print(f"  Method           : {method}")
    print(f"  Total features   : {len(feature_cols)}")
    print(f"  Features selected: {k}")
    print()
    print(scores_df.to_string(index=False))
    print(f"\n  Selected: {selected_features}")
    print(f"{'='*50}\n")

    return selected_features, scores_df
