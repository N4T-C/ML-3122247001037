"""
eda.py
======
Roll No : 3122247001061
Course  : ICS1512 — Machine Learning Laboratory, Semester 5

PURPOSE
-------
Provides ONE reusable EDA function:
    run_eda(df, target_column, figures_path)

The notebook imports and calls this single function.
It prints a structured summary and saves required plots as EPS files.

PLOTS GENERATED
---------------
1. Class Distribution  — bar chart of target class counts
2. Histograms          — one histogram per numerical feature
3. Correlation Heatmap — Pearson correlation between numerical features
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Faculty-required plot formatting
FONT_FAMILY = "Times New Roman"
FONT_SIZE   = 15
FIGURE_DPI  = 600
FIGURE_FMT  = "eps"


def run_eda(df, target_column, figures_path="../figures/"):
    """
    Perform Exploratory Data Analysis on a tabular classification dataset.

    This function is called ONCE in the notebook with the RAW dataset.
    It does NOT modify the data — that is preprocessing.py's job.

    Steps performed
    ---------------
    1. Dataset overview   — shape, column types, non-null counts
    2. Missing values     — count and percentage per column
    3. Duplicate rows     — total count and percentage
    4. Summary statistics — describe() for numerical and categorical columns
    5. Class distribution — bar chart of target class counts  [SAVED]
    6. Histograms         — distribution of each numerical feature [SAVED]
    7. Correlation heatmap— Pearson correlation between numerical columns [SAVED]

    Parameters
    ----------
    df            : raw pd.DataFrame (straight from pd.read_csv)
    target_column : name of the label column (e.g. "Loan_Status")
    figures_path  : directory where EPS plots are saved (created if missing)

    Returns
    -------
    dict with keys:
        "shape"         : (rows, columns)
        "missing"       : DataFrame of missing counts and percentages
        "duplicates"    : number of duplicate rows
        "class_counts"  : Series of target class value counts
    """

    os.makedirs(figures_path, exist_ok=True)

    # Apply faculty-required matplotlib style once
    matplotlib.rcParams.update({
        "font.family"     : FONT_FAMILY,
        "font.size"       : FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    # Separate numerical and categorical feature columns (excluding target)
    feature_cols     = [col for col in df.columns if col != target_column]
    numerical_cols   = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include="object").columns.tolist()

    # -------------------------------------------------------------------------
    # 1. DATASET OVERVIEW
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Rows    : {df.shape[0]}")
    print(f"  Columns : {df.shape[1]}")
    print()

    # Show each column's type and non-null count
    overview = pd.DataFrame({
        "Column"       : df.columns,
        "Dtype"        : df.dtypes.values,
        "Non-Null"     : df.notnull().sum().values,
    })
    print(overview.to_string(index=False))
    print()

    # -------------------------------------------------------------------------
    # 2. MISSING VALUES
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("2. MISSING VALUES")
    print("=" * 60)

    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df) * 100).round(2)
    missing_df    = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %"    : missing_pct,
    })
    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if missing_df.empty:
        print("  No missing values found.\n")
    else:
        print(missing_df.to_string())
        print()

    # -------------------------------------------------------------------------
    # 3. DUPLICATE ROWS
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("3. DUPLICATE ROWS")
    print("=" * 60)

    dup_count = int(df.duplicated().sum())
    dup_pct   = round(dup_count / len(df) * 100, 2)
    print(f"  Duplicate rows : {dup_count}  ({dup_pct}%)")
    if dup_count > 0:
        print("  → Remove duplicates in preprocessing.")
    print()

    # -------------------------------------------------------------------------
    # 4. SUMMARY STATISTICS
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("4. SUMMARY STATISTICS")
    print("=" * 60)

    if numerical_cols:
        print("  Numerical Features:")
        print(df[numerical_cols].describe().T.to_string())
        print()

    if categorical_cols:
        print("  Categorical Features:")
        print(df[categorical_cols].describe(include="object").T.to_string())
        print()

    # -------------------------------------------------------------------------
    # 5. CLASS DISTRIBUTION — bar chart
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("5. CLASS DISTRIBUTION")
    print("=" * 60)

    class_counts = df[target_column].value_counts()
    class_pct    = df[target_column].value_counts(normalize=True) * 100

    for cls in class_counts.index:
        print(f"  {str(cls):<20}: {class_counts[cls]:>5} samples  ({class_pct[cls]:.1f}%)")

    # Warn if any class has less than 30% of samples (class imbalance)
    if class_pct.min() < 30:
        print("\n  ⚠ Class imbalance detected (minority class < 30%).")
    print()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(class_counts))
    bars   = ax.bar(class_counts.index.astype(str), class_counts.values,
                    color=colors, edgecolor="white")

    # Annotate each bar with count and percentage
    for bar, count, pct in zip(bars, class_counts.values, class_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + class_counts.max() * 0.01,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=FONT_SIZE - 2)

    ax.set_xlabel(target_column)
    ax.set_ylabel("Count")
    ax.set_title(f"Class Distribution — {target_column}")
    ax.set_ylim(0, class_counts.max() * 1.25)
    plt.tight_layout()

    save_path = os.path.join(figures_path, f"class_distribution.{FIGURE_FMT}")
    fig.savefig(save_path, format=FIGURE_FMT, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  [Saved] {save_path}")
    plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. HISTOGRAMS — one per numerical feature
    # -------------------------------------------------------------------------
    if numerical_cols:
        print("=" * 60)
        print("6. FEATURE DISTRIBUTIONS (Histograms)")
        print("=" * 60)

        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols  # ceiling division

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 4 * n_rows),
                                 squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i],
                         color="#4C72B0", edgecolor="white")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")
            axes[i].set_title(f"Distribution of {col}")

        # Hide unused subplot slots
        for j in range(len(numerical_cols), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Feature Histograms with KDE", y=1.02)
        plt.tight_layout()

        save_path = os.path.join(figures_path, f"histograms.{FIGURE_FMT}")
        fig.savefig(save_path, format=FIGURE_FMT, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  [Saved] {save_path}\n")
        plt.show()
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 7. CORRELATION HEATMAP — Pearson correlation
    # -------------------------------------------------------------------------
    if len(numerical_cols) >= 2:
        print("=" * 60)
        print("7. CORRELATION HEATMAP")
        print("=" * 60)

        corr = df[numerical_cols].corr(method="pearson")

        # Mask the upper triangle (it mirrors the lower triangle)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        n       = len(numerical_cols)
        fig, ax = plt.subplots(figsize=(max(8, n), max(7, n - 1)))

        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, ax=ax,
                    annot_kws={"size": FONT_SIZE - 2})

        ax.set_title("Pearson Correlation Heatmap")
        plt.tight_layout()

        save_path = os.path.join(figures_path, f"correlation_heatmap.{FIGURE_FMT}")
        fig.savefig(save_path, format=FIGURE_FMT, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  [Saved] {save_path}\n")
        plt.show()
        plt.close(fig)

    # -------------------------------------------------------------------------
    # RETURN structured summary for use in observations
    # -------------------------------------------------------------------------
    return {
        "shape"       : df.shape,
        "missing"     : missing_df,
        "duplicates"  : dup_count,
        "class_counts": class_counts,
    }
