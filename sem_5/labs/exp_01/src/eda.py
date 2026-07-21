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
It prints a structured summary and saves ONE consolidated EDA figure
named eda_summary.eps containing exactly 12 subplots on a single page.

SUBPLOTS (3 rows x 4 cols)
---------------------------
[0,0] Dataset Overview (text table)
[0,1] Missing Value Bar Chart
[0,2] Class Distribution Bar Chart
[0,3] Pearson Correlation Heatmap
[1,0] Histogram — numerical col 1
[1,1] Histogram — numerical col 2
[1,2] Histogram — numerical col 3
[1,3] Histogram — numerical col 4
[2,0] Boxplot   — numerical col 1
[2,1] Boxplot   — numerical col 2
[2,2] Boxplot   — numerical col 3
[2,3] Scatter   — col 1 vs col 2 (coloured by class)
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

    Generates ONE Matplotlib figure with exactly 12 subplots arranged in a
    3-row x 4-column grid and saves it as eda_summary.eps.

    This function is called ONCE in the notebook with the RAW dataset.
    It does NOT modify the data — that is preprocessing.py's job.

    Parameters
    ----------
    df            : raw pd.DataFrame (straight from pd.read_csv)
    target_column : name of the label column (e.g. "Loan_Status")
    figures_path  : directory where the EPS file is saved (created if missing)

    Returns
    -------
    dict with keys:
        "shape"        : (rows, columns)
        "missing"      : DataFrame of missing counts and percentages
        "duplicates"   : number of duplicate rows
        "class_counts" : Series of target class value counts
    """

    os.makedirs(figures_path, exist_ok=True)

    # Apply faculty-required matplotlib style once
    matplotlib.rcParams.update({
        "font.family"     : FONT_FAMILY,
        "font.size"       : FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "legend.fontsize" : FONT_SIZE - 2,
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

    overview = pd.DataFrame({
        "Column"  : df.columns,
        "Dtype"   : df.dtypes.values,
        "Non-Null": df.notnull().sum().values,
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
    # 5. CLASS DISTRIBUTION — console summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("5. CLASS DISTRIBUTION")
    print("=" * 60)

    class_counts = df[target_column].value_counts()
    class_pct    = df[target_column].value_counts(normalize=True) * 100

    for cls in class_counts.index:
        print(f"  {str(cls):<20}: {class_counts[cls]:>5} samples  ({class_pct[cls]:.1f}%)")

    if class_pct.min() < 30:
        print("\n  ⚠ Class imbalance detected (minority class < 30%).")
    print()

    # =========================================================================
    # ONE FIGURE — 3 rows x 4 columns = 12 subplots
    # =========================================================================
    palette = sns.color_palette("Set2")

    fig, axes = plt.subplots(3, 4, figsize=(28, 18))
    fig.suptitle("EDA Summary", fontsize=FONT_SIZE + 4, fontweight="bold",
                 fontfamily=FONT_FAMILY, y=1.01)

    # -------------------------------------------------------------------------
    # [0, 0] Dataset Overview — text table
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("Dataset Overview", pad=10)
    overview_data = [
        ["Rows",             str(df.shape[0])],
        ["Columns",          str(df.shape[1])],
        ["Numerical cols",   str(len(numerical_cols))],
        ["Categorical cols", str(len(categorical_cols))],
        ["Target column",    target_column],
        ["Duplicate rows",   str(dup_count)],
        ["Total missing",    str(int(df.isnull().sum().sum()))],
        ["Classes",          str(df[target_column].nunique())],
    ]
    tbl = ax.table(cellText=overview_data,
                   colLabels=["Attribute", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT_SIZE - 3)
    tbl.scale(1, 1.6)

    # -------------------------------------------------------------------------
    # [0, 1] Missing Value Bar Chart
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    all_missing = df.isnull().sum()
    all_missing = all_missing[all_missing > 0]
    if all_missing.empty:
        ax.text(0.5, 0.5, "No Missing Values",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=FONT_SIZE - 1)
    else:
        ax.bar(all_missing.index, all_missing.values,
               color=palette[1], edgecolor="white")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("Missing Value Analysis")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing Count")

    # -------------------------------------------------------------------------
    # [0, 2] Class Distribution Bar Chart
    # -------------------------------------------------------------------------
    ax = axes[0, 2]
    colors = sns.color_palette("Set2", len(class_counts))
    bars   = ax.bar(class_counts.index.astype(str), class_counts.values,
                    color=colors, edgecolor="white")
    for bar, count, pct in zip(bars, class_counts.values, class_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + class_counts.max() * 0.01,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=FONT_SIZE - 4)
    ax.set_title("Class Distribution")
    ax.set_xlabel(target_column)
    ax.set_ylabel("Count")
    ax.set_ylim(0, class_counts.max() * 1.3)

    # -------------------------------------------------------------------------
    # [0, 3] Pearson Correlation Heatmap
    # -------------------------------------------------------------------------
    ax = axes[0, 3]
    if len(numerical_cols) >= 2:
        corr = df[numerical_cols].corr(method="pearson")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    linewidths=0.4, ax=ax,
                    annot_kws={"size": FONT_SIZE - 5},
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Heatmap")
        ax.tick_params(axis="x", rotation=45, labelsize=FONT_SIZE - 5)
        ax.tick_params(axis="y", rotation=0,  labelsize=FONT_SIZE - 5)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Need >= 2 numerical\ncolumns",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Correlation Heatmap")

    # -------------------------------------------------------------------------
    # [1, 0..3] Histograms — up to 4 numerical features
    # -------------------------------------------------------------------------
    hist_cols = numerical_cols[:4]
    for i in range(4):
        ax = axes[1, i]
        if i < len(hist_cols):
            col = hist_cols[i]
            sns.histplot(df[col].dropna(), kde=True, ax=ax,
                         color=palette[i % len(palette)], edgecolor="white")
            ax.set_title(f"Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
        else:
            ax.axis("off")

    # -------------------------------------------------------------------------
    # [2, 0..2] Boxplots — up to 3 numerical features
    # -------------------------------------------------------------------------
    box_cols = numerical_cols[:3]
    for i in range(3):
        ax = axes[2, i]
        if i < len(box_cols):
            col = box_cols[i]
            ax.boxplot(df[col].dropna(), patch_artist=True,
                       boxprops=dict(facecolor=palette[i % len(palette)],
                                     color="black"),
                       medianprops=dict(color="black", linewidth=2),
                       whiskerprops=dict(color="black"),
                       capprops=dict(color="black"),
                       flierprops=dict(marker="o", markerfacecolor="grey",
                                       markersize=4, linestyle="none"))
            ax.set_title(f"Boxplot: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Value")
        else:
            ax.axis("off")

    # -------------------------------------------------------------------------
    # [2, 3] Scatter plot — numerical col 0 vs col 1, coloured by class
    # -------------------------------------------------------------------------
    ax = axes[2, 3]
    if len(numerical_cols) >= 2:
        x_col, y_col    = numerical_cols[0], numerical_cols[1]
        scatter_palette = sns.color_palette("Set1", df[target_column].nunique())
        for idx, cls in enumerate(df[target_column].unique()):
            mask_cls = df[target_column] == cls
            ax.scatter(df.loc[mask_cls, x_col], df.loc[mask_cls, y_col],
                       label=str(cls), alpha=0.6,
                       color=scatter_palette[idx % len(scatter_palette)],
                       s=30, edgecolors="none")
        ax.set_title(f"Scatter: {x_col} vs {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        legend = ax.legend(title=target_column, framealpha=0.7,
                           prop={"family": FONT_FAMILY, "size": FONT_SIZE - 2})
        legend.get_title().set_fontfamily(FONT_FAMILY)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Need >= 2 numerical\ncolumns for scatter",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Scatter Plot")

    # =========================================================================
    # Save ONE file: eda_summary.eps
    # =========================================================================
    plt.tight_layout()

    save_path = os.path.join(figures_path, "eda_summary.eps")
    fig.savefig(save_path, format=FIGURE_FMT, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  [Saved] {save_path}")

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
