"""
eda.py
======

Exploratory Data Analysis (EDA) module for tabular classification datasets.

RESPONSIBILITY
--------------
This module is responsible for ONE thing only: performing EDA.
It does NOT preprocess data, split data, train models, or evaluate models.

PUBLIC INTERFACE
----------------
The notebook should call exactly ONE function from this module:

    >>> eda_results = classification_eda(df, target_col="Loan_Status")

Everything else in this module is an internal implementation detail.
Functions prefixed with a single underscore (_) are private by convention.
Do not call them directly from the notebook.

PLOT FORMATTING (Faculty Requirements)
---------------------------------------
    Font Family  : Times New Roman
    Font Size    : 15
    Axis Labels  : Bold
    Legend Font  : Times New Roman, size 15
    Save Format  : EPS
    Save DPI     : 600

These constants are defined once at the top of this module.
Changing one value here updates every plot in the framework.

ASSUMPTIONS (Version 1)
-----------------------
    - Dataset is tabular (CSV-based).
    - Target column is categorical (classification problems only).
    - NLP and image datasets are out of scope for this module.
    - Future experiments may require eda_text.py or eda_image.py
      as separate modules alongside this one.

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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
# PLOT FORMATTING CONSTANTS  (Faculty Requirements)
#
# Why here and not in utils.py?
# -> For Version 1, this module is the only module with plots.
# -> If a future experiment introduces new plot-heavy modules, we will
#    extract these into a shared plot_utils.py at that time (YAGNI).
# =============================================================================

_FONT_FAMILY = "Times New Roman"
_FONT_SIZE   = 15
_FIGURE_DPI  = 600
_FIGURE_FMT  = "eps"


# =============================================================================
# PRIVATE HELPER: Apply Plot Style
#
# Why a private helper instead of repeating 6 lines in every function?
# -> DRY (Don't Repeat Yourself) within this module.
# -> If the faculty changes the font to Arial, one line changes here.
# =============================================================================

def _apply_plot_style():
    """
    Apply faculty-required matplotlib formatting globally.

    Called at the start of every plot function in this module.
    Uses matplotlib's rcParams system to set global defaults.

    Note
    ----
    rcParams are global -- they persist for the entire notebook session.
    This is intentional: all plots in one experiment share the same style.
    """
    matplotlib.rcParams.update({
        "font.family"       : _FONT_FAMILY,
        "font.size"         : _FONT_SIZE,
        "axes.labelweight"  : "bold",
        "axes.titleweight"  : "bold",
        "legend.fontsize"   : _FONT_SIZE,
        "xtick.labelsize"   : _FONT_SIZE,
        "ytick.labelsize"   : _FONT_SIZE,
    })


def _save_figure(fig, figures_path, filename):
    """
    Save a matplotlib figure as an EPS file at 600 DPI.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    figures_path : str
        Directory path where the file will be saved.
        Created automatically if it does not exist.
    filename : str
        Filename WITHOUT extension (e.g., "class_distribution").
        Extension is added automatically based on _FIGURE_FMT.

    Side Effects
    ------------
    Creates the directory if it does not exist.
    Saves the file to disk.
    Closes the figure to free memory.
    Prints the saved file path.
    """
    os.makedirs(figures_path, exist_ok=True)
    filepath = os.path.join(figures_path, f"{filename}.{_FIGURE_FMT}")
    fig.savefig(filepath, format=_FIGURE_FMT, dpi=_FIGURE_DPI, bbox_inches="tight")
    print(f"    [Saved] {filepath}")
    plt.close(fig)


# =============================================================================
# PRIVATE FUNCTION 1: Dataset Overview
#
# PURPOSE  : Give the first look at the dataset before any analysis.
# PRINTS   : Shape, column names, data types, memory usage.
# RETURNS  : dict -- for use in observation summaries.
# PLOTS    : None.
# =============================================================================

def _dataset_overview(df):
    """
    Print a structured overview of the dataset.

    Covers shape, column data types, and memory usage.
    This is always the FIRST step in EDA -- understand what you have
    before you try to analyze it.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.

    Returns
    -------
    dict
        Keys: rows, columns, dtypes, memory_kb.
    """
    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Rows         : {df.shape[0]}")
    print(f"  Columns      : {df.shape[1]}")
    print(f"  Memory Usage : {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print()
    print("  Column Data Types:")
    type_df = pd.DataFrame({
        "Column"         : df.columns,
        "Dtype"          : df.dtypes.values,
        "Non-Null Count" : df.notnull().sum().values,
    })
    print(type_df.to_string(index=False))
    print()

    return {
        "rows"      : df.shape[0],
        "columns"   : df.shape[1],
        "dtypes"    : df.dtypes.to_dict(),
        "memory_kb" : round(df.memory_usage(deep=True).sum() / 1024, 2),
    }


# =============================================================================
# PRIVATE FUNCTION 2: Missing Value Analysis
#
# PURPOSE  : Identify which columns have missing data and how much.
# PRINTS   : Column-wise missing count and percentage.
# RETURNS  : dict -- structured missing value summary.
# PLOTS    : Horizontal bar chart (only if missing values exist).
#
# Design note: Missing value analysis is done BEFORE preprocessing.
# We analyze the raw dataset -- not the cleaned one.
# This tells us what preprocessing decisions to make.
# =============================================================================

def _missing_value_analysis(df, figures_path):
    """
    Identify and visualize missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
    figures_path : str

    Returns
    -------
    dict
        Keys: "Missing Count" and "Missing (%)" as sub-dicts keyed by column name.
        Returns an empty dict if no missing values exist.
    """
    print("=" * 60)
    print("2. MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df)) * 100

    missing_df = pd.DataFrame({
        "Missing Count" : missing_count,
        "Missing (%)"   : missing_pct.round(2),
    }).sort_values("Missing (%)", ascending=False)

    # Filter to only columns that actually have missing values
    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if missing_df.empty:
        print("  No missing values found in any column.")
        print()
        return {}

    print(missing_df.to_string())
    print()

    # Plot: horizontal bar chart of missing percentages
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.7)))
    ax.barh(
        missing_df.index,
        missing_df["Missing (%)"],
        color="#4C72B0",
        edgecolor="white",
    )
    ax.set_xlabel("Missing Values (%)")
    ax.set_ylabel("Feature")
    ax.set_title("Missing Value Analysis")
    ax.invert_yaxis()  # Highest missing at the top

    # Annotate each bar with the percentage
    for i, (col, row) in enumerate(missing_df.iterrows()):
        ax.text(
            row["Missing (%)"] + 0.3,
            i,
            f'{row["Missing (%)"]:.1f}%',
            va="center",
            fontsize=_FONT_SIZE - 2,
            fontfamily=_FONT_FAMILY,
        )

    plt.tight_layout()
    _save_figure(fig, figures_path, "missing_values")

    return missing_df.to_dict()


# =============================================================================
# PRIVATE FUNCTION 3: Duplicate Analysis
#
# PURPOSE  : Detect rows that are exact copies of another row.
# PRINTS   : Count and percentage of duplicate rows.
# RETURNS  : dict
# PLOTS    : None -- a number is sufficient here.
#
# Design note: Duplicates can silently inflate model accuracy if they
# appear in both training and test sets. Identifying them early is important.
# =============================================================================

def _duplicate_analysis(df):
    """
    Identify and report duplicate rows in the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Keys: duplicate_count, duplicate_percentage.
    """
    print("=" * 60)
    print("3. DUPLICATE ANALYSIS")
    print("=" * 60)

    dup_count = int(df.duplicated().sum())
    dup_pct   = round((dup_count / len(df)) * 100, 2)

    print(f"  Total Rows     : {len(df)}")
    print(f"  Duplicate Rows : {dup_count} ({dup_pct}%)")

    if dup_count > 0:
        print("  Note: Duplicates should be removed during preprocessing.")
    else:
        print("  No duplicates found.")

    print()

    return {
        "duplicate_count"      : dup_count,
        "duplicate_percentage" : dup_pct,
    }


# =============================================================================
# PRIVATE FUNCTION 4: Summary Statistics
#
# PURPOSE  : Understand the scale, spread, and central tendency of features.
# PRINTS   : describe() for numerical and categorical columns separately.
# RETURNS  : dict
# PLOTS    : None.
#
# Design note: We separate numerical and categorical describe() outputs
# because pandas' describe() on mixed types is misleading.
# =============================================================================

def _summary_statistics(df, numerical_cols, categorical_cols):
    """
    Print descriptive statistics for numerical and categorical features.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str
        Numerical feature column names.
    categorical_cols : list of str
        Categorical feature column names.

    Returns
    -------
    dict
        Contains 'numerical_stats' and 'categorical_stats'.
    """
    print("=" * 60)
    print("4. SUMMARY STATISTICS")
    print("=" * 60)

    result = {}

    if numerical_cols:
        print("  Numerical Features (count, mean, std, min, quartiles, max):")
        num_stats = df[numerical_cols].describe().T
        print(num_stats.to_string())
        print()
        result["numerical_stats"] = num_stats.to_dict()
    else:
        print("  No numerical features found.")
        result["numerical_stats"] = {}

    if categorical_cols:
        print("  Categorical Features (count, unique, top, freq):")
        cat_stats = df[categorical_cols].describe(include="object").T
        print(cat_stats.to_string())
        print()
        result["categorical_stats"] = cat_stats.to_dict()
    else:
        print("  No categorical features found.")
        result["categorical_stats"] = {}

    return result


# =============================================================================
# PRIVATE FUNCTION 5: Class Distribution
#
# PURPOSE  : Understand whether the target classes are balanced or skewed.
# PRINTS   : Class counts and percentages. Imbalance warning if needed.
# RETURNS  : dict
# PLOTS    : Bar chart with count + percentage labels on each bar.
#
# Design note: Class imbalance (e.g., 95% vs 5%) is one of the most
# common problems in real-world classification. Detecting it early
# influences which evaluation metrics to use (accuracy alone is misleading).
# =============================================================================

def _class_distribution(df, target_col, figures_path):
    """
    Visualize and analyze the target class distribution.

    Detects class imbalance: flags if any class has < 30% of total samples.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Name of the target (label) column.
    figures_path : str

    Returns
    -------
    dict
        Keys: class_counts, class_percentages, imbalanced (bool).
    """
    print("=" * 60)
    print("5. CLASS DISTRIBUTION")
    print("=" * 60)

    class_counts = df[target_col].value_counts()
    class_pct    = df[target_col].value_counts(normalize=True) * 100

    for cls in class_counts.index:
        print(f"  {str(cls):<25}: {class_counts[cls]:>5} samples  ({class_pct[cls]:.1f}%)")

    is_imbalanced = bool(class_pct.min() < 30)
    if is_imbalanced:
        print("\n  WARNING: Class imbalance detected (minority class < 30%).")
        print("  Consider stratified train-test split and appropriate metrics.")
    else:
        print("\n  Classes are approximately balanced.")
    print()

    # Plot: bar chart with annotated counts and percentages
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("Set2", len(class_counts))
    bars = ax.bar(
        class_counts.index.astype(str),
        class_counts.values,
        color=palette,
        edgecolor="white",
        linewidth=1.2,
    )

    for bar, count, pct in zip(bars, class_counts.values, class_pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(class_counts.values) * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=_FONT_SIZE - 2,
            fontfamily=_FONT_FAMILY,
        )

    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    ax.set_title(f"Class Distribution -- {target_col}")
    ax.set_ylim(0, max(class_counts.values) * 1.2)
    plt.tight_layout()
    _save_figure(fig, figures_path, "class_distribution")

    return {
        "class_counts"      : class_counts.to_dict(),
        "class_percentages" : class_pct.round(2).to_dict(),
        "imbalanced"        : is_imbalanced,
    }


# =============================================================================
# PRIVATE FUNCTION 6: Histograms
#
# PURPOSE  : Understand the distribution shape of each numerical feature.
# PRINTS   : Nothing (plot is self-explanatory).
# RETURNS  : None
# PLOTS    : Grid of histograms with KDE overlay, one per numerical column.
#
# Design note: The KDE (Kernel Density Estimate) line on top of the
# histogram gives a smooth approximation of the distribution.
# It helps identify skewness (asymmetry) which affects preprocessing
# decisions like log-transform or normalization.
# =============================================================================

def _histograms(df, numerical_cols, figures_path):
    """
    Plot histograms with KDE overlay for all numerical features.

    Arranged in a grid: maximum 3 plots per row.
    squeeze=False ensures we always get a 2D array of axes,
    even when there is only 1 plot.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str
    figures_path : str
    """
    if not numerical_cols:
        print("  No numerical columns found. Skipping histograms.")
        return

    _apply_plot_style()

    n_cols = min(3, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        squeeze=False,  # Always return 2D array even for a 1x1 grid
    )
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.histplot(
            df[col].dropna(),
            kde=True,
            ax=axes[i],
            color="#4C72B0",
            edgecolor="white",
        )
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"Distribution of {col}")

    # Hide any unused subplot slots
    for j in range(len(numerical_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions -- Histograms with KDE", y=1.02, fontsize=_FONT_SIZE)
    plt.tight_layout()
    _save_figure(fig, figures_path, "histograms")


# =============================================================================
# PRIVATE FUNCTION 7: Box Plots
#
# PURPOSE  : Visualize spread, median, and outliers per feature, split by class.
# PRINTS   : Nothing.
# RETURNS  : None.
# PLOTS    : Grid of box plots, one per numerical column, grouped by target class.
#
# Design note: Box plots grouped by target class answer the question:
# "Does this feature have different distributions for different classes?"
# If yes, that feature is likely informative for the model.
# =============================================================================

def _boxplots(df, numerical_cols, target_col, figures_path):
    """
    Plot box plots for each numerical feature, grouped by target class.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str
    target_col : str
    figures_path : str
    """
    if not numerical_cols:
        print("  No numerical columns found. Skipping box plots.")
        return

    _apply_plot_style()

    n_cols = min(3, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    # Convert target to string to ensure consistent categorical axis behavior
    plot_df = df.copy()
    plot_df[target_col] = plot_df[target_col].astype(str)

    for i, col in enumerate(numerical_cols):
        sns.boxplot(
            data=plot_df,
            x=target_col,
            y=col,
            ax=axes[i],
            palette="Set2",
        )
        axes[i].set_xlabel(target_col)
        axes[i].set_ylabel(col)
        axes[i].set_title(f"{col}  by  {target_col}")

    for j in range(len(numerical_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Box Plots -- Feature Distribution by Class", y=1.02, fontsize=_FONT_SIZE)
    plt.tight_layout()
    _save_figure(fig, figures_path, "boxplots")


# =============================================================================
# PRIVATE FUNCTION 8: Correlation Heatmap
#
# PURPOSE  : Reveal linear relationships between numerical features.
# PRINTS   : Nothing.
# RETURNS  : pd.DataFrame -- the full correlation matrix (used by observations).
# PLOTS    : Lower-triangle heatmap of Pearson correlation coefficients.
#
# Design note: We show only the lower triangle because the matrix is
# symmetric -- corr(A,B) == corr(B,A). The upper triangle adds no new info.
#
# Pearson correlation ranges from -1 to +1:
#   +1 = perfect positive linear relationship
#   -1 = perfect negative linear relationship
#    0 = no linear relationship
#
# High correlation (|r| > 0.7) between two features suggests redundancy.
# One of them may be dropped during feature selection.
# =============================================================================

def _correlation_heatmap(df, numerical_cols, figures_path):
    """
    Plot a Pearson correlation heatmap for all numerical features.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str
    figures_path : str

    Returns
    -------
    pd.DataFrame
        The full correlation matrix. Returns an empty DataFrame if
        fewer than 2 numerical columns exist.
    """
    if len(numerical_cols) < 2:
        print("  Not enough numerical columns for correlation heatmap. Skipping.")
        return pd.DataFrame()

    _apply_plot_style()

    corr_matrix = df[numerical_cols].corr(method="pearson")

    # Mask the upper triangle -- it mirrors the lower triangle exactly
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    n        = len(numerical_cols)
    fig_size = max(8, n)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size - 1))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": _FONT_SIZE - 2, "family": _FONT_FAMILY},
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Pearson Correlation Heatmap")
    plt.tight_layout()
    _save_figure(fig, figures_path, "correlation_heatmap")

    return corr_matrix


# =============================================================================
# PRIVATE FUNCTION 9: Scatter Plots
#
# PURPOSE  : Show the relationship between pairs of features, colored by class.
# PRINTS   : Nothing.
# RETURNS  : None.
# PLOTS    : Up to 3 scatter plots for the most correlated feature pairs.
#
# Design note: We auto-select the top 3 feature pairs by absolute Pearson
# correlation. This focuses attention on the most interesting relationships
# rather than showing all combinations (which can be overwhelming).
# =============================================================================

def _scatterplots(df, numerical_cols, target_col, figures_path):
    """
    Plot scatter plots for the top feature pairs, colored by target class.

    Automatically selects up to 3 feature pairs with the highest absolute
    Pearson correlation between them.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str
    target_col : str
    figures_path : str
    """
    if len(numerical_cols) < 2:
        print("  Not enough numerical columns for scatter plots. Skipping.")
        return

    # Build list of all unique pairs with their correlation value
    corr  = df[numerical_cols].corr().abs()
    pairs = []
    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            pairs.append((
                numerical_cols[i],
                numerical_cols[j],
                corr.iloc[i, j],
            ))

    # Sort pairs by correlation (descending) and take top 3
    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:3]

    _apply_plot_style()

    n_plots = len(pairs)
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(6 * n_plots, 5),
        squeeze=False,
    )
    axes = axes.flatten()

    # Assign a consistent color to each unique class value
    target_classes = df[target_col].astype(str).unique()
    palette        = sns.color_palette("Set2", len(target_classes))
    color_map      = dict(zip(target_classes, palette))

    for idx, (col_x, col_y, corr_val) in enumerate(pairs):
        for cls in target_classes:
            mask = df[target_col].astype(str) == cls
            axes[idx].scatter(
                df.loc[mask, col_x],
                df.loc[mask, col_y],
                label=cls,
                color=color_map[cls],
                alpha=0.6,
                s=30,
                edgecolors="none",
            )
        axes[idx].set_xlabel(col_x)
        axes[idx].set_ylabel(col_y)
        axes[idx].set_title(f"{col_x} vs {col_y}\n(r = {corr_val:.2f})")
        axes[idx].legend(title=target_col, fontsize=_FONT_SIZE - 3)

    plt.suptitle(
        "Scatter Plots -- Top Correlated Feature Pairs by Class",
        y=1.04,
        fontsize=_FONT_SIZE,
    )
    plt.tight_layout()
    _save_figure(fig, figures_path, "scatterplots")


# =============================================================================
# PRIVATE FUNCTION 10: Pair Plot
#
# PURPOSE  : Show all pairwise feature relationships in one grid view.
# PRINTS   : Nothing.
# RETURNS  : None.
# PLOTS    : Seaborn pairplot (scatter on off-diagonal, KDE on diagonal).
#
# Design note: Pairplot is the "big picture" view. While scatterplots
# show the 3 most correlated pairs, pairplot shows all combinations.
# We limit it to the first 5 numerical columns because pairplot with
# 10+ columns becomes unreadable and very slow to render.
# =============================================================================

def _pairplot(df, numerical_cols, target_col, figures_path):
    """
    Plot a pairwise scatter plot grid colored by target class.

    Limited to the first 5 numerical columns for readability.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str
    target_col : str
    figures_path : str
    """
    if len(numerical_cols) < 2:
        print("  Not enough numerical columns for pairplot. Skipping.")
        return

    # Limit to first 5 columns to keep plot readable and fast to render
    cols_to_plot = numerical_cols[:5]

    _apply_plot_style()

    plot_df = df[cols_to_plot + [target_col]].dropna().copy()
    plot_df[target_col] = plot_df[target_col].astype(str)

    g = sns.pairplot(
        plot_df,
        hue=target_col,
        palette="Set2",
        diag_kind="kde",          # KDE on the diagonal instead of histogram
        plot_kws={"alpha": 0.6, "s": 20},
    )
    g.fig.suptitle(
        "Pairplot -- All Feature Pair Relationships by Class",
        y=1.02,
    )

    _save_figure(g.fig, figures_path, "pairplot")


# =============================================================================
# PRIVATE FUNCTION 11: Outlier Analysis
#
# PURPOSE  : Detect and quantify outliers in numerical features.
# PRINTS   : Outlier count and percentage per column.
# RETURNS  : dict -- outlier summary per column.
# PLOTS    : None (box plots already visualize outliers as flier points).
#
# METHOD   : Interquartile Range (IQR) Rule.
#   A value is an outlier if it falls outside:
#       [Q1 - 1.5 * IQR,  Q3 + 1.5 * IQR]
#   where IQR = Q3 - Q1 (the middle 50% of the data).
#
# Design note: IQR is robust to extreme values, unlike standard deviation
# which is itself affected by outliers. It is the most common method used
# in introductory ML courses.
# =============================================================================

def _outlier_analysis(df, numerical_cols):
    """
    Detect outliers in numerical features using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str

    Returns
    -------
    dict
        Per-column dict with keys: count, percentage, lower_bound, upper_bound.
    """
    if not numerical_cols:
        print("  No numerical columns found. Skipping outlier analysis.")
        return {}

    print("=" * 60)
    print("11. OUTLIER ANALYSIS  (IQR Rule: Q1 - 1.5*IQR  to  Q3 + 1.5*IQR)")
    print("=" * 60)

    outlier_summary = {}

    for col in numerical_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        n_outliers = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        pct        = round((n_outliers / len(df)) * 100, 2)

        flag = "  <-- HIGH" if pct > 5 else ""
        print(f"  {col:<30} : {n_outliers:>4} outliers  ({pct:.2f}%){flag}")

        outlier_summary[col] = {
            "count"       : n_outliers,
            "percentage"  : pct,
            "lower_bound" : round(float(lower_bound), 4),
            "upper_bound" : round(float(upper_bound), 4),
        }

    print()
    return outlier_summary


# =============================================================================
# PRIVATE FUNCTION 12: Generate Observations
#
# PURPOSE  : Surface key statistical findings so YOU can write observations.
# PRINTS   : Auto-detected patterns -- imbalance, missing values, outliers,
#            high correlations. These are starting points, not conclusions.
# RETURNS  : dict -- structured EDA findings for the notebook observation cells.
#
# IMPORTANT: This function does NOT write your observations for you.
#            It gives you the numbers. Your job is to INTERPRET them.
#            Faculty expects YOUR analysis, not auto-generated text.
# =============================================================================

def _generate_observations(overview, missing_summary, duplicate_summary,
                            class_dist, outlier_summary, corr_matrix,
                            numerical_cols):
    """
    Surface key numerical findings from EDA to assist observation writing.

    Parameters
    ----------
    overview : dict
        From _dataset_overview().
    missing_summary : dict
        From _missing_value_analysis().
    duplicate_summary : dict
        From _duplicate_analysis().
    class_dist : dict
        From _class_distribution().
    outlier_summary : dict
        From _outlier_analysis().
    corr_matrix : pd.DataFrame
        From _correlation_heatmap(). Can be an empty DataFrame.
    numerical_cols : list of str

    Returns
    -------
    dict
        Consolidated EDA findings for use in notebook observation cells.
    """
    print("=" * 60)
    print("12. AUTO-GENERATED OBSERVATION SUMMARY")
    print("    (Use these numbers to write your manual observations.)")
    print("=" * 60)

    # -- Dataset Size ----------------------------------------------------------
    print(f"\n  Dataset       : {overview['rows']} rows x {overview['columns']} columns")

    # -- Missing Values --------------------------------------------------------
    if not missing_summary:
        print("  Missing Values: None")
    else:
        pct_dict     = missing_summary.get("Missing (%)", {})
        high_missing = {col: pct for col, pct in pct_dict.items() if pct > 5}
        if high_missing:
            print(f"\n  Columns with >5% missing values:")
            for col, pct in sorted(high_missing.items(), key=lambda x: -x[1]):
                print(f"    --> {col}: {pct:.1f}%")
        else:
            print("  Missing Values: Present but all < 5% (low severity)")

    # -- Duplicates ------------------------------------------------------------
    dup_count = duplicate_summary.get("duplicate_count", 0)
    dup_pct   = duplicate_summary.get("duplicate_percentage", 0)
    if dup_count > 0:
        print(f"\n  Duplicate Rows : {dup_count} ({dup_pct}%) --> Remove in preprocessing")
    else:
        print("  Duplicate Rows : None")

    # -- Class Balance ---------------------------------------------------------
    if class_dist.get("imbalanced"):
        print("\n  Class Balance  : IMBALANCED")
        print("    --> Use stratify=y in train_test_split()")
        print("    --> Use F1-score / Recall instead of Accuracy alone")
    else:
        print("\n  Class Balance  : Balanced")

    # -- Outliers --------------------------------------------------------------
    high_outlier_cols = [
        col for col, v in outlier_summary.items() if v["percentage"] > 5
    ]
    if high_outlier_cols:
        print(f"\n  High Outlier Columns (>5%): {high_outlier_cols}")
        print("    --> Consider IQR capping or robust scaling in preprocessing")
    else:
        print("\n  Outliers       : No column exceeds 5% outlier rate")

    # -- High Correlations -----------------------------------------------------
    high_corr_pairs = []
    if isinstance(corr_matrix, pd.DataFrame) and not corr_matrix.empty:
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = abs(corr_matrix.iloc[i, j])
                if val > 0.7:
                    high_corr_pairs.append((cols[i], cols[j], round(float(val), 3)))

        if high_corr_pairs:
            print(f"\n  Highly Correlated Pairs (|r| > 0.7):")
            for col1, col2, r in sorted(high_corr_pairs, key=lambda x: -x[2]):
                print(f"    --> {col1}  <-->  {col2}  :  r = {r}")
            print("    --> Consider dropping one from each pair in feature selection")
        else:
            print("\n  Correlations   : No pair with |r| > 0.7 found")

    print()
    print("  " + "-" * 56)
    print("  Write your detailed manual observations in the notebook")
    print("  Markdown cells BELOW each plot output.")
    print("  " + "-" * 56)
    print()

    return {
        "overview"          : overview,
        "missing_summary"   : missing_summary,
        "duplicate_summary" : duplicate_summary,
        "class_dist"        : class_dist,
        "outlier_summary"   : outlier_summary,
        "high_corr_pairs"   : high_corr_pairs,
    }


# =============================================================================
# PUBLIC FUNCTION -- The ONLY function the notebook should call.
#
# This is the FACADE: one clean interface over 11 internal EDA steps.
# The notebook sees one function. The complexity lives below.
#
# DESIGN PATTERN : Facade Pattern
# WHY            : The notebook is an orchestration document, not an
#                  implementation document. Keeping it clean lets the
#                  reader focus on the ML workflow, not the EDA details.
# =============================================================================

def classification_eda(df, target_col, figures_path="../figures/"):
    """
    Perform complete Exploratory Data Analysis for a tabular classification dataset.

    This is the ONLY function this module exposes to the notebook.
    It internally orchestrates all 12 EDA steps in order.

    Workflow
    --------
    1.  Dataset Overview
    2.  Missing Value Analysis
    3.  Duplicate Analysis
    4.  Summary Statistics
    5.  Class Distribution
    6.  Histograms
    7.  Box Plots
    8.  Correlation Heatmap
    9.  Scatter Plots
    10. Pair Plot
    11. Outlier Analysis
    12. Auto-generated Observations

    Parameters
    ----------
    df : pd.DataFrame
        The RAW dataset loaded directly from CSV.
        Do NOT preprocess the data before running EDA.
        EDA must reflect the true state of the raw data.
    target_col : str
        Name of the column containing the class label (target variable).
    figures_path : str, optional
        Directory where all EDA plots will be saved as EPS files.
        Defaults to "../figures/" -- correct when the notebook is at
        exp_01/notebooks/experiment_1.ipynb.

    Returns
    -------
    dict
        A structured dictionary of EDA findings. Assign the return value
        to a variable (e.g., eda_results) and reference specific findings
        in your Markdown observation cells.

    Example
    -------
    >>> eda_results = classification_eda(df, target_col="Loan_Status")
    >>> eda_results = classification_eda(df, target_col="Outcome",
    ...                                  figures_path="../figures/")

    Notes
    -----
    - All plots are saved as EPS files at 600 DPI.
    - Column types are auto-detected. You do not need to specify them.
    - The pairplot is limited to the first 5 numerical columns.
    - This function is designed for TABULAR CLASSIFICATION datasets only.
      For NLP or image datasets, use eda_text.py or eda_image.py (future).
    """
    print()
    print("=" * 60)
    print("  CLASSIFICATION EDA")
    print(f"  Target Column : {target_col}")
    print(f"  Figures Path  : {figures_path}")
    print("=" * 60)
    print()

    # -- Auto-detect column types ----------------------------------------------
    # We exclude the target column so it does not appear in feature analyses.
    feature_cols     = [col for col in df.columns if col != target_col]
    numerical_cols   = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"  Numerical Features   ({len(numerical_cols)}) : {numerical_cols}")
    print(f"  Categorical Features ({len(categorical_cols)}) : {categorical_cols}")
    print()

    # -- Step 1 ---------------------------------------------------------------
    overview = _dataset_overview(df)

    # -- Step 2 ---------------------------------------------------------------
    missing_summary = _missing_value_analysis(df, figures_path)

    # -- Step 3 ---------------------------------------------------------------
    duplicate_summary = _duplicate_analysis(df)

    # -- Step 4 ---------------------------------------------------------------
    _summary_statistics(df, numerical_cols, categorical_cols)

    # -- Step 5 ---------------------------------------------------------------
    class_dist = _class_distribution(df, target_col, figures_path)

    # -- Step 6 ---------------------------------------------------------------
    print("  Generating histograms...")
    _histograms(df, numerical_cols, figures_path)

    # -- Step 7 ---------------------------------------------------------------
    print("  Generating box plots...")
    _boxplots(df, numerical_cols, target_col, figures_path)

    # -- Step 8 ---------------------------------------------------------------
    print("  Generating correlation heatmap...")
    corr_matrix = _correlation_heatmap(df, numerical_cols, figures_path)

    # -- Step 9 ---------------------------------------------------------------
    print("  Generating scatter plots...")
    _scatterplots(df, numerical_cols, target_col, figures_path)

    # -- Step 10 --------------------------------------------------------------
    print("  Generating pairplot (limited to first 5 numerical columns)...")
    _pairplot(df, numerical_cols, target_col, figures_path)

    # -- Step 11 --------------------------------------------------------------
    outlier_summary = _outlier_analysis(df, numerical_cols)

    # -- Step 12 --------------------------------------------------------------
    observations = _generate_observations(
        overview, missing_summary, duplicate_summary,
        class_dist, outlier_summary, corr_matrix, numerical_cols,
    )

    print(f"  EDA Complete. All plots saved to: {figures_path}")
    print()

    return observations
