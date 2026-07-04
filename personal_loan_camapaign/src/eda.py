"""
Exploratory Data Analysis (EDA) Pipeline
========================================

This module performs comprehensive exploratory data analysis on the
Personal Loan Campaign dataset.

Features
--------
- Structured logging
- Automatic output directory creation
- Dataset summary generation
- Univariate analysis
- Bivariate analysis
- Correlation heatmap
- Production-quality modular implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import BASE_DIR, RAW_DATA_PATH
from src.logger import setup_app_logger

# =============================================================================
# Logging
# =============================================================================

logger = setup_app_logger("eda")

# =============================================================================
# Paths
# =============================================================================

EDA_DIR = Path(BASE_DIR) / "assets" / "eda"

EDA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =============================================================================
# Plot Configuration
# =============================================================================

sns.set_theme(
    style="whitegrid",
)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "figure.titlesize": 14,
    }
)

# =============================================================================
# Dataset Loading
# =============================================================================


def load_dataset() -> pd.DataFrame:
    """
    Load the raw dataset.

    Returns
    -------
    pd.DataFrame
        Raw customer dataset.
    """

    logger.info("=" * 80)
    logger.info("Loading Dataset")
    logger.info("=" * 80)

    try:

        df = pd.read_csv(RAW_DATA_PATH)

        logger.info(
            "Dataset loaded successfully."
        )

        logger.info(
            "Shape : %s",
            df.shape,
        )

        return df

    except Exception:

        logger.exception(
            "Unable to load dataset."
        )

        raise


# =============================================================================
# Dataset Preparation
# =============================================================================


def prepare_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare dataset for visualization.

    - Rename ZIP Code
    - Remove ID columns
    - Fix negative Experience values
    """

    logger.info("Preparing dataset...")

    df = df.copy()

    if "ZIP Code" in df.columns:

        df.rename(
            columns={
                "ZIP Code": "ZIPCode",
            },
            inplace=True,
        )

    columns_to_drop = [
        column
        for column in [
            "ID",
            "ZIPCode",
        ]
        if column in df.columns
    ]

    if columns_to_drop:

        logger.info(
            "Dropping columns: %s",
            columns_to_drop,
        )

        df.drop(
            columns=columns_to_drop,
            inplace=True,
        )

    if "Experience" in df.columns:

        negative_count = (
            df["Experience"] < 0
        ).sum()

        if negative_count > 0:

            logger.info(
                "Correcting %d negative Experience values.",
                negative_count,
            )

            df["Experience"] = (
                df["Experience"].abs()
            )

    return df


# =============================================================================
# Dataset Summary
# =============================================================================


def generate_dataset_summary(
    df: pd.DataFrame,
) -> None:
    """
    Save dataset summary to CSV files.
    """

    logger.info("=" * 80)
    logger.info("Generating Dataset Summary")
    logger.info("=" * 80)

    summary = pd.DataFrame(
        {
            "Data Type": df.dtypes,
            "Missing Values": df.isna().sum(),
            "Unique Values": df.nunique(),
        }
    )

    summary.to_csv(
        EDA_DIR / "dataset_summary.csv"
    )

    df.describe(
        include="all"
    ).transpose().to_csv(
        EDA_DIR / "dataset_statistics.csv"
    )

    logger.info(
        "Dataset summary exported."
    )


# =============================================================================
# Feature Identification
# =============================================================================


def get_feature_lists(
    df: pd.DataFrame,
) -> Tuple[list, list]:
    """
    Return continuous and categorical features.
    """

    continuous_features = [
        column
        for column in [
            "Age",
            "Experience",
            "Income",
            "CCAvg",
            "Mortgage",
        ]
        if column in df.columns
    ]

    categorical_features = [
        column
        for column in [
            "Family",
            "Education",
            "Securities_Account",
            "CD_Account",
            "Online",
            "CreditCard",
        ]
        if column in df.columns
    ]

    return (
        continuous_features,
        categorical_features,
    )


# =============================================================================
# Plot Utilities
# =============================================================================


def save_plot(
    filename: str,
) -> None:
    """
    Save and close current matplotlib figure.
    """

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def annotate_countplot(
    ax,
) -> None:
    """
    Add count labels to a seaborn countplot.
    """

    for patch in ax.patches:

        height = patch.get_height()

        ax.annotate(
            f"{int(height)}",
            (
                patch.get_x()
                + patch.get_width() / 2,
                height,
            ),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )

# =============================================================================
# Univariate Analysis
# =============================================================================

def plot_target_distribution(
    df: pd.DataFrame,
) -> None:
    """
    Plot the target variable distribution.
    """

    logger.info("Generating target variable distribution...")

    if "Personal_Loan" not in df.columns:
        logger.warning("Target column 'Personal_Loan' not found.")
        return

    plt.figure(figsize=(6, 4))

    ax = sns.countplot(
        data=df,
        x="Personal_Loan",
        hue="Personal_Loan",
        palette="viridis",
        legend=False,
    )

    annotate_countplot(ax)

    plt.title(
        "Target Variable Distribution",
        fontweight="bold",
    )

    plt.xlabel("Personal Loan")
    plt.ylabel("Customer Count")

    save_plot(
        "univariate_target_distribution.png"
    )


# =============================================================================
# Continuous Feature Analysis
# =============================================================================

def plot_continuous_feature(
    df: pd.DataFrame,
    feature: str,
) -> None:
    """
    Plot histogram and boxplot for a continuous feature.
    """

    logger.info("Plotting %s...", feature)

    fig, (ax_box, ax_hist) = plt.subplots(
        2,
        sharex=True,
        figsize=(8, 5),
        gridspec_kw={
            "height_ratios": (0.2, 0.8),
        },
    )

    sns.boxplot(
        x=df[feature],
        ax=ax_box,
        color="skyblue",
    )

    sns.histplot(
        data=df,
        x=feature,
        kde=True,
        ax=ax_hist,
        color="teal",
    )

    ax_box.set(
        xlabel="",
    )

    ax_box.set_title(
        f"{feature} Distribution",
        fontweight="bold",
    )

    ax_hist.set_xlabel(feature)
    ax_hist.set_ylabel("Count")

    save_plot(
        f"univariate_{feature.lower()}.png"
    )


def plot_all_continuous_features(
    df: pd.DataFrame,
    continuous_features: list[str],
) -> None:
    """
    Plot all continuous variables.
    """

    logger.info("=" * 80)
    logger.info("Generating Continuous Variable Analysis")
    logger.info("=" * 80)

    for feature in continuous_features:

        if feature not in df.columns:
            continue

        plot_continuous_feature(
            df,
            feature,
        )


# =============================================================================
# Categorical Feature Analysis
# =============================================================================

def plot_categorical_feature(
    df: pd.DataFrame,
    feature: str,
) -> None:
    """
    Plot countplot for a categorical feature.
    """

    logger.info("Plotting %s...", feature)

    plt.figure(figsize=(6, 4))

    ax = sns.countplot(
        data=df,
        x=feature,
        hue=feature,
        palette="magma",
        legend=False,
    )

    annotate_countplot(ax)

    plt.title(
        f"{feature} Distribution",
        fontweight="bold",
    )

    plt.xlabel(feature)
    plt.ylabel("Customer Count")

    save_plot(
        f"univariate_{feature.lower()}.png"
    )


def plot_all_categorical_features(
    df: pd.DataFrame,
    categorical_features: list[str],
) -> None:
    """
    Plot all categorical variables.
    """

    logger.info("=" * 80)
    logger.info("Generating Categorical Variable Analysis")
    logger.info("=" * 80)

    for feature in categorical_features:

        if feature not in df.columns:
            continue

        plot_categorical_feature(
            df,
            feature,
        )


# =============================================================================
# Complete Univariate Analysis
# =============================================================================

def run_univariate_analysis(
    df: pd.DataFrame,
) -> None:
    """
    Execute the complete univariate analysis pipeline.
    """

    logger.info("=" * 80)
    logger.info("Starting Univariate Analysis")
    logger.info("=" * 80)

    continuous_features, categorical_features = get_feature_lists(df)

    plot_target_distribution(df)

    plot_all_continuous_features(
        df,
        continuous_features,
    )

    plot_all_categorical_features(
        df,
        categorical_features,
    )

    logger.info("Univariate analysis completed successfully.")

# =============================================================================
# Bivariate Analysis
# =============================================================================

def plot_continuous_vs_target(
    df: pd.DataFrame,
    feature: str,
) -> None:
    """
    Plot a continuous feature against the target variable.
    """

    if "Personal_Loan" not in df.columns:
        logger.warning("Target column 'Personal_Loan' not found.")
        return

    logger.info("Generating %s vs Personal_Loan...", feature)

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="Personal_Loan",
        y=feature,
        hue="Personal_Loan",
        palette="Set2",
        legend=False,
    )

    plt.title(
        f"{feature} vs Personal Loan",
        fontweight="bold",
    )

    plt.xlabel("Personal Loan")
    plt.ylabel(feature)

    save_plot(
        f"bivariate_box_{feature.lower()}_vs_target.png"
    )


def plot_categorical_vs_target(
    df: pd.DataFrame,
    feature: str,
) -> None:
    """
    Plot a categorical feature against the target variable.
    """

    if "Personal_Loan" not in df.columns:
        logger.warning("Target column 'Personal_Loan' not found.")
        return

    logger.info("Generating %s vs Personal_Loan...", feature)

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x=feature,
        hue="Personal_Loan",
        palette="coolwarm",
    )

    plt.title(
        f"{feature} vs Personal Loan",
        fontweight="bold",
    )

    plt.xlabel(feature)
    plt.ylabel("Customer Count")

    save_plot(
        f"bivariate_bar_{feature.lower()}_vs_target.png"
    )


def run_bivariate_analysis(
    df: pd.DataFrame,
) -> None:
    """
    Execute complete bivariate analysis.
    """

    logger.info("=" * 80)
    logger.info("Starting Bivariate Analysis")
    logger.info("=" * 80)

    continuous_features = [
        "Income",
        "CCAvg",
        "Mortgage",
        "Age",
    ]

    categorical_features = [
        "Education",
        "Family",
        "CD_Account",
    ]

    for feature in continuous_features:

        if feature in df.columns:

            plot_continuous_vs_target(
                df,
                feature,
            )

    for feature in categorical_features:

        if feature in df.columns:

            plot_categorical_vs_target(
                df,
                feature,
            )

    logger.info("Bivariate analysis completed.")


# =============================================================================
# Correlation Analysis
# =============================================================================

def plot_correlation_matrix(
    df: pd.DataFrame,
) -> None:
    """
    Generate and save the feature correlation heatmap.
    """

    logger.info("=" * 80)
    logger.info("Generating Correlation Matrix")
    logger.info("=" * 80)

    plt.figure(figsize=(12, 9))

    correlation = df.corr(
        numeric_only=True,
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="Spectral",
        fmt=".2f",
        linewidths=0.5,
        annot_kws={
            "size": 8,
        },
    )

    plt.title(
        "Feature Correlation Matrix",
        fontweight="bold",
    )

    save_plot(
        "correlation_matrix.png"
    )

    logger.info("Correlation matrix saved.")


# =============================================================================
# Main EDA Pipeline
# =============================================================================

def run_comprehensive_eda() -> pd.DataFrame:
    """
    Execute the complete EDA pipeline.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset used for visualization.
    """

    logger.info("=" * 80)
    logger.info("Starting Comprehensive EDA Pipeline")
    logger.info("=" * 80)

    try:

        # -------------------------------------------------------------
        # Load dataset
        # -------------------------------------------------------------
        df = load_dataset()

        # -------------------------------------------------------------
        # Prepare dataset
        # -------------------------------------------------------------
        df = prepare_dataset(df)

        # -------------------------------------------------------------
        # Dataset summary
        # -------------------------------------------------------------
        generate_dataset_summary(df)

        # -------------------------------------------------------------
        # Univariate Analysis
        # -------------------------------------------------------------
        run_univariate_analysis(df)

        # -------------------------------------------------------------
        # Bivariate Analysis
        # -------------------------------------------------------------
        run_bivariate_analysis(df)

        # -------------------------------------------------------------
        # Correlation Matrix
        # -------------------------------------------------------------
        plot_correlation_matrix(df)

        logger.info("=" * 80)
        logger.info("EDA Pipeline Completed Successfully")
        logger.info("=" * 80)

        return df

    except Exception:

        logger.exception(
            "EDA pipeline failed."
        )

        raise


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    """
    Entry point for executing the EDA pipeline.
    """

    run_comprehensive_eda()


if __name__ == "__main__":
    main()