"""
Preprocessing module.

Responsibilities
----------------
1. Read raw dataset.
2. Clean data.
3. Fix negative Experience values.
4. Drop ID column.
5. Train/Test split.
6. Save processed datasets.
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET,
)


def load_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load dataset.

    Parameters
    ----------
    filepath : Path

    Returns
    -------
    DataFrame
    """

    return pd.read_csv(filepath)


def fix_negative_experience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace negative Experience values with absolute values.

    Notebook logic:
    Negative experience is considered a data-entry issue.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame
    """

    df["Experience"] = df["Experience"].abs()

    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns not useful for prediction.
    """

    columns_to_drop = ["ID"]

    return df.drop(columns=columns_to_drop)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute preprocessing pipeline.
    """

    df = fix_negative_experience(df)

    df = drop_columns(df)

    return df


def split_data(df: pd.DataFrame):
    """
    Split dataset into train and test.
    """

    X = df.drop(columns=[TARGET])

    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Save processed datasets.
    """

    train = X_train.copy()

    train[TARGET] = y_train.values

    test = X_test.copy()

    test[TARGET] = y_test.values

    TRAIN_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        TRAIN_DATA_PATH,
        index=False,
    )

    test.to_csv(
        TEST_DATA_PATH,
        index=False,
    )


def main():

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    df = load_data()

    print(df.shape)

    print("=" * 60)
    print("Cleaning Dataset")
    print("=" * 60)

    df = preprocess(df)

    print(df.shape)

    X_train, X_test, y_train, y_test = split_data(df)

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print()

    print("Train Shape :", X_train.shape)

    print("Test Shape  :", X_test.shape)

    print()

    print("Processed datasets saved successfully.")


if __name__ == "__main__":
    main()