"""
Model evaluation script.

Evaluates the trained model on the held-out test dataset.
"""

import logging

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import (
    TEST_DATA_PATH,
    MODEL_PATH,
    TARGET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_test_data():

    df = pd.read_csv(TEST_DATA_PATH)

    X_test = df.drop(columns=[TARGET])
    y_test = df[TARGET]

    return X_test, y_test


def load_model():

    logger.info("Loading trained model...")

    return joblib.load(MODEL_PATH)


def evaluate():

    X_test, y_test = load_test_data()

    model = load_model()

    predictions = model.predict(X_test)

    logger.info("=" * 60)
    logger.info("Evaluation Metrics")
    logger.info("=" * 60)

    logger.info(
        "Accuracy : %.4f",
        accuracy_score(y_test, predictions),
    )

    logger.info(
        "Precision : %.4f",
        precision_score(y_test, predictions),
    )

    logger.info(
        "Recall : %.4f",
        recall_score(y_test, predictions),
    )

    logger.info(
        "F1 Score : %.4f",
        f1_score(y_test, predictions),
    )

    logger.info("\nConfusion Matrix")
    logger.info("\n%s", confusion_matrix(y_test, predictions))

    logger.info("\nClassification Report")
    logger.info(
        "\n%s",
        classification_report(
            y_test,
            predictions,
        ),
    )


if __name__ == "__main__":
    evaluate()