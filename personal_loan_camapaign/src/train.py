"""
Production Training Pipeline for Personal Loan Campaign

Features
--------
- Load training data only once
- Structured logging
- Weights & Biases experiment tracking
- Utility functions for evaluation
- Modular architecture
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import wandb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from config import (
    MODEL_PATH,
    N_TRIALS,
    RANDOM_STATE,
    TARGET,
    TRAIN_DATA_PATH,
    WANDB_ENTITY,
    WANDB_JOB_TYPE,
    WANDB_PROJECT,
)

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# =============================================================================
# Paths
# =============================================================================

MODEL_DIR = MODEL_PATH.parent

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"

OPTUNA_DB_PATH = MODEL_DIR / "optuna.db"

OPTUNA_STORAGE = f"sqlite:///{OPTUNA_DB_PATH}"

# =============================================================================
# Global Variables
# =============================================================================

X_train = None
y_train = None

# =============================================================================
# Data Loading
# =============================================================================


def load_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed training dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target labels.
    """

    logger.info("Loading training dataset...")

    df = pd.read_csv(TRAIN_DATA_PATH)

    logger.info("Dataset shape : %s", df.shape)

    X = df.drop(columns=[TARGET])

    y = df[TARGET]

    logger.info("Number of features : %d", X.shape[1])

    logger.info("Target distribution")

    logger.info("\n%s", y.value_counts())

    return X, y


def initialize_dataset() -> None:
    """
    Load dataset only once.

    Stores data in global variables to avoid repeatedly
    reading the CSV during Optuna optimization.
    """

    global X_train
    global y_train

    if X_train is None or y_train is None:
        X_train, y_train = load_training_data()

        logger.info("Training data cached successfully.")


# =============================================================================
# Metric Utilities
# =============================================================================


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def generate_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Return classification report as DataFrame.
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "class",
            }
        )
    )

    return report_df


# =============================================================================
# Weights & Biases
# =============================================================================


def initialize_wandb():
    """
    Initialize W&B experiment.
    """

    logger.info("Initializing Weights & Biases...")

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type=WANDB_JOB_TYPE,
        config={
            "model": "DecisionTreeClassifier",
            "random_state": RANDOM_STATE,
            "trials": N_TRIALS,
        },
        reinit=True,
    )

    logger.info("W&B initialized.")

    return run


# =============================================================================
# Feature Importance
# =============================================================================


def get_feature_importance(
    model: DecisionTreeClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Return sorted feature importance dataframe.
    """

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    )

    importance = (
        importance.sort_values(
            by="Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance


def log_feature_importance(
    model: DecisionTreeClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Log feature importance to W&B.
    """

    importance = get_feature_importance(
        model,
        feature_names,
    )

    logger.info("=" * 70)
    logger.info("Top 10 Important Features")
    logger.info("=" * 70)

    logger.info("\n%s", importance.head(10))

    wandb.log(
        {
            "feature_importance": wandb.Table(
                dataframe=importance,
            )
        }
    )

    return importance


# =============================================================================
# Optuna Objective Function
# =============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Uses Stratified K-Fold cross-validation and reports intermediate
    fold scores to enable Optuna pruning.
    """

    params = {
        "criterion": trial.suggest_categorical(
            "criterion",
            ["gini", "entropy", "log_loss"],
        ),
        "splitter": trial.suggest_categorical(
            "splitter",
            ["best", "random"],
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            20,
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            2,
            20,
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            1,
            10,
        ),
        "max_features": trial.suggest_categorical(
            "max_features",
            [
                None,
                "sqrt",
                "log2",
            ],
        ),
        "ccp_alpha": trial.suggest_float(
            "ccp_alpha",
            0.0,
            0.02,
        ),
        "random_state": RANDOM_STATE,
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[valid_idx]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[valid_idx]

        model = DecisionTreeClassifier(**params)

        model.fit(X_tr, y_tr)

        predictions = model.predict(X_val)

        fold_f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        fold_scores.append(fold_f1)

        trial.report(
            fold_f1,
            step=fold,
        )

        if trial.should_prune():

            logger.info(
                "Trial %d pruned at fold %d",
                trial.number,
                fold,
            )

            raise optuna.TrialPruned()

    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))

    trial.set_user_attr(
        "cv_f1_mean",
        mean_f1,
    )

    trial.set_user_attr(
        "cv_f1_std",
        std_f1,
    )

    logger.info(
        "Trial %03d | Mean F1 = %.4f | Std = %.4f",
        trial.number,
        mean_f1,
        std_f1,
    )

    wandb.log(
        {
            "trial": trial.number,
            "cv_f1": mean_f1,
            "cv_f1_std": std_f1,
            **params,
        }
    )

    return mean_f1


# =============================================================================
# Optuna Hyperparameter Optimization
# =============================================================================

def run_optuna() -> optuna.study.Study:
    """
    Run Optuna hyperparameter optimization.

    Returns
    -------
    optuna.study.Study
        Optimized study object.
    """

    logger.info("=" * 80)
    logger.info("Starting Optuna Hyperparameter Optimization")
    logger.info("=" * 80)

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=2,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name="decision_tree_optimization",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    logger.info("=" * 80)
    logger.info("Optimization Finished")
    logger.info("=" * 80)

    logger.info(
        "Completed Trials : %d",
        len(study.trials),
    )

    logger.info(
        "Best CV F1 : %.4f",
        study.best_value,
    )

    logger.info("Best Parameters")

    for key, value in study.best_params.items():

        logger.info(
            "  %-20s : %s",
            key,
            value,
        )

    wandb.summary["best_cv_f1"] = study.best_value
    wandb.summary["completed_trials"] = len(study.trials)

    return study


# =============================================================================
# Save Best Hyperparameters
# =============================================================================

def save_best_parameters(
    study: optuna.study.Study,
) -> None:
    """
    Save best hyperparameters as JSON.
    """

    with open(
        BEST_PARAMS_PATH,
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            study.best_params,
            fp,
            indent=4,
        )

    logger.info(
        "Best parameters saved to %s",
        BEST_PARAMS_PATH,
    )

# =============================================================================
# Train Final Model
# =============================================================================

def train_best_model(
    study: optuna.study.Study,
) -> DecisionTreeClassifier:
    """
    Train the final Decision Tree model using the best
    hyperparameters found by Optuna.
    """

    logger.info("=" * 80)
    logger.info("Training Final Model")
    logger.info("=" * 80)

    best_params = study.best_params.copy()
    best_params["random_state"] = RANDOM_STATE

    logger.info("Best Hyperparameters")

    for key, value in best_params.items():
        logger.info("%-20s : %s", key, value)

    model = DecisionTreeClassifier(**best_params)

    model.fit(
        X_train,
        y_train,
    )

    logger.info("Final model trained successfully.")

    return model


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_model(
    model: DecisionTreeClassifier,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate the trained model.

    Returns
    -------
    metrics : dict
        Accuracy, Precision, Recall and F1.

    report_df : DataFrame
        Classification report.
    """

    logger.info("=" * 80)
    logger.info("Evaluating Final Model")
    logger.info("=" * 80)

    predictions = model.predict(X_train)

    metrics = calculate_metrics(
        y_train,
        predictions,
    )

    logger.info("Training Metrics")

    for metric, value in metrics.items():

        logger.info(
            "%-15s : %.4f",
            metric,
            value,
        )

    report_df = generate_classification_report(
        y_train,
        predictions,
    )

    logger.info("\nClassification Report")
    logger.info("\n%s", report_df)

    cm = confusion_matrix(
        y_train,
        predictions,
    )

    logger.info("\nConfusion Matrix")
    logger.info("\n%s", cm)

    # ---------------------------------------------------------
    # W&B Logging
    # ---------------------------------------------------------

    wandb.log(metrics)

    wandb.log(
        {
            "classification_report": wandb.Table(
                dataframe=report_df
            )
        }
    )

    wandb.log(
        {
            "confusion_matrix":
                wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_train,
                    preds=predictions,
                )
        }
    )

    wandb.summary["train_accuracy"] = metrics["accuracy"]
    wandb.summary["train_precision"] = metrics["precision"]
    wandb.summary["train_recall"] = metrics["recall"]
    wandb.summary["train_f1"] = metrics["f1"]

    return metrics, report_df


# =============================================================================
# Feature Importance
# =============================================================================

def log_model_feature_importance(
    model: DecisionTreeClassifier,
) -> pd.DataFrame:
    """
    Compute and log feature importance.
    """

    importance_df = get_feature_importance(
        model,
        X_train.columns.tolist(),
    )

    logger.info("=" * 80)
    logger.info("Top 10 Important Features")
    logger.info("=" * 80)

    logger.info("\n%s", importance_df.head(10))

    wandb.log(
        {
            "feature_importance_table":
                wandb.Table(
                    dataframe=importance_df
                )
        }
    )

    # Optional bar chart
    wandb.log(
        {
            "feature_importance":
                wandb.plot.bar(
                    wandb.Table(
                        dataframe=importance_df
                    ),
                    "Feature",
                    "Importance",
                    title="Decision Tree Feature Importance",
                )
        }
    )

    return importance_df


# =============================================================================
# Study Summary
# =============================================================================

def log_study_statistics(
    study: optuna.study.Study,
) -> None:
    """
    Log Optuna study statistics.
    """

    logger.info("=" * 80)
    logger.info("Optuna Study Summary")
    logger.info("=" * 80)

    logger.info(
        "Finished Trials : %d",
        len(study.trials),
    )

    logger.info(
        "Best CV F1 : %.4f",
        study.best_value,
    )

    logger.info("Best Parameters")

    for key, value in study.best_params.items():

        logger.info(
            "%-20s : %s",
            key,
            value,
        )

    wandb.summary["best_cv_score"] = study.best_value
    wandb.summary["number_of_trials"] = len(study.trials)

# =============================================================================
# Save Model
# =============================================================================

def save_model(
    model: DecisionTreeClassifier,
) -> None:
    """
    Save the trained model to disk.
    """

    logger.info("=" * 80)
    logger.info("Saving Model")
    logger.info("=" * 80)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    logger.info(
        "Model saved successfully to %s",
        MODEL_PATH,
    )


# =============================================================================
# W&B Artifact Logging
# =============================================================================

def log_model_artifact() -> None:
    """
    Upload the trained model as a W&B artifact.
    """

    logger.info("Uploading model artifact to W&B...")

    artifact = wandb.Artifact(
        name="decision_tree_model",
        type="model",
        description="Decision Tree optimized using Optuna",
    )

    artifact.add_file(str(MODEL_PATH))

    if BEST_PARAMS_PATH.exists():
        artifact.add_file(str(BEST_PARAMS_PATH))

    wandb.log_artifact(artifact)

    logger.info("Model artifact uploaded successfully.")


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    """
    Execute the complete training pipeline.
    """

    logger.info("=" * 80)
    logger.info("Personal Loan Campaign - Training Pipeline")
    logger.info("=" * 80)

    run = None

    try:

        # -------------------------------------------------------------
        # Initialize W&B
        # -------------------------------------------------------------
        run = initialize_wandb()

        # -------------------------------------------------------------
        # Load training data (only once)
        # -------------------------------------------------------------
        initialize_dataset()

        # -------------------------------------------------------------
        # Hyperparameter Optimization
        # -------------------------------------------------------------
        study = run_optuna()

        # -------------------------------------------------------------
        # Save Best Parameters
        # -------------------------------------------------------------
        save_best_parameters(study)

        # -------------------------------------------------------------
        # Train Final Model
        # -------------------------------------------------------------
        model = train_best_model(study)

        # -------------------------------------------------------------
        # Evaluate Model
        # -------------------------------------------------------------
        metrics, report_df = evaluate_model(model)

        # -------------------------------------------------------------
        # Feature Importance
        # -------------------------------------------------------------
        log_model_feature_importance(model)

        # -------------------------------------------------------------
        # Study Statistics
        # -------------------------------------------------------------
        log_study_statistics(study)

        # -------------------------------------------------------------
        # Save Model
        # -------------------------------------------------------------
        save_model(model)

        # -------------------------------------------------------------
        # Upload Artifact
        # -------------------------------------------------------------
        log_model_artifact()

        logger.info("=" * 80)
        logger.info("Training Completed Successfully")
        logger.info("=" * 80)

    except Exception:

        logger.exception("Training pipeline failed.")
        raise

    finally:

        if run is not None:
            run.finish()

        logger.info("Weights & Biases run closed.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()