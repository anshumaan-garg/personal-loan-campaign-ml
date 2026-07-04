"""
Configuration file for the Personal Loan Campaign project.

Contains:
- Project paths
- Random seed
- Train/Test split configuration
- W&B configuration
- Optuna configuration
"""

from pathlib import Path

# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "Loan_Modelling.csv"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "decision_tree_model.joblib"

# =============================================================================
# RANDOM SEED
# =============================================================================

RANDOM_STATE = 42

# =============================================================================
# TRAIN TEST SPLIT
# =============================================================================

TEST_SIZE = 0.30

TARGET = "Personal_Loan"

# =============================================================================
# W&B
# =============================================================================

WANDB_PROJECT = "personal-loan-campaign"

WANDB_ENTITY = None
# Add your username/team here if required

WANDB_JOB_TYPE = "training"

# =============================================================================
# OPTUNA
# =============================================================================

N_TRIALS = 50

OPTUNA_DIRECTION = "maximize"

OPTUNA_METRIC = "f1"

# =============================================================================
# Decision Tree Search Space
# =============================================================================

MAX_DEPTH = (2, 20)

MIN_SAMPLES_SPLIT = (2, 30)

MIN_SAMPLES_LEAF = (1, 20)

CRITERION = [
    "gini",
    "entropy",
    "log_loss",
]

SPLITTER = [
    "best",
    "random",
]