"""Central paths and constants for GridGuard."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_FULL = RAW_DIR / "sgcc_full.csv"
RAW_SMALL = RAW_DIR / "sgcc_small.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_CACHE = PROCESSED_DIR / "features_full.csv"
FIXTURE_CSV = REPO_ROOT / "tests" / "data" / "fixture.csv"

MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
WEB_PUBLIC_MODEL = REPO_ROOT / "web" / "public" / "model"
WEB_DATA = REPO_ROOT / "web" / "src" / "data"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Weekday index (0=Sunday, JS Date.getDay convention) of the first day of the
# SGCC series (2014-01-01, a Wednesday -> 3).
DATASET_DOW0 = 3
