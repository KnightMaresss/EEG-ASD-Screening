"""Configuration management for the application."""

from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models directory
MODELS_DIR = ROOT_DIR / "models"

# Database path
DATABASE_DIR = DATA_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "predictions.db"

# Ensure directories exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# App settings
APP_NAME = "EEG ASD Screening Tool"
APP_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Signal processing constants
# ---------------------------------------------------------------------------
EPOCH_DURATION_S = 2.0            # seconds per epoch
MAX_EPOCHS_PER_SUBJECT = 40       # cap epochs per subject during training/inference
DEFAULT_SFREQ = 512.0             # Sheffield dataset sampling rate (Hz)
BANDPASS_LOW_HZ = 1.0             # high-pass cutoff
BANDPASS_HIGH_HZ = 50.0           # low-pass cutoff

# ---------------------------------------------------------------------------
# Model / training constants
# ---------------------------------------------------------------------------
CV_FOLDS = 5
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RANDOM_STATE = 42
# ---------------------------------------------------------------------------
# Plot constants
# ---------------------------------------------------------------------------
PLOT_CHANNEL_COUNT = 8
PLOT_SAMPLE_COUNT = 5000