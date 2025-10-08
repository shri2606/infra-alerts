"""
CloudInfraAI Configuration
==========================

Central configuration file for all project settings including file paths,
model parameters, and system configurations.

Author: CloudInfraAI Team
Date: 2024
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "saved_models"
MODEL_ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_OUTPUT_DIR = OUTPUTS_DIR / "analysis"
FEATURES_OUTPUT_DIR = OUTPUTS_DIR / "features"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Scripts
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Dataset configuration
class DataConfig:
    """Data processing configuration."""

    # Input dataset
    RAW_DATASET_PATH = RAW_DATA_DIR / "OpenStack_2k.log_structured.csv"

    # Processed outputs
    PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "processed_dataset_with_labels.csv"
    ENGINEERED_FEATURES_PATH = PROCESSED_DATA_DIR / "engineered_features.csv"

    # Output directories
    ANALYSIS_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR
    FEATURES_OUTPUT_DIR = FEATURES_OUTPUT_DIR

    # Analysis outputs
    ANALYSIS_RESULTS_PATH = ANALYSIS_OUTPUT_DIR / "analysis_results.json"
    ANALYSIS_SUMMARY_PATH = ANALYSIS_OUTPUT_DIR / "analysis_summary.txt"
    ANALYSIS_PLOTS_PATH = ANALYSIS_OUTPUT_DIR / "analysis_visualizations.png"

    # Feature outputs
    TRAIN_DATA_PATH = FEATURES_OUTPUT_DIR / "train_data.pt"
    VAL_DATA_PATH = FEATURES_OUTPUT_DIR / "val_data.pt"
    TEST_DATA_PATH = FEATURES_OUTPUT_DIR / "test_data.pt"
    ENCODERS_PATH = FEATURES_OUTPUT_DIR / "encoders.json"
    SCALERS_PATH = FEATURES_OUTPUT_DIR / "scalers.json"
    FEATURE_CONFIG_PATH = FEATURES_OUTPUT_DIR / "feature_config.json"
    FEATURE_STATS_PATH = FEATURES_OUTPUT_DIR / "feature_stats.json"

class ModelConfig:
    """Model architecture and training configuration."""

    # Model architecture
    MODEL_TYPE = "transformer"  # Options: "transformer", "lstm"
    D_MODEL = 128               # Transformer dimension (optimized for M2 Pro)
    NUM_LAYERS = 2              # Number of transformer/LSTM layers
    NUM_HEADS = 4               # Number of attention heads
    DROPOUT = 0.1               # Dropout rate

    # Sequence parameters
    SEQUENCE_LENGTH = 50        # Maximum events per sequence
    WINDOW_SIZE_MINUTES = 5     # Time window for sequence creation

    # Training parameters
    BATCH_SIZE = 32             # Batch size (optimized for M2 Pro 16GB)
    LEARNING_RATE = 1e-4        # Initial learning rate
    NUM_EPOCHS = 100            # Maximum training epochs
    EARLY_STOPPING_PATIENCE = 10  # Early stopping patience

    # Data splits
    TEST_SIZE = 0.2             # Test set proportion
    VAL_SIZE = 0.2              # Validation set proportion
    RANDOM_STATE = 42           # Random seed for reproducibility

    # Anomaly detection thresholds
    MEMORY_SPIKE_THRESHOLD = 2560  # MB
    API_LATENCY_THRESHOLD = 0.5    # seconds

    # Model saving
    MODEL_SAVE_PATH = MODELS_DIR / "anomaly_detector_v1.pth"
    CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
    BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

class DashboardConfig:
    """Streamlit dashboard configuration."""

    # Dashboard settings
    PAGE_TITLE = "CloudInfraAI - Infrastructure Monitoring"
    PAGE_ICON = "🔍"
    LAYOUT = "wide"

    # Refresh settings
    AUTO_REFRESH_SECONDS = 5    # Dashboard auto-refresh interval
    MAX_EVENTS_DISPLAY = 100    # Maximum events to show in timeline

    # Chart settings
    CHART_HEIGHT = 400          # Default chart height
    COLOR_NORMAL = "#2E8B57"    # Green for normal events
    COLOR_ANOMALY = "#DC143C"   # Red for anomalies
    COLOR_WARNING = "#FF8C00"   # Orange for warnings

    # Data refresh paths
    LIVE_DATA_PATH = PROCESSED_DATA_DIR / "live_data.csv"
    LIVE_PREDICTIONS_PATH = OUTPUTS_DIR / "live_predictions.json"

class AlertingConfig:
    """Alerting and notification configuration."""

    # Slack configuration
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    SLACK_CHANNEL = "#cloudinfra-alerts"
    SLACK_USERNAME = "CloudInfraAI Bot"
    SLACK_ICON_EMOJI = ":warning:"

    # Alert thresholds
    ALERT_CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence for alerts
    ALERT_COOLDOWN_MINUTES = 5         # Minimum time between similar alerts

    # Alert severity levels
    SEVERITY_CRITICAL = "critical"      # System down, major outages
    SEVERITY_HIGH = "high"             # Performance degradation
    SEVERITY_MEDIUM = "medium"         # Minor issues, warnings
    SEVERITY_LOW = "low"               # Informational alerts

class SystemConfig:
    """System and hardware configuration."""

    # Hardware detection
    import platform
    PLATFORM = platform.system()       # Darwin, Linux, Windows

    # PyTorch device configuration
    import torch
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")    # Apple M2 Pro acceleration
        DEVICE_NAME = "Apple MPS"
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")   # NVIDIA GPU
        DEVICE_NAME = "CUDA GPU"
    else:
        DEVICE = torch.device("cpu")    # CPU fallback
        DEVICE_NAME = "CPU"

    # Memory and performance
    MAX_MEMORY_GB = 16                  # Assuming M2 Pro 16GB
    NUM_WORKERS = 4                     # Data loader workers

    # Logging
    LOG_LEVEL = "INFO"                  # Logging level
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "cloudinfra_ai.log"

class FeatureConfig:
    """Feature engineering configuration."""

    # Sequence parameters
    SEQUENCE_LENGTH = ModelConfig.SEQUENCE_LENGTH
    WINDOW_SIZE_MINUTES = ModelConfig.WINDOW_SIZE_MINUTES
    WINDOW_OVERLAP = 0                  # No overlap between windows
    MIN_EVENTS_PER_WINDOW = 1           # Minimum events to create sequence

    # Anomaly thresholds
    MEMORY_SPIKE_THRESHOLD = ModelConfig.MEMORY_SPIKE_THRESHOLD
    API_LATENCY_THRESHOLD = ModelConfig.API_LATENCY_THRESHOLD

    # Feature processing
    PADDING_TOKEN = "PAD"               # Padding token for sequences
    UNKNOWN_TOKEN = "UNK"               # Unknown token for categories
    MAX_VOCAB_SIZE = 1000               # Maximum vocabulary size

    # Normalization
    NUMERICAL_SCALER = "StandardScaler" # StandardScaler or MinMaxScaler
    CLIP_OUTLIERS = True                # Clip extreme values
    TIME_DELTA_MAX = 300                # Maximum time delta (5 minutes)

# Ensure directories exist
def create_directories():
    """Create all necessary directories."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, MODEL_ARTIFACTS_DIR, ModelConfig.CHECKPOINT_DIR,
        OUTPUTS_DIR, ANALYSIS_OUTPUT_DIR, FEATURES_OUTPUT_DIR,
        LOGS_DIR, NOTEBOOKS_DIR, SCRIPTS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Create directories on import
create_directories()

# Environment validation
def validate_environment():
    """Validate the environment and configuration."""
    import warnings

    # Check dataset exists
    if not DataConfig.RAW_DATASET_PATH.exists():
        warnings.warn(f"Raw dataset not found at {DataConfig.RAW_DATASET_PATH}")

    # Check device compatibility
    if SystemConfig.DEVICE.type == "mps":
        print(f"✅ Using Apple M2 Pro acceleration ({SystemConfig.DEVICE_NAME})")
    elif SystemConfig.DEVICE.type == "cuda":
        print(f"✅ Using GPU acceleration ({SystemConfig.DEVICE_NAME})")
    else:
        print(f"⚠️  Using CPU processing ({SystemConfig.DEVICE_NAME})")

    # Check Slack configuration
    if not AlertingConfig.SLACK_WEBHOOK_URL:
        warnings.warn("Slack webhook URL not configured. Set SLACK_WEBHOOK_URL environment variable.")

    return True

if __name__ == "__main__":
    print("CloudInfraAI Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Device: {SystemConfig.DEVICE_NAME}")
    print(f"Platform: {SystemConfig.PLATFORM}")
    validate_environment()