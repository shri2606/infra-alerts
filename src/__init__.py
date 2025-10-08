"""
CloudInfraAI - AI-powered OpenStack Infrastructure Monitoring
===========================================================

This package contains the core modules for the CloudInfraAI system:
- Data processing and feature engineering
- Machine learning models (Transformer-based anomaly detection)
- Real-time dashboard and visualization
- Alert management and notifications

Author: CloudInfraAI Team
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "CloudInfraAI Team"

# Import main modules for easy access
from . import data_processing
from . import model
from . import dashboard
from . import alerting
from . import utils

__all__ = [
    "data_processing",
    "model",
    "dashboard",
    "alerting",
    "utils"
]