"""
Data Processing Module
=====================

Contains all data processing, analysis, and feature engineering functionality
for OpenStack log data.

Modules:
- data_analyzer: Pattern detection and anomaly identification
- feature_engineer: Feature extraction and sequence creation
- loader: Data loading utilities
"""

from .data_analyzer import OpenStackLogAnalyzer
from .feature_engineer import OpenStackFeatureEngineer, FeatureConfig

__all__ = [
    "OpenStackLogAnalyzer",
    "OpenStackFeatureEngineer",
    "FeatureConfig"
]