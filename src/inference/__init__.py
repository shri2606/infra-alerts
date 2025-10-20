"""
Inference Pipeline Module
==========================

Real-time anomaly detection inference for new OpenStack logs.

Components:
- predictor: Load trained model and make predictions
- log_processor: Process new logs into model-ready features (TODO)
"""

from .predictor import AnomalyPredictor
# from .log_processor import LogProcessor  # TODO: Implement later

__all__ = ['AnomalyPredictor']
