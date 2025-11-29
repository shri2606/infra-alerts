"""
Streaming Anomaly Predictor
============================

Real-time anomaly detection with sliding window buffer.
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Callable
import logging

from .predictor import AnomalyPredictor
from .feature_extractor import RealTimeFeatureExtractor


logger = logging.getLogger(__name__)


class StreamingAnomalyPredictor:
    """
    Real-time anomaly detection with sliding window buffer.

    Maintains a buffer of recent events and triggers predictions
    when enough events have accumulated.
    """

    def __init__(self,
                 window_size: int = 50,
                 stride: int = 10,
                 model_path: Optional[str] = None,
                 artifacts_dir: str = "outputs_experiment"):
        """
        Initialize streaming predictor.

        Args:
            window_size: Number of events per prediction window
            stride: Number of events to slide before next prediction
            model_path: Path to trained model (uses default if None)
            artifacts_dir: Directory with encoders/scalers
        """
        self.window_size = window_size
        self.stride = stride

        # Event buffer (sliding window)
        self.buffer = deque(maxlen=window_size)

        # Events since last prediction
        self.events_since_prediction = 0

        # Total events processed
        self.total_events = 0

        # Prediction counter
        self.prediction_count = 0

        # Initialize predictor and feature extractor
        self.predictor = AnomalyPredictor(model_path=model_path)
        self.feature_extractor = RealTimeFeatureExtractor(artifacts_dir=artifacts_dir)

        # Callback for predictions (optional)
        self.prediction_callback = None

        logger.info(f"StreamingPredictor initialized: window={window_size}, stride={stride}")

    def set_prediction_callback(self, callback: Callable):
        """
        Set a callback function to be called when predictions are made.

        Args:
            callback: Function(predictions, events, metadata) to call
        """
        self.prediction_callback = callback

    def is_warming_up(self) -> bool:
        """Check if buffer is still warming up."""
        return len(self.buffer) < self.window_size

    def get_warmup_progress(self) -> float:
        """Get warmup progress as percentage."""
        return (len(self.buffer) / self.window_size) * 100

    def should_predict(self) -> bool:
        """Check if we should trigger a prediction."""
        if self.is_warming_up():
            return False

        return self.events_since_prediction >= self.stride

    def process_event(self, event: Dict) -> Optional[Dict]:
        """
        Process a single event and optionally trigger prediction.

        Args:
            event: Raw log event dictionary

        Returns:
            Prediction results if prediction was triggered, None otherwise
        """
        # Add event to buffer
        self.buffer.append(event)
        self.total_events += 1
        self.events_since_prediction += 1

        # Check if we should predict
        if self.should_predict():
            return self._trigger_prediction()

        return None

    def process_events(self, events: List[Dict]) -> List[Dict]:
        """
        Process multiple events and return all predictions.

        Args:
            events: List of raw log event dictionaries

        Returns:
            List of prediction results
        """
        predictions = []

        for event in events:
            result = self.process_event(event)
            if result is not None:
                predictions.append(result)

        return predictions

    def _trigger_prediction(self) -> Dict:
        """
        Trigger a prediction on the current buffer.

        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Triggering prediction #{self.prediction_count + 1} "
                   f"(buffer size: {len(self.buffer)}, stride: {self.events_since_prediction})")

        # Extract features from buffered events
        events_list = list(self.buffer)
        features = self.feature_extractor.process_events_to_features(events_list)

        # Prepare features for model
        model_features = self._prepare_model_features(features)

        # Run prediction
        predictions = self.predictor.predict_single_sequence(
            model_features,
            sequence_length=len(events_list),
            return_scores=True
        )

        # Reset stride counter
        self.events_since_prediction = 0
        self.prediction_count += 1

        # Prepare result
        result = {
            'prediction_id': self.prediction_count,
            'total_events_processed': self.total_events,
            'window_size': len(events_list),
            'predictions': predictions['predictions'],
            'scores': predictions['scores'],
            'num_anomalies': predictions['num_anomalies'],
            'anomaly_rate': predictions['anomaly_rate'],
            'anomaly_indices': predictions['anomaly_indices'],
            'events': events_list
        }

        # Call callback if set
        if self.prediction_callback is not None:
            self.prediction_callback(result)

        return result

    def _prepare_model_features(self, features: Dict) -> Dict:
        """
        Prepare features for model input.

        Args:
            features: Dictionary with feature arrays (already has batch dimension from feature extractor)

        Returns:
            Dictionary with torch tensors
        """
        # Convert to tensors (batch dimension already added by feature extractor)
        numerical = torch.tensor(features['numerical'], dtype=torch.float32)
        binary = torch.tensor(features['binary'], dtype=torch.float32)
        aggregates = torch.tensor(features['aggregates'], dtype=torch.float32)

        categorical = {}
        for key, value in features['categorical'].items():
            categorical[key] = torch.tensor(value, dtype=torch.long)

        # Get sequence length from the shape
        sequence_length = features['numerical'].shape[1]
        sequence_lengths = torch.tensor([sequence_length], dtype=torch.long)

        return {
            'numerical': numerical,
            'binary': binary,
            'aggregates': aggregates,
            'categorical': categorical,
            'sequence_lengths': sequence_lengths
        }

    def get_status(self) -> Dict:
        """
        Get current status of the streaming predictor.

        Returns:
            Dictionary with status information
        """
        return {
            'total_events': self.total_events,
            'buffer_size': len(self.buffer),
            'window_size': self.window_size,
            'is_warming_up': self.is_warming_up(),
            'warmup_progress': self.get_warmup_progress(),
            'events_since_prediction': self.events_since_prediction,
            'predictions_made': self.prediction_count,
            'next_prediction_in': self.stride - self.events_since_prediction
        }

    def reset(self):
        """Reset the predictor state."""
        self.buffer.clear()
        self.events_since_prediction = 0
        self.total_events = 0
        self.prediction_count = 0
        logger.info("StreamingPredictor reset")
