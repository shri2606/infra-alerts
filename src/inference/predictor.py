#!/usr/bin/env python3
"""
Anomaly Predictor
==================

Load trained model and make predictions on new log data.

Author: CloudInfraAI Team
Date: 2024
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np

from config import ModelConfig, DataConfig
from src.model.transformer_model import create_model

logger = logging.getLogger(__name__)


class AnomalyPredictor:
    """Predict anomalies in new log data using trained model."""

    # Optimized threshold from 2-min windows experiment
    # Achieves F1=76%, Precision=72%, Recall=81% on test set
    DEFAULT_THRESHOLD = 0.7

    def __init__(
        self,
        model_path: Path = None,
        encoders_path: Path = None,
        scalers_path: Path = None,
        feature_config_path: Path = None,
        threshold: float = None,
        device: str = None
    ):
        """
        Initialize anomaly predictor.

        Args:
            model_path: Path to trained model checkpoint (.pth file)
            encoders_path: Path to saved encoders (JSON)
            scalers_path: Path to saved scalers (JSON)
            feature_config_path: Path to feature configuration (JSON)
            threshold: Classification threshold (default: 0.7, optimized from experiment)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        # Use experimental model by default
        default_model_path = Path("saved_models/experiment_2min/best_model.pth")
        default_encoders_path = Path("outputs_experiment/encoders.json")
        default_scalers_path = Path("outputs_experiment/scalers.json")
        default_feature_config_path = Path("outputs_experiment/feature_config.json")

        # Default paths - prefer experimental model if available
        if model_path is None and default_model_path.exists():
            self.model_path = default_model_path
            logger.info("Using experimental model (2-min windows, 30-sec stride)")
        else:
            self.model_path = model_path or (ModelConfig.CHECKPOINT_DIR / "best_model.pth")

        if encoders_path is None and default_encoders_path.exists():
            self.encoders_path = default_encoders_path
        else:
            self.encoders_path = encoders_path or DataConfig.ENCODERS_PATH

        if scalers_path is None and default_scalers_path.exists():
            self.scalers_path = default_scalers_path
        else:
            self.scalers_path = scalers_path or DataConfig.SCALERS_PATH

        if feature_config_path is None and default_feature_config_path.exists():
            self.feature_config_path = default_feature_config_path
        else:
            self.feature_config_path = feature_config_path or DataConfig.FEATURE_CONFIG_PATH

        # Threshold for binary classification (use optimized 0.7 by default)
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        
        # Device configuration
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing predictor on device: {self.device}")
        
        # Load artifacts
        self.encoders = self._load_encoders()
        self.scalers = self._load_scalers()
        self.feature_config = self._load_feature_config()
        self.model = self._load_model()
        
        logger.info(f"Predictor initialized with threshold={threshold}")
    
    def _load_encoders(self) -> Dict:
        """Load categorical encoders."""
        with open(self.encoders_path, 'r') as f:
            encoders = json.load(f)
        logger.info(f"Loaded encoders for {len(encoders)} categorical features")
        return encoders
    
    def _load_scalers(self) -> Dict:
        """Load numerical scalers."""
        with open(self.scalers_path, 'r') as f:
            scalers = json.load(f)
        logger.info(f"Loaded scalers for {len(scalers)} numerical features")
        return scalers
    
    def _load_feature_config(self) -> Dict:
        """Load feature configuration."""
        with open(self.feature_config_path, 'r') as f:
            config = json.load(f)
        logger.info("Loaded feature configuration")
        return config
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract categorical vocab sizes from encoders
        categorical_vocab_sizes = {}
        for feature_name, encoder_data in self.encoders.items():
            categorical_vocab_sizes[feature_name] = len(encoder_data['classes_'])
        
        # Create model configuration (same as training)
        model_config = {
            'd_model': ModelConfig.D_MODEL,
            'nhead': ModelConfig.NUM_HEADS,
            'num_layers': ModelConfig.NUM_LAYERS,
            'dim_feedforward': ModelConfig.D_MODEL * 4,
            'dropout': ModelConfig.DROPOUT,
            'numerical_dim': 14,
            'binary_dim': 8,
            'aggregate_dim': 9,
            'categorical_vocab_sizes': categorical_vocab_sizes,
            'max_seq_len': ModelConfig.SEQUENCE_LENGTH
        }
        
        # Create model
        model = create_model(model_config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded model with {total_params:,} parameters")
        logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        logger.info(f"Best validation F1: {checkpoint.get('val_f1', 'unknown'):.4f}")
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        features: Dict[str, torch.Tensor],
        return_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict anomalies for new log sequences.
        
        Args:
            features: Dictionary containing preprocessed features:
                - 'numerical': [batch_size, seq_len, 14]
                - 'binary': [batch_size, seq_len, 8]
                - 'categorical': Dict of [batch_size, seq_len] tensors
                - 'aggregates': [batch_size, 9]
                - 'sequence_lengths': [batch_size]
            return_scores: If True, return raw scores along with predictions
        
        Returns:
            predictions: [batch_size, seq_len] binary predictions (0 or 1)
            scores: [batch_size, seq_len] anomaly scores (0.0 to 1.0) if return_scores=True
        """
        # Move features to device
        features = self._to_device(features)
        
        # Forward pass
        logits, _ = self.model(features)  # [batch_size, seq_len, 1]
        
        # Apply sigmoid to get probabilities
        scores = torch.sigmoid(logits).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply threshold to get binary predictions
        predictions = (scores >= self.threshold).long()
        
        if return_scores:
            return predictions, scores
        else:
            return predictions, None
    
    def predict_single_sequence(
        self,
        features: Dict[str, torch.Tensor],
        sequence_length: int,
        return_scores: bool = True
    ) -> Dict[str, any]:
        """
        Predict anomalies for a single sequence and return detailed results.
        
        Args:
            features: Preprocessed features for one sequence
            sequence_length: Actual length of the sequence (excluding padding)
            return_scores: Whether to include anomaly scores
        
        Returns:
            Dictionary containing:
                - 'predictions': List of binary predictions for each event
                - 'scores': List of anomaly scores (if return_scores=True)
                - 'num_anomalies': Count of detected anomalies
                - 'anomaly_indices': Indices of anomalous events
                - 'anomaly_rate': Percentage of anomalous events
        """
        predictions, scores = self.predict(features, return_scores=True)
        
        # Convert to numpy and extract actual sequence (remove padding)
        preds = predictions[0, :sequence_length].cpu().numpy()
        score_vals = scores[0, :sequence_length].cpu().numpy() if scores is not None else None
        
        # Analyze results
        anomaly_indices = np.where(preds == 1)[0].tolist()
        num_anomalies = len(anomaly_indices)
        anomaly_rate = (num_anomalies / sequence_length) * 100 if sequence_length > 0 else 0.0
        
        result = {
            'predictions': preds.tolist(),
            'num_anomalies': num_anomalies,
            'anomaly_indices': anomaly_indices,
            'anomaly_rate': anomaly_rate,
            'sequence_length': sequence_length
        }
        
        if return_scores and score_vals is not None:
            result['scores'] = score_vals.tolist()
        
        return result
    
    def _to_device(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move all feature tensors to the target device."""
        device_features = {}
        
        for key, value in features.items():
            if isinstance(value, dict):
                # Nested dict (e.g., categorical features)
                device_features[key] = {k: v.to(self.device) for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                device_features[key] = value.to(self.device)
            else:
                device_features[key] = value
        
        return device_features
    
    def set_threshold(self, threshold: float):
        """Update classification threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        logger.info(f"Updated threshold: {self.threshold:.2f} -> {threshold:.2f}")
        self.threshold = threshold
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        return {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'threshold': self.threshold,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'val_f1': checkpoint.get('val_f1', 'unknown'),
            'val_loss': checkpoint.get('val_loss', 'unknown'),
            'config': {
                'd_model': ModelConfig.D_MODEL,
                'num_layers': ModelConfig.NUM_LAYERS,
                'num_heads': ModelConfig.NUM_HEADS,
                'dropout': ModelConfig.DROPOUT
            }
        }


if __name__ == "__main__":
    # Test predictor initialization
    logging.basicConfig(level=logging.INFO)
    
    try:
        predictor = AnomalyPredictor()
        info = predictor.get_model_info()
        
        print("\n" + "="*60)
        print("ANOMALY PREDICTOR - MODEL INFO")
        print("="*60)
        print(f"Model Path: {info['model_path']}")
        print(f"Device: {info['device']}")
        print(f"Threshold: {info['threshold']}")
        print(f"Parameters: {info['total_parameters']:,}")
        print(f"Trained Epochs: {info['epoch']}")
        print(f"Val F1: {info['val_f1']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")
        raise
