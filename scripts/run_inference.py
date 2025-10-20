#!/usr/bin/env python3
"""
Inference Script
=================

Run inference on test data to validate the inference pipeline.

Author: CloudInfraAI Team
Date: 2024

Usage:
    source .venv/bin/activate
    python scripts/run_inference.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
import logging
import numpy as np
from datetime import datetime

from config import DataConfig, ModelConfig
from src.inference.predictor import AnomalyPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_test_data():
    """Load test data for inference."""
    logger.info("Loading test data...")

    # Try to load experimental test data first, fall back to baseline
    experimental_test_path = Path("outputs_experiment/test_data.pt")
    if experimental_test_path.exists():
        logger.info("Using experimental test data (2-min windows)")
        test_data = torch.load(experimental_test_path, weights_only=False)
    else:
        logger.info("Using baseline test data")
        test_data = torch.load(DataConfig.TEST_DATA_PATH, weights_only=False)

    sequences, labels = test_data

    logger.info(f"Loaded {len(labels)} test sequences")
    logger.info(f"First sequence length: {sequences['numerical'][0].shape[0]}")

    return sequences, labels


def run_inference(predictor, sequences, labels):
    """Run inference on test sequences and evaluate results."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING INFERENCE ON TEST DATA")
    logger.info("="*60)
    
    num_sequences = len(labels)
    all_predictions = []
    all_true_labels = []
    all_scores = []
    
    for seq_idx in range(num_sequences):
        logger.info(f"\n--- Sequence {seq_idx + 1}/{num_sequences} ---")
        
        # Extract single sequence features
        seq_features = {
            'numerical': sequences['numerical'][seq_idx].unsqueeze(0),
            'binary': sequences['binary'][seq_idx].unsqueeze(0),
            'aggregates': sequences['aggregates'][seq_idx].unsqueeze(0),
            'sequence_lengths': sequences['sequence_lengths'][seq_idx].unsqueeze(0),
            'categorical': {
                key: val[seq_idx].unsqueeze(0) 
                for key, val in sequences['categorical'].items()
            }
        }
        
        # Get sequence length
        seq_length = sequences['sequence_lengths'][seq_idx].item()
        
        # True labels for this sequence
        true_labels = labels[seq_idx, :seq_length].numpy()
        
        # Make predictions
        result = predictor.predict_single_sequence(
            seq_features,
            sequence_length=seq_length,
            return_scores=True
        )
        
        predictions = np.array(result['predictions'])
        scores = np.array(result['scores'])
        
        # Store for overall evaluation
        all_predictions.extend(predictions.tolist())
        all_true_labels.extend(true_labels.tolist())
        all_scores.extend(scores.tolist())
        
        # Calculate metrics for this sequence
        correct = (predictions == true_labels).sum()
        accuracy = (correct / seq_length) * 100
        
        # Count true positives, false positives, false negatives
        tp = ((predictions == 1) & (true_labels == 1)).sum()
        fp = ((predictions == 1) & (true_labels == 0)).sum()
        fn = ((predictions == 0) & (true_labels == 1)).sum()
        tn = ((predictions == 0) & (true_labels == 0)).sum()
        
        # Print results
        logger.info(f"Sequence Length: {seq_length} events")
        logger.info(f"True Anomalies: {true_labels.sum()}")
        logger.info(f"Predicted Anomalies: {result['num_anomalies']}")
        logger.info(f"Anomaly Rate: {result['anomaly_rate']:.1f}%")
        logger.info(f"Accuracy: {accuracy:.1f}%")
        logger.info(f"True Positives: {tp}, False Positives: {fp}")
        logger.info(f"True Negatives: {tn}, False Negatives: {fn}")
        
        if result['anomaly_indices']:
            logger.info(f"Anomaly detected at indices: {result['anomaly_indices']}")
            logger.info(f"Anomaly scores: {[f'{scores[i]:.3f}' for i in result['anomaly_indices']]}")
    
    return np.array(all_predictions), np.array(all_true_labels), np.array(all_scores)


def calculate_overall_metrics(predictions, true_labels, scores):
    """Calculate overall performance metrics."""
    logger.info("\n" + "="*60)
    logger.info("OVERALL TEST SET PERFORMANCE")
    logger.info("="*60)
    
    # Basic counts
    tp = ((predictions == 1) & (true_labels == 1)).sum()
    fp = ((predictions == 1) & (true_labels == 0)).sum()
    fn = ((predictions == 0) & (true_labels == 1)).sum()
    tn = ((predictions == 0) & (true_labels == 0)).sum()
    
    # Calculate metrics
    accuracy = (tp + tn) / len(predictions) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    logger.info(f"Total Events: {len(predictions)}")
    logger.info(f"True Anomalies: {true_labels.sum()}")
    logger.info(f"Predicted Anomalies: {predictions.sum()}")
    logger.info(f"")
    logger.info(f"Confusion Matrix:")
    logger.info(f"  True Positives (TP):  {tp}")
    logger.info(f"  False Positives (FP): {fp}")
    logger.info(f"  True Negatives (TN):  {tn}")
    logger.info(f"  False Negatives (FN): {fn}")
    logger.info(f"")
    logger.info(f"Performance Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.2f}%")
    logger.info(f"  Precision: {precision:.2f}%")
    logger.info(f"  Recall:    {recall:.2f}%")
    logger.info(f"  F1-Score:  {f1:.2f}%")
    logger.info("="*60)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_events': int(len(predictions)),
        'true_anomalies': int(true_labels.sum()),
        'predicted_anomalies': int(predictions.sum()),
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    }
    
    return results


def main():
    """Main inference function."""
    logger.info("="*60)
    logger.info("CLOUDINFRAAI - INFERENCE PIPELINE TEST")
    logger.info("="*60)
    
    # Initialize predictor (uses experimental model + threshold 0.7 by default)
    logger.info("\nInitializing predictor...")
    predictor = AnomalyPredictor()  # Uses DEFAULT_THRESHOLD = 0.7
    
    # Print model info
    model_info = predictor.get_model_info()
    logger.info(f"\nModel Info:")
    logger.info(f"  Device: {model_info['device']}")
    logger.info(f"  Parameters: {model_info['total_parameters']:,}")
    logger.info(f"  Trained Epochs: {model_info['epoch']}")
    logger.info(f"  Validation F1: {model_info['val_f1']:.4f}")
    logger.info(f"  Threshold: {model_info['threshold']}")
    
    # Load test data
    sequences, labels = load_test_data()
    
    # Run inference
    predictions, true_labels, scores = run_inference(predictor, sequences, labels)
    
    # Calculate overall metrics
    results = calculate_overall_metrics(predictions, true_labels, scores)
    
    # Save results to file
    output_path = project_root / "outputs" / "inference_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    logger.info("\n✅ Inference pipeline test completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        sys.exit(1)
