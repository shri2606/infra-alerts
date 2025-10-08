#!/usr/bin/env python3
"""
Threshold Optimization Script
==============================

Find the optimal classification threshold for the anomaly detection model.
Tests different thresholds on validation set and reports metrics.

Author: CloudInfraAI Team
Date: 2024

Usage:
    source .venv/bin/activate
    python scripts/optimize_threshold.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import json
import logging
from typing import Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, auc

from config import ModelConfig, DataConfig, SystemConfig
from src.model.transformer_model import create_model
from src.model.model_trainer import OpenStackSequenceDataset, collate_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_predictions_and_labels(model, data_loader, device):
    """
    Get model predictions (probabilities) and true labels.

    Args:
        model: Trained model
        data_loader: DataLoader for the dataset
        device: Device to run on

    Returns:
        probabilities: Model output probabilities [num_events]
        labels: True labels [num_events]
    """
    model.eval()
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            # Move to device
            features_device = {
                'numerical': features['numerical'].to(device),
                'binary': features['binary'].to(device),
                'categorical': {
                    key: value.to(device) for key, value in features['categorical'].items()
                },
                'aggregates': features['aggregates'].to(device),
                'sequence_lengths': features['sequence_lengths'].to(device)
            }
            labels = labels.to(device)  # [batch_size, seq_len]

            # Forward pass
            logits, _ = model(features_device)  # [batch_size, seq_len, 1]

            # Get probabilities
            probabilities = torch.sigmoid(logits.squeeze(-1))  # [batch_size, seq_len]

            # Create mask for non-padded positions
            mask = torch.arange(labels.size(1), device=device).expand(labels.size(0), -1)
            mask = mask < features_device['sequence_lengths'].unsqueeze(1)

            # Flatten and apply mask
            probs_flat = probabilities[mask].cpu().numpy()
            labels_flat = labels[mask].cpu().numpy()

            all_probabilities.extend(probs_flat)
            all_labels.extend(labels_flat)

    return np.array(all_probabilities), np.array(all_labels)


def evaluate_threshold(probabilities, labels, threshold):
    """
    Evaluate metrics at a specific threshold.

    Args:
        probabilities: Prediction probabilities
        labels: True labels
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def find_optimal_thresholds(probabilities, labels):
    """
    Find optimal thresholds for different objectives.

    Args:
        probabilities: Prediction probabilities
        labels: True labels

    Returns:
        Dictionary of optimal thresholds for different objectives
    """
    logger.info("Testing thresholds from 0.1 to 0.9...")

    thresholds_to_test = np.arange(0.1, 0.95, 0.05)
    results = []

    for threshold in thresholds_to_test:
        metrics = evaluate_threshold(probabilities, labels, threshold)
        results.append(metrics)

        logger.info(
            f"Threshold {threshold:.2f}: "
            f"Acc={metrics['accuracy']:.3f}, "
            f"P={metrics['precision']:.3f}, "
            f"R={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}"
        )

    # Find best thresholds for different objectives
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
    best_acc_idx = max(range(len(results)), key=lambda i: results[i]['accuracy'])
    best_precision_idx = max(range(len(results)), key=lambda i: results[i]['precision'])
    best_recall_idx = max(range(len(results)), key=lambda i: results[i]['recall'])

    # Find balanced threshold (best combination of precision and recall)
    best_balanced_idx = max(
        range(len(results)),
        key=lambda i: results[i]['precision'] * results[i]['recall']
    )

    optimal_thresholds = {
        'best_f1': results[best_f1_idx],
        'best_accuracy': results[best_acc_idx],
        'best_precision': results[best_precision_idx],
        'best_recall': results[best_recall_idx],
        'best_balanced': results[best_balanced_idx],
        'default_0.5': evaluate_threshold(probabilities, labels, 0.5),
        'all_results': results
    }

    return optimal_thresholds


def calculate_auroc(probabilities, labels):
    """Calculate AUROC (Area Under ROC Curve)."""
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present in labels, cannot calculate AUROC")
        return 0.0

    fpr, tpr, _ = roc_curve(labels, probabilities)
    auroc = auc(fpr, tpr)
    return auroc


def main():
    """Main threshold optimization function."""
    logger.info("="*60)
    logger.info("THRESHOLD OPTIMIZATION")
    logger.info("="*60)

    # Device configuration
    device = SystemConfig.DEVICE
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading data...")
    val_data = torch.load(DataConfig.VAL_DATA_PATH)
    test_data = torch.load(DataConfig.TEST_DATA_PATH)

    # Create data loaders
    val_dataset = OpenStackSequenceDataset(*val_data)
    test_dataset = OpenStackSequenceDataset(*test_data)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # Load trained model
    logger.info("Loading trained model...")
    checkpoint_path = ModelConfig.CHECKPOINT_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        logger.error(f"Model checkpoint not found at {checkpoint_path}")
        logger.error("Please train the model first using: python scripts/train_model.py")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model config
    with open(DataConfig.ENCODERS_PATH, 'r') as f:
        encoders = json.load(f)

    categorical_vocab_sizes = {
        feature_name: len(encoder_data['classes_'])
        for feature_name, encoder_data in encoders.items()
    }

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

    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    # Get predictions on validation set
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SET - THRESHOLD OPTIMIZATION")
    logger.info("="*60)

    val_probs, val_labels = get_predictions_and_labels(model, val_loader, device)
    logger.info(f"Validation set: {len(val_labels)} events, {val_labels.sum():.0f} anomalies")

    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(val_probs, val_labels)

    # Calculate AUROC
    val_auroc = calculate_auroc(val_probs, val_labels)
    logger.info(f"\nValidation AUROC: {val_auroc:.4f}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("OPTIMAL THRESHOLDS SUMMARY")
    logger.info("="*60)

    logger.info("\n1. Best F1-Score (Balanced Performance):")
    best_f1 = optimal_thresholds['best_f1']
    logger.info(f"   Threshold: {best_f1['threshold']:.2f}")
    logger.info(f"   Accuracy:  {best_f1['accuracy']:.1%}")
    logger.info(f"   Precision: {best_f1['precision']:.1%}")
    logger.info(f"   Recall:    {best_f1['recall']:.1%}")
    logger.info(f"   F1-Score:  {best_f1['f1']:.1%} ⭐")

    logger.info("\n2. Best Accuracy:")
    best_acc = optimal_thresholds['best_accuracy']
    logger.info(f"   Threshold: {best_acc['threshold']:.2f}")
    logger.info(f"   Accuracy:  {best_acc['accuracy']:.1%} ⭐")
    logger.info(f"   Precision: {best_acc['precision']:.1%}")
    logger.info(f"   Recall:    {best_acc['recall']:.1%}")
    logger.info(f"   F1-Score:  {best_acc['f1']:.1%}")

    logger.info("\n3. Best Balanced (P×R):")
    best_bal = optimal_thresholds['best_balanced']
    logger.info(f"   Threshold: {best_bal['threshold']:.2f}")
    logger.info(f"   Accuracy:  {best_bal['accuracy']:.1%}")
    logger.info(f"   Precision: {best_bal['precision']:.1%}")
    logger.info(f"   Recall:    {best_bal['recall']:.1%}")
    logger.info(f"   F1-Score:  {best_bal['f1']:.1%}")

    logger.info("\n4. Default (0.5):")
    default = optimal_thresholds['default_0.5']
    logger.info(f"   Threshold: {default['threshold']:.2f}")
    logger.info(f"   Accuracy:  {default['accuracy']:.1%}")
    logger.info(f"   Precision: {default['precision']:.1%}")
    logger.info(f"   Recall:    {default['recall']:.1%}")
    logger.info(f"   F1-Score:  {default['f1']:.1%}")

    # Evaluate on test set with optimal thresholds
    logger.info("\n" + "="*60)
    logger.info("TEST SET EVALUATION")
    logger.info("="*60)

    test_probs, test_labels = get_predictions_and_labels(model, test_loader, device)
    logger.info(f"Test set: {len(test_labels)} events, {test_labels.sum():.0f} anomalies")

    test_auroc = calculate_auroc(test_probs, test_labels)
    logger.info(f"Test AUROC: {test_auroc:.4f}")

    # Evaluate with different thresholds on test set
    test_results = {}

    for name, val_result in optimal_thresholds.items():
        if name == 'all_results':
            continue
        threshold = val_result['threshold']
        test_metrics = evaluate_threshold(test_probs, test_labels, threshold)
        test_results[name] = test_metrics

    logger.info("\n" + "="*60)
    logger.info("TEST SET RESULTS WITH OPTIMAL THRESHOLDS")
    logger.info("="*60)

    logger.info("\n1. Best F1 Threshold (from validation):")
    result = test_results['best_f1']
    logger.info(f"   Threshold: {result['threshold']:.2f}")
    logger.info(f"   Accuracy:  {result['accuracy']:.1%}")
    logger.info(f"   Precision: {result['precision']:.1%}")
    logger.info(f"   Recall:    {result['recall']:.1%}")
    logger.info(f"   F1-Score:  {result['f1']:.1%}")

    logger.info("\n2. Best Accuracy Threshold (from validation):")
    result = test_results['best_accuracy']
    logger.info(f"   Threshold: {result['threshold']:.2f}")
    logger.info(f"   Accuracy:  {result['accuracy']:.1%}")
    logger.info(f"   Precision: {result['precision']:.1%}")
    logger.info(f"   Recall:    {result['recall']:.1%}")
    logger.info(f"   F1-Score:  {result['f1']:.1%}")

    logger.info("\n3. Default Threshold (0.5):")
    result = test_results['default_0.5']
    logger.info(f"   Threshold: {result['threshold']:.2f}")
    logger.info(f"   Accuracy:  {result['accuracy']:.1%}")
    logger.info(f"   Precision: {result['precision']:.1%}")
    logger.info(f"   Recall:    {result['recall']:.1%}")
    logger.info(f"   F1-Score:  {result['f1']:.1%}")

    # Save results
    output_results = {
        'validation': {
            'auroc': float(val_auroc),
            'optimal_thresholds': {
                k: {key: float(val) if isinstance(val, (np.floating, float)) else val
                    for key, val in v.items()}
                for k, v in optimal_thresholds.items() if k != 'all_results'
            }
        },
        'test': {
            'auroc': float(test_auroc),
            'results_by_threshold': {
                k: {key: float(val) if isinstance(val, (np.floating, float)) else val
                    for key, val in v.items()}
                for k, v in test_results.items()
            }
        }
    }

    from config import MODELS_DIR
    output_path = MODELS_DIR / "threshold_optimization_results.json"
    with open(output_path, 'w') as f:
        json.dump(output_results, f, indent=2)

    logger.info(f"\n✅ Results saved to: {output_path}")

    # Print recommendation
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATION")
    logger.info("="*60)

    recommended_threshold = optimal_thresholds['best_f1']['threshold']
    logger.info(f"\nRecommended threshold: {recommended_threshold:.2f}")
    logger.info(f"This maximizes F1-score on validation set.")
    logger.info(f"\nTo use this threshold in inference:")
    logger.info(f"  predictions = (probabilities >= {recommended_threshold:.2f})")

    logger.info("\n" + "="*60)
    logger.info("✅ THRESHOLD OPTIMIZATION COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nOptimization interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        sys.exit(1)
