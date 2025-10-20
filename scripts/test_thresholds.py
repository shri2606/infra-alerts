#!/usr/bin/env python3
"""
Test Different Thresholds
==========================

Find optimal threshold balancing precision and recall.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import logging

from config import DataConfig
from src.inference.predictor import AnomalyPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_threshold(predictor, sequences, labels, threshold):
    """Evaluate model at given threshold."""
    predictor.set_threshold(threshold)
    
    # Get predictions for all sequences
    all_preds = []
    all_labels = []
    
    for seq_idx in range(len(labels)):
        seq_features = {
            'numerical': sequences['numerical'][seq_idx].unsqueeze(0),
            'binary': sequences['binary'][seq_idx].unsqueeze(0),
            'aggregates': sequences['aggregates'][seq_idx].unsqueeze(0),
            'sequence_lengths': sequences['sequence_lengths'][seq_idx].unsqueeze(0),
            'categorical': {k: v[seq_idx].unsqueeze(0) for k, v in sequences['categorical'].items()}
        }
        
        seq_length = sequences['sequence_lengths'][seq_idx].item()
        result = predictor.predict_single_sequence(seq_features, seq_length, return_scores=True)
        
        all_preds.extend(result['predictions'])
        all_labels.extend(labels[seq_idx, :seq_length].numpy().tolist())
    
    preds = np.array(all_preds)
    labels_arr = np.array(all_labels)
    
    # Calculate metrics
    tp = ((preds == 1) & (labels_arr == 1)).sum()
    fp = ((preds == 1) & (labels_arr == 0)).sum()
    fn = ((preds == 0) & (labels_arr == 1)).sum()
    tn = ((preds == 0) & (labels_arr == 0)).sum()
    
    accuracy = (tp + tn) / len(preds) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def main():
    logger.info("Loading test data...")
    test_data = torch.load(DataConfig.TEST_DATA_PATH, weights_only=False)
    sequences, labels = test_data
    
    logger.info("Initializing predictor...")
    predictor = AnomalyPredictor()
    
    # Test different thresholds
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*80)
    print(f"{'Thresh':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'TP':<5} {'FP':<5} {'FN':<5}")
    print("-"*80)
    
    best_f1 = 0
    best_result = None
    
    for threshold in thresholds:
        result = evaluate_threshold(predictor, sequences, labels, threshold)
        
        print(f"{result['threshold']:<8.2f} "
              f"{result['accuracy']:<8.1f} "
              f"{result['precision']:<8.1f} "
              f"{result['recall']:<8.1f} "
              f"{result['f1']:<8.1f} "
              f"{result['tp']:<5} "
              f"{result['fp']:<5} "
              f"{result['fn']:<5}")
        
        if result['f1'] > best_f1:
            best_f1 = result['f1']
            best_result = result
    
    print("="*80)
    print(f"\nBest Threshold: {best_result['threshold']:.2f}")
    print(f"  Accuracy:  {best_result['accuracy']:.1f}%")
    print(f"  Precision: {best_result['precision']:.1f}%")
    print(f"  Recall:    {best_result['recall']:.1f}%")
    print(f"  F1-Score:  {best_result['f1']:.1f}%")


if __name__ == "__main__":
    main()
