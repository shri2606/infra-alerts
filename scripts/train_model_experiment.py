#!/usr/bin/env python3
"""
Experimental Model Training Script
===================================

Train the Transformer model on experimental data (2-min windows, 30-sec stride).

Experiment Goals:
- Improve recall from 33% to 55-65%
- Maintain precision above 85%
- Leverage 39 training anomalies (vs baseline 2)

Author: CloudInfraAI Team
Date: October 2024

Usage:
    source .venv/bin/activate
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python scripts/train_model_experiment.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
import logging
from datetime import datetime

from config import ModelConfig, SystemConfig
from src.model.transformer_model import create_model
from src.model.model_trainer import ModelTrainer, OpenStackSequenceDataset, collate_fn, calculate_pos_weight
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_experimental_data():
    """Load experimental training data."""
    logger.info("Loading experimental data...")

    experiment_dir = Path("outputs_experiment")

    # Load feature outputs
    train_data = torch.load(experiment_dir / "train_data.pt")
    val_data = torch.load(experiment_dir / "val_data.pt")
    test_data = torch.load(experiment_dir / "test_data.pt")

    # Load encoders for vocab sizes
    with open(experiment_dir / "encoders.json", 'r') as f:
        encoders = json.load(f)

    # Load feature config
    with open(experiment_dir / "feature_config.json", 'r') as f:
        feature_config = json.load(f)

    logger.info(f"Loaded training data: {len(train_data[1])} samples")
    logger.info(f"Loaded validation data: {len(val_data[1])} samples")
    logger.info(f"Loaded test data: {len(test_data[1])} samples")

    # Calculate anomaly counts
    train_anomalies = sum(
        int(train_data[1][i][:train_data[0]['sequence_lengths'][i]].sum().item())
        for i in range(len(train_data[1]))
    )
    val_anomalies = sum(
        int(val_data[1][i][:val_data[0]['sequence_lengths'][i]].sum().item())
        for i in range(len(val_data[1]))
    )
    test_anomalies = sum(
        int(test_data[1][i][:test_data[0]['sequence_lengths'][i]].sum().item())
        for i in range(len(test_data[1]))
    )

    logger.info(f"Training anomalies: {train_anomalies}")
    logger.info(f"Validation anomalies: {val_anomalies}")
    logger.info(f"Test anomalies: {test_anomalies}")

    return train_data, val_data, test_data, encoders, feature_config


def create_dataloaders(train_data, val_data, batch_size=32):
    """Create PyTorch DataLoaders."""
    logger.info("Creating data loaders...")

    # Create datasets
    train_dataset = OpenStackSequenceDataset(*train_data)
    val_dataset = OpenStackSequenceDataset(*val_data)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for MPS compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    logger.info(f"Created data loaders with batch_size={batch_size}")

    return train_loader, val_loader


def main():
    """Main training function."""
    logger.info("="*60)
    logger.info("EXPERIMENTAL MODEL TRAINING")
    logger.info("Configuration: 2-min windows, 30-sec stride")
    logger.info("="*60)

    # Device configuration
    device = SystemConfig.DEVICE
    logger.info(f"Using device: {device} ({SystemConfig.DEVICE_NAME})")

    # Load experimental data
    train_data, val_data, test_data, encoders, feature_config = load_experimental_data()

    # Extract categorical vocab sizes
    categorical_vocab_sizes = {}
    for feature_name, encoder_data in encoders.items():
        categorical_vocab_sizes[feature_name] = len(encoder_data['classes_'])

    # Create model configuration
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
    logger.info("Creating model...")
    model = create_model(model_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total ({trainable_params:,} trainable)")

    # Calculate class weights for imbalanced dataset
    pos_weight = calculate_pos_weight(train_data[1])
    logger.info(f"Calculated pos_weight: {pos_weight:.2f}")

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=ModelConfig.LEARNING_RATE,
        weight_decay=1e-5,
        pos_weight=pos_weight,
        threshold=0.5
    )

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, batch_size=ModelConfig.BATCH_SIZE
    )

    # Train model
    logger.info("Starting training...")
    experiment_checkpoint_dir = Path("saved_models/experiment_2min")
    experiment_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=ModelConfig.NUM_EPOCHS,
        early_stopping_patience=ModelConfig.EARLY_STOPPING_PATIENCE,
        checkpoint_dir=experiment_checkpoint_dir
    )

    # Save final model
    final_model_path = experiment_checkpoint_dir / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'history': history,
        'device': str(device),
        'experiment': '2min_windows_30sec_stride'
    }, final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Save training history
    history_path = experiment_checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        history_serializable = {
            key: [float(v) for v in values] for key, values in history.items()
        }
        json.dump(history_serializable, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_dataset = OpenStackSequenceDataset(*test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=ModelConfig.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    test_loss, test_acc, test_precision, test_recall, test_f1 = trainer.validate(test_loader)

    logger.info("="*60)
    logger.info("EXPERIMENTAL TEST SET RESULTS")
    logger.info("="*60)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    logger.info(f"Test F1-Score: {test_f1:.4f}")
    logger.info("="*60)

    # Comparison with baseline (96% acc, 100% precision, 33% recall, 50% F1)
    logger.info("COMPARISON WITH BASELINE:")
    logger.info(f"- Accuracy: 96% → {test_acc*100:.0f}%")
    logger.info(f"- Precision: 100% → {test_precision*100:.0f}%")
    logger.info(f"- Recall: 33% → {test_recall*100:.0f}% ({(test_recall-0.33)*100:+.0f}pp)")
    logger.info(f"- F1-Score: 50% → {test_f1*100:.0f}% ({(test_f1-0.50)*100:+.0f}pp)")
    logger.info("="*60)

    # Success criteria evaluation
    logger.info("SUCCESS CRITERIA EVALUATION:")
    success_criteria = {
        'minimum_success': {
            'recall_target': 0.50,
            'precision_target': 0.85,
            'f1_target': 0.60,
            'status': 'Proceed with experiment'
        },
        'target_success': {
            'recall_target': 0.60,
            'precision_target': 0.88,
            'f1_target': 0.70,
            'status': 'Use experimental model'
        },
        'outstanding_success': {
            'recall_target': 0.70,
            'precision_target': 0.90,
            'f1_target': 0.75,
            'status': 'Publish results'
        }
    }

    achieved_level = "None"
    for level, criteria in success_criteria.items():
        if (test_recall >= criteria['recall_target'] and
            test_precision >= criteria['precision_target'] and
            test_f1 >= criteria['f1_target']):
            achieved_level = level
            logger.info(f"✅ {level.replace('_', ' ').title()}: {criteria['status']}")
        else:
            logger.info(f"❌ {level.replace('_', ' ').title()}: Not achieved")

    logger.info("="*60)

    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'timestamp': datetime.now().isoformat(),
        'experiment': '2min_windows_30sec_stride',
        'baseline_comparison': {
            'accuracy_change': float(test_acc - 0.96),
            'precision_change': float(test_precision - 1.00),
            'recall_change': float(test_recall - 0.33),
            'f1_change': float(test_f1 - 0.50)
        },
        'success_level': achieved_level
    }

    results_path = experiment_checkpoint_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Test results saved to: {results_path}")

    logger.info("\n✅ Experimental training completed successfully!")
    logger.info(f"Best model saved at: {experiment_checkpoint_dir / 'best_model.pth'}")
    logger.info(f"Final model saved at: {final_model_path}")

    return history, test_results


if __name__ == "__main__":
    try:
        history, test_results = main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)
