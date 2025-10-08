#!/usr/bin/env python3
"""
Model Training Script
=====================

Train the Transformer-based anomaly detection model on processed OpenStack logs.

Author: CloudInfraAI Team
Date: 2024

Usage:
    source .venv/bin/activate
    python scripts/train_model.py
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

from config import ModelConfig, DataConfig, SystemConfig
from src.model.transformer_model import create_model
from src.model.model_trainer import ModelTrainer, OpenStackSequenceDataset, collate_fn, calculate_pos_weight
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load processed training data."""
    logger.info("Loading processed data...")

    # Load feature outputs
    train_data = torch.load(DataConfig.TRAIN_DATA_PATH)
    val_data = torch.load(DataConfig.VAL_DATA_PATH)
    test_data = torch.load(DataConfig.TEST_DATA_PATH)

    # Load encoders for vocab sizes
    with open(DataConfig.ENCODERS_PATH, 'r') as f:
        encoders = json.load(f)

    # Load feature config
    with open(DataConfig.FEATURE_CONFIG_PATH, 'r') as f:
        feature_config = json.load(f)

    logger.info(f"Loaded training data: {len(train_data[1])} samples")
    logger.info(f"Loaded validation data: {len(val_data[1])} samples")
    logger.info(f"Loaded test data: {len(test_data[1])} samples")

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
    logger.info("CLOUDINFRAAI - MODEL TRAINING")
    logger.info("="*60)

    # Device configuration
    device = SystemConfig.DEVICE
    logger.info(f"Using device: {device} ({SystemConfig.DEVICE_NAME})")

    # Load data
    train_data, val_data, test_data, encoders, feature_config = load_data()

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

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=ModelConfig.LEARNING_RATE,
        weight_decay=1e-5,
        pos_weight=pos_weight
    )

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, batch_size=ModelConfig.BATCH_SIZE
    )

    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=ModelConfig.NUM_EPOCHS,
        early_stopping_patience=ModelConfig.EARLY_STOPPING_PATIENCE,
        checkpoint_dir=ModelConfig.CHECKPOINT_DIR
    )

    # Save final model
    final_model_path = ModelConfig.MODEL_SAVE_PATH
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'history': history,
        'device': str(device)
    }, final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Save training history
    from config import MODELS_DIR
    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        # Convert any numpy types to Python types for JSON serialization
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
    logger.info("TEST SET RESULTS")
    logger.info("="*60)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    logger.info(f"Test F1-Score: {test_f1:.4f}")
    logger.info("="*60)

    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'timestamp': datetime.now().isoformat()
    }

    results_path = MODELS_DIR / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Test results saved to: {results_path}")

    logger.info("\n✅ Training completed successfully!")
    logger.info(f"Best model saved at: {ModelConfig.CHECKPOINT_DIR / 'best_model.pth'}")
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
