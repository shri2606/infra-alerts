#!/usr/bin/env python3
"""
Model Training Pipeline
=======================

Training pipeline for the Transformer-based anomaly detection model.
Includes training loop, validation, early stopping, and checkpointing.

Author: CloudInfraAI Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class OpenStackSequenceDataset(Dataset):
    """PyTorch Dataset for OpenStack log sequences."""

    def __init__(self, features: Dict[str, torch.Tensor], labels: torch.Tensor):
        """
        Initialize dataset.

        Args:
            features: Dictionary of feature tensors
            labels: Label tensor [num_samples]
        """
        self.features = features
        self.labels = labels
        self.num_samples = len(labels)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single sample."""
        sample_features = {
            'numerical': self.features['numerical'][idx],
            'binary': self.features['binary'][idx],
            'categorical': {
                key: value[idx] for key, value in self.features['categorical'].items()
            },
            'aggregates': self.features['aggregates'][idx],
            'sequence_lengths': self.features['sequence_lengths'][idx]
        }

        return sample_features, self.labels[idx]


def collate_fn(batch: List[Tuple[Dict, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Custom collate function for batching.

    Args:
        batch: List of (features, label) tuples

    Returns:
        Batched features and labels
    """
    features_list, labels_list = zip(*batch)

    # Stack features
    batched_features = {
        'numerical': torch.stack([f['numerical'] for f in features_list]),
        'binary': torch.stack([f['binary'] for f in features_list]),
        'categorical': {},
        'aggregates': torch.stack([f['aggregates'] for f in features_list]),
        'sequence_lengths': torch.stack([f['sequence_lengths'] for f in features_list])
    }

    # Stack categorical features
    categorical_keys = features_list[0]['categorical'].keys()
    for key in categorical_keys:
        batched_features['categorical'][key] = torch.stack(
            [f['categorical'][key] for f in features_list]
        )

    # Stack labels
    batched_labels = torch.stack(labels_list)

    return batched_features, batched_labels


class ModelTrainer:
    """Trainer class for anomaly detection model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pos_weight: Optional[float] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            device: Device to train on (cpu/cuda/mps)
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            pos_weight: Positive class weight for imbalanced dataset
        """
        self.model = model.to(device)
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function with optional class weighting
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rates': []
        }

        logger.info(f"Trainer initialized on device: {device}")
        if pos_weight is not None:
            logger.info(f"Using positive class weight: {pos_weight:.2f}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average loss and accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move to device
            features = self._move_to_device(features)
            labels = labels.to(self.device)  # [batch_size, seq_len]

            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(features)  # [batch_size, seq_len, 1]

            # Create mask for non-padded positions
            mask = torch.arange(labels.size(1), device=labels.device).expand(labels.size(0), -1)
            mask = mask < features['sequence_lengths'].unsqueeze(1)  # [batch_size, seq_len]

            # Flatten and apply mask
            logits_flat = logits.squeeze(-1)[mask]  # [num_valid_events]
            labels_flat = labels[mask]  # [num_valid_events]

            # Compute loss only on valid (non-padded) positions
            loss = self.criterion(logits_flat, labels_flat)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = (torch.sigmoid(logits_flat) >= 0.5).float()
            correct += (predictions == labels_flat).sum().item()
            total += labels_flat.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average loss, accuracy, precision, recall, F1-score
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                # Move to device
                features = self._move_to_device(features)
                labels = labels.to(self.device)  # [batch_size, seq_len]

                # Forward pass
                logits, _ = self.model(features)  # [batch_size, seq_len, 1]

                # Create mask for non-padded positions
                mask = torch.arange(labels.size(1), device=labels.device).expand(labels.size(0), -1)
                mask = mask < features['sequence_lengths'].unsqueeze(1)

                # Flatten and apply mask
                logits_flat = logits.squeeze(-1)[mask]
                labels_flat = labels[mask]

                # Compute loss
                loss = self.criterion(logits_flat, labels_flat)
                total_loss += loss.item()

                # Store predictions and labels (event-level)
                predictions = (torch.sigmoid(logits_flat) >= 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_flat.cpu().numpy())

        # Calculate metrics
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()

        accuracy = (all_predictions == all_labels).mean()

        # Precision, recall, F1
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_loss = total_loss / len(val_loader)

        return avg_loss, accuracy, precision, recall, f1

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[Path] = None
    ) -> Dict:
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        best_val_f1 = 0.0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_f1)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val F1: {val_f1:.4f}, LR: {current_lr:.6f}"
            )

            # Early stopping and checkpointing
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0

                # Save best model
                if checkpoint_dir:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / "best_model.pth"
                    self.save_checkpoint(checkpoint_path, epoch, val_f1)
                    logger.info(f"✅ Saved best model (F1: {val_f1:.4f})")

            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        logger.info(f"Best validation F1: {best_val_f1:.4f}")

        return self.history

    def save_checkpoint(self, path: Path, epoch: int, val_f1: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': val_f1,
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def _move_to_device(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move feature dictionary to device."""
        return {
            'numerical': features['numerical'].to(self.device),
            'binary': features['binary'].to(self.device),
            'categorical': {
                key: value.to(self.device) for key, value in features['categorical'].items()
            },
            'aggregates': features['aggregates'].to(self.device),
            'sequence_lengths': features['sequence_lengths'].to(self.device)
        }


def calculate_pos_weight(labels: torch.Tensor) -> float:
    """
    Calculate positive class weight for imbalanced dataset.

    Args:
        labels: Training labels

    Returns:
        Positive class weight
    """
    num_positive = labels.sum().item()
    num_negative = len(labels) - num_positive

    if num_positive == 0:
        return 1.0

    pos_weight = num_negative / num_positive
    logger.info(f"Class distribution - Positive: {num_positive}, Negative: {num_negative}")
    logger.info(f"Calculated pos_weight: {pos_weight:.2f}")

    return pos_weight


if __name__ == "__main__":
    # Test training pipeline
    logging.basicConfig(level=logging.INFO)

    print("Testing training pipeline...")
    print("✅ ModelTrainer class ready for use")
