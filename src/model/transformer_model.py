#!/usr/bin/env python3
"""
Transformer-based Anomaly Detection Model
==========================================

Multi-modal Transformer architecture optimized for OpenStack log anomaly detection.
Handles numerical, categorical, and binary features with temporal awareness.

Author: CloudInfraAI Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequence information."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [seq_len, batch_size, d_model]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiModalEmbedding(nn.Module):
    """Embedding layer for multi-modal input features."""

    def __init__(
        self,
        d_model: int,
        numerical_dim: int = 14,
        binary_dim: int = 8,
        categorical_vocab_sizes: Dict[str, int] = None,
        embedding_dim: int = 32,
        dropout: float = 0.1
    ):
        """
        Initialize multi-modal embedding layer.

        Args:
            d_model: Target dimension for combined embeddings
            numerical_dim: Number of numerical features
            binary_dim: Number of binary features
            categorical_vocab_sizes: Dictionary mapping categorical feature names to vocab sizes
            embedding_dim: Dimension for categorical embeddings
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.numerical_dim = numerical_dim
        self.binary_dim = binary_dim

        # Numerical feature projection
        self.numerical_projection = nn.Linear(numerical_dim, d_model // 2)

        # Binary feature projection
        self.binary_projection = nn.Linear(binary_dim, d_model // 4)

        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict()
        categorical_vocab_sizes = categorical_vocab_sizes or {}

        for feature_name, vocab_size in categorical_vocab_sizes.items():
            self.categorical_embeddings[feature_name] = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )

        # Calculate total categorical embedding dimension
        total_cat_dim = len(categorical_vocab_sizes) * embedding_dim

        # Categorical projection to fit remaining d_model space
        remaining_dim = d_model - (d_model // 2) - (d_model // 4)
        self.categorical_projection = nn.Linear(total_cat_dim, remaining_dim) if total_cat_dim > 0 else None

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        logger.info(f"MultiModalEmbedding initialized: d_model={d_model}, "
                   f"numerical_dim={numerical_dim}, binary_dim={binary_dim}, "
                   f"categorical_features={len(categorical_vocab_sizes)}")

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Embed multi-modal features.

        Args:
            features: Dictionary containing:
                - 'numerical': [batch_size, seq_len, numerical_dim]
                - 'binary': [batch_size, seq_len, binary_dim]
                - 'categorical': Dict of [batch_size, seq_len] tensors

        Returns:
            Combined embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = features['numerical'].shape[:2]

        # Process numerical features
        numerical_emb = self.numerical_projection(features['numerical'])  # [batch, seq, d_model//2]

        # Process binary features
        binary_emb = self.binary_projection(features['binary'])  # [batch, seq, d_model//4]

        # Process categorical features
        categorical_embs = []
        if 'categorical' in features and self.categorical_projection is not None:
            for feature_name, embedding_layer in self.categorical_embeddings.items():
                if feature_name in features['categorical']:
                    cat_emb = embedding_layer(features['categorical'][feature_name])  # [batch, seq, embedding_dim]
                    categorical_embs.append(cat_emb)

        # Combine embeddings
        if categorical_embs:
            categorical_combined = torch.cat(categorical_embs, dim=-1)  # [batch, seq, total_cat_dim]
            categorical_emb = self.categorical_projection(categorical_combined)  # [batch, seq, remaining_dim]
            combined = torch.cat([numerical_emb, binary_emb, categorical_emb], dim=-1)
        else:
            # Pad to d_model if no categorical features
            pad_dim = self.d_model - numerical_emb.size(-1) - binary_emb.size(-1)
            padding = torch.zeros(batch_size, seq_len, pad_dim, device=numerical_emb.device)
            combined = torch.cat([numerical_emb, binary_emb, padding], dim=-1)

        # Layer norm and dropout
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        return combined


class AnomalyDetectionTransformer(nn.Module):
    """
    Transformer-based model for anomaly detection in OpenStack logs.

    Architecture:
        1. Multi-modal embedding layer (numerical + categorical + binary)
        2. Positional encoding for temporal information
        3. Transformer encoder (2 layers, 4 attention heads)
        4. Aggregate feature integration
        5. Classification head for binary anomaly detection
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        numerical_dim: int = 14,
        binary_dim: int = 8,
        aggregate_dim: int = 9,
        categorical_vocab_sizes: Dict[str, int] = None,
        max_seq_len: int = 50
    ):
        """
        Initialize Transformer model.

        Args:
            d_model: Dimension of the model (128 optimized for M2 Pro)
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            numerical_dim: Number of numerical features
            binary_dim: Number of binary features
            aggregate_dim: Number of aggregate sequence features
            categorical_vocab_sizes: Vocabulary sizes for categorical features
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Multi-modal embedding
        self.embedding = MultiModalEmbedding(
            d_model=d_model,
            numerical_dim=numerical_dim,
            binary_dim=binary_dim,
            categorical_vocab_sizes=categorical_vocab_sizes,
            dropout=dropout
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Aggregate feature projection
        self.aggregate_projection = nn.Linear(aggregate_dim, d_model)

        # Event-level classification head (one prediction per event)
        self.event_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Binary classification per event
        )

        # Initialize weights
        self._init_weights()

        logger.info(f"AnomalyDetectionTransformer initialized with d_model={d_model}, "
                   f"nhead={nhead}, num_layers={num_layers}")

    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, sequence_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create padding mask for variable-length sequences.

        Args:
            sequence_lengths: Tensor of actual sequence lengths [batch_size]
            max_len: Maximum sequence length

        Returns:
            Padding mask [batch_size, max_len] (True for padding positions)
        """
        batch_size = sequence_lengths.size(0)
        mask = torch.arange(max_len, device=sequence_lengths.device).expand(batch_size, max_len)
        mask = mask >= sequence_lengths.unsqueeze(1)
        return mask

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            features: Dictionary containing:
                - 'numerical': [batch_size, seq_len, numerical_dim]
                - 'binary': [batch_size, seq_len, binary_dim]
                - 'categorical': Dict of [batch_size, seq_len] tensors
                - 'aggregates': [batch_size, aggregate_dim]
                - 'sequence_lengths': [batch_size] actual lengths (optional)

            return_attention: Whether to return attention weights

        Returns:
            - logits: [batch_size, seq_len, 1] event-level anomaly scores
            - attention_weights: Attention weights if return_attention=True
        """
        batch_size, seq_len = features['numerical'].shape[:2]

        # Embed multi-modal features
        embedded = self.embedding(features)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        # Transformer expects [seq_len, batch_size, d_model] for pos encoding
        embedded = embedded.transpose(0, 1)  # [seq_len, batch_size, d_model]
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Create padding mask if sequence_lengths provided
        src_key_padding_mask = None
        if 'sequence_lengths' in features:
            src_key_padding_mask = self.create_padding_mask(
                features['sequence_lengths'], seq_len
            )

        # Transformer encoder
        encoder_output = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )  # [batch_size, seq_len, d_model]

        # Event-level classification: predict for each event
        logits = self.event_classifier(encoder_output)  # [batch_size, seq_len, 1]

        # Return with or without attention weights
        attention_weights = None
        if return_attention:
            # Extract attention weights from last encoder layer
            # Note: This requires modifying TransformerEncoder to return attention weights
            # For now, return None
            attention_weights = None

        return logits, attention_weights

    def predict(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict event-level anomaly probabilities.

        Args:
            features: Feature dictionary

        Returns:
            - predictions: Binary predictions [batch_size, seq_len] (0 or 1)
            - probabilities: Anomaly probabilities [batch_size, seq_len] [0, 1]
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(features)  # [batch_size, seq_len, 1]
            probabilities = torch.sigmoid(logits).squeeze(-1)  # [batch_size, seq_len]
            predictions = (probabilities >= 0.5).long()

        return predictions, probabilities


def create_model(config: Dict) -> AnomalyDetectionTransformer:
    """
    Create model from configuration.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        Initialized model
    """
    # Extract categorical vocab sizes from encoders
    categorical_vocab_sizes = {}
    if 'categorical_vocab_sizes' in config:
        categorical_vocab_sizes = config['categorical_vocab_sizes']

    model = AnomalyDetectionTransformer(
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 2),
        dim_feedforward=config.get('dim_feedforward', 512),
        dropout=config.get('dropout', 0.1),
        numerical_dim=config.get('numerical_dim', 14),
        binary_dim=config.get('binary_dim', 8),
        aggregate_dim=config.get('aggregate_dim', 9),
        categorical_vocab_sizes=categorical_vocab_sizes,
        max_seq_len=config.get('max_seq_len', 50)
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model created with {total_params:,} total parameters "
               f"({trainable_params:,} trainable)")

    return model


if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)

    # Example configuration
    config = {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'numerical_dim': 14,
        'binary_dim': 8,
        'aggregate_dim': 9,
        'categorical_vocab_sizes': {
            'Level': 2,
            'Component': 9,
            'EventId': 38,
            'http_method': 4,
            'vm_event_type': 5,
            'instance_id_hash': 6
        },
        'max_seq_len': 50
    }

    # Create model
    model = create_model(config)

    # Create dummy input
    batch_size = 2
    seq_len = 50

    dummy_features = {
        'numerical': torch.randn(batch_size, seq_len, 14),
        'binary': torch.randn(batch_size, seq_len, 8),
        'categorical': {
            'Level': torch.randint(0, 2, (batch_size, seq_len)),
            'Component': torch.randint(0, 9, (batch_size, seq_len)),
            'EventId': torch.randint(0, 38, (batch_size, seq_len)),
            'http_method': torch.randint(0, 4, (batch_size, seq_len)),
            'vm_event_type': torch.randint(0, 5, (batch_size, seq_len)),
            'instance_id_hash': torch.randint(0, 6, (batch_size, seq_len))
        },
        'aggregates': torch.randn(batch_size, 9),
        'sequence_lengths': torch.tensor([45, 50])
    }

    # Forward pass
    print("\nTesting forward pass...")
    logits, _ = model(dummy_features)
    print(f"Output shape: {logits.shape}")
    print(f"Output logits: {logits.squeeze()}")

    # Test prediction
    predictions, probabilities = model.predict(dummy_features)
    print(f"\nPredictions: {predictions}")
    print(f"Probabilities: {probabilities}")

    print("\n✅ Model architecture test passed!")
