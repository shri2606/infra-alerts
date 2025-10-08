# Transformer Model Architecture

## Overview

Multi-modal Transformer-based anomaly detection model optimized for OpenStack infrastructure logs.

## Model Architecture

### Components

1. **Multi-Modal Embedding Layer**
   - Numerical features (14): Memory metrics, API latency, temporal features
   - Binary features (8): Error flags, spike indicators
   - Categorical features (6): Components, EventIds, HTTP methods
   - Combined into d_model=128 dimensional space

2. **Positional Encoding**
   - Sinusoidal encoding for temporal sequence information
   - Captures event order within 5-minute windows

3. **Transformer Encoder**
   - 2 encoder layers
   - 4 attention heads
   - Feedforward dimension: 512
   - Handles variable-length sequences with padding mask

4. **Aggregate Feature Integration**
   - Sequence-level statistics (9 features)
   - Combined with transformer output

5. **Classification Head**
   - 3-layer MLP with ReLU activations
   - Binary output (normal/anomaly)
   - Sigmoid activation for probability

### Model Statistics

- **Total Parameters**: ~450K parameters
- **Model Size**: <5MB
- **Training Time**: 10-15 minutes on M2 Pro
- **Inference Time**: <1 second per sequence

## Files

```
src/model/
├── transformer_model.py    # Model architecture
├── model_trainer.py        # Training pipeline
└── README.md              # This file
```

## Usage

### Create Model

```python
from src.model.transformer_model import create_model

config = {
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'categorical_vocab_sizes': {
        'Level': 2,
        'Component': 9,
        'EventId': 38,
        # ... other features
    }
}

model = create_model(config)
```

### Train Model

```bash
# Activate virtual environment
source .venv/bin/activate

# Run training
python scripts/train_model.py
```

### Model Input Format

```python
features = {
    'numerical': torch.Tensor([batch, seq_len, 14]),
    'binary': torch.Tensor([batch, seq_len, 8]),
    'categorical': {
        'Level': torch.LongTensor([batch, seq_len]),
        'Component': torch.LongTensor([batch, seq_len]),
        # ... other categorical features
    },
    'aggregates': torch.Tensor([batch, 9]),
    'sequence_lengths': torch.LongTensor([batch])
}
```

### Model Output

```python
logits, attention_weights = model(features)
# logits: [batch, 1] - raw scores
# Convert to probabilities: torch.sigmoid(logits)
```

## Training Configuration

**Optimizer**: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-5

**Loss Function**: BCEWithLogitsLoss
- Positive class weighting for imbalanced dataset

**Scheduler**: ReduceLROnPlateau
- Monitor: validation F1-score
- Factor: 0.5
- Patience: 5 epochs

**Early Stopping**:
- Monitor: validation F1-score
- Patience: 10 epochs

## Performance Targets

- **Accuracy**: >85%
- **F1-Score**: >0.80
- **Precision**: >0.75
- **Recall**: >0.75
- **False Positive Rate**: <10%

## Hardware Optimization

### Apple M2 Pro (16GB RAM)
- MPS acceleration enabled
- Batch size: 32 (optimized for memory)
- d_model: 128 (balanced complexity)
- Training time: ~10-15 minutes

### CPU Fallback
- Automatic device detection
- Slightly slower but functional

## Model Interpretability

The model supports attention weight extraction for interpretability:

```python
logits, attention_weights = model(features, return_attention=True)
# Visualize which events the model focused on
```

## Next Steps

1. Train model: `python scripts/train_model.py`
2. Evaluate performance on test set
3. Deploy for real-time inference
4. Integrate with dashboard and alerting

## Notes

- Model handles variable-length sequences with padding
- Class imbalance handled through weighted loss
- Small model size (<5MB) enables fast deployment
- Optimized for M2 Pro but works on any device
