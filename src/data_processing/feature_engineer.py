#!/usr/bin/env python3
"""
Feature Engineering Pipeline for OpenStack Logs
===============================================

This script transforms raw log data into features suitable for training
the Transformer-based anomaly detection model.

Author: CloudInfraAI Team
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureConfig:
    """Configuration for feature engineering."""

    # Sequence parameters
    SEQUENCE_LENGTH = 50  # Maximum events per sequence
    WINDOW_SIZE_MINUTES = 5  # Time window for creating sequences

    # Anomaly thresholds
    MEMORY_SPIKE_THRESHOLD = 2560  # MB
    API_LATENCY_THRESHOLD = 0.5    # seconds

    # Model parameters
    BATCH_SIZE = 32
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2

    # Feature dimensions
    MAX_VOCAB_SIZE = 1000

class OpenStackFeatureEngineer:
    """Transform OpenStack logs into ML-ready features."""

    def __init__(self, config: FeatureConfig = None):
        """Initialize feature engineer with configuration."""
        self.config = config or FeatureConfig()
        self.encoders = {}
        self.scalers = {}
        self.feature_stats = {}

    def load_processed_data(self, csv_path: str) -> pd.DataFrame:
        """Load the processed dataset with labels."""
        logger.info(f"Loading processed data from {csv_path}")

        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} events")
        return df

    def extract_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numerical features from log content."""
        logger.info("Extracting numerical features...")

        # Initialize feature columns
        df['memory_claim_mb'] = 0.0
        df['memory_used_mb'] = 0.0
        df['memory_total_mb'] = 0.0
        df['memory_free_mb'] = 0.0
        df['memory_utilization_pct'] = 0.0
        df['api_response_time'] = 0.0
        df['api_content_length'] = 0.0
        df['http_status_code'] = 200  # Default to success

        # Memory features
        memory_patterns = {
            'claim': r'memory (\d+) MB',
            'used': r'used: (\d+\.\d+) MB',
            'total': r'Total memory: (\d+) MB',
            'free': r'free: (\d+\.\d+) MB',
            'spike': r'used_ram=(\d+)MB'
        }

        # API features
        api_pattern = r'"(GET|POST|DELETE)\s+[^"]+"\s+status:\s+(\d+)\s+len:\s+(\d+)\s+time:\s+(\d+\.\d+)'

        for idx, row in df.iterrows():
            content = str(row['Content'])

            # Extract memory features
            for feature, pattern in memory_patterns.items():
                match = re.search(pattern, content)
                if match:
                    value = float(match.group(1))
                    if feature == 'claim':
                        df.at[idx, 'memory_claim_mb'] = value
                    elif feature == 'used':
                        df.at[idx, 'memory_used_mb'] = value
                    elif feature == 'total':
                        df.at[idx, 'memory_total_mb'] = value
                    elif feature == 'free':
                        df.at[idx, 'memory_free_mb'] = value
                    elif feature == 'spike':
                        df.at[idx, 'memory_used_mb'] = value

            # Calculate memory utilization
            if df.at[idx, 'memory_total_mb'] > 0:
                df.at[idx, 'memory_utilization_pct'] = (
                    df.at[idx, 'memory_used_mb'] / df.at[idx, 'memory_total_mb']
                ) * 100

            # Extract API features
            api_match = re.search(api_pattern, content)
            if api_match:
                df.at[idx, 'http_status_code'] = int(api_match.group(2))
                df.at[idx, 'api_content_length'] = int(api_match.group(3))
                df.at[idx, 'api_response_time'] = float(api_match.group(4))

        logger.info("Numerical features extracted")
        return df

    def extract_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and encode categorical features."""
        logger.info("Extracting categorical features...")

        # Extract HTTP method
        df['http_method'] = df['Content'].str.extract(r'"(GET|POST|DELETE|PUT|PATCH)')[0].fillna('UNKNOWN')

        # Extract VM event type
        df['vm_event_type'] = df['Content'].str.extract(r'VM (Started|Paused|Resumed|Stopped)')[0].fillna('NONE')

        # Extract instance ID (hash for privacy)
        instance_pattern = r'instance: ([a-f0-9-]{36})'
        df['instance_id_hash'] = df['Content'].str.extract(instance_pattern)[0].fillna('NO_INSTANCE')

        # Create binary flags
        df['has_memory_info'] = (df['memory_claim_mb'] > 0).astype(int)
        df['has_api_info'] = (df['api_response_time'] > 0).astype(int)
        df['is_vm_event'] = (df['vm_event_type'] != 'NONE').astype(int)
        df['is_warning'] = (df['Level'] == 'WARNING').astype(int)
        df['is_error'] = (df['Level'] == 'ERROR').astype(int)
        df['is_slow_api'] = (df['api_response_time'] >= self.config.API_LATENCY_THRESHOLD).astype(int)
        df['is_memory_spike'] = (df['memory_used_mb'] >= self.config.MEMORY_SPIKE_THRESHOLD).astype(int)
        df['is_http_error'] = (df['http_status_code'] >= 400).astype(int)

        logger.info("Categorical features extracted")
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from timestamps."""
        logger.info("Creating temporal features...")

        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        df['time_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

        # Time deltas between events
        df['time_delta'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
        df['time_delta'] = df['time_delta'].clip(0, 300)  # Cap at 5 minutes

        # Sequence position (within time windows)
        df['sequence_position'] = 0

        logger.info("Temporal features created")
        return df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[List[Dict], List[List[int]]]:
        """Create sequences from the processed dataframe with event-level labels."""
        logger.info("Creating sequences with event-level labels...")

        sequences = []
        labels = []

        # Create time windows
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        current_time = start_time
        window_delta = timedelta(minutes=self.config.WINDOW_SIZE_MINUTES)

        while current_time < end_time:
            window_end = current_time + window_delta

            # Get events in this window
            window_mask = (df['timestamp'] >= current_time) & (df['timestamp'] < window_end)
            window_events = df[window_mask].copy()

            if len(window_events) > 0:
                # Add sequence position
                window_events['sequence_position'] = range(len(window_events))

                # Pad or truncate to fixed length
                if len(window_events) > self.config.SEQUENCE_LENGTH:
                    window_events = window_events.head(self.config.SEQUENCE_LENGTH)

                # Create sequence features
                sequence_data = self._create_sequence_features(window_events)

                # Event-level labels: Get label for each event in the sequence
                event_labels = window_events['is_anomaly'].values.tolist()

                # Pad labels to match sequence length
                pad_length = self.config.SEQUENCE_LENGTH - len(event_labels)
                event_labels += [0] * pad_length  # Pad with 0 (normal)

                sequences.append(sequence_data)
                labels.append(event_labels)

            current_time = window_end

        # Calculate statistics
        total_events = sum(seq['sequence_length'] for seq in sequences)
        total_anomalies = sum(sum(label[:seq['sequence_length']]) for label, seq in zip(labels, sequences))

        logger.info(f"Created {len(sequences)} sequences")
        logger.info(f"Total events: {total_events}, Anomalies: {total_anomalies}")
        logger.info(f"Event-level anomaly ratio: {total_anomalies/total_events*100:.1f}%")

        return sequences, labels

    def _create_sequence_features(self, window_events: pd.DataFrame) -> Dict[str, Any]:
        """Create features for a single sequence window."""

        # Numerical features
        numerical_features = [
            'memory_claim_mb', 'memory_used_mb', 'memory_total_mb', 'memory_free_mb',
            'memory_utilization_pct', 'api_response_time', 'api_content_length',
            'http_status_code', 'hour', 'minute', 'second', 'time_since_start',
            'time_delta', 'sequence_position'
        ]

        # Categorical features
        categorical_features = [
            'Level', 'Component', 'EventId', 'http_method', 'vm_event_type', 'instance_id_hash'
        ]

        # Binary features
        binary_features = [
            'has_memory_info', 'has_api_info', 'is_vm_event', 'is_warning',
            'is_error', 'is_slow_api', 'is_memory_spike', 'is_http_error'
        ]

        # Pad sequences to fixed length
        sequence_length = len(window_events)
        pad_length = self.config.SEQUENCE_LENGTH - sequence_length

        sequence_data = {
            'sequence_length': sequence_length,
            'numerical': [],
            'categorical': {},
            'binary': []
        }

        # Process numerical features
        for feature in numerical_features:
            values = window_events[feature].values.tolist()
            values += [0.0] * pad_length  # Pad with zeros
            sequence_data['numerical'].append(values)

        # Process categorical features
        for feature in categorical_features:
            values = window_events[feature].fillna('UNK').values.tolist()
            values += ['PAD'] * pad_length  # Pad with special token
            sequence_data['categorical'][feature] = values

        # Process binary features
        for feature in binary_features:
            values = window_events[feature].values.tolist()
            values += [0] * pad_length  # Pad with zeros
            sequence_data['binary'].append(values)

        # Aggregate features for the entire sequence
        sequence_data['aggregates'] = {
            'total_events': sequence_length,
            'memory_events': window_events['has_memory_info'].sum(),
            'api_events': window_events['has_api_info'].sum(),
            'vm_events': window_events['is_vm_event'].sum(),
            'warning_events': window_events['is_warning'].sum(),
            'error_events': window_events['is_error'].sum(),
            'avg_response_time': window_events['api_response_time'].mean(),
            'max_memory_used': window_events['memory_used_mb'].max(),
            'unique_components': window_events['Component'].nunique()
        }

        return sequence_data

    def encode_categorical_features(self, sequences: List[Dict]) -> List[Dict]:
        """Encode categorical features using label encoders."""
        logger.info("Encoding categorical features...")

        # Collect all categorical values
        categorical_values = {}
        for seq in sequences:
            for feature, values in seq['categorical'].items():
                if feature not in categorical_values:
                    categorical_values[feature] = set()
                categorical_values[feature].update(values)

        # Create encoders
        for feature, values in categorical_values.items():
            encoder = LabelEncoder()
            encoder.fit(list(values))
            self.encoders[feature] = encoder

        # Apply encoding
        for seq in sequences:
            for feature, values in seq['categorical'].items():
                encoded_values = self.encoders[feature].transform(values)
                seq['categorical'][feature] = encoded_values.tolist()

        logger.info("Categorical encoding complete")
        return sequences

    def normalize_numerical_features(self, sequences: List[Dict]) -> List[Dict]:
        """Normalize numerical features."""
        logger.info("Normalizing numerical features...")

        # Collect all numerical values
        all_numerical = []
        for seq in sequences:
            numerical_array = np.array(seq['numerical'])  # Shape: (num_features, sequence_length)
            all_numerical.append(numerical_array)

        # Stack all sequences and compute statistics
        all_data = np.stack(all_numerical, axis=0)  # Shape: (num_sequences, num_features, sequence_length)

        # Normalize per feature across all sequences and time steps
        num_features = all_data.shape[1]
        for feature_idx in range(num_features):
            feature_data = all_data[:, feature_idx, :].flatten()

            # Remove padding zeros for better statistics
            non_zero_data = feature_data[feature_data != 0]

            if len(non_zero_data) > 0:
                scaler = StandardScaler()
                scaler.fit(non_zero_data.reshape(-1, 1))
                self.scalers[f'numerical_{feature_idx}'] = scaler

                # Apply normalization
                for seq_idx, seq in enumerate(sequences):
                    values = np.array(seq['numerical'][feature_idx])
                    # Only normalize non-zero values (preserve padding)
                    non_zero_mask = values != 0
                    if non_zero_mask.any():
                        values[non_zero_mask] = scaler.transform(values[non_zero_mask].reshape(-1, 1)).flatten()
                    seq['numerical'][feature_idx] = values.tolist()

        logger.info("Numerical normalization complete")
        return sequences

    def prepare_model_inputs(self, sequences: List[Dict], labels: List[List[int]]) -> Tuple[Dict, torch.Tensor]:
        """Prepare inputs for PyTorch model with event-level labels."""
        logger.info("Preparing model inputs with event-level labels...")

        batch_size = len(sequences)
        seq_len = self.config.SEQUENCE_LENGTH

        # Prepare numerical features
        numerical_features = []
        for seq in sequences:
            numerical_array = np.array(seq['numerical']).T  # Transpose to (seq_len, num_features)
            numerical_features.append(numerical_array)

        numerical_tensor = torch.tensor(np.stack(numerical_features), dtype=torch.float32)

        # Prepare categorical features
        categorical_tensors = {}
        categorical_feature_names = list(sequences[0]['categorical'].keys())

        for feature_name in categorical_feature_names:
            feature_sequences = []
            for seq in sequences:
                feature_sequences.append(seq['categorical'][feature_name])
            categorical_tensors[feature_name] = torch.tensor(feature_sequences, dtype=torch.long)

        # Prepare binary features
        binary_features = []
        for seq in sequences:
            binary_array = np.array(seq['binary']).T  # Transpose to (seq_len, num_binary_features)
            binary_features.append(binary_array)

        binary_tensor = torch.tensor(np.stack(binary_features), dtype=torch.float32)

        # Prepare aggregate features
        aggregate_features = []
        aggregate_feature_names = list(sequences[0]['aggregates'].keys())
        for seq in sequences:
            aggregate_values = [seq['aggregates'][name] for name in aggregate_feature_names]
            aggregate_features.append(aggregate_values)

        aggregate_tensor = torch.tensor(aggregate_features, dtype=torch.float32)

        # Prepare sequence lengths
        sequence_lengths = torch.tensor([seq['sequence_length'] for seq in sequences], dtype=torch.long)

        # Prepare event-level labels [batch_size, seq_len]
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        model_inputs = {
            'numerical': numerical_tensor,
            'categorical': categorical_tensors,
            'binary': binary_tensor,
            'aggregates': aggregate_tensor,
            'sequence_lengths': sequence_lengths
        }

        logger.info(f"Model inputs prepared - Batch size: {batch_size}, Sequence length: {seq_len}")

        return model_inputs, labels_tensor

    def create_data_splits(self, model_inputs: Dict, labels: torch.Tensor) -> Tuple[Dict, Dict, Dict]:
        """Create train/validation/test splits."""
        logger.info("Creating data splits...")

        # Get indices for splitting
        n_samples = len(labels)
        indices = np.arange(n_samples)

        # For small datasets, simple split without stratification
        if n_samples <= 3:
            logger.warning(f"Very small dataset ({n_samples} samples). Using simple split without stratification.")
            # Simple split for tiny datasets
            train_indices = np.array([0])
            val_indices = np.array([1]) if n_samples > 1 else np.array([0])
            test_indices = np.array([2]) if n_samples > 2 else np.array([0])
        else:
            # First split: train+val / test
            train_val_indices, test_indices = train_test_split(
                indices, test_size=self.config.TEST_SIZE, random_state=42,
                stratify=labels.numpy() if len(np.unique(labels.numpy())) > 1 else None
            )

            # Second split: train / val
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=self.config.VAL_SIZE/(1-self.config.TEST_SIZE),
                random_state=42,
                stratify=labels[train_val_indices].numpy() if len(np.unique(labels[train_val_indices].numpy())) > 1 else None
            )

        # Create splits
        splits = {}
        for split_name, split_indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
            split_data = {}

            # Split each input tensor
            split_data['numerical'] = model_inputs['numerical'][split_indices]
            split_data['binary'] = model_inputs['binary'][split_indices]
            split_data['aggregates'] = model_inputs['aggregates'][split_indices]
            split_data['sequence_lengths'] = model_inputs['sequence_lengths'][split_indices]

            # Split categorical features
            split_data['categorical'] = {}
            for feature_name, feature_tensor in model_inputs['categorical'].items():
                split_data['categorical'][feature_name] = feature_tensor[split_indices]

            splits[split_name] = (split_data, labels[split_indices])

        logger.info(f"Data splits created:")
        logger.info(f"- Train: {len(train_indices)} samples ({len(train_indices)/n_samples*100:.1f}%)")
        logger.info(f"- Validation: {len(val_indices)} samples ({len(val_indices)/n_samples*100:.1f}%)")
        logger.info(f"- Test: {len(test_indices)} samples ({len(test_indices)/n_samples*100:.1f}%)")

        return splits['train'], splits['val'], splits['test']

    def save_feature_artifacts(self, output_dir: str):
        """Save encoders, scalers, and configuration."""
        logger.info("Saving feature artifacts...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save encoders
        encoders_dict = {}
        for name, encoder in self.encoders.items():
            encoders_dict[name] = {
                'classes_': encoder.classes_.tolist()
            }

        with open(output_path / "encoders.json", "w") as f:
            json.dump(encoders_dict, f, indent=2)

        # Save scalers
        scalers_dict = {}
        for name, scaler in self.scalers.items():
            scalers_dict[name] = {
                'mean_': scaler.mean_.tolist() if hasattr(scaler.mean_, 'tolist') else float(scaler.mean_),
                'scale_': scaler.scale_.tolist() if hasattr(scaler.scale_, 'tolist') else float(scaler.scale_)
            }

        with open(output_path / "scalers.json", "w") as f:
            json.dump(scalers_dict, f, indent=2)

        # Save configuration
        config_dict = {
            'SEQUENCE_LENGTH': self.config.SEQUENCE_LENGTH,
            'WINDOW_SIZE_MINUTES': self.config.WINDOW_SIZE_MINUTES,
            'MEMORY_SPIKE_THRESHOLD': self.config.MEMORY_SPIKE_THRESHOLD,
            'API_LATENCY_THRESHOLD': self.config.API_LATENCY_THRESHOLD,
            'BATCH_SIZE': self.config.BATCH_SIZE
        }

        with open(output_path / "feature_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Feature artifacts saved to {output_path}")

    def process_full_pipeline(self, csv_path: str, output_dir: str = "features_output") -> Tuple[Dict, Dict, Dict]:
        """Run the complete feature engineering pipeline."""
        logger.info("Starting full feature engineering pipeline...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Step 1: Load data
        df = self.load_processed_data(csv_path)

        # Step 2: Extract features
        df = self.extract_numerical_features(df)
        df = self.extract_categorical_features(df)
        df = self.create_temporal_features(df)

        # Step 3: Create sequences
        sequences, labels = self.create_sequences(df)

        # Step 4: Encode and normalize
        sequences = self.encode_categorical_features(sequences)
        sequences = self.normalize_numerical_features(sequences)

        # Step 5: Prepare model inputs
        model_inputs, labels_tensor = self.prepare_model_inputs(sequences, labels)

        # Step 6: Create splits
        train_data, val_data, test_data = self.create_data_splits(model_inputs, labels_tensor)

        # Step 7: Save artifacts
        self.save_feature_artifacts(output_dir)

        # Save processed data
        torch.save(train_data, output_path / "train_data.pt")
        torch.save(val_data, output_path / "val_data.pt")
        torch.save(test_data, output_path / "test_data.pt")

        # Save feature statistics
        # Calculate event-level anomaly ratio
        total_events = sum(seq['sequence_length'] for seq in sequences)
        total_anomalies = sum(sum(label[:seq['sequence_length']]) for label, seq in zip(labels, sequences))

        feature_stats = {
            'total_sequences': len(sequences),
            'total_events': total_events,
            'total_anomalies': total_anomalies,
            'event_level_anomaly_ratio': total_anomalies / total_events if total_events > 0 else 0,
            'numerical_features': len(sequences[0]['numerical']),
            'categorical_features': len(sequences[0]['categorical']),
            'binary_features': len(sequences[0]['binary']),
            'aggregate_features': len(sequences[0]['aggregates']),
            'vocab_sizes': {name: len(encoder.classes_) for name, encoder in self.encoders.items()}
        }

        with open(output_path / "feature_stats.json", "w") as f:
            json.dump(feature_stats, f, indent=2)

        logger.info("Feature engineering pipeline complete!")
        logger.info(f"Results saved to {output_path}")

        return train_data, val_data, test_data

def main():
    """Main execution function."""
    # Configuration
    from config import DataConfig
    PROCESSED_CSV_PATH = str(DataConfig.PROCESSED_DATASET_PATH)
    OUTPUT_DIR = "features_output"

    # Initialize feature engineer
    config = FeatureConfig()
    engineer = OpenStackFeatureEngineer(config)

    try:
        # Run full pipeline
        train_data, val_data, test_data = engineer.process_full_pipeline(
            PROCESSED_CSV_PATH, OUTPUT_DIR
        )

        print(f"\nFeature engineering complete!")
        print(f"Data splits created:")
        print(f"- Training samples: {len(train_data[1])}")
        print(f"- Validation samples: {len(val_data[1])}")
        print(f"- Test samples: {len(test_data[1])}")
        print(f"Check '{OUTPUT_DIR}' directory for all outputs.")

    except FileNotFoundError:
        print(f"Error: Could not find {PROCESSED_CSV_PATH}")
        print("Please run data_analyzer.py first to create the processed dataset.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()