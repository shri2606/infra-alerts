"""
Real-Time Feature Extractor
============================

Extract features from raw log events for real-time inference.
Strictly mirrors the logic in OpenStackFeatureEngineer to ensure alignment.
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class RealTimeFeatureExtractor:
    """
    Extract features from raw log events for real-time inference.
    Strictly mirrors the logic in OpenStackFeatureEngineer to ensure alignment.
    """

    def __init__(self, artifacts_dir: str = "outputs_experiment"):
        self.artifacts_dir = Path(artifacts_dir)
        self.encoders = self._load_encoders()
        self.scalers = self._load_scalers()
        self.config = self._load_feature_config()

        # FEATURE DEFINITIONS (MUST MATCH TRAINING EXACTLY)
        self.NUMERICAL_FEATURES = [
            'memory_claim_mb', 'memory_used_mb', 'memory_total_mb', 'memory_free_mb',
            'memory_utilization_pct', 'api_response_time', 'api_content_length',
            'http_status_code', 'hour', 'minute', 'second', 'time_since_start',
            'time_delta', 'sequence_position'
        ]

        self.CATEGORICAL_FEATURES = [
            'Level', 'Component', 'EventId', 'http_method', 'vm_event_type', 'instance_id_hash'
        ]

        self.BINARY_FEATURES = [
            'has_memory_info', 'has_api_info', 'is_vm_event', 'is_warning',
            'is_error', 'is_slow_api', 'is_memory_spike', 'is_http_error'
        ]

        self.AGGREGATE_KEYS = [
            'total_events', 'memory_events', 'api_events', 'vm_events',
            'warning_events', 'error_events', 'avg_response_time',
            'max_memory_used', 'unique_components'
        ]

    def _load_encoders(self) -> Dict:
        path = self.artifacts_dir / "encoders.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing")
        with open(path, 'r') as f:
            return json.load(f)

    def _load_scalers(self) -> Dict:
        path = self.artifacts_dir / "scalers.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing")
        with open(path, 'r') as f:
            return json.load(f)

    def _load_feature_config(self) -> Dict:
        path = self.artifacts_dir / "feature_config.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing")
        with open(path, 'r') as f:
            return json.load(f)

    def process_events_to_features(self, events: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Main entry point: Transforms a list of raw log dicts into model input tensors.
        Expects a sequence of events (usually 50).
        """
        # Convert to DataFrame to reuse vectorization logic
        df = pd.DataFrame(events)

        # Ensure timestamp is datetime
        if 'Date' in df.columns and 'Time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        elif 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()

        # Extract Base Features
        df = self._extract_base_features(df)

        # Create Sequence-Level Features
        return self._transform_sequence(df)

    def _extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mirror of extract_numerical/categorical/temporal from feature_engineer.py"""

        # Initialize columns with defaults
        for col in self.NUMERICAL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        # Regex Patterns
        patterns = {
            'claim': r'memory (\d+) MB',
            'used': r'used: (\d+\.\d+) MB',
            'total': r'Total memory: (\d+) MB',
            'free': r'free: (\d+\.\d+) MB',
            'spike': r'used_ram=(\d+)MB',
            'api': r'"(GET|POST|DELETE)\s+[^"]+"\s+status:\s+(\d+)\s+len:\s+(\d+)\s+time:\s+(\d+\.\d+)',
            'instance': r'instance: ([a-f0-9-]{36})',
            'vm_event': r'VM (Started|Paused|Resumed|Stopped)'
        }

        start_time = df['timestamp'].min()

        for idx, row in df.iterrows():
            content = str(row.get('Content', ''))

            # Numerical (Memory)
            if 'claim' in content:
                m = re.search(patterns['claim'], content)
                if m:
                    df.at[idx, 'memory_claim_mb'] = float(m.group(1))

            if 'used:' in content:
                m = re.search(patterns['used'], content)
                if m:
                    df.at[idx, 'memory_used_mb'] = float(m.group(1))

            if 'Total memory:' in content:
                m = re.search(patterns['total'], content)
                if m:
                    df.at[idx, 'memory_total_mb'] = float(m.group(1))

            if 'used_ram=' in content:
                m = re.search(patterns['spike'], content)
                if m:
                    df.at[idx, 'memory_used_mb'] = float(m.group(1))

            # Memory Utilization
            if df.at[idx, 'memory_total_mb'] > 0:
                df.at[idx, 'memory_utilization_pct'] = (df.at[idx, 'memory_used_mb'] / df.at[idx, 'memory_total_mb']) * 100

            # Numerical (API)
            m_api = re.search(patterns['api'], content)
            if m_api:
                df.at[idx, 'http_method'] = m_api.group(1)
                df.at[idx, 'http_status_code'] = int(m_api.group(2))
                df.at[idx, 'api_content_length'] = int(m_api.group(3))
                df.at[idx, 'api_response_time'] = float(m_api.group(4))
            else:
                df.at[idx, 'http_method'] = 'UNKNOWN'

            # Categorical Extras
            m_vm = re.search(patterns['vm_event'], content)
            df.at[idx, 'vm_event_type'] = m_vm.group(1) if m_vm else 'NONE'

            m_inst = re.search(patterns['instance'], content)
            df.at[idx, 'instance_id_hash'] = m_inst.group(1) if m_inst else 'NO_INSTANCE'

            # Temporal
            ts = row['timestamp']
            df.at[idx, 'hour'] = ts.hour
            df.at[idx, 'minute'] = ts.minute
            df.at[idx, 'second'] = ts.second
            df.at[idx, 'time_since_start'] = (ts - start_time).total_seconds()

            # Time Delta
            if idx > 0:
                prev_ts = df.iloc[idx-1]['timestamp']
                df.at[idx, 'time_delta'] = (ts - prev_ts).total_seconds()
            else:
                df.at[idx, 'time_delta'] = 0.0

            df.at[idx, 'sequence_position'] = idx

        # Binary Features
        df['has_memory_info'] = (df['memory_claim_mb'] > 0).astype(int)
        df['has_api_info'] = (df['api_response_time'] > 0).astype(int)
        df['is_vm_event'] = (df['vm_event_type'] != 'NONE').astype(int)
        df['is_warning'] = (df['Level'] == 'WARNING').astype(int)
        df['is_error'] = (df['Level'] == 'ERROR').astype(int)
        df['is_slow_api'] = (df['api_response_time'] >= self.config.get('API_LATENCY_THRESHOLD', 0.5)).astype(int)
        df['is_memory_spike'] = (df['memory_used_mb'] >= self.config.get('MEMORY_SPIKE_THRESHOLD', 2560)).astype(int)
        df['is_http_error'] = (df['http_status_code'] >= 400).astype(int)

        return df

    def _transform_sequence(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Apply scaling and encoding to match training artifacts."""

        # Numerical Scaling
        numerical_data = []
        for i, feature in enumerate(self.NUMERICAL_FEATURES):
            values = df[feature].values.astype(float)

            # Use the saved scaler
            scaler_name = f"numerical_{i}"
            if scaler_name in self.scalers:
                s = self.scalers[scaler_name]
                mean = s['mean_'][0]
                scale = s['scale_'][0]
                if scale != 0:
                    values = (values - mean) / scale

            numerical_data.append(values)

        # Categorical Encoding
        categorical_data = {}
        for feature in self.CATEGORICAL_FEATURES:
            raw_values = df[feature].fillna('UNK').astype(str).values
            encoded_values = []

            if feature in self.encoders:
                classes = self.encoders[feature]['classes_']
                class_map = {label: idx for idx, label in enumerate(classes)}

                for v in raw_values:
                    encoded_values.append(class_map.get(v, 0))
            else:
                encoded_values = [0] * len(raw_values)

            categorical_data[feature] = np.array(encoded_values)

        # Binary Features
        binary_data = []
        for feature in self.BINARY_FEATURES:
            binary_data.append(df[feature].values)

        # Aggregates
        aggregates = [
            len(df),
            df['has_memory_info'].sum(),
            df['has_api_info'].sum(),
            df['is_vm_event'].sum(),
            df['is_warning'].sum(),
            df['is_error'].sum(),
            df['api_response_time'].mean(),
            df['memory_used_mb'].max(),
            df['Component'].nunique()
        ]

        # Format Output
        return {
            'numerical': np.array(numerical_data).T[np.newaxis, ...].astype(np.float32),
            'categorical': {k: v[np.newaxis, ...] for k, v in categorical_data.items()},
            'binary': np.array(binary_data).T[np.newaxis, ...].astype(np.float32),
            'aggregates': np.array(aggregates)[np.newaxis, ...].astype(np.float32)
        }
