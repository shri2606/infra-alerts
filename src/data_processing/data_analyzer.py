#!/usr/bin/env python3
"""
OpenStack Log Data Analyzer
============================

This script analyzes OpenStack log data to extract patterns, anomalies,
and insights for training the CloudInfraAI model.

Author: CloudInfraAI Team
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenStackLogAnalyzer:
    """Analyze OpenStack logs to identify patterns and anomalies."""

    def __init__(self, csv_path: str):
        """Initialize analyzer with dataset path."""
        self.csv_path = csv_path
        self.df = None
        self.analysis_results = {}

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the OpenStack log data."""
        logger.info(f"Loading data from {self.csv_path}")

        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} log entries")

            # Convert timestamp columns
            self.df['timestamp'] = pd.to_datetime(
                self.df['Date'] + ' ' + self.df['Time'],
                format='%Y-%m-%d %H:%M:%S.%f'
            )

            # Sort by timestamp
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)

            return self.df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def analyze_memory_patterns(self) -> Dict:
        """Analyze memory usage patterns in the logs."""
        logger.info("Analyzing memory patterns...")

        memory_patterns = {
            'memory_claims': [],
            'memory_usage': [],
            'memory_spikes': [],
            'baseline_usage': []
        }

        # Extract memory claims
        memory_claim_pattern = r'memory (\d+) MB'
        memory_usage_pattern = r'used: (\d+\.\d+) MB'
        memory_total_pattern = r'Total memory: (\d+) MB'
        memory_spike_pattern = r'used_ram=(\d+)MB'

        for idx, row in self.df.iterrows():
            content = str(row['Content'])

            # Memory claims
            claim_match = re.search(memory_claim_pattern, content)
            if claim_match:
                memory_patterns['memory_claims'].append({
                    'timestamp': row['timestamp'],
                    'memory_mb': int(claim_match.group(1)),
                    'instance_id': self._extract_instance_id(content),
                    'line_id': row['LineId']
                })

            # Memory usage
            usage_match = re.search(memory_usage_pattern, content)
            total_match = re.search(memory_total_pattern, content)
            if usage_match and total_match:
                used_mb = float(usage_match.group(1))
                total_mb = int(total_match.group(1))
                memory_patterns['memory_usage'].append({
                    'timestamp': row['timestamp'],
                    'used_mb': used_mb,
                    'total_mb': total_mb,
                    'utilization_pct': (used_mb / total_mb) * 100,
                    'line_id': row['LineId']
                })

                # Check for baseline vs spike
                if used_mb <= 512:
                    memory_patterns['baseline_usage'].append(used_mb)

            # Memory spikes
            spike_match = re.search(memory_spike_pattern, content)
            if spike_match:
                spike_mb = int(spike_match.group(1))
                if spike_mb >= 2560:  # Anomaly threshold
                    memory_patterns['memory_spikes'].append({
                        'timestamp': row['timestamp'],
                        'spike_mb': spike_mb,
                        'line_id': row['LineId']
                    })

        # Calculate statistics
        claims_df = pd.DataFrame(memory_patterns['memory_claims'])
        if not claims_df.empty:
            self.analysis_results['memory_claims_stats'] = {
                'total_claims': len(claims_df),
                'avg_claim_mb': claims_df['memory_mb'].mean(),
                'std_claim_mb': claims_df['memory_mb'].std(),
                'most_common_claim': claims_df['memory_mb'].mode().iloc[0] if not claims_df['memory_mb'].mode().empty else None
            }

        usage_df = pd.DataFrame(memory_patterns['memory_usage'])
        if not usage_df.empty:
            self.analysis_results['memory_usage_stats'] = {
                'total_measurements': len(usage_df),
                'avg_utilization_pct': usage_df['utilization_pct'].mean(),
                'max_utilization_pct': usage_df['utilization_pct'].max(),
                'baseline_measurements': len([u for u in usage_df['used_mb'] if u <= 512])
            }

        self.analysis_results['memory_spikes_count'] = len(memory_patterns['memory_spikes'])
        logger.info(f"Found {len(memory_patterns['memory_spikes'])} memory spikes")

        return memory_patterns

    def analyze_api_patterns(self) -> Dict:
        """Analyze API request patterns and response times."""
        logger.info("Analyzing API patterns...")

        api_patterns = {
            'requests': [],
            'slow_requests': [],
            'error_requests': []
        }

        # API request pattern: "METHOD /path" status: XXX len: XXX time: X.XXXXX
        api_pattern = r'"(GET|POST|DELETE)\s+([^"]+)"\s+status:\s+(\d+)\s+len:\s+(\d+)\s+time:\s+(\d+\.\d+)'

        for idx, row in self.df.iterrows():
            content = str(row['Content'])

            api_match = re.search(api_pattern, content)
            if api_match:
                method = api_match.group(1)
                path = api_match.group(2)
                status = int(api_match.group(3))
                length = int(api_match.group(4))
                time_sec = float(api_match.group(5))

                request_data = {
                    'timestamp': row['timestamp'],
                    'method': method,
                    'path': path,
                    'status': status,
                    'response_length': length,
                    'response_time': time_sec,
                    'line_id': row['LineId']
                }

                api_patterns['requests'].append(request_data)

                # Check for slow requests (anomaly threshold)
                if time_sec >= 0.5:
                    api_patterns['slow_requests'].append(request_data)

                # Check for error status codes
                if status >= 400:
                    api_patterns['error_requests'].append(request_data)

        # Calculate statistics
        requests_df = pd.DataFrame(api_patterns['requests'])
        if not requests_df.empty:
            self.analysis_results['api_stats'] = {
                'total_requests': len(requests_df),
                'avg_response_time': requests_df['response_time'].mean(),
                'median_response_time': requests_df['response_time'].median(),
                'slow_requests_count': len(api_patterns['slow_requests']),
                'error_requests_count': len(api_patterns['error_requests']),
                'method_distribution': requests_df['method'].value_counts().to_dict(),
                'status_distribution': requests_df['status'].value_counts().to_dict()
            }

        logger.info(f"Found {len(api_patterns['requests'])} API requests")
        logger.info(f"Found {len(api_patterns['slow_requests'])} slow requests")
        logger.info(f"Found {len(api_patterns['error_requests'])} error requests")

        return api_patterns

    def analyze_vm_lifecycle(self) -> Dict:
        """Analyze VM lifecycle events and patterns."""
        logger.info("Analyzing VM lifecycle patterns...")

        lifecycle_patterns = {
            'events': [],
            'sequences': {},
            'anomalies': []
        }

        # VM lifecycle event pattern
        vm_event_pattern = r'VM (Started|Paused|Resumed|Stopped) \(Lifecycle Event\)'

        for idx, row in self.df.iterrows():
            content = str(row['Content'])

            vm_match = re.search(vm_event_pattern, content)
            if vm_match:
                event_type = vm_match.group(1)
                instance_id = self._extract_instance_id(content)

                event_data = {
                    'timestamp': row['timestamp'],
                    'event_type': event_type,
                    'instance_id': instance_id,
                    'line_id': row['LineId']
                }

                lifecycle_patterns['events'].append(event_data)

                # Track sequences per instance
                if instance_id not in lifecycle_patterns['sequences']:
                    lifecycle_patterns['sequences'][instance_id] = []

                lifecycle_patterns['sequences'][instance_id].append({
                    'timestamp': row['timestamp'],
                    'event_type': event_type,
                    'line_id': row['LineId']
                })

        # Analyze sequences for anomalies
        for instance_id, events in lifecycle_patterns['sequences'].items():
            events.sort(key=lambda x: x['timestamp'])

            # Check for duplicate Resume events (anomaly pattern)
            resume_events = [e for e in events if e['event_type'] == 'Resumed']
            if len(resume_events) > 2:  # More than expected Resume events
                lifecycle_patterns['anomalies'].append({
                    'instance_id': instance_id,
                    'anomaly_type': 'duplicate_resume',
                    'resume_count': len(resume_events),
                    'events': events
                })

            # Check for incomplete sequences
            event_types = [e['event_type'] for e in events]
            if 'Started' in event_types and 'Stopped' not in event_types:
                lifecycle_patterns['anomalies'].append({
                    'instance_id': instance_id,
                    'anomaly_type': 'incomplete_sequence',
                    'missing_event': 'Stopped',
                    'events': events
                })

        # Calculate statistics
        self.analysis_results['vm_lifecycle_stats'] = {
            'total_events': len(lifecycle_patterns['events']),
            'unique_instances': len(lifecycle_patterns['sequences']),
            'anomalies_count': len(lifecycle_patterns['anomalies']),
            'event_type_distribution': {}
        }

        events_df = pd.DataFrame(lifecycle_patterns['events'])
        if not events_df.empty:
            self.analysis_results['vm_lifecycle_stats']['event_type_distribution'] = \
                events_df['event_type'].value_counts().to_dict()

        logger.info(f"Found {len(lifecycle_patterns['events'])} VM lifecycle events")
        logger.info(f"Found {len(lifecycle_patterns['anomalies'])} lifecycle anomalies")

        return lifecycle_patterns

    def analyze_system_health(self) -> Dict:
        """Analyze system health indicators and warnings."""
        logger.info("Analyzing system health patterns...")

        health_patterns = {
            'warnings': [],
            'errors': [],
            'component_stats': {},
            'event_template_stats': {}
        }

        # Count by log level
        level_counts = self.df['Level'].value_counts().to_dict()

        # Extract warnings and errors
        warnings_df = self.df[self.df['Level'] == 'WARNING']
        errors_df = self.df[self.df['Level'] == 'ERROR']

        for idx, row in warnings_df.iterrows():
            health_patterns['warnings'].append({
                'timestamp': row['timestamp'],
                'component': row['Component'],
                'content': row['Content'],
                'line_id': row['LineId']
            })

        for idx, row in errors_df.iterrows():
            health_patterns['errors'].append({
                'timestamp': row['timestamp'],
                'component': row['Component'],
                'content': row['Content'],
                'line_id': row['LineId']
            })

        # Component statistics
        health_patterns['component_stats'] = self.df['Component'].value_counts().to_dict()

        # Event template statistics
        health_patterns['event_template_stats'] = self.df['EventId'].value_counts().to_dict()

        # Special warning patterns
        unknown_file_warnings = [
            w for w in health_patterns['warnings']
            if 'Unknown base file' in str(w['content'])
        ]

        self.analysis_results['system_health_stats'] = {
            'log_level_distribution': level_counts,
            'warnings_count': len(health_patterns['warnings']),
            'errors_count': len(health_patterns['errors']),
            'unknown_file_warnings': len(unknown_file_warnings),
            'most_active_component': max(health_patterns['component_stats'].items(), key=lambda x: x[1])[0],
            'most_common_event_template': max(health_patterns['event_template_stats'].items(), key=lambda x: x[1])[0]
        }

        logger.info(f"Found {len(health_patterns['warnings'])} warnings")
        logger.info(f"Found {len(health_patterns['errors'])} errors")
        logger.info(f"Found {len(unknown_file_warnings)} unknown file warnings")

        return health_patterns

    def _extract_instance_id(self, content: str) -> Optional[str]:
        """Extract instance ID from log content."""
        instance_pattern = r'instance: ([a-f0-9-]{36})'
        match = re.search(instance_pattern, content)
        return match.group(1) if match else None

    def create_anomaly_labels(self) -> pd.DataFrame:
        """Create binary labels for anomaly detection."""
        logger.info("Creating anomaly labels...")

        # Initialize all as normal
        self.df['is_anomaly'] = 0

        # Memory spike anomalies
        memory_spike_mask = self.df['Content'].str.contains(r'used_ram=2560MB|used_ram=[3-9]\d{3}MB', na=False, regex=True)
        self.df.loc[memory_spike_mask, 'is_anomaly'] = 1

        # API latency anomalies
        api_slow_mask = self.df['Content'].str.contains(r'time: [0-9]\.[5-9]|time: [1-9]\.\d+', na=False, regex=True)
        self.df.loc[api_slow_mask, 'is_anomaly'] = 1

        # HTTP error anomalies
        http_error_mask = self.df['Content'].str.contains(r'status: 4\d{2}|status: 5\d{2}', na=False, regex=True)
        self.df.loc[http_error_mask, 'is_anomaly'] = 1

        # Warning level anomalies
        warning_mask = self.df['Level'] == 'WARNING'
        self.df.loc[warning_mask, 'is_anomaly'] = 1

        # Error level anomalies
        error_mask = self.df['Level'] == 'ERROR'
        self.df.loc[error_mask, 'is_anomaly'] = 1

        anomaly_count = self.df['is_anomaly'].sum()
        normal_count = len(self.df) - anomaly_count

        logger.info(f"Created labels: {normal_count} normal, {anomaly_count} anomalies")
        logger.info(f"Anomaly ratio: {anomaly_count/len(self.df)*100:.1f}%")

        return self.df

    def generate_analysis_report(self, output_dir: str = "analysis_output"):
        """Generate comprehensive analysis report."""
        logger.info("Generating analysis report...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Run all analyses
        memory_patterns = self.analyze_memory_patterns()
        api_patterns = self.analyze_api_patterns()
        lifecycle_patterns = self.analyze_vm_lifecycle()
        health_patterns = self.analyze_system_health()
        labeled_df = self.create_anomaly_labels()

        # Save analysis results
        with open(output_path / "analysis_results.json", "w") as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        # Save processed dataset with labels
        from config import DataConfig
        labeled_df.to_csv(DataConfig.PROCESSED_DATASET_PATH, index=False)

        # Generate visualizations
        self._create_visualizations(output_path)

        # Generate summary report
        self._create_summary_report(output_path)

        logger.info(f"Analysis complete! Results saved to {output_path}")

        return self.analysis_results

    def _create_visualizations(self, output_path: Path):
        """Create visualization plots."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Anomaly distribution
        anomaly_counts = self.df['is_anomaly'].value_counts()
        axes[0, 0].pie(anomaly_counts.values, labels=['Normal', 'Anomaly'], autopct='%1.1f%%')
        axes[0, 0].set_title('Dataset Distribution: Normal vs Anomaly')

        # 2. Log level distribution
        level_counts = self.df['Level'].value_counts()
        axes[0, 1].bar(level_counts.index, level_counts.values)
        axes[0, 1].set_title('Log Level Distribution')
        axes[0, 1].set_xlabel('Log Level')
        axes[0, 1].set_ylabel('Count')

        # 3. Component activity
        component_counts = self.df['Component'].value_counts().head(10)
        axes[1, 0].barh(range(len(component_counts)), component_counts.values)
        axes[1, 0].set_yticks(range(len(component_counts)))
        axes[1, 0].set_yticklabels([c.split('.')[-1] for c in component_counts.index], fontsize=8)
        axes[1, 0].set_title('Top 10 Most Active Components')
        axes[1, 0].set_xlabel('Event Count')

        # 4. Timeline of events
        hourly_counts = self.df.set_index('timestamp').resample('5T').size()
        axes[1, 1].plot(hourly_counts.index, hourly_counts.values)
        axes[1, 1].set_title('Event Timeline (5-minute intervals)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Event Count')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / "analysis_visualizations.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_summary_report(self, output_path: Path):
        """Create text summary report."""
        with open(output_path / "analysis_summary.txt", "w") as f:
            f.write("OpenStack Log Analysis Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Dataset Overview:\n")
            f.write(f"- Total log entries: {len(self.df)}\n")
            f.write(f"- Time range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}\n")
            f.write(f"- Duration: {self.df['timestamp'].max() - self.df['timestamp'].min()}\n\n")

            anomaly_count = self.df['is_anomaly'].sum()
            f.write(f"Anomaly Detection:\n")
            f.write(f"- Normal events: {len(self.df) - anomaly_count} ({(len(self.df) - anomaly_count)/len(self.df)*100:.1f}%)\n")
            f.write(f"- Anomalous events: {anomaly_count} ({anomaly_count/len(self.df)*100:.1f}%)\n\n")

            if 'memory_spikes_count' in self.analysis_results:
                f.write(f"Memory Analysis:\n")
                f.write(f"- Memory spikes detected: {self.analysis_results['memory_spikes_count']}\n")

            if 'api_stats' in self.analysis_results:
                f.write(f"- API requests analyzed: {self.analysis_results['api_stats']['total_requests']}\n")
                f.write(f"- Slow requests: {self.analysis_results['api_stats']['slow_requests_count']}\n")
                f.write(f"- Error requests: {self.analysis_results['api_stats']['error_requests_count']}\n")

            if 'vm_lifecycle_stats' in self.analysis_results:
                f.write(f"- VM lifecycle events: {self.analysis_results['vm_lifecycle_stats']['total_events']}\n")
                f.write(f"- Lifecycle anomalies: {self.analysis_results['vm_lifecycle_stats']['anomalies_count']}\n")

            f.write(f"\nRecommendations for Model Training:\n")
            f.write(f"- Use 5-minute sliding windows for sequence creation\n")
            f.write(f"- Focus on memory usage, API latency, and lifecycle patterns\n")
            f.write(f"- Apply class weighting due to imbalanced dataset\n")
            f.write(f"- Consider data augmentation for minority class (anomalies)\n")

def main():
    """Main execution function."""
    # Configuration
    CSV_PATH = "dataset/OpenStack_2k.log_structured.csv"
    OUTPUT_DIR = "analysis_output"

    # Initialize analyzer
    analyzer = OpenStackLogAnalyzer(CSV_PATH)

    # Load data
    df = analyzer.load_data()

    # Generate complete analysis
    results = analyzer.generate_analysis_report(OUTPUT_DIR)

    print(f"\nAnalysis complete! Check the '{OUTPUT_DIR}' directory for results.")
    print(f"Key findings:")
    print(f"- Total log entries: {len(df)}")
    print(f"- Anomaly ratio: {df['is_anomaly'].sum()/len(df)*100:.1f}%")
    print(f"- Memory spikes detected: {results.get('memory_spikes_count', 'N/A')}")

if __name__ == "__main__":
    main()