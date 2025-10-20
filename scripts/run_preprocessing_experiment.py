#!/usr/bin/env python3
"""
Experimental Preprocessing Pipeline: 2-Minute Windows with 30-Second Stride
===========================================================================

This script runs the experimental preprocessing with 2-minute windows and
30-second stride to test if more training examples improve recall.

Experiment Details:
- Window size: 2 minutes (changed from 5)
- Stride: 30 seconds (75% overlap)
- Expected: ~24 sequences (vs baseline 3)
- Output: outputs_experiment/ directory

Author: CloudInfraAI Team
Date: October 2024
"""

import os
import sys
import time
import logging
from pathlib import Path
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configuration and modules
from config import DataConfig
from src.data_processing import OpenStackFeatureEngineer, FeatureConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_experimental_feature_engineering():
    """Run feature engineering with experimental configuration."""
    logger.info("="*60)
    logger.info("EXPERIMENTAL FEATURE ENGINEERING")
    logger.info("Configuration: 2-min windows, 30-sec stride")
    logger.info("="*60)

    start_time = time.time()

    try:
        # Use experimental configuration (already set in FeatureConfig)
        config = FeatureConfig()

        logger.info(f"Window size: {config.WINDOW_SIZE_MINUTES} minutes")
        logger.info(f"Window stride: {config.WINDOW_STRIDE_SECONDS} seconds")
        logger.info(f"Sequence length: {config.SEQUENCE_LENGTH} events")

        engineer = OpenStackFeatureEngineer(config)

        # Run pipeline with experimental output directory
        output_dir = "outputs_experiment"
        Path(output_dir).mkdir(exist_ok=True)

        train_data, val_data, test_data = engineer.process_full_pipeline(
            str(DataConfig.PROCESSED_DATASET_PATH),
            output_dir
        )

        feature_time = time.time() - start_time
        logger.info(f"Experimental feature engineering completed in {feature_time:.2f} seconds")

        # Log results
        logger.info("="*60)
        logger.info("EXPERIMENTAL RESULTS:")
        logger.info(f"- Training samples: {len(train_data[1])}")
        logger.info(f"- Validation samples: {len(val_data[1])}")
        logger.info(f"- Test samples: {len(test_data[1])}")

        # Calculate anomaly counts per split
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

        logger.info(f"- Training anomalies: {train_anomalies}")
        logger.info(f"- Validation anomalies: {val_anomalies}")
        logger.info(f"- Test anomalies: {test_anomalies}")
        logger.info("="*60)

        # Load and display feature stats
        with open(f"{output_dir}/feature_stats.json", "r") as f:
            stats = json.load(f)

        logger.info("Feature Statistics:")
        logger.info(f"- Total sequences: {stats['total_sequences']}")
        logger.info(f"- Total events: {stats['total_events']}")
        logger.info(f"- Total anomalies: {stats['total_anomalies']}")
        logger.info(f"- Event-level anomaly ratio: {stats['event_level_anomaly_ratio']*100:.1f}%")

        # Comparison with baseline (expected 3 sequences, 2 train anomalies)
        logger.info("="*60)
        logger.info("COMPARISON WITH BASELINE:")
        logger.info(f"- Sequences: 3 → {stats['total_sequences']} ({stats['total_sequences']/3:.1f}x increase)")
        logger.info(f"- Training anomalies: 2 → {train_anomalies} ({train_anomalies/2:.1f}x increase)")
        logger.info("="*60)

        # Validation checks
        logger.info("VALIDATION CHECKS:")

        # Check 1: Sequence count
        expected_min = 20
        expected_max = 30
        if expected_min <= stats['total_sequences'] <= expected_max:
            logger.info(f"✅ Sequence count within expected range: {stats['total_sequences']}")
        else:
            logger.warning(f"⚠️  Sequence count outside expected range: {stats['total_sequences']} (expected {expected_min}-{expected_max})")

        # Check 2: Anomaly ratio
        if 0.04 <= stats['event_level_anomaly_ratio'] <= 0.08:
            logger.info(f"✅ Anomaly ratio looks reasonable: {stats['event_level_anomaly_ratio']*100:.1f}%")
        else:
            logger.warning(f"⚠️  Anomaly ratio unusual: {stats['event_level_anomaly_ratio']*100:.1f}% (expected 4-8%)")

        # Check 3: Training anomalies
        if train_anomalies >= 30:
            logger.info(f"✅ Sufficient training anomalies: {train_anomalies}")
        else:
            logger.warning(f"⚠️  Low training anomalies: {train_anomalies} (expected ≥30)")

        logger.info("="*60)

        return True

    except Exception as e:
        logger.error(f"Experimental feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution."""
    pipeline_start = time.time()

    logger.info("🧪 Starting Experimental Preprocessing Pipeline")
    logger.info(f"Experiment: 2-Minute Windows with 30-Second Stride")
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run experimental preprocessing
    if not run_experimental_feature_engineering():
        logger.error("❌ Experimental preprocessing failed. Exiting.")
        sys.exit(1)

    pipeline_time = time.time() - pipeline_start
    logger.info(f"🎉 Experimental preprocessing completed in {pipeline_time:.2f} seconds")
    logger.info(f"📁 Outputs saved to: outputs_experiment/")
    logger.info(f"📋 Next step: Run training with experimental data")

if __name__ == "__main__":
    main()
