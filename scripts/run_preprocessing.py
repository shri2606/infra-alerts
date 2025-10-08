#!/usr/bin/env python3
"""
Complete Data Preprocessing Pipeline Runner
==========================================

This script runs the complete data analysis and feature engineering pipeline
for the CloudInfraAI project. It orchestrates both data_analyzer.py and
feature_engineer.py to produce model-ready datasets.

Author: CloudInfraAI Team
Date: 2024

Usage:
    python scripts/run_preprocessing.py

Requirements:
    - data/raw/OpenStack_2k.log_structured.csv
    - Python 3.9+ with required packages
    - Apple M2 Pro recommended (16GB RAM)
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
from config import DataConfig, SystemConfig, create_directories
from src.data_processing import OpenStackLogAnalyzer, OpenStackFeatureEngineer, FeatureConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all required files and dependencies exist."""
    logger.info("Checking prerequisites...")

    # Check dataset
    if not DataConfig.RAW_DATASET_PATH.exists():
        logger.error(f"Dataset not found: {DataConfig.RAW_DATASET_PATH}")
        logger.error("Please ensure the dataset is in the correct location.")
        return False

    # Ensure output directories exist
    create_directories()

    # Check Python packages
    required_packages = [
        "pandas", "numpy", "sklearn", "torch",
        "matplotlib", "seaborn"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Install with: pip install " + " ".join(missing_packages))
        return False

    logger.info("All prerequisites satisfied ✅")
    return True

def run_data_analysis():
    """Run the data analysis step."""
    logger.info("="*60)
    logger.info("STEP 1: DATA ANALYSIS")
    logger.info("="*60)

    start_time = time.time()

    try:
        # Initialize analyzer
        analyzer = OpenStackLogAnalyzer(str(DataConfig.RAW_DATASET_PATH))

        # Load data
        df = analyzer.load_data()

        # Generate complete analysis
        results = analyzer.generate_analysis_report(str(DataConfig.ANALYSIS_OUTPUT_DIR.parent))

        analysis_time = time.time() - start_time
        logger.info(f"Data analysis completed in {analysis_time:.2f} seconds")

        # Log key findings
        logger.info("Key Analysis Results:")
        logger.info(f"- Total log entries: {len(df)}")
        logger.info(f"- Anomaly ratio: {df['is_anomaly'].sum()/len(df)*100:.1f}%")
        logger.info(f"- Memory spikes detected: {results.get('memory_spikes_count', 'N/A')}")

        return True

    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        return False

def run_feature_engineering():
    """Run the feature engineering step."""
    logger.info("="*60)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("="*60)

    start_time = time.time()

    try:
        # Initialize feature engineer
        config = FeatureConfig()
        engineer = OpenStackFeatureEngineer(config)

        # Run full pipeline
        train_data, val_data, test_data = engineer.process_full_pipeline(
            str(DataConfig.PROCESSED_DATASET_PATH), str(DataConfig.FEATURES_OUTPUT_DIR.parent)
        )

        feature_time = time.time() - start_time
        logger.info(f"Feature engineering completed in {feature_time:.2f} seconds")

        # Log results
        logger.info("Feature Engineering Results:")
        logger.info(f"- Training samples: {len(train_data[1])}")
        logger.info(f"- Validation samples: {len(val_data[1])}")
        logger.info(f"- Test samples: {len(test_data[1])}")

        return True

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return False

def validate_outputs():
    """Validate that all expected outputs were generated."""
    logger.info("="*60)
    logger.info("STEP 3: OUTPUT VALIDATION")
    logger.info("="*60)

    validation_passed = True

    # Check analysis outputs
    analysis_files = [
        "analysis_output/analysis_results.json",
        "analysis_output/processed_dataset_with_labels.csv",
        "analysis_output/analysis_visualizations.png",
        "analysis_output/analysis_summary.txt"
    ]

    for file_path in analysis_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            logger.info(f"✅ {file_path} ({size_mb:.2f} MB)")
        else:
            logger.error(f"❌ Missing: {file_path}")
            validation_passed = False

    # Check feature outputs
    feature_files = [
        "features_output/train_data.pt",
        "features_output/val_data.pt",
        "features_output/test_data.pt",
        "features_output/encoders.json",
        "features_output/scalers.json",
        "features_output/feature_config.json",
        "features_output/feature_stats.json"
    ]

    for file_path in feature_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            logger.info(f"✅ {file_path} ({size_mb:.2f} MB)")
        else:
            logger.error(f"❌ Missing: {file_path}")
            validation_passed = False

    # Validate feature statistics
    try:
        with open("features_output/feature_stats.json", "r") as f:
            stats = json.load(f)

        logger.info("Feature Statistics:")
        logger.info(f"- Total sequences: {stats['total_sequences']}")
        logger.info(f"- Anomaly ratio: {stats['anomaly_ratio']:.3f}")
        logger.info(f"- Numerical features: {stats['numerical_features']}")
        logger.info(f"- Categorical features: {stats['categorical_features']}")
        logger.info(f"- Binary features: {stats['binary_features']}")

        # Quality checks
        if stats['total_sequences'] < 20:
            logger.warning("⚠️  Low sequence count. Consider reducing window size.")

        if stats['anomaly_ratio'] < 0.05 or stats['anomaly_ratio'] > 0.5:
            logger.warning("⚠️  Unusual anomaly ratio. Check labeling logic.")

    except Exception as e:
        logger.error(f"Could not validate feature statistics: {e}")
        validation_passed = False

    return validation_passed

def generate_final_report():
    """Generate a final summary report."""
    logger.info("="*60)
    logger.info("GENERATING FINAL REPORT")
    logger.info("="*60)

    try:
        # Read analysis and feature statistics
        with open("analysis_output/analysis_results.json", "r") as f:
            analysis_stats = json.load(f)

        with open("features_output/feature_stats.json", "r") as f:
            feature_stats = json.load(f)

        # Create comprehensive report
        report_content = f"""
CloudInfraAI Data Preprocessing Pipeline - Final Report
======================================================

Execution Summary:
- Pipeline Status: ✅ COMPLETED SUCCESSFULLY
- Total Processing Time: See individual step timings above
- Target Platform: Apple M2 Pro (16GB RAM)
- Dataset Size: 2,001 OpenStack log entries

Data Analysis Results:
- Memory Anomalies: {analysis_stats.get('memory_spikes_count', 'N/A')} spikes detected
- API Anomalies: {analysis_stats.get('api_stats', {}).get('slow_requests_count', 'N/A')} slow requests
- System Warnings: {analysis_stats.get('system_health_stats', {}).get('warnings_count', 'N/A')} warning events
- HTTP Errors: {analysis_stats.get('api_stats', {}).get('error_requests_count', 'N/A')} error responses

Feature Engineering Results:
- Total Sequences Generated: {feature_stats['total_sequences']}
- Anomaly Ratio: {feature_stats['anomaly_ratio']:.1%}
- Feature Dimensions:
  * Numerical Features: {feature_stats['numerical_features']}
  * Categorical Features: {feature_stats['categorical_features']}
  * Binary Features: {feature_stats['binary_features']}
  * Aggregate Features: {feature_stats['aggregate_features']}

Data Splits:
- Training Sequences: Available in features_output/train_data.pt
- Validation Sequences: Available in features_output/val_data.pt
- Test Sequences: Available in features_output/test_data.pt

Next Steps:
1. Train Transformer model using features_output/ data
2. Implement real-time inference pipeline
3. Integrate with Streamlit dashboard
4. Deploy for demo environment

Files Ready for Model Training:
✅ features_output/train_data.pt       - Training tensors
✅ features_output/val_data.pt         - Validation tensors
✅ features_output/test_data.pt        - Test tensors
✅ features_output/encoders.json       - Categorical encoders
✅ features_output/scalers.json        - Numerical scalers
✅ features_output/feature_config.json - Model configuration

For detailed analysis, see:
- analysis_output/analysis_summary.txt
- analysis_output/analysis_visualizations.png
- preprocessing_pipeline.log

Ready to proceed with model training! 🚀
"""

        # Save report
        with open("preprocessing_final_report.txt", "w") as f:
            f.write(report_content)

        # Print summary
        print("\n" + "="*60)
        print("🎉 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Generated {feature_stats['total_sequences']} sequences from 2,001 log entries")
        print(f"📈 Anomaly ratio: {feature_stats['anomaly_ratio']:.1%}")
        print(f"📁 All outputs saved to analysis_output/ and features_output/")
        print(f"📋 Full report: preprocessing_final_report.txt")
        print("="*60)
        print("Ready for model training! 🚀")

    except Exception as e:
        logger.error(f"Could not generate final report: {e}")

def main():
    """Main pipeline execution."""
    pipeline_start = time.time()

    logger.info("🚀 Starting CloudInfraAI Data Preprocessing Pipeline")
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Platform: {sys.platform}")

    # Step 0: Prerequisites
    if not check_prerequisites():
        logger.error("❌ Prerequisites not met. Exiting.")
        sys.exit(1)

    # Step 1: Data Analysis
    if not run_data_analysis():
        logger.error("❌ Data analysis failed. Exiting.")
        sys.exit(1)

    # Step 2: Feature Engineering
    if not run_feature_engineering():
        logger.error("❌ Feature engineering failed. Exiting.")
        sys.exit(1)

    # Step 3: Validation
    if not validate_outputs():
        logger.error("❌ Output validation failed. Check logs.")
        sys.exit(1)

    # Step 4: Final Report
    generate_final_report()

    pipeline_time = time.time() - pipeline_start
    logger.info(f"🎉 Complete pipeline finished in {pipeline_time:.2f} seconds")

if __name__ == "__main__":
    main()