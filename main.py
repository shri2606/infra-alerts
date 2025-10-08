#!/usr/bin/env python3
"""
CloudInfraAI - Main Entry Point
===============================

Main entry point for the CloudInfraAI system. Provides command-line interface
for running different components of the system.

Usage:
    python main.py --help                    # Show help
    python main.py preprocess               # Run data preprocessing
    python main.py train                    # Train the model (future)
    python main.py dashboard                # Launch dashboard (future)
    python main.py analyze                  # Run data analysis only

Author: CloudInfraAI Team
Date: 2024
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import configuration
from config import SystemConfig, validate_environment

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, SystemConfig.LOG_LEVEL),
        format=SystemConfig.LOG_FORMAT,
        handlers=[
            logging.FileHandler(SystemConfig.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_preprocessing():
    """Run the complete data preprocessing pipeline."""
    print("🚀 Starting CloudInfraAI Data Preprocessing Pipeline")

    try:
        # Import and run preprocessing script
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(project_root / "scripts" / "run_preprocessing.py")
        ], check=True, capture_output=True, text=True)

        print("✅ Preprocessing completed successfully!")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("❌ Preprocessing failed!")
        print(e.stderr)
        sys.exit(1)

def run_analysis_only():
    """Run data analysis only."""
    print("📊 Running Data Analysis")

    try:
        from src.data_processing import OpenStackLogAnalyzer
        from config import DataConfig

        # Initialize and run analyzer
        analyzer = OpenStackLogAnalyzer(str(DataConfig.RAW_DATASET_PATH))
        df = analyzer.load_data()
        results = analyzer.generate_analysis_report(str(DataConfig.ANALYSIS_OUTPUT_DIR))

        print("✅ Data analysis completed!")
        print(f"📊 Analyzed {len(df)} log entries")
        print(f"📈 Anomaly ratio: {df['is_anomaly'].sum()/len(df)*100:.1f}%")
        print(f"📁 Results saved to {DataConfig.ANALYSIS_OUTPUT_DIR}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

def run_training():
    """Train the machine learning model."""
    print("🧠 Model Training")
    print("⚠️  Model training not yet implemented.")
    print("💡 Coming in the next phase of development!")

def run_dashboard():
    """Launch the Streamlit dashboard."""
    print("📱 Dashboard")
    print("⚠️  Dashboard not yet implemented.")
    print("💡 Coming in the next phase of development!")

def main():
    """Main function with CLI argument parsing."""

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Validate environment
    try:
        validate_environment()
    except Exception as e:
        logger.warning(f"Environment validation warning: {e}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="CloudInfraAI - AI-powered OpenStack Infrastructure Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py preprocess        # Run complete preprocessing pipeline
    python main.py analyze          # Run data analysis only
    python main.py train            # Train the ML model (future)
    python main.py dashboard        # Launch monitoring dashboard (future)

Project Structure:
    data/raw/                       # Raw OpenStack log data
    data/processed/                 # Processed datasets
    outputs/analysis/               # Analysis results and visualizations
    outputs/features/               # ML-ready features and tensors
    saved_models/                   # Trained model artifacts
    notebooks/                      # Jupyter notebooks for exploration
    src/                           # Source code modules
    """
    )

    parser.add_argument(
        "command",
        choices=["preprocess", "analyze", "train", "dashboard"],
        help="Command to execute"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file (future feature)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Display system information
    print("CloudInfraAI System Information")
    print("=" * 50)
    print(f"🖥️  Platform: {SystemConfig.PLATFORM}")
    print(f"🧠 Device: {SystemConfig.DEVICE_NAME}")
    print(f"📁 Project Root: {project_root}")
    print("=" * 50)

    # Execute command
    try:
        if args.command == "preprocess":
            run_preprocessing()
        elif args.command == "analyze":
            run_analysis_only()
        elif args.command == "train":
            run_training()
        elif args.command == "dashboard":
            run_dashboard()
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()