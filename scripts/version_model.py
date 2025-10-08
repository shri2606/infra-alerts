#!/usr/bin/env python3
"""
Model Versioning Utility
=========================

Save trained models with versions to prevent overwriting.

Usage:
    python scripts/version_model.py --message "Best model: 96% accuracy"
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import shutil
from datetime import datetime
from config import MODELS_DIR

def create_model_version(message: str = ""):
    """Create a versioned copy of the current model."""

    # Create timestamp-based version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"v_{timestamp}"
    version_dir = MODELS_DIR / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating model version: {version_name}")

    # Files to version
    files_to_copy = [
        "checkpoints/best_model.pth",
        "anomaly_detector_v1.pth",
        "test_results.json",
        "training_history.json",
        "threshold_optimization_results.json"
    ]

    copied_files = []
    for file_name in files_to_copy:
        src = MODELS_DIR / file_name
        if src.exists():
            dst = version_dir / Path(file_name).name
            shutil.copy2(src, dst)
            copied_files.append(file_name)
            print(f"  ✓ Copied: {file_name}")
        else:
            print(f"  ⊗ Skipped (not found): {file_name}")

    # Load test results for metadata
    results_path = MODELS_DIR / "test_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            test_results = json.load(f)
    else:
        test_results = {}

    # Create metadata
    metadata = {
        "version": version_name,
        "created_at": datetime.now().isoformat(),
        "message": message,
        "files": copied_files,
        "metrics": test_results
    }

    # Save metadata
    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model version created: {version_dir}")
    print(f"   Message: {message}")
    if test_results:
        print(f"   Accuracy: {test_results.get('test_accuracy', 'N/A')}")
        print(f"   F1-Score: {test_results.get('test_f1', 'N/A')}")

    # Update versions index
    update_versions_index(version_name, metadata)

    return version_dir


def update_versions_index(version_name: str, metadata: dict):
    """Maintain an index of all model versions."""
    index_path = MODELS_DIR / "versions.json"

    if index_path.exists():
        with open(index_path, 'r') as f:
            versions = json.load(f)
    else:
        versions = {"versions": []}

    # Add new version
    versions["versions"].append({
        "name": version_name,
        "created_at": metadata["created_at"],
        "message": metadata["message"],
        "metrics": metadata.get("metrics", {})
    })

    # Save updated index
    with open(index_path, 'w') as f:
        json.dump(versions, f, indent=2)


def list_versions():
    """List all saved model versions."""
    index_path = MODELS_DIR / "versions.json"

    if not index_path.exists():
        print("No model versions found.")
        return

    with open(index_path, 'r') as f:
        versions = json.load(f)

    print("\n" + "="*70)
    print("MODEL VERSIONS")
    print("="*70)

    for i, version in enumerate(versions["versions"], 1):
        print(f"\n{i}. {version['name']}")
        print(f"   Created: {version['created_at']}")
        print(f"   Message: {version['message']}")

        metrics = version.get('metrics', {})
        if metrics:
            acc = metrics.get('test_accuracy', 'N/A')
            f1 = metrics.get('test_f1', 'N/A')
            print(f"   Accuracy: {acc:.2%}" if isinstance(acc, float) else f"   Accuracy: {acc}")
            print(f"   F1-Score: {f1:.2%}" if isinstance(f1, float) else f"   F1-Score: {f1}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Version control for trained models")
    parser.add_argument(
        '--message', '-m',
        type=str,
        default="",
        help="Version message/description"
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help="List all model versions"
    )

    args = parser.parse_args()

    if args.list:
        list_versions()
    else:
        create_model_version(args.message)


if __name__ == "__main__":
    main()
