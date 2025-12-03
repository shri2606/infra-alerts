#!/usr/bin/env python3
"""
Dashboard Runner Script
=======================

Launches the Streamlit dashboard for real-time monitoring.
"""

import os
import sys
import subprocess


def main():
    """Launch the Streamlit dashboard."""
    # Get project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Path to dashboard app
    dashboard_path = os.path.join(project_root, 'src', 'dashboard', 'app.py')

    # Set environment variable for MPS fallback
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Add project root to Python path
    os.environ['PYTHONPATH'] = project_root

    print("=" * 80)
    print("CloudInfraAI - Real-Time Monitoring Dashboard")
    print("=" * 80)
    print()
    print("Starting Streamlit dashboard...")
    print(f"Dashboard path: {dashboard_path}")
    print()
    print("The dashboard will open in your web browser automatically.")
    print("Press Ctrl+C to stop the dashboard.")
    print()
    print("=" * 80)

    # Launch streamlit
    try:
        subprocess.run([
            sys.executable,
            '-m',
            'streamlit',
            'run',
            dashboard_path,
            '--server.port=8501',
            '--server.headless=false',
            '--browser.gatherUsageStats=false'
        ])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    except Exception as e:
        print(f"\nError launching dashboard: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Streamlit is installed: pip install streamlit")
        print("2. Check that the virtual environment is activated")
        print("3. Verify all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
