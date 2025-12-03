#!/bin/bash

echo "=================================================="
echo "CloudInfraAI - Starting Dashboard"
echo "=================================================="
echo ""

# Activate virtual environment
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Set MPS fallback for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Launch dashboard
echo "Launching dashboard..."
echo "Dashboard will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

python scripts/run_dashboard.py
