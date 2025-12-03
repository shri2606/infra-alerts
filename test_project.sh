#!/bin/bash

echo "=================================================="
echo "CloudInfraAI - Project Flow Test"
echo "=================================================="
echo ""

# Activate virtual environment
echo "Step 1: Activating virtual environment..."
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo "OK: Virtual environment activated"
echo ""

# Check Python version
echo "Step 2: Checking Python version..."
python --version
echo ""

# Check required modules
echo "Step 3: Checking required Python modules..."
python -c "
import streamlit
import plotly
import torch
import pandas
import numpy
import sklearn
print('streamlit:', streamlit.__version__)
print('plotly:', plotly.__version__)
print('torch:', torch.__version__)
print('pandas:', pandas.__version__)
print('numpy:', numpy.__version__)
print('sklearn:', sklearn.__version__)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to import required modules"
    exit 1
fi
echo "OK: All modules imported successfully"
echo ""

# Check dataset
echo "Step 4: Checking dataset..."
if [ -f "data/raw/OpenStack_2k.log_structured.csv" ]; then
    ls -lh data/raw/OpenStack_2k.log_structured.csv
    echo "OK: Dataset found"
else
    echo "WARNING: Dataset not found at data/raw/OpenStack_2k.log_structured.csv"
fi
echo ""

# Check model
echo "Step 5: Checking trained model..."
if [ -f "saved_models/experiment_2min/best_model.pth" ]; then
    ls -lh saved_models/experiment_2min/best_model.pth
    echo "OK: Model found"
else
    echo "WARNING: Model not found at saved_models/experiment_2min/best_model.pth"
fi
echo ""

# Check feature artifacts
echo "Step 6: Checking feature artifacts..."
if [ -f "outputs_experiment/encoders.json" ] && [ -f "outputs_experiment/scalers.json" ]; then
    echo "OK: encoders.json found"
    echo "OK: scalers.json found"
else
    echo "WARNING: Feature artifacts not found in outputs_experiment/"
fi
echo ""

# Test model loading
echo "Step 7: Testing model loading..."
python -c "
from src.inference.predictor import AnomalyPredictor
predictor = AnomalyPredictor()
print('OK: Model loaded successfully')
print('Model parameters:', sum(p.numel() for p in predictor.model.parameters()))
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load model"
    exit 1
fi
echo ""

# Test feature extractor
echo "Step 8: Testing feature extractor..."
python -c "
from src.inference.feature_extractor import RealTimeFeatureExtractor
extractor = RealTimeFeatureExtractor()
print('OK: Feature extractor initialized')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to initialize feature extractor"
    exit 1
fi
echo ""

# Test log simulator
echo "Step 9: Testing log simulator (10 events)..."
python -c "
from src.simulation.log_simulator import LogSimulator
simulator = LogSimulator()
events = simulator.generate_normal_stream(duration_seconds=2)
print('OK: Generated', len(events), 'events')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate logs"
    exit 1
fi
echo ""

# Test streaming predictor
echo "Step 10: Testing streaming predictor initialization..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -c "
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from src.inference.streaming_predictor import StreamingAnomalyPredictor
predictor = StreamingAnomalyPredictor(window_size=50, stride=10)
print('OK: Streaming predictor initialized')
print('Window size:', predictor.window_size)
print('Stride:', predictor.stride)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to initialize streaming predictor"
    exit 1
fi
echo ""

echo "=================================================="
echo "All checks completed successfully!"
echo "=================================================="
echo ""
echo "To run the dashboard:"
echo "1. source .venv/bin/activate"
echo "2. export PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "3. python scripts/run_dashboard.py"
echo ""
echo "Or use the quick start script:"
echo "./start_dashboard.sh"
echo ""
