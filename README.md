# CloudInfraAI - AI-Powered OpenStack Infrastructure Monitoring

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset
cp OpenStack_2k.log_structured.csv data/raw/

# 3. Run complete pipeline
python main.py preprocess
```

**Expected Runtime:** 3-5 minutes on Apple M2 Pro
**Output:** Model-ready PyTorch tensors for Transformer training

---

## 📁 Project Structure

```
CloudInfraAI/
│
├── 📂 data/
│   ├── 📄 raw/
│   │   ├── OpenStack_2k.log_structured.csv    # Original OpenStack logs
│   │   └── .gitkeep
│   └── 📄 processed/
│       ├── processed_dataset_with_labels.csv  # Logs with anomaly labels
│       ├── engineered_features.csv            # Extracted features
│       └── .gitkeep
│
├── 📂 notebooks/
│   └── (Future: Jupyter notebooks for EDA)
│
├── 📂 saved_models/
│   └── (Future: Trained PyTorch model files)
│
├── 📂 outputs/
│   ├── 📂 analysis/
│   │   ├── analysis_results.json              # Pattern analysis results
│   │   ├── analysis_visualizations.png        # Data distribution plots
│   │   ├── analysis_summary.txt               # Human-readable insights
│   │   └── .gitkeep
│   └── 📂 features/
│       ├── train_data.pt                      # Training tensors
│       ├── val_data.pt                        # Validation tensors
│       ├── test_data.pt                       # Test tensors
│       ├── encoders.json                      # Categorical encoders
│       ├── scalers.json                       # Numerical scalers
│       ├── feature_config.json                # Pipeline configuration
│       ├── feature_stats.json                 # Feature statistics
│       └── .gitkeep
│
├── 📂 src/
│   ├── __init__.py                            # Main package
│   │
│   ├── 📂 data_processing/
│   │   ├── __init__.py
│   │   ├── data_analyzer.py                   # Pattern detection & analysis
│   │   └── feature_engineer.py                # Feature extraction pipeline
│   │
│   ├── 📂 model/
│   │   ├── __init__.py
│   │   ├── architecture.py                    # Transformer model (future)
│   │   ├── train.py                          # Training pipeline (future)
│   │   └── predict.py                        # Inference functions (future)
│   │
│   ├── 📂 dashboard/
│   │   ├── __init__.py
│   │   └── app.py                            # Streamlit dashboard (future)
│   │
│   ├── 📂 alerting/
│   │   ├── __init__.py
│   │   └── slack.py                          # Slack notifications (future)
│   │
│   └── 📂 utils/
│       ├── __init__.py
│       └── (Common utilities - future)
│
├── 📂 scripts/
│   └── run_preprocessing.py                   # Complete preprocessing pipeline
│
├── 📂 docs/
│   ├── CloudInfraAI_PRD.md                   # Main project requirements
│   ├── Data_Processing_PRD.md                # Data processing specifications
│   └── README.md                             # Comprehensive documentation
│
├── 📂 tests/
│   └── (Unit tests - future)
│
├── 📂 configs/
│   └── (Configuration files - future)
│
├── 📂 logs/
│   └── (Application logs)
│
├── 📄 main.py                                # Main CLI entry point
├── 📄 config.py                              # Central configuration
├── 📄 requirements.txt                       # Python dependencies
└── 📄 .gitignore                             # Git ignore rules
```

---

## 🎯 Usage Options

### Option 1: Complete Pipeline (Recommended)
```bash
python main.py preprocess
```
- Runs data analysis + feature engineering
- Full validation and reporting
- Ready for model training

### Option 2: Analysis Only
```bash
python main.py analyze
```
- Data pattern detection
- Anomaly identification
- Visualization generation

### Option 3: Individual Scripts
```bash
# Direct script execution
python scripts/run_preprocessing.py
```

### Option 4: Future Commands
```bash
python main.py train          # Train ML model (coming soon)
python main.py dashboard      # Launch Streamlit app (coming soon)
```

---

## 📊 Detected Anomaly Patterns

### 🔴 High Priority Anomalies
1. **Memory Spikes**
   - Normal: 512MB baseline
   - Anomaly: ≥2560MB (5x spike)
   - Count: ~9 events in dataset

2. **API Latency**
   - Normal: 0.2-0.3 seconds
   - Anomaly: ≥0.5 seconds
   - Count: ~10 slow requests

### 🟡 Medium Priority Anomalies
3. **HTTP Errors**
   - Normal: 200, 202, 204 status
   - Anomaly: 404, 500+ errors
   - Count: ~15 error responses

4. **System Warnings**
   - Normal: INFO level logs
   - Anomaly: WARNING/ERROR levels
   - Count: ~31 warning events

---

## ⚙️ Configuration

The project uses a centralized configuration in `config.py`:

```python
# Key configurations
DataConfig.RAW_DATASET_PATH        # Input dataset location
ModelConfig.SEQUENCE_LENGTH = 50   # Events per sequence
ModelConfig.D_MODEL = 128          # Transformer dimension
SystemConfig.DEVICE               # Auto-detects M2 Pro/GPU/CPU
```

### Hardware Optimization
- **Apple M2 Pro:** Uses MPS acceleration automatically
- **NVIDIA GPU:** Uses CUDA if available
- **CPU Fallback:** Works on any system

---

## 🔧 Development Workflow

### Setting Up Development Environment
```bash
# Clone and setup
git clone <repository>
cd CloudInfraAI
pip install -r requirements.txt

# Place your dataset
cp your_dataset.csv data/raw/OpenStack_2k.log_structured.csv

# Run analysis
python main.py analyze
```

### Adding New Features
1. **Data Processing:** Add modules to `src/data_processing/`
2. **Models:** Add architectures to `src/model/`
3. **Dashboard:** Add components to `src/dashboard/`
4. **Utilities:** Add helpers to `src/utils/`

### Project Conventions
- **Import Structure:** Use relative imports within packages
- **Configuration:** Central config in `config.py`
- **Logging:** Structured logging to `logs/` directory
- **Documentation:** Comprehensive docstrings and comments

---

## 📈 Expected Results

### Processing Metrics (M2 Pro)
- **Execution Time:** 3-5 minutes
- **Memory Usage:** <3GB RAM
- **Sequences Generated:** 50-100 from 2k logs
- **Feature Count:** 25+ per event
- **Anomaly Ratio:** ~15%

### Quality Validation
```
✅ Data loaded: 2,001 log entries
✅ Patterns detected: Memory, API, lifecycle, health
✅ Features extracted: Numerical, categorical, binary
✅ Sequences created: 5-minute windows
✅ Labels generated: Binary anomaly classification
✅ Artifacts saved: Encoders, scalers, configs
```

---

## 🚨 Troubleshooting

### Common Issues

#### Dataset Not Found
```bash
Error: Dataset not found at data/raw/OpenStack_2k.log_structured.csv
```
**Solution:** Place dataset in correct location
```bash
mkdir -p data/raw
cp OpenStack_2k.log_structured.csv data/raw/
```

#### PyTorch MPS Errors (M2 Pro)
```bash
# Force CPU if MPS issues
export PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py preprocess
```

#### Import Errors
```bash
# Ensure you're in project root
cd CloudInfraAI
python main.py preprocess
```

#### Memory Issues
```python
# Edit config.py for lower memory usage
ModelConfig.SEQUENCE_LENGTH = 30    # Reduce from 50
ModelConfig.BATCH_SIZE = 16         # Reduce from 32
```

---

## 🔮 Future Development Phases

### Phase 2: Model Training
- Transformer architecture implementation
- Training pipeline with M2 Pro optimization
- Model evaluation and validation

### Phase 3: Real-time Dashboard
- Streamlit application
- Live monitoring interface
- Interactive visualizations

### Phase 4: Production Deployment
- Slack alert integration
- Docker containerization
- Kubernetes deployment ready

---

## 📚 Documentation

- **Main PRD:** `docs/CloudInfraAI_PRD.md` - Project requirements
- **Data PRD:** `docs/Data_Processing_PRD.md` - Technical specifications
- **API Docs:** Generated from code docstrings
- **Notebooks:** Analysis and exploration in `notebooks/`

---

## 🤝 Contributing

### Development Setup
```bash
# Setup development environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run tests (when implemented)
python -m pytest tests/

# Format code
black src/ scripts/ main.py config.py
```

### Code Standards
- **Python 3.9+** with type hints
- **PEP 8** formatting with black
- **Comprehensive docstrings** for all functions
- **Unit tests** for core functionality

---

## 📄 License

This project is part of the CloudInfraAI capstone project. See project documentation for licensing details.

---

**Ready to detect OpenStack infrastructure anomalies with AI! 🚀**

Need help? Run `python main.py --help` for command options.