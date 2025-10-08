# CloudInfraAI - Project Structure

## 📁 Complete Directory Tree

```
CloudInfraAI/
│
├── 📂 data/                                    # Data storage
│   ├── 📂 raw/                                # Original, untouched data
│   │   ├── OpenStack_2k.log_structured.csv   # Your original dataset
│   │   └── .gitkeep                          # Keep directory in git
│   └── 📂 processed/                         # Processed datasets
│       ├── processed_dataset_with_labels.csv # Logs with anomaly labels
│       ├── engineered_features.csv          # Extracted features (future)
│       └── .gitkeep                          # Keep directory in git
│
├── 📂 notebooks/                              # Jupyter notebooks
│   └── (Future: EDA and experimentation)
│
├── 📂 saved_models/                           # Model artifacts
│   └── (Future: anomaly_detector_v1.pth)
│
├── 📂 outputs/                                # Generated outputs
│   ├── 📂 analysis/                          # Analysis results
│   │   ├── analysis_results.json            # Detailed statistics
│   │   ├── analysis_visualizations.png      # Data plots
│   │   ├── analysis_summary.txt             # Summary report
│   │   └── .gitkeep
│   └── 📂 features/                          # ML-ready features
│       ├── train_data.pt                    # Training tensors
│       ├── val_data.pt                      # Validation tensors
│       ├── test_data.pt                     # Test tensors
│       ├── encoders.json                    # Categorical encoders
│       ├── scalers.json                     # Numerical scalers
│       ├── feature_config.json              # Pipeline config
│       ├── feature_stats.json               # Feature statistics
│       └── .gitkeep
│
├── 📂 src/                                   # Source code
│   ├── __init__.py                          # Main package init
│   │
│   ├── 📂 data_processing/                  # Data processing module
│   │   ├── __init__.py                      # Module exports
│   │   ├── data_analyzer.py                # Pattern detection
│   │   └── feature_engineer.py             # Feature extraction
│   │
│   ├── 📂 model/                           # ML models (future)
│   │   ├── __init__.py
│   │   ├── architecture.py                 # Transformer/LSTM models
│   │   ├── train.py                        # Training pipeline
│   │   └── predict.py                      # Inference functions
│   │
│   ├── 📂 dashboard/                       # Web dashboard (future)
│   │   ├── __init__.py
│   │   └── app.py                          # Streamlit application
│   │
│   ├── 📂 alerting/                        # Notification system (future)
│   │   ├── __init__.py
│   │   └── slack.py                        # Slack integration
│   │
│   └── 📂 utils/                           # Utilities (future)
│       ├── __init__.py
│       ├── logging_config.py               # Logging setup
│       ├── file_helpers.py                 # File operations
│       └── validation.py                   # Data validation
│
├── 📂 scripts/                             # Standalone scripts
│   └── run_preprocessing.py                # Complete preprocessing pipeline
│
├── 📂 docs/                                # Documentation
│   ├── CloudInfraAI_PRD.md                # Main project requirements
│   ├── Data_Processing_PRD.md              # Data processing specs
│   └── README.md                           # Original comprehensive docs
│
├── 📂 tests/                               # Unit tests (future)
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_models.py
│   └── test_utils.py
│
├── 📂 configs/                             # Configuration files (future)
│   ├── model_config.yaml
│   ├── dashboard_config.yaml
│   └── deployment_config.yaml
│
├── 📂 logs/                                # Application logs
│   └── cloudinfra_ai.log                  # Main log file
│
├── 📄 main.py                              # Main CLI entry point
├── 📄 config.py                            # Central configuration
├── 📄 requirements.txt                     # Python dependencies
├── 📄 README.md                            # Project overview & usage
├── 📄 .gitignore                           # Git ignore rules
└── 📄 PROJECT_STRUCTURE.md                 # This file
```

## 🚀 Entry Points

### Primary Entry Point
- **`main.py`** - Command-line interface for all operations

### Secondary Entry Points
- **`scripts/run_preprocessing.py`** - Direct preprocessing execution
- **`src/data_processing/`** - Individual module imports

## 📦 Package Structure

### Core Packages
1. **`src.data_processing`** - Data analysis and feature engineering
2. **`src.model`** - Machine learning models (future)
3. **`src.dashboard`** - Web interface (future)
4. **`src.alerting`** - Notifications (future)
5. **`src.utils`** - Common utilities (future)

### Configuration
- **`config.py`** - Centralized settings for all modules
- **Environment variables** - For sensitive configurations (Slack tokens, etc.)

## 🔄 Data Flow

```
Raw Data → Data Processing → Feature Engineering → Model Training → Dashboard
    ↓           ↓                   ↓               ↓             ↓
data/raw/   outputs/         outputs/        saved_models/  dashboard/
            analysis/        features/
```

## 🎯 Current Status

### ✅ Implemented
- **Folder structure** - Professional organization
- **Data processing** - Analysis and feature engineering
- **Configuration** - Centralized settings
- **Documentation** - Comprehensive guides
- **CLI interface** - Main entry point

### 🔄 In Progress
- **Testing** - Unit tests for core functionality

### 📋 Future Phases
- **Model training** - Transformer architecture
- **Dashboard** - Streamlit web interface
- **Alerting** - Slack integration
- **Deployment** - Docker and Kubernetes

## 📐 Design Principles

### 1. **Modularity**
- Each component is a separate package
- Clear separation of concerns
- Reusable modules

### 2. **Configuration-Driven**
- Central configuration in `config.py`
- Environment-specific settings
- Easy parameter tuning

### 3. **Professional Standards**
- Proper `__init__.py` files
- Comprehensive documentation
- Consistent naming conventions

### 4. **Scalability**
- Designed for future expansion
- Easy to add new components
- Production-ready structure

This structure follows industry best practices and makes the project maintainable, scalable, and professional! 🚀