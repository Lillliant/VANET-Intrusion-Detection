# COMP-4990-Intrusion-Detection

Machine learning models for intrusion detection in vehicular networks using the VeReMi dataset.

## Overview

This repository implements three machine learning models for detecting intrusions in V2V (Vehicle-to-Vehicle) communication networks:

1. **Random Forest** - Ensemble learning method using decision trees
2. **XGBoost** - Gradient boosting framework with tree-based models
3. **Convolutional Neural Network (CNN)** - Deep learning model for pattern recognition

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# With your own dataset
python src/main.py data/your_dataset.csv outputs

# Generate and use sample data
python generate_sample_data.py
python src/main.py data/sample_dataset.csv outputs
```

### Testing the Models

```bash
python test_models.py
```

## Dataset Format

The models are designed for the VeReMi dataset (Vehicle Reference Misbehavior dataset):
- Dataset URL: https://data.mendeley.com/datasets/k62n4z9gdz/1
- Format: CSV with numerical features and a label column
- Label column should be named: `label`, `target`, `class`, or be the last column

## Model Performance

On sample data (5000 samples, 50 features):
- **Random Forest**: 87.7% accuracy
- **XGBoost**: 92.5% accuracy
- **CNN**: 91.0% accuracy

## Documentation

See [MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md) for detailed information about:
- Model architectures
- Hyperparameter configurations
- API usage
- Output formats

## Project Structure

```
COMP-4990-Intrusion-Detection/
├── src/
│   ├── main.py                 # ML pipeline
│   ├── param.py                # Configurations
│   └── model/
│       ├── random_forest.py    # Random Forest
│       ├── xgboost_model.py    # XGBoost
│       └── cnn.py              # CNN
├── data/                       # Dataset directory
├── outputs/                    # Results and saved models
├── requirements.txt            # Dependencies
└── MODEL_DOCUMENTATION.md      # Detailed docs
```

## License

This project is part of COMP-4990 coursework.