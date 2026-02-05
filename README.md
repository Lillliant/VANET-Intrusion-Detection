# COMP-4990-Intrusion-Detection

Machine learning models for intrusion detection in vehicular networks using the VeReMi dataset.

## Overview

This repository implements six machine learning models for detecting intrusions in V2V (Vehicle-to-Vehicle) communication networks:

**Tree-based & Ensemble Models:**
1. **Random Forest** - Ensemble learning method using decision trees
2. **XGBoost** - Gradient boosting framework with tree-based models
3. **LightGBM** - Efficient gradient boosting optimized for large datasets

**Classical ML Models:**
4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
5. **Logistic Regression** - Linear model for classification

**Deep Learning:**
6. **Convolutional Neural Network (CNN)** - Deep learning model for pattern recognition

All models support **multi-class classification** and are optimized for large datasets (~3 million samples).

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
# Test all original models
python test_models.py

# Test new models (Naive Bayes, Logistic Regression, LightGBM)
python test_new_models.py

# Test with larger multi-class dataset
python test_large_dataset.py
```

## Dataset Format

The models are designed for the VeReMi dataset (Vehicle Reference Misbehavior dataset):
- Dataset URL: https://data.mendeley.com/datasets/k62n4z9gdz/1
- Format: CSV with numerical features and a label column
- Label column should be named: `label`, `target`, `class`, or be the last column
- Supports both binary and multi-class classification

## Model Performance

### Sample Binary Classification (5000 samples, 50 features):
- **Random Forest**: 87.7% accuracy
- **XGBoost**: 92.5% accuracy
- **CNN**: 91.0% accuracy

### Sample Multi-Class (10K samples, 50 features, 5 classes):
- **Naive Bayes**: 55.8% accuracy (very fast)
- **Logistic Regression**: 54.9% accuracy (fast, interpretable)
- **LightGBM**: 78.5% accuracy (best for large datasets)

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
│   ├── main.py                      # ML pipeline
│   ├── param.py                     # Configurations
│   └── model/
│       ├── random_forest.py         # Random Forest
│       ├── xgboost_model.py         # XGBoost
│       ├── lightgbm_model.py        # LightGBM
│       ├── naive_bayes.py           # Naive Bayes
│       ├── logistic_regression.py   # Logistic Regression
│       └── cnn.py                   # CNN
├── data/                            # Dataset directory
├── outputs/                         # Results and saved models
├── requirements.txt                 # Dependencies
└── MODEL_DOCUMENTATION.md           # Detailed docs
```

## License

This project is part of COMP-4990 coursework.