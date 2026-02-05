# Intrusion Detection Machine Learning Models

This repository contains implementations of three machine learning models for intrusion detection:
1. **Random Forest** - Ensemble learning method using decision trees
2. **XGBoost** - Gradient boosting framework with tree-based models
3. **Convolutional Neural Network (CNN)** - Deep learning model for pattern recognition

## Dataset

The models are designed to work with the VeReMi dataset (Vehicle Reference Misbehavior dataset) available at:
https://data.mendeley.com/datasets/k62n4z9gdz/1

The dataset contains vehicular network data for intrusion detection in V2V (Vehicle-to-Vehicle) communications.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete pipeline with all three models:

```bash
python src/main.py data/dataset.csv outputs
```

Arguments:
- First argument: Path to your CSV dataset file
- Second argument (optional): Output directory for results (default: `outputs`)

### Dataset Format

The CSV file should have:
- Feature columns (numerical values)
- A label/target column with one of these names: `label`, `target`, `class`, or as the last column

### Output

The pipeline generates:
1. **Results JSON file**: Contains metrics for all models with timestamp
   - Location: `outputs/results_YYYYMMDD_HHMMSS.json`
   - Includes: accuracy, precision, recall, F1-score, confusion matrix

2. **Trained Models**: Saved models for future use
   - Location: `outputs/models/`
   - Random Forest and XGBoost: `.pkl` files
   - CNN: `.h5` file (model) and `_scaler.pkl` (scaler)

3. **Console Output**: Real-time training progress and evaluation metrics

## Model Configurations

Models can be configured via `src/param.py`:

### Random Forest Hyperparameters
```python
'RandomForest': {
    'n_estimators': 100,        # Number of trees
    'max_depth': 20,            # Maximum tree depth
    'min_samples_split': 2,     # Minimum samples to split
    'min_samples_leaf': 1,      # Minimum samples per leaf
    'random_state': 42,
    'n_jobs': -1                # Use all CPU cores
}
```

### XGBoost Hyperparameters
```python
'XGBoost': {
    'n_estimators': 100,        # Number of boosting rounds
    'max_depth': 10,            # Maximum tree depth
    'learning_rate': 0.1,       # Step size shrinkage
    'subsample': 0.8,           # Subsample ratio
    'colsample_bytree': 0.8,    # Feature subsample ratio
    'random_state': 42,
    'n_jobs': -1
}
```

### CNN Hyperparameters
```python
'CNN': {
    'epochs': 50,               # Training epochs
    'batch_size': 32,           # Batch size
    'learning_rate': 0.001,     # Learning rate
    'num_classes': 2            # Number of output classes
}
```

## Architecture Details

### Random Forest
- Ensemble of decision trees
- Handles high-dimensional data well
- Provides feature importance
- Robust to overfitting

### XGBoost
- Gradient boosted decision trees
- Fast training with GPU support
- Built-in regularization
- Early stopping for optimal performance
- Feature importance analysis

### CNN
- 2D Convolutional architecture
- Three convolutional blocks with max pooling
- Dropout layers for regularization
- Dense layers for classification
- Automatic feature scaling
- Early stopping callback

**CNN Architecture:**
```
Input → Reshape → Conv2D(32) → MaxPool → Dropout
     → Conv2D(64) → MaxPool → Dropout
     → Conv2D(128) → MaxPool → Dropout
     → Flatten → Dense(128) → Dropout
     → Dense(64) → Dropout
     → Output (Softmax/Sigmoid)
```

## Model API

All models inherit from the `Base` class and implement:

### Methods

- **`train(X_train, y_train, ...)`**: Train the model
- **`predict(X)`**: Make predictions on new data
- **`predict_proba(X)`**: Get prediction probabilities
- **`evaluate(X_test, y_test)`**: Evaluate model performance

### Example Usage

```python
from model import RandomForest, XGBoostModel, CNN

# Random Forest
rf = RandomForest(n_estimators=100, max_depth=20)
rf.train(X_train, y_train)
predictions = rf.predict(X_test)
metrics = rf.evaluate(X_test, y_test)

# XGBoost
xgb = XGBoostModel(n_estimators=100, learning_rate=0.1)
xgb.train(X_train, y_train, eval_set=[(X_val, y_val)])
predictions = xgb.predict(X_test)
feature_importance = xgb.get_feature_importance()

# CNN
cnn = CNN(epochs=50, batch_size=32)
cnn.train(X_train, y_train, validation_data=(X_val, y_val))
predictions = cnn.predict(X_test)
probabilities = cnn.predict_proba(X_test)
```

## Performance Metrics

Each model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/False positives and negatives

## Data Preprocessing

The pipeline automatically:
1. Splits data into train (70%), validation (10%), and test (20%) sets
2. Applies stratified sampling to maintain class distribution
3. Scales features (for CNN only)
4. Handles missing values and data formatting

## Project Structure

```
COMP-4990-Intrusion-Detection/
├── src/
│   ├── main.py                 # Main pipeline
│   ├── param.py                # Model configurations
│   └── model/
│       ├── __init__.py         # Package initialization
│       ├── base.py             # Base model class
│       ├── random_forest.py    # Random Forest implementation
│       ├── xgboost_model.py    # XGBoost implementation
│       └── cnn.py              # CNN implementation
├── data/
│   └── readme.md               # Data documentation
├── outputs/                    # Generated results (created automatically)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- XGBoost >= 1.5.0
- TensorFlow >= 2.10.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

## License

This project is part of COMP-4990 coursework.

## References

- VeReMi Dataset: https://data.mendeley.com/datasets/k62n4z9gdz/1
- Random Forest: Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
- XGBoost: Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system.
- CNN: LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition.
