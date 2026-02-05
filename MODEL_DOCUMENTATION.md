# Intrusion Detection Machine Learning Models

This repository contains implementations of six machine learning models for intrusion detection:

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

Run the complete pipeline with all models:

```bash
python src/main.py data/dataset.csv outputs
```

By default, all six models will be trained. You can modify `src/param.py` to select specific models.

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
   - Sklearn models (Random Forest, XGBoost, Naive Bayes, Logistic Regression, LightGBM): `.pkl` files
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

### LightGBM Hyperparameters
```python
'LightGBM': {
    'objective': 'multiclass',  # Multi-class classification
    'boosting_type': 'gbdt',    # Gradient boosting
    'num_leaves': 31,           # Maximum leaves per tree
    'learning_rate': 0.05,      # Learning rate
    'n_estimators': 100,        # Number of boosting rounds
    'subsample': 0.8,           # Subsample ratio
    'colsample_bytree': 0.8,    # Feature subsample ratio
    'reg_alpha': 0.1,           # L1 regularization
    'reg_lambda': 0.1,          # L2 regularization
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```

### Naive Bayes Hyperparameters
```python
'NaiveBayes': {
    'var_smoothing': 1e-9       # Variance smoothing parameter
}
```

### Logistic Regression Hyperparameters
```python
'LogisticRegression': {
    'max_iter': 1000,           # Maximum iterations
    'solver': 'lbfgs',          # Optimization algorithm
    'C': 1.0,                   # Inverse regularization strength
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

### LightGBM
- Histogram-based gradient boosting
- Optimized for large datasets (millions of samples)
- Faster training than XGBoost on large data
- Lower memory usage
- Excellent for multi-class classification
- Feature importance analysis

### Naive Bayes
- Probabilistic classifier (Gaussian Naive Bayes)
- Extremely fast training and prediction
- Works well with high-dimensional data
- Assumes feature independence
- Ideal for very large datasets

### Logistic Regression
- Linear classification model
- Fast and interpretable
- Supports multi-class via one-vs-rest or multinomial
- L2 regularization by default
- Efficient for large datasets with parallel processing

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
from model import RandomForest, XGBoostModel, CNN, NaiveBayes, LogisticRegressionModel, LightGBMModel

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

# LightGBM (optimized for large datasets)
lgb = LightGBMModel(n_estimators=100, num_leaves=31)
lgb.train(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
predictions = lgb.predict(X_test)
feature_importance = lgb.get_feature_importance()

# Naive Bayes (fast for large datasets)
nb = NaiveBayes(var_smoothing=1e-9)
nb.train(X_train, y_train)
predictions = nb.predict(X_test)
probabilities = nb.predict_proba(X_test)

# Logistic Regression (interpretable linear model)
lr = LogisticRegressionModel(max_iter=1000, C=1.0)
lr.train(X_train, y_train)
predictions = lr.predict(X_test)
coefficients = lr.get_coefficients()

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
│   ├── main.py                      # Main pipeline
│   ├── param.py                     # Model configurations
│   └── model/
│       ├── __init__.py              # Package initialization
│       ├── base.py                  # Base model class
│       ├── random_forest.py         # Random Forest implementation
│       ├── xgboost_model.py         # XGBoost implementation
│       ├── lightgbm_model.py        # LightGBM implementation
│       ├── naive_bayes.py           # Naive Bayes implementation
│       ├── logistic_regression.py   # Logistic Regression implementation
│       └── cnn.py                   # CNN implementation
├── data/
│   └── readme.md                    # Data documentation
├── outputs/                         # Generated results (created automatically)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- XGBoost >= 1.5.0
- LightGBM >= 4.6.0
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
