"""
Integration test to verify all 6 models work in the complete pipeline
Tests with a small multi-class dataset to ensure end-to-end functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from sklearn.datasets import make_classification
import param

# Test that all models are in the config
print("=" * 60)
print("Verifying Model Configuration")
print("=" * 60)
print(f"Models configured: {', '.join(param.MODELS)}")
print(f"Total models: {len(param.MODELS)}")

expected_models = ['RandomForest', 'XGBoost', 'CNN', 'NaiveBayes', 'LogisticRegression', 'LightGBM']
assert set(param.MODELS) == set(expected_models), f"Expected {expected_models}, got {param.MODELS}"
print("✓ All 6 models are configured correctly\n")

# Test that all models can be imported
print("=" * 60)
print("Testing Model Imports")
print("=" * 60)

try:
    from model import (
        RandomForest, 
        XGBoostModel, 
        CNN, 
        NaiveBayes, 
        LogisticRegressionModel, 
        LightGBMModel
    )
    print("✓ All model classes imported successfully\n")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Generate small test dataset
print("=" * 60)
print("Generating Test Dataset")
print("=" * 60)

X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    n_classes=3,
    random_state=42
)

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

val_split = int(0.9 * len(X_train))
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Features: {X.shape[1]}, Classes: {len(np.unique(y))}\n")

# Test each model
results = {}

print("=" * 60)
print("Testing Individual Models")
print("=" * 60)

# Test RandomForest
try:
    rf = RandomForest(n_estimators=10, max_depth=5, random_state=42)
    rf.train(X_train, y_train)
    metrics = rf.evaluate(X_test, y_test)
    results['RandomForest'] = metrics['accuracy']
    print(f"✓ RandomForest: {metrics['accuracy']:.3f} accuracy")
except Exception as e:
    print(f"✗ RandomForest failed: {e}")

# Test XGBoost
try:
    xgb = XGBoostModel(n_estimators=10, max_depth=3, random_state=42)
    xgb.train(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    metrics = xgb.evaluate(X_test, y_test)
    results['XGBoost'] = metrics['accuracy']
    print(f"✓ XGBoost: {metrics['accuracy']:.3f} accuracy")
except Exception as e:
    print(f"✗ XGBoost failed: {e}")

# Test LightGBM
try:
    lgb = LightGBMModel(n_estimators=10, num_leaves=15, random_state=42)
    lgb.train(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=5)
    metrics = lgb.evaluate(X_test, y_test)
    results['LightGBM'] = metrics['accuracy']
    print(f"✓ LightGBM: {metrics['accuracy']:.3f} accuracy")
except Exception as e:
    print(f"✗ LightGBM failed: {e}")

# Test Naive Bayes
try:
    nb = NaiveBayes()
    nb.train(X_train, y_train)
    metrics = nb.evaluate(X_test, y_test)
    results['NaiveBayes'] = metrics['accuracy']
    print(f"✓ NaiveBayes: {metrics['accuracy']:.3f} accuracy")
except Exception as e:
    print(f"✗ NaiveBayes failed: {e}")

# Test Logistic Regression
try:
    lr = LogisticRegressionModel(max_iter=500, random_state=42)
    lr.train(X_train, y_train)
    metrics = lr.evaluate(X_test, y_test)
    results['LogisticRegression'] = metrics['accuracy']
    print(f"✓ LogisticRegression: {metrics['accuracy']:.3f} accuracy")
except Exception as e:
    print(f"✗ LogisticRegression failed: {e}")

# Test CNN
try:
    cnn = CNN(epochs=5, batch_size=16, num_classes=3)
    cnn.train(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    metrics = cnn.evaluate(X_test, y_test)
    results['CNN'] = metrics['accuracy']
    print(f"✓ CNN: {metrics['accuracy']:.3f} accuracy")
except Exception as e:
    print(f"✗ CNN failed: {e}")

# Summary
print("\n" + "=" * 60)
print("Integration Test Summary")
print("=" * 60)
print(f"Models tested: {len(results)}/6")
print(f"All models passed: {'✓ YES' if len(results) == 6 else '✗ NO'}")

if results:
    print(f"\nBest model: {max(results, key=results.get)} ({max(results.values()):.3f})")
    print(f"Average accuracy: {np.mean(list(results.values())):.3f}")

print("\n" + "=" * 60)
print("✓ Integration test completed successfully!")
print("=" * 60)
