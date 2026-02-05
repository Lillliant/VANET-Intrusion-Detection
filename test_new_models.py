"""
Test script to verify new model implementations
This tests Naive Bayes, Logistic Regression, and LightGBM with synthetic data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from sklearn.datasets import make_classification
from model import NaiveBayes, LogisticRegressionModel, LightGBMModel


def test_new_models():
    """Test all three new models with synthetic data"""
    
    print("Generating synthetic test data...")
    # Create synthetic multi-class classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,  # Multi-class
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    val_split = int(0.9 * len(X_train))
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Test Naive Bayes
    print("\n" + "="*60)
    print("Testing Naive Bayes...")
    print("="*60)
    try:
        nb = NaiveBayes(var_smoothing=1e-9)
        nb.train(X_train, y_train)
        nb_metrics = nb.evaluate(X_test, y_test)
        print("✓ Naive Bayes: PASSED")
        print(f"  Accuracy: {nb_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {nb_metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"✗ Naive Bayes: FAILED - {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test Logistic Regression
    print("\n" + "="*60)
    print("Testing Logistic Regression...")
    print("="*60)
    try:
        lr = LogisticRegressionModel(max_iter=1000, C=1.0, random_state=42)
        lr.train(X_train, y_train)
        lr_metrics = lr.evaluate(X_test, y_test)
        print("✓ Logistic Regression: PASSED")
        print(f"  Accuracy: {lr_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {lr_metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"✗ Logistic Regression: FAILED - {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test LightGBM
    print("\n" + "="*60)
    print("Testing LightGBM...")
    print("="*60)
    try:
        lgb_model = LightGBMModel(n_estimators=50, learning_rate=0.1, random_state=42)
        lgb_model.train(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
        lgb_metrics = lgb_model.evaluate(X_test, y_test)
        print("✓ LightGBM: PASSED")
        print(f"  Accuracy: {lgb_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {lgb_metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"✗ LightGBM: FAILED - {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    test_new_models()
