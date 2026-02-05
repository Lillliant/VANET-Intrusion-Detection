"""
Test new models with larger multi-class dataset
This simulates a more realistic scenario with more samples
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from sklearn.datasets import make_classification
from model import NaiveBayes, LogisticRegressionModel, LightGBMModel
import time


def test_large_dataset():
    """Test models with larger multi-class dataset"""
    
    print("Generating larger synthetic multi-class dataset...")
    # Create a larger dataset (10K samples, 5 classes)
    X, y = make_classification(
        n_samples=10000,
        n_features=50,
        n_informative=40,
        n_redundant=5,
        n_classes=5,  # 5-class problem
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    val_split = int(0.9 * len(X_train))
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]
    
    print(f"Train samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    results = {}
    
    # Test Naive Bayes
    print("\n" + "="*60)
    print("Testing Naive Bayes on large dataset...")
    print("="*60)
    try:
        start = time.time()
        nb = NaiveBayes()
        nb.train(X_train, y_train)
        train_time = time.time() - start
        
        start = time.time()
        nb_metrics = nb.evaluate(X_test, y_test)
        eval_time = time.time() - start
        
        results['NaiveBayes'] = nb_metrics
        print("✓ Naive Bayes: PASSED")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Evaluation time: {eval_time:.2f}s")
        print(f"  Accuracy: {nb_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {nb_metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"✗ Naive Bayes: FAILED - {str(e)}")
    
    # Test Logistic Regression
    print("\n" + "="*60)
    print("Testing Logistic Regression on large dataset...")
    print("="*60)
    try:
        start = time.time()
        lr = LogisticRegressionModel(max_iter=500)
        lr.train(X_train, y_train)
        train_time = time.time() - start
        
        start = time.time()
        lr_metrics = lr.evaluate(X_test, y_test)
        eval_time = time.time() - start
        
        results['LogisticRegression'] = lr_metrics
        print("✓ Logistic Regression: PASSED")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Evaluation time: {eval_time:.2f}s")
        print(f"  Accuracy: {lr_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {lr_metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"✗ Logistic Regression: FAILED - {str(e)}")
    
    # Test LightGBM
    print("\n" + "="*60)
    print("Testing LightGBM on large dataset...")
    print("="*60)
    try:
        start = time.time()
        lgb_model = LightGBMModel(n_estimators=100, num_leaves=31)
        lgb_model.train(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
        train_time = time.time() - start
        
        start = time.time()
        lgb_metrics = lgb_model.evaluate(X_test, y_test)
        eval_time = time.time() - start
        
        results['LightGBM'] = lgb_metrics
        print("✓ LightGBM: PASSED")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Evaluation time: {eval_time:.2f}s")
        print(f"  Accuracy: {lgb_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {lgb_metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"✗ LightGBM: FAILED - {str(e)}")
    
    # Print comparison
    print("\n" + "="*60)
    print("Performance Comparison on Large Multi-Class Dataset")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1-Score':<12}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f}")
    
    print("\n" + "="*60)
    print("All large dataset tests completed!")
    print("="*60)


if __name__ == "__main__":
    test_large_dataset()
