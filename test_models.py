"""
Test script to verify model implementations
This creates synthetic data to test that all models can be instantiated and trained
"""

import numpy as np
from sklearn.datasets import make_classification
from model import RandomForest, XGBoostModel, CNN


def test_models():
    """Test all three models with synthetic data"""
    
    print("Generating synthetic test data...")
    # Create synthetic binary classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
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
    
    # Test Random Forest
    print("\n" + "="*60)
    print("Testing Random Forest...")
    print("="*60)
    try:
        rf = RandomForest(n_estimators=10, max_depth=5, random_state=42)
        rf.train(X_train, y_train)
        rf_metrics = rf.evaluate(X_test, y_test)
        print("✓ Random Forest: PASSED")
        print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"✗ Random Forest: FAILED - {str(e)}")
    
    # Test XGBoost
    print("\n" + "="*60)
    print("Testing XGBoost...")
    print("="*60)
    try:
        xgb = XGBoostModel(n_estimators=10, max_depth=5, random_state=42)
        xgb.train(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_metrics = xgb.evaluate(X_test, y_test)
        print("✓ XGBoost: PASSED")
        print(f"  Accuracy: {xgb_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"✗ XGBoost: FAILED - {str(e)}")
    
    # Test CNN
    print("\n" + "="*60)
    print("Testing CNN...")
    print("="*60)
    try:
        cnn = CNN(epochs=5, batch_size=32, learning_rate=0.001, num_classes=2)
        cnn.train(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
        cnn_metrics = cnn.evaluate(X_test, y_test)
        print("✓ CNN: PASSED")
        print(f"  Accuracy: {cnn_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"✗ CNN: FAILED - {str(e)}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    test_models()
