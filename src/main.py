"""
Machine Learning Pipeline for Intrusion Detection
This framework trains and evaluates Random Forest, XGBoost, and CNN models on the VeReMi dataset.
"""

import os
import json
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from model.random_forest import RandomForest
from model.xgboost_model import XGBoostModel
from model.cnn import CNN
import param


def load(data_path):
    """
    Load the dataset from CSV file
    
    Args:
        data_path: Path to the CSV data file
        
    Returns:
        X (features), y (labels)
    """
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load CSV data
    df = pd.read_csv(data_path)
    
    # Assuming the last column is the label/target
    # Adjust this based on actual dataset structure
    if 'label' in df.columns:
        y = df['label'].values
        X = df.drop('label', axis=1).values
    elif 'target' in df.columns:
        y = df['target'].values
        X = df.drop('target', axis=1).values
    elif 'class' in df.columns:
        y = df['class'].values
        X = df.drop('class', axis=1).values
    else:
        # Assume last column is the label
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X, y


def preprocess(X, y):
    """
    Preprocess the data: split into train, validation, and test sets
    
    Args:
        X: Features
        y: Labels
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("Preprocessing data...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=param.DATA_PARAMS['test_size'],
        random_state=param.DATA_PARAMS['random_state'],
        stratify=y
    )
    
    # Second split: separate validation set from training set
    val_size = param.DATA_PARAMS['validation_size'] / (1 - param.DATA_PARAMS['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=param.DATA_PARAMS['random_state'],
        stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train(model_name, X_train, y_train, X_val, y_val):
    """
    Train a specific model
    
    Args:
        model_name: Name of the model to train ('RandomForest', 'XGBoost', or 'CNN')
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Trained model instance
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} model...")
    print(f"{'='*60}")
    
    # Get hyperparameters for this model
    hyperparams = param.HYPERPARAMETERS.get(model_name, {})
    
    # Initialize model
    if model_name == 'RandomForest':
        model = RandomForest(**hyperparams)
        model.train(X_train, y_train)
        
    elif model_name == 'XGBoost':
        model = XGBoostModel(**hyperparams)
        # Use validation set for early stopping
        eval_set = [(X_val, y_val)]
        model.train(X_train, y_train, eval_set=eval_set, verbose=True)
        
    elif model_name == 'CNN':
        model = CNN(**hyperparams)
        # Use validation set
        validation_data = (X_val, y_val)
        model.train(X_train, y_train, validation_data=validation_data)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"{model_name} training completed!")
    return model


def validate(model, X_test, y_test):
    """
    Validate/evaluate a trained model
    
    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating {model.name}...")
    metrics = model.evaluate(X_test, y_test)
    
    print(f"\nResults for {model.name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    return metrics


def aggregate(results, output_dir):
    """
    Aggregate and save final statistics
    
    Args:
        results: Dictionary of results for all models
        output_dir: Directory to save results
    """
    print(f"\n{'='*60}")
    print("Aggregating Results")
    print(f"{'='*60}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models': param.MODELS,
        'hyperparameters': param.HYPERPARAMETERS,
        'results': {}
    }
    
    # Format results
    for model_name, metrics in results.items():
        summary['results'][model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
    
    # Save summary to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"results_{timestamp}.json")
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {summary_path}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")
    
    return summary


def save_models(models, output_dir):
    """
    Save trained models
    
    Args:
        models: Dictionary of trained models
        output_dir: Directory to save models
    """
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, model_instance in models.items():
        model_path = os.path.join(models_dir, f"{model_name}_{timestamp}")
        
        if model_name == 'CNN':
            # Save Keras model
            model_instance.model.save(f"{model_path}.h5")
            # Save scaler separately
            with open(f"{model_path}_scaler.pkl", 'wb') as f:
                pickle.dump(model_instance.scaler, f)
        else:
            # Save sklearn/xgboost models with pickle
            with open(f"{model_path}.pkl", 'wb') as f:
                pickle.dump(model_instance, f)
        
        print(f"Saved {model_name} model to {model_path}")


def main(data_path, output_dir='outputs'):
    """
    Main pipeline execution
    
    Args:
        data_path: Path to the dataset CSV file
        output_dir: Directory to save outputs
    """
    print("="*60)
    print("Intrusion Detection ML Pipeline")
    print("="*60)
    print(f"Models to train: {', '.join(param.MODELS)}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Load data
    X, y = load(data_path)
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(X, y)
    
    # Train all models
    trained_models = {}
    results = {}
    
    for model_name in param.MODELS:
        try:
            # Train model
            model = train(model_name, X_train, y_train, X_val, y_val)
            trained_models[model_name] = model
            
            # Evaluate model
            metrics = validate(model, X_test, y_test)
            results[model_name] = metrics
            
        except Exception as e:
            print(f"\nError training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Aggregate results
    if results:
        aggregate(results, output_dir)
        
        # Save trained models
        save_models(trained_models, output_dir)
    else:
        print("\nNo models were successfully trained.")
    
    print("\n" + "="*60)
    print("Pipeline execution completed!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Check if data path is provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default data path
        data_path = "data/dataset.csv"
    
    # Check if output directory is provided
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "outputs"
    
    # Run the pipeline
    main(data_path, output_dir)

