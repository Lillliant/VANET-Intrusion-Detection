import os
import json
import pickle
from datetime import datetime
import time
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from model.base import Base
import param
import util.metrics
from util.util import print_results

def load(data_path):
    """Load dataset from CSV file."""

    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    y = df['class'].values
    X = df.drop('class', axis=1).values
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X, y


def preprocess(X, y):
    """Create train/validation/test split."""
    print("Preprocessing data...")

    # Filter out the desired class if specified in parameters
    # In this case, we will perform binary classification
    if param.DATA_PARAMS['class'] is not None:
        print(f"Filtering to examine only class {param.DATA_PARAMS['class']}...")
        mask = (y == param.DATA_PARAMS['class']) | (y == 0)  # Keep the specified class and the normal class (0)
        X = X[mask]
        y = y[mask]
        y = np.where(y == param.DATA_PARAMS['class'], 1, 0) # Convert to binary labels
        print(f"Filtered dataset has {X.shape[0]} samples")

    # Filter out the specified number of samples if defined in parameters, stratified by class to maintain distribution
    if param.DATA_PARAMS['samples'] is not None and X.shape[0] > param.DATA_PARAMS['samples']:
        print(f"Sampling {param.DATA_PARAMS['samples']} samples from the dataset...")
        X, _, y, _ = train_test_split(
            X, y,
            train_size=param.DATA_PARAMS['samples'],
            random_state=param.DATA_PARAMS['random_state'],
            stratify=y
        )
        print(f"Sampled dataset has {X.shape[0]} samples")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=param.DATA_PARAMS['test_size'],
        random_state=param.DATA_PARAMS['random_state'],
        stratify=y
    )
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


def get_estimator(model_name):
    hp = param.HYPERPARAMETERS.get(model_name, {})
    if model_name == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**hp)
    if model_name == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**hp)
    if model_name == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**hp)
    if model_name == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**hp)
    if model_name == 'NaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(**hp)
    if model_name == 'XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(**hp)
    raise ValueError(f"Unknown model: {model_name}")


def train(model_name, X_train, y_train, X_val, y_val):
    """
    If a grid parameter is given, run GridSearchCV; otherwise, 
    fit the estimator directly using default hyperparameters.
    """
    estimator = get_estimator(model_name)
    param_grid = param.GRID_PARAMS.get(model_name, None) if hasattr(param, 'GRID_PARAMS') else None

    # Build scoring dict for GridSearchCV
    scoring = {m: m for m in param.METRICS}
    # Use a fixed validation set for GridSearchCV
    cv = PredefinedSplit(test_fold=[-1]*len(X_train) + [0]*len(X_val))

    if param_grid:
        print(f"\n{'='*60}")
        print(f"Tuning and training {model_name} with GridSearchCV...")
        print(f"{'='*60}")
        gs = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring=scoring,
            refit=param.REFIT_METRIC,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        t0 = time.perf_counter()
        gs.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
        total_time = time.perf_counter() - t0
        best = gs.best_estimator_
        wrapped = Base(model_name, best)
        wrapped.trained = True
        wrapped.training_time = total_time
        print(f"Best params for {model_name}: {gs.best_params_}")
        return wrapped, gs
    else:
        # No grid provided: fit the estimator directly using default hyperparameters
        # Does not use the validation set since no tuning is performed.
        print(f"\n{'='*60}")
        print(f"Tuning and training {model_name} with default parameters...")
        print(f"{'='*60}")
        hyperparams = param.HYPERPARAMETERS.get(model_name, {})
        wrapped = Base(model_name, estimator)
        wrapped.train(X_train, y_train, **hyperparams)
        return wrapped, None


def validate(model_wrapper, X_test, y_test):
    """Evaluate using metrics selected in `param.METRICS` and include confusion matrix."""
    print(f"\nEvaluating {model_wrapper.name}...")

    scorers = util.metrics.get_scorers(param.METRICS, multiclass=param.DATA_PARAMS['class'] is None)
    metrics = model_wrapper.evaluate(X_test, y_test, scorers=scorers)

    print(f"\nResults for {model_wrapper.name}:")
    print_results(metrics)

    return metrics


def aggregate(results, output_dir):
    """
    Aggregate and save final statistics
    """
    print(f"\n{'='*60}")
    print("Aggregating Results")
    print(f"{'='*60}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the results and summary to JSON file
    summary = save_results(results, output_dir)
    print(f"\nResults saved to {os.path.join(output_dir, 'results.json')}")

    # Print comparison table
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    # Print the available metrics as columns in the table, except confusion matrix
    print(f"{'Model':<15} " + " ".join([f"{metric:<12}" for metric in param.METRICS if metric != 'confusion_matrix']))
    print("-"*60)
    
    for model_name, metrics in results.items():
        metric_values = " ".join([f"{metrics.get(metric, 'N/A'):<12.4f}" for metric in param.METRICS if metric != 'confusion_matrix'])
        print(f"{model_name:<15} {metric_values}")
    
    return summary

def save_params(output_dir):
    # Copy param.py to the output directory for traceability
    param_copy_path = os.path.join(output_dir, f"param.py")
    shutil.copy('src/param.py', param_copy_path)
    print(f"Copied param.py to {param_copy_path}")

def save_models(models, output_dir):
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    for model_name, model_instance in models.items():
        model_path = os.path.join(models_dir, f"{model_name}")
        # Save sklearn/xgboost objects via pickle
        start_time = time.perf_counter()
        with open(f"{model_path}.pkl", 'wb') as f:
            pickle.dump(model_instance, f)
        elapsed_time = time.perf_counter() - start_time
        print(f"Saved {model_name} model to {model_path}.pkl in {elapsed_time:.2f}s")

def save_results(results, output_dir):
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models': list(results.keys()),
        'class': param.DATA_PARAMS['class'],
        'hyperparameters': {model: param.HYPERPARAMETERS.get(model, {}) for model in results.keys()},
        'results': {}
    }
    
    # Format results
    for model_name, metrics in results.items():
        entry = {}
        for k, v in metrics.items():
            if k == 'confusion_matrix':
                entry[k] = v.tolist()
            else:
                try:
                    entry[k] = float(v)
                except Exception:
                    entry[k] = v
        summary['results'][model_name] = entry
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, f"results.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

def main(data_path, output_dir='outputs'):
    print("="*60)
    print("Intrusion Detection ML Pipeline")
    print("="*60)
    print(f"Models to train: {', '.join(param.MODELS)}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Load data and preprocess
    X, y = load(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(X, y)
    save_params(output_dir)

    results = {}

    for model_name in param.MODELS:
        try:
            # Depend on given parameters, either run GridSearchCV or fit directly with default hyperparameters
            wrapped, gs = train(model_name, X_train, y_train, X_val, y_val)
            metrics = validate(wrapped, X_test, y_test)
            # attach best params and cv results
            metrics['best_params'] = getattr(gs, 'best_params_', {})
            metrics['cv_results'] = getattr(gs, 'cv_results_', {})
            results[model_name] = metrics
            # Save models immediately after they are done for partial results
            save_models({model_name: wrapped}, output_dir)
            save_results(results, output_dir)
        except Exception as e:
            print(f"\nError training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    if results:
        aggregate(results, output_dir)
    else:
        print("\nNo models were successfully trained.")
    
    print("\n" + "="*60)
    print("Pipeline execution completed!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    """
    Usage:
        python main.py [data_path] [output_dir] [timestamp]
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs='?', default="../data/mixalldata_clean.csv", help="Path to the input CSV data file")
    parser.add_argument('--output_path', nargs='?', default="outputs", help="Directory to save outputs")
    parser.add_argument('--timestamp', nargs='?', default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Optional timestamp string for output directory naming")
    args = parser.parse_args()

    # Check if a timestamp string is provided for the outputs directory
    # This allows easier movement of Colab-generated logs into the outputs directory
    if args.timestamp:
        output_dir = os.path.join(args.output_path, "run " + args.timestamp)
    else:
        output_dir = os.path.join(args.output_path, "run " + datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Create directory for storing output
    os.makedirs(output_dir, exist_ok=True)

    # Run the pipeline
    main(args.data_path, output_dir)
