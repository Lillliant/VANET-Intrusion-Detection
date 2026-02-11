import os
import json
import pickle
from datetime import datetime
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import confusion_matrix
from model.base import Base
import param
import util.metrics

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
    if param.DATA_PARAMS['class'] is not None:
        print(f"Filtering to examine only class {param.DATA_PARAMS['class']}...")
        mask = (y == param.DATA_PARAMS['class'])
        X = X[mask]
        y = y[mask]
        print(f"Filtered dataset has {X.shape[0]} samples")

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
    print(f"\n{'='*60}")
    print(f"Tuning and training {model_name} with GridSearchCV...")
    print(f"{'='*60}")

    estimator = get_estimator(model_name)
    param_grid = param.GRID_PARAMS.get(model_name, None)

    # Build scoring dict for GridSearchCV
    scoring = {m: m for m in param.METRICS}
    # Use a fixed validation set for GridSearchCV
    cv = PredefinedSplit(test_fold=[-1]*len(X_train) + [0]*len(X_val))

    if param_grid:
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
        hyperparams = param.HYPERPARAMETERS.get(model_name, {})
        wrapped = Base(model_name, estimator)
        wrapped.train(X_train, y_train, **hyperparams)
        return wrapped, None


def validate(model_wrapper, X_test, y_test):
    """Evaluate using metrics selected in `param.METRICS` and include confusion matrix."""
    print(f"\nEvaluating {model_wrapper.name}...")

    scorers = util.metrics.get_scorers(param.METRICS, multiclass=param.DATA_PARAMS['class'] is None)
    metrics = model_wrapper.evaluate(X_test, y_test, scorers=scorers)

    # Add confusion matrix separately
    y_pred = model_wrapper.predict(X_test)
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)

    print(f"\nResults for {model_wrapper.name}:")
    for k, v in metrics.items():
        if k == 'confusion_matrix':
            print(f"\nConfusion Matrix:\n{v}")
        elif isinstance(v, float) or isinstance(v, np.floating):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


def aggregate(results, output_dir):
    """
    Aggregate and save final statistics to a timestamped JSON file
    """
    print(f"\n{'='*60}")
    print("Aggregating Results")
    print(f"{'='*60}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models': list(results.keys()),
        'hyperparameters': param.HYPERPARAMETERS,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {summary_path}")

    # Copy a set of param.py parameters to the output directory for traceability
    param_copy_path = os.path.join(output_dir, f"param_{timestamp}.py")
    with open(param_copy_path, 'w') as f:
        f.write(json.dumps(param, indent=4))
    
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
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    for model_name, model_instance in models.items():
        model_path = os.path.join(models_dir, f"{model_name}")
        # Save sklearn/xgboost objects via pickle
        with open(f"{model_path}.pkl", 'wb') as f:
            pickle.dump(model_instance, f)
        print(f"Saved {model_name} model to {model_path}.pkl")


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

    # Prepare timestamped output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(output_dir, "run " + ts)
    os.makedirs(output_root, exist_ok=True)

    trained_models = {}
    results = {}

    for model_name in param.MODELS:
        try:
            # Depend on given parameters, either run GridSearchCV or fit directly with default hyperparameters
            wrapped, gs = train(model_name, X_train, y_train, X_val, y_val)
            trained_models[model_name] = wrapped
            metrics = validate(wrapped, X_test, y_test)
            # attach best params for traceability
            metrics['best_params'] = getattr(gs, 'best_params_', {})
            results[model_name] = metrics
        except Exception as e:
            print(f"\nError training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    if results:
        aggregate(results, output_root)
        save_models(trained_models, output_root)
    else:
        print("\nNo models were successfully trained.")
    
    print("\n" + "="*60)
    print("Pipeline execution completed!")
    print("="*60)


if __name__ == "__main__":
    import sys
    """
    Usage:
        python main.py <data_path> <output_dir>
    """
    
    # Check if data path is provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default data path
        data_path = "../data/mixalldata_clean.csv"
    
    # Check if output directory is provided
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "outputs"
    
    # Run the pipeline
    main(data_path, output_dir)

