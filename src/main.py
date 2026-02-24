# src/main.py
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

    # Filter out the desired class if specified in parameters (binary: class vs normal(0))
    if param.DATA_PARAMS['class'] is not None:
        print(f"Filtering to examine only class {param.DATA_PARAMS['class']}...")
        mask = (y == param.DATA_PARAMS['class']) | (y == 0)
        X = X[mask]
        y = y[mask]
        y = np.where(y == param.DATA_PARAMS['class'], 1, 0)
        print(f"Filtered dataset has {X.shape[0]} samples")

    # Optionally subsample the dataset (stratified)
    if param.DATA_PARAMS.get('samples') is not None and X.shape[0] > param.DATA_PARAMS.get('samples'):
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

    # --- optional: oversample training set (SMOTE) to fix imbalance ---
    if param.DATA_PARAMS.get('oversample', False):
        try:
            from imblearn.over_sampling import SMOTE
        except Exception as e:
            raise RuntimeError("imbalanced-learn required for oversampling. Install it in the runtime: pip install imbalanced-learn") from e

        print("Applying SMOTE to training set (oversample=True)...")
        sm = SMOTE(random_state=param.DATA_PARAMS.get('random_state', 42), n_jobs=-1)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        unique, counts = np.unique(y_train, return_counts=True)
        print("Post-SMOTE train class counts:", dict(zip(unique, counts)))

    # --- optional: compute scale_pos_weight for XGBoost automatically ---
    try:
        if 'XGBoost' in param.HYPERPARAMETERS:
            if 'scale_pos_weight' not in param.HYPERPARAMETERS['XGBoost']:
                unique, counts = np.unique(y_train, return_counts=True)
                if len(counts) == 2:
                    # find negative and positive counts
                    if unique[0] == 0:
                        neg = int(counts[0]); pos = int(counts[1])
                    else:
                        neg = int(counts[1]); pos = int(counts[0])
                    if pos > 0:
                        param.HYPERPARAMETERS['XGBoost']['scale_pos_weight'] = float(neg) / float(pos)
                        print("Set XGBoost scale_pos_weight to", param.HYPERPARAMETERS['XGBoost']['scale_pos_weight'])
    except Exception:
        pass

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_estimator(model_name):
    """
    Return an estimator instance for a given model_name based on param.HYPERPARAMETERS.
    For LR and KNN we return a Pipeline that includes StandardScaler to ensure correct behaviour.
    """
    hp = param.HYPERPARAMETERS.get(model_name, {})

    if model_name == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**hp)

    if model_name == 'LogisticRegression':
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        hp_lr = dict(hp)  # copy to avoid mutating param
        if 'class_weight' not in hp_lr:
            hp_lr['class_weight'] = 'balanced'
        if 'solver' not in hp_lr:
            # saga supports large datasets and class_weight
            hp_lr['solver'] = 'saga'
        if 'max_iter' not in hp_lr:
            hp_lr['max_iter'] = 2000
        return Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(**hp_lr))])

    if model_name == 'KNN':
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        hp_knn = dict(hp)
        if 'weights' not in hp_knn:
            hp_knn['weights'] = 'distance'
        return Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(**hp_knn))])

    if model_name == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**hp)

    if model_name == 'NaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(**hp)

    if model_name == 'XGBoost':
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError("xgboost is required for XGBoost model. Install it in the runtime: pip install xgboost") from e
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
    cv = PredefinedSplit(test_fold=[-1] * len(X_train) + [0] * len(X_val))

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
    # attempt to copy project param; use src/param.py if present
    src_param = 'src/param.py' if os.path.exists('src/param.py') else 'param.py'
    shutil.copy(src_param, param_copy_path)
    print(f"Copied param.py to {param_copy_path}")


def save_models(models, output_dir):
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    for model_name, model_instance in models.items():
        model_path = os.path.join(models_dir, f"{model_name}")
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
                try:
                    entry[k] = v.tolist()
                except Exception:
                    entry[k] = v
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
    return summary


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
            wrapped, gs = train(model_name, X_train, y_train, X_val, y_val)
            metrics = validate(wrapped, X_test, y_test)
            metrics['best_params'] = getattr(gs, 'best_params_', {})
            metrics['cv_results'] = getattr(gs, 'cv_results_', {})
            results[model_name] = metrics
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
        python main.py --data_path="data/mixalldata_clean.csv" --output_path="outputs" --timestamp=<ts>
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs='?', default="../data/mixalldata_clean.csv", help="Path to the input CSV data file")
    parser.add_argument('--output_path', nargs='?', default="outputs", help="Directory to save outputs")
    parser.add_argument('--timestamp', nargs='?', default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Optional timestamp string for output directory naming")
    args = parser.parse_args()

    # create output dir path
    if args.timestamp:
        output_dir = os.path.join(args.output_path, "run " + args.timestamp)
    else:
        output_dir = os.path.join(args.output_path, "run " + datetime.now().strftime("%Y%m%d_%H%M%S"))

    os.makedirs(output_dir, exist_ok=True)
    main(args.data_path, output_dir)
