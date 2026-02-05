# This file acts as a configuration file listing the hyperparameters for the machine learning framework.

# Models to train
MODELS = ['RandomForest', 'XGBoost', 'CNN']

# Hyperparameters for each model
HYPERPARAMETERS = {
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    },
    'XGBoost': {
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    },
    'CNN': {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_classes': 2  # Binary classification by default
    }
}

# Data preprocessing parameters
DATA_PARAMS = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42
}
