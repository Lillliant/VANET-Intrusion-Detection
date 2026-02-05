# This file acts as a configuration file listing the hyperparameters for the machine learning framework.

# Models to train
MODELS = ['RandomForest', 'XGBoost', 'CNN', 'NaiveBayes', 'LogisticRegression', 'LightGBM']

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
    },
    'NaiveBayes': {
        # Gaussian Naive Bayes has minimal hyperparameters
        # var_smoothing can be adjusted for numerical stability
        'var_smoothing': 1e-9
    },
    'LogisticRegression': {
        'max_iter': 1000,
        'solver': 'lbfgs',  # Good for multi-class, efficient for large datasets
        'multi_class': 'auto',
        'C': 1.0,  # Inverse regularization strength
        'random_state': 42,
        'n_jobs': -1
    },
    'LightGBM': {
        'objective': 'multiclass',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 0.1,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
}

# Data preprocessing parameters
DATA_PARAMS = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42
}
