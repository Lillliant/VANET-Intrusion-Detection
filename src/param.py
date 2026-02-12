# Models to train
MODELS = [
	'XGBoost',
    'RandomForest',
    'LogisticRegression',
    'KNN',
    'DecisionTree',
]

# Default hyperparameters used to initialize estimators (non-grid defaults)
HYPERPARAMETERS = {
	'RandomForest': {
		'n_estimators': 100,
		'max_depth': 10,
		'random_state': 42,
		'n_jobs': -1
	},
	'XGBoost': {
		'n_estimators': 100,
		'max_depth': 10,
		'learning_rate': 0.1,
		'eval_metric': 'logloss',
		'random_state': 42
	},
	'LogisticRegression': {
		'max_iter': 1000,
		'solver': 'lbfgs',
		'C': 1.0,
		'random_state': 42
	},
	'NaiveBayes': {
		'var_smoothing': 1e-9
	},
    'KNN': {
        'n_neighbors': 5,
        'weights': 'uniform'
    },
    'DecisionTree': {
        'max_depth': 10,
        'min_samples_split': 500,
        'random_state': 42
    }
}

# Parameter grids for GridSearchCV
"""
GRID_PARAMS = {
	'RandomForest': {
		'n_estimators': [100, 200],
		'max_depth': [None, 10, 20]
	},
	'XGBoost': {
		'n_estimators': [50, 100],
		'max_depth': [3, 6],
		'learning_rate': [0.1, 0.01]
	},
	'LogisticRegression': {
		'C': [0.01, 0.1, 1.0, 10.0]
	},
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [100, 500, 1000]
    },
	'NaiveBayes': {
		'var_smoothing': [1e-9, 1e-8, 1e-7]
	}
}
"""

# Data and CV configuration
DATA_PARAMS = {
	'test_size': 0.15,
    'class': 15, # Examine only one class; Otherwise, examine all classes (set to None)
    'validation_size': 0.15,
	'random_state': 42,
}

# Metrics selection for evaluation and GridSearchCV
# The first entry is used as the primary metric to refit in GridSearchCV
METRICS = ['f1_score', 'precision', 'recall', 'accuracy']
REFIT_METRIC = METRICS[0]

# Misc
RANDOM_STATE = 42

