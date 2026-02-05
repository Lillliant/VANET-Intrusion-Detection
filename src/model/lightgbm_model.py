from .base import Base
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


class LightGBMModel(Base):
    """LightGBM model for intrusion detection
    
    Gradient boosting framework optimized for large datasets.
    Uses histogram-based algorithms for efficient training.
    Excellent for multi-class classification with millions of samples.
    """
    
    def __init__(self, name="LightGBM", **kwargs):
        """
        Initialize LightGBM model
        
        Args:
            name: Model name
            **kwargs: Hyperparameters for LGBMClassifier
        """
        # Set default parameters optimized for large datasets
        default_params = {
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
            'verbose': -1  # Suppress warnings
        }
        default_params.update(kwargs)
        
        model = lgb.LGBMClassifier(**default_params)
        super().__init__(name, model)
        self.trained = False
    
    def preprocess(self, X, y=None):
        """
        Preprocess data for LightGBM
        
        Args:
            X: Features
            y: Labels (optional)
            
        Returns:
            Preprocessed X, y (if provided)
        """
        # LightGBM can handle numerical data directly
        X = np.array(X)
        if y is not None:
            y = np.array(y)
            return X, y
        return X
    
    def train(self, X_train, y_train, eval_set=None, **kwargs):
        """
        Train the LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            eval_set: Optional evaluation set for early stopping
            **kwargs: Additional training parameters
        """
        X_train, y_train = self.preprocess(X_train, y_train)
        
        if eval_set is not None:
            eval_set = [(self.preprocess(X, y)) for X, y in eval_set]
        
        # Add early stopping if eval_set is provided
        callbacks = []
        if eval_set is not None and kwargs.get('early_stopping_rounds'):
            callbacks.append(lgb.early_stopping(kwargs.get('early_stopping_rounds', 10)))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
            **{k: v for k, v in kwargs.items() if k != 'early_stopping_rounds'}
        )
        
        self.trained = True
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        X = self.preprocess(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        X = self.preprocess(X)
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        
        Returns:
            Feature importance array
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importance")
        return self.model.feature_importances_
