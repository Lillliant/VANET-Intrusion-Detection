from .base import Base
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


class XGBoostModel(Base):
    """XGBoost model for intrusion detection"""
    
    def __init__(self, name="XGBoost", **kwargs):
        """
        Initialize XGBoost model
        
        Args:
            name: Model name
            **kwargs: Hyperparameters for XGBClassifier
        """
        # Set default parameters if not provided
        # Don't set objective here - let XGBClassifier infer it
        default_params = {
            'use_label_encoder': False
        }
        default_params.update(kwargs)
        
        model = xgb.XGBClassifier(**default_params)
        super().__init__(name, model)
        self.trained = False
    
    def preprocess(self, X, y=None):
        """
        Preprocess data for XGBoost
        
        Args:
            X: Features
            y: Labels (optional)
            
        Returns:
            Preprocessed X, y (if provided)
        """
        # XGBoost can handle numerical data directly
        X = np.array(X)
        if y is not None:
            y = np.array(y)
            return X, y
        return X
    
    def train(self, X_train, y_train, eval_set=None, **kwargs):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            eval_set: Optional evaluation set for early stopping
            **kwargs: Additional training parameters
        """
        X_train, y_train = self.preprocess(X_train, y_train)
        
        if eval_set is not None:
            eval_set = [(self.preprocess(X, y)) for X, y in eval_set]
        
        self.model.fit(X_train, y_train, eval_set=eval_set, **kwargs)
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
