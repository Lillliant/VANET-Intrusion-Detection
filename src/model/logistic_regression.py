from .base import Base
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


class LogisticRegressionModel(Base):
    """Logistic Regression model for intrusion detection
    
    Suitable for multi-class classification with large datasets.
    Uses L2 regularization and supports parallel processing.
    """
    
    def __init__(self, name="LogisticRegression", **kwargs):
        """
        Initialize Logistic Regression model
        
        Args:
            name: Model name
            **kwargs: Hyperparameters for LogisticRegression
        """
        # Set default parameters optimized for large datasets
        default_params = {
            'max_iter': 1000,
            'solver': 'lbfgs',  # Good for multi-class problems
            'multi_class': 'auto',
            'n_jobs': -1,  # Use all CPU cores
            'random_state': 42
        }
        default_params.update(kwargs)
        
        model = LogisticRegression(**default_params)
        super().__init__(name, model)
        self.trained = False
    
    def preprocess(self, X, y=None):
        """
        Preprocess data for Logistic Regression
        
        Args:
            X: Features
            y: Labels (optional)
            
        Returns:
            Preprocessed X, y (if provided)
        """
        # Logistic Regression works with numerical data
        X = np.array(X)
        if y is not None:
            y = np.array(y)
            return X, y
        return X
    
    def train(self, X_train, y_train):
        """
        Train the Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        X_train, y_train = self.preprocess(X_train, y_train)
        self.model.fit(X_train, y_train)
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
    
    def get_coefficients(self):
        """
        Get model coefficients
        
        Returns:
            Model coefficients
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting coefficients")
        return self.model.coef_
