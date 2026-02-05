from .base import Base
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


class RandomForest(Base):
    """Random Forest model for intrusion detection"""
    
    def __init__(self, name="RandomForest", **kwargs):
        """
        Initialize Random Forest model
        
        Args:
            name: Model name
            **kwargs: Hyperparameters for RandomForestClassifier
        """
        model = RandomForestClassifier(**kwargs)
        super().__init__(name, model)
        self.trained = False
    
    def preprocess(self, X, y=None):
        """
        Preprocess data for Random Forest
        
        Args:
            X: Features
            y: Labels (optional)
            
        Returns:
            Preprocessed X, y (if provided)
        """
        # Random Forest can handle numerical data directly
        # Ensure data is in the right format
        X = np.array(X)
        if y is not None:
            y = np.array(y)
            return X, y
        return X
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model
        
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
