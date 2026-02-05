from .base import Base
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np


class CNN(Base):
    """Convolutional Neural Network model for intrusion detection"""
    
    def __init__(self, name="CNN", input_shape=None, num_classes=2, **kwargs):
        """
        Initialize CNN model
        
        Args:
            name: Model name
            input_shape: Shape of input data (will be reshaped for CNN)
            num_classes: Number of output classes
            **kwargs: Additional hyperparameters
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.scaler = StandardScaler()
        self.trained = False
        
        # Extract hyperparameters
        self.epochs = kwargs.get('epochs', 50)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        
        # Model will be built when we know the input shape
        model = None
        super().__init__(name, model)
    
    def build_model(self, input_dim):
        """
        Build the CNN architecture
        
        Args:
            input_dim: Number of input features
        """
        # Reshape input for CNN (add channel dimension)
        # For 1D CNN, we'll reshape features into a sequence
        reshaped_dim = int(np.sqrt(input_dim))
        if reshaped_dim * reshaped_dim < input_dim:
            reshaped_dim += 1
        
        # Pad input to make it square if needed
        self.reshaped_dim = reshaped_dim
        self.input_dim = input_dim
        
        model = models.Sequential([
            # Reshape to 2D for Conv2D
            layers.Reshape((self.reshaped_dim, self.reshaped_dim, 1), 
                          input_shape=(self.reshaped_dim * self.reshaped_dim,)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = 'sparse_categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess(self, X, y=None, fit_scaler=False):
        """
        Preprocess data for CNN
        
        Args:
            X: Features
            y: Labels (optional)
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            Preprocessed X, y (if provided)
        """
        X = np.array(X)
        
        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        # Pad to square shape if needed
        if not hasattr(self, 'reshaped_dim'):
            input_dim = X.shape[1]
            reshaped_dim = int(np.sqrt(input_dim))
            if reshaped_dim * reshaped_dim < input_dim:
                reshaped_dim += 1
            self.reshaped_dim = reshaped_dim
            self.input_dim = input_dim
        
        # Pad features to match reshaped_dim^2
        target_size = self.reshaped_dim * self.reshaped_dim
        if X.shape[1] < target_size:
            padding = np.zeros((X.shape[0], target_size - X.shape[1]))
            X = np.hstack([X, padding])
        
        if y is not None:
            y = np.array(y)
            return X, y
        return X
    
    def train(self, X_train, y_train, validation_data=None, **kwargs):
        """
        Train the CNN model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Optional tuple of (X_val, y_val)
            **kwargs: Additional training parameters
        """
        # Preprocess data
        X_train, y_train = self.preprocess(X_train, y_train, fit_scaler=True)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Preprocess validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val, y_val = self.preprocess(X_val, y_val, fit_scaler=False)
            validation_data = (X_val, y_val)
        
        # Early stopping callback
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Train model
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
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
        
        X = self.preprocess(X, fit_scaler=False)
        predictions = self.model.predict(X)
        
        # Convert probabilities to class labels
        if self.num_classes > 2:
            return np.argmax(predictions, axis=1)
        else:
            return (predictions > 0.5).astype(int).flatten()
    
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
        
        X = self.preprocess(X, fit_scaler=False)
        return self.model.predict(X)
    
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
