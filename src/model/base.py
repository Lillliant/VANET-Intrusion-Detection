import time

class Base:
    """
    A base class for all machine learning models.
    This is used as an easy method for integrating different machine learning frameworks
    from different libraries (e.g., PyTorch, Scikit-learn).
    """
    
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.trained = False
        self.training_time = None
        self.prediction_time = None

    def train(self, X_train, y_train, **kwargs):            
        start_time = time.perf_counter()
        self.model.fit(X_train, y_train)
        self.training_time = time.perf_counter() - start_time # training time in seconds
        print(f"{self.name} training completed in {self.training_time:.2f} seconds.")
        self.trained = True
        return self

    def predict(self, X):
        if not self.trained: 
                raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, scorers=None):
        if not self.trained:
                raise ValueError("Model must be trained before evaluation")
        if scorers is None:
                raise ValueError("Scorers must be provided for evaluation")
        
        start_time = time.perf_counter()
        y_pred = self.predict(X_test)
        self.prediction_time = time.perf_counter() - start_time # evaluation time in seconds
        print(f"{self.name} prediction completed in {self.prediction_time:.2f} seconds.")
        metrics = {'training_time': self.training_time, 'prediction_time': self.prediction_time}
        for score_name, scorer in scorers.items():
                start_time = time.perf_counter()
                metrics[score_name] = scorer(y_test, y_pred)
                print(f"{self.name} {score_name} calculated in {time.perf_counter() - start_time:.4f} seconds.")
        return metrics
