class Base:
  # A base class for all machine learning models
  # This is used as a easy method for integrating different machine learning frameworks
  # from different libraries (e.g., PyTorch, Scikit-learn).
  
  def __init__(self, name, model):
    self.name = name
    self.model = model

  def preprocess(self):
    pass

  def train(self):
    pass

  def predict(self):
    pass

  def evaluate(self):
    pass
