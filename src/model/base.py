class Base:
  # A base class for all machine learning models
  def __init__(self, name, model):
    self.name = name
    self.model = model

  def train(self):
    pass

  def predict(self):
    pass
