"""
Model package for intrusion detection
"""

from .base import Base
from .random_forest import RandomForest
from .xgboost_model import XGBoostModel
from .cnn import CNN

__all__ = ['Base', 'RandomForest', 'XGBoostModel', 'CNN']
