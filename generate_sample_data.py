"""
Generate a sample dataset for testing the ML pipeline
This creates a synthetic intrusion detection dataset in CSV format
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic data
print("Generating sample intrusion detection dataset...")

# Create a larger dataset for better model training
n_samples = 5000
n_features = 50

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=40,
    n_redundant=5,
    n_repeated=0,
    n_classes=2,
    weights=[0.7, 0.3],  # Imbalanced dataset (70% normal, 30% intrusion)
    flip_y=0.01,  # 1% label noise
    random_state=42
)

# Create feature names
feature_names = [f'feature_{i+1}' for i in range(n_features)]

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y

# Save to CSV
output_path = 'data/sample_dataset.csv'
df.to_csv(output_path, index=False)

print(f"Dataset saved to {output_path}")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  Classes: {df['label'].nunique()}")
print(f"  Class distribution:")
print(f"    Normal (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"    Intrusion (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
