"""Preprocess the data into pickle files for training and testing. This is used for the binary classification aspect."""
from main import load, preprocess
import os
import argparse
import pickle
import json


"""
Configurations
"""
methods = ['smote_tomek'] #'tomek_links', 'neighbourhood_cleaning_rule', 'smote_tomek', 'smote', 'random_under_sample'
classes = [i for i in range(1, 20)] # 0-4 for all classes; Otherwise, specify a class (e.g., 4)
statistics = {m: {} for m in methods}


# Get the path to the dataset from the command line arguments
parser = argparse.ArgumentParser(description='Preprocess the dataset and save the resampled training splits and original test splits as pickle files.')
parser.add_argument('--data_path', type=str, default="data/mixalldata_clean.csv", help='Path to the dataset CSV file')
parser.add_argument('--output_path', type=str, default="data/resampled_data", help='Path to save the resampled training splits and original test splits as pickle files')
args = parser.parse_args()
data_path = args.data_path
destination_path = args.output_path

X, y = load(data_path)
print("Dataset loaded successfully.")

for c in classes:
    for m in methods:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess(X, y, y_class=c, resamp_method=m)
        statistics[m][f'class_{c}'] = {
            'train': {
                'X': X_train.shape,
                'y': y_train.shape,
                'positive': int(sum(y_train == 1)),
                'negative': int(sum(y_train == 0)),
            },
            'val': {
                'X': X_val.shape if X_val is not None else None,
                'y': y_val.shape if y_val is not None else None,
                'positive': int(sum(y_val == 1)) if y_val is not None else None,
                'negative': int(sum(y_val == 0)) if y_val is not None else None,
            },
            'test': {
                'X': X_test.shape,
                'y': y_test.shape,
                'positive': int(sum(y_test == 1)),
                'negative': int(sum(y_test == 0))
            }
        }
        print(f"Preprocessing completed for class {c} with resampling method {m}.")

        # Save the resampled splits as pickle files in the output path, ordered by resampling method
        output_dir = f"{destination_path}/{m}/class_{c}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/train.pkl", "wb") as f:
            pickle.dump((X_train, y_train), f)
        if X_val is not None and y_val is not None:
            with open(f"{output_dir}/val.pkl", "wb") as f:
                pickle.dump((X_val, y_val), f)
        with open(f"{output_dir}/test.pkl", "wb") as f:
            pickle.dump((X_test, y_test), f)
        print(f"Pickle files saved for class {c} with resampling method {m}.")
    
# Save the statistics as a json file in the output path
with open(f"{destination_path}/statistics.json", "w") as f:
    json.dump(statistics, f, indent=2)
print("Statistics saved successfully.")