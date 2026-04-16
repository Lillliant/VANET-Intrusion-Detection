"""Preprocess the data into pickle files for training and testing."""
from main import load, preprocess
methods = ['smote'] #'tomek_links', 'neighbourhood_cleaning_rule', 'smote_tomek'
classes = [i for i in range(1, 20)] # 0-4 for all classes; Otherwise, specify a class (e.g., 4)
X_test_list = []
y_test_list = []

"""
Load dataset
"""
data_path = "data/mixalldata_clean.csv"
df = load(data_path)
print("Dataset loaded successfully.")

for c in classes:
    for method in methods:
        """
        1. filter the dataset to only include the specified class and the negative class (0)
        2. apply the specified resampling method to the training split
        3. save the resampled training split and the original test split as pickle files
        4. organize the pickle files into folders based on the resampling method
        """
        X, y = load(data_path)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess(X, y, y_class=c)
        X_test_list.append(X_test)
        y_test_list.append(y_test)

# Check if the test splits are the same across all classes and methods
for i in range(1, len(X_test_list)):
    assert (X_test_list[i] == X_test_list[0]).all(), "Test splits are not the same across all classes and methods."
    assert (y_test_list[i] == y_test_list[0]).all(), "Test splits are not the same across all classes and methods."
print("Test splits are the same across all classes and methods.")
