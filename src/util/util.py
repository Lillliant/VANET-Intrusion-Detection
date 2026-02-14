# File to store any utility functions used across files

def get_project_base(__file__):
    """
    Get the base directory of the project, which consists of the folders src, util, etc.
    The calling file must be within the project directory for this to work.
    """
    import os
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    while True: # Kepp going up the directory tree until we find a directory that contains 'src'
        if 'src' in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the root directory
            raise FileNotFoundError("Project base directory not found.")
        current_dir = parent_dir

def print_results(metrics):
    """
    Helper function to print the evaluation results in a readable format.
    """
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        if metric in ['training_time', 'prediction_time']:
            print(f"{metric}: {value:.4f} seconds")
        elif metric == 'confusion_matrix':
            print(f"{metric}:\n{value}")
        elif isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")