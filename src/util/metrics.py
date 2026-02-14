from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def get_scorers(metric_names, multiclass=False):
    """
    Return a dictionary of scorer functions based on the provided metric names.
    If multiclass is True, use 'weighted' averaging for precision, recall, and f1.
    """
    
    # Lambda functions are used to create partial functions to handle multiclass vs binary cases
    scorer_map = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted' if multiclass else 'binary'),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted' if multiclass else 'binary'),
        'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted' if multiclass else 'binary'),
        'confusion_matrix': lambda y_true, y_pred: confusion_matrix(y_true, y_pred)
    }
    return {m: scorer_map[m] for m in metric_names if m in scorer_map}