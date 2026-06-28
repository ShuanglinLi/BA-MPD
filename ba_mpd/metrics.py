import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def macro_accuracy(y_true, y_pred, num_classes=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.arange(int(num_classes)) if num_classes is not None else np.unique(y_true)
    values = []
    for cls in classes:
        mask = y_true == cls
        if mask.any():
            values.append(float((y_pred[mask] == y_true[mask]).mean()))
    return float(np.mean(values)) if values else 0.0


def classification_metrics(y_true, y_pred, num_classes=None):
    labels = None if num_classes is None else np.arange(int(num_classes))
    return {
        "macro_accuracy": macro_accuracy(y_true, y_pred, num_classes=num_classes),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "micro_accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
    }

