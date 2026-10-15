import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os
from typing import List, Dict, Any

def calculate_classification_metrics(y_true: List[int], y_pred: List[int], classes: List[str]) -> Dict[str, Any]:
    """Calculates rigorous classification metrics preventing metric fabrication."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist()
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classes": classes
    }
    return metrics

def save_metrics(metrics: Dict[str, Any], output_path: str):
    """Saves metrics to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
