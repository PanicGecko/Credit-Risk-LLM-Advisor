import numpy as np
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
    }
    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1 Score:  {metrics['f1_score']}")
    return metrics

def check_thresholds(metrics, thresholds):
    for metric, threshold in thresholds.items():
        if metrics[metric] < threshold:
            print(f"\nWARNING: {metric} {metrics[metric]} is below threshold {threshold}")
        else:
            print(f"\n{metric} {metrics[metric]} is above threshold {threshold}")

    