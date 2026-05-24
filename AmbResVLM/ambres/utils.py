"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import wandb
import numpy as np


def classification_metrics(
    pred: np.ndarray, gt: np.ndarray, verbose=False, log_wandb=False
) -> dict:
    TOT = len(pred)  # Total
    P = np.sum(gt)  # Positive
    N = np.sum(1 - gt)  # Negative
    TP = np.sum(pred * gt)  # True Positive
    TN = np.sum((1 - pred) * (1 - gt))  # True Negative
    FP = np.sum(pred) - TP  # False Positive
    FN = np.sum(1 - pred) - TN  # False Negative

    TPR = TP / P  # True Positive Rate
    TNR = TN / N  # True Negative Rate
    FNR = FN / P  # False Negative Rate
    FPR = FP / N  # False Positive Rate
    ACC = (TP + TN) / TOT  # Accuracy
    Precision = TP / (TP + FP)  # Precision
    Recall = TP / (TP + FN)  # Recall
    F1 = 2 * Precision * Recall / (Precision + Recall)  # F1 Score

    metrics = {
        "TPR": TPR,
        "TNR": TNR,
        "FNR": FNR,
        "FPR": FPR,
        "ACC": ACC,
        "Precision": Precision,
        "Recall": Recall,
        "F1": F1,
    }
    for key, value in metrics.items():
        if verbose:
            print(f"{key}: {value}")
        if log_wandb:
            wandb.run.summary[key] = value
    return metrics
