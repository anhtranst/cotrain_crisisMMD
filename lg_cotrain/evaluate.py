"""Evaluation metrics and ensemble prediction."""

from collections import Counter
from typing import Dict, List


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute error rate, F1, precision, recall (macro + weighted), and per-class F1.

    Args:
        y_true: Ground truth label ids (list or array).
        y_pred: Predicted label ids (list or array).

    Returns:
        Dict with error_rate (%), macro/weighted F1/precision/recall,
        and per_class_f1 (list).
    """
    try:
        import numpy as np
        from sklearn.metrics import f1_score, precision_score, recall_score
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        error_rate = 100.0 * (1.0 - np.mean(y_true == y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        weighted_precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        weighted_recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    except ImportError:
        # Pure-Python fallback
        n = len(y_true)
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        error_rate = 100.0 * (1.0 - correct / n) if n > 0 else 0.0
        macro_f1, per_class_f1 = _compute_f1_pure(list(y_true), list(y_pred))
        macro_precision, macro_recall = _compute_precision_recall_pure(list(y_true), list(y_pred), "macro")
        weighted_f1 = macro_f1  # approximate
        weighted_precision, weighted_recall = _compute_precision_recall_pure(list(y_true), list(y_pred), "weighted")

    return {
        "error_rate": error_rate,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "per_class_f1": per_class_f1,
    }


def _compute_f1_pure(y_true: list, y_pred: list):
    """Pure-Python macro-F1 and per-class F1 computation."""
    classes = sorted(set(y_true) | set(y_pred))
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return macro_f1, f1s


def _compute_precision_recall_pure(y_true: list, y_pred: list, average: str = "macro"):
    """Pure-Python precision and recall computation."""
    classes = sorted(set(y_true) | set(y_pred))
    precisions, recalls, supports = [], [], []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        supports.append(sum(1 for t in y_true if t == cls))
    if average == "weighted" and sum(supports) > 0:
        total = sum(supports)
        avg_p = sum(p * s for p, s in zip(precisions, supports)) / total
        avg_r = sum(r * s for r, s in zip(recalls, supports)) / total
    else:
        avg_p = sum(precisions) / len(precisions) if precisions else 0.0
        avg_r = sum(recalls) / len(recalls) if recalls else 0.0
    return avg_p, avg_r


def compute_ece(y_true, y_probs, n_bins=15):
    """Compute Expected Calibration Error.

    Args:
        y_true: Ground truth label ids (list or array), length n.
        y_probs: Predicted probability matrix, shape (n, n_classes).
                 Can be a numpy array or list of lists.
        n_bins: Number of equal-width bins over [0, 1].

    Returns:
        ECE as a float in [0, 1].
    """
    try:
        import numpy as np
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs, dtype=float)
        confidences = y_probs.max(axis=1)
        predictions = y_probs.argmax(axis=1)
        correct = (predictions == y_true).astype(float)
        n = len(y_true)
        if n == 0:
            return 0.0
        ece = 0.0
        for i in range(n_bins):
            lo = i / n_bins
            hi = (i + 1) / n_bins
            mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
            count = mask.sum()
            if count > 0:
                acc = correct[mask].mean()
                conf = confidences[mask].mean()
                ece += (count / n) * abs(acc - conf)
        return float(ece)
    except ImportError:
        return _compute_ece_pure(list(y_true), y_probs, n_bins)


def _compute_ece_pure(y_true, y_probs, n_bins=15):
    """Pure-Python ECE computation."""
    n = len(y_true)
    if n == 0:
        return 0.0
    # Extract max prob and predicted class per sample
    confidences = []
    predictions = []
    for row in y_probs:
        row = list(row)
        max_p = max(row)
        confidences.append(max_p)
        predictions.append(row.index(max_p))
    correct = [1.0 if p == t else 0.0 for p, t in zip(predictions, y_true)]
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        indices = []
        for j in range(n):
            if i == 0:
                if confidences[j] <= hi:
                    indices.append(j)
            else:
                if lo < confidences[j] <= hi:
                    indices.append(j)
        count = len(indices)
        if count > 0:
            acc = sum(correct[j] for j in indices) / count
            conf = sum(confidences[j] for j in indices) / count
            ece += (count / n) * abs(acc - conf)
    return ece


def _predict_proba_batch(model, batch, device, modality="text_only"):
    """Call model.predict_proba with the right inputs for the modality."""
    if modality == "image_only":
        return model.predict_proba(batch["pixel_values"].to(device))
    elif modality == "text_image":
        return model.predict_proba(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["pixel_values"].to(device),
        )
    else:  # text_only
        return model.predict_proba(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )


def ensemble_predict(model1, model2, loader, device, modality="text_only"):
    """Ensemble prediction: average softmax of two models, then argmax.

    Requires torch. Returns (predictions, true_labels, probabilities) as
    numpy arrays. Probabilities has shape (n_samples, n_classes).
    """
    import numpy as np
    import torch

    model1.eval()
    model2.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]

            probs1 = _predict_proba_batch(model1, batch, device, modality)
            probs2 = _predict_proba_batch(model2, batch, device, modality)

            avg_probs = (probs1 + probs2) / 2.0
            preds = avg_probs.argmax(dim=-1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_probs.append(avg_probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def evaluate_pseudo_labels(class_label, predicted_label) -> float:
    """Compute accuracy of pseudo-labels against ground truth."""
    n = len(class_label)
    correct = sum(1 for a, b in zip(class_label, predicted_label) if a == b)
    return (correct / n) * 100.0 if n > 0 else 0.0
