from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader

from .model import PGM3F
from .utils import choose_device


def _compute_metrics(y_true: List[int], y_pred: List[int], classes: List[int]) -> Dict[str, object]:
    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "per_class_recall": [0.0 for _ in classes],
            "classes": classes,
            "confusion_matrix": [[0 for _ in classes] for _ in classes],
            "num_samples": 0,
        }

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_recall": recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0).tolist(),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(y_true),
    }


def evaluate(
    model: PGM3F,
    loader: DataLoader,
    device_str: str = "cuda",
    class_labels: Optional[List[int]] = None,
    return_window_metrics: bool = True,
) -> Dict[str, object]:
    model.eval()
    device = choose_device(device_str)

    y_true_window: List[int] = []
    y_pred_window: List[int] = []

    file_prob_sum: Dict[str, np.ndarray] = {}
    file_counts: Dict[str, int] = {}
    file_true: Dict[str, int] = {}

    inferred_num_classes: Optional[int] = None

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            logits = model(batch_dev)
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)

            probs_np = probs.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy().tolist()
            true_np = batch["fault_label"].detach().cpu().numpy().tolist()
            file_ids = [str(v) for v in batch["file_id"]]

            if inferred_num_classes is None:
                inferred_num_classes = int(logits.shape[-1])

            y_pred_window.extend(int(v) for v in pred_np)
            y_true_window.extend(int(v) for v in true_np)

            for i, fid in enumerate(file_ids):
                if fid not in file_prob_sum:
                    file_prob_sum[fid] = np.zeros(probs_np.shape[1], dtype=np.float64)
                    file_counts[fid] = 0
                file_prob_sum[fid] += probs_np[i]
                file_counts[fid] += 1
                if fid not in file_true:
                    file_true[fid] = int(true_np[i])

    num_classes = inferred_num_classes or (max(class_labels) + 1 if class_labels else 0)
    classes = class_labels if class_labels is not None else list(range(num_classes))

    file_ids_sorted = sorted(file_prob_sum.keys())
    y_true_file: List[int] = []
    y_pred_file: List[int] = []
    for fid in file_ids_sorted:
        avg_prob = file_prob_sum[fid] / max(1, file_counts[fid])
        y_pred_file.append(int(np.argmax(avg_prob)))
        y_true_file.append(int(file_true[fid]))

    file_metrics = _compute_metrics(y_true_file, y_pred_file, classes=classes)
    out: Dict[str, object] = {
        "granularity": "file",
        "aggregation": "mean_softmax_by_file",
        "num_files": len(file_ids_sorted),
        "num_windows": len(y_true_window),
        "accuracy": file_metrics["accuracy"],
        "macro_f1": file_metrics["macro_f1"],
        "balanced_accuracy": file_metrics["balanced_accuracy"],
        "per_class_recall": file_metrics["per_class_recall"],
        "classes": file_metrics["classes"],
        "confusion_matrix": file_metrics["confusion_matrix"],
        "num_samples": file_metrics["num_samples"],
    }

    if return_window_metrics:
        out["window_metrics"] = _compute_metrics(y_true_window, y_pred_window, classes=classes)

    return out
