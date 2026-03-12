from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pgm3f.evaluate import evaluate


class ToyEvalDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        file_ids = ["a", "a", "b", "b"]
        labels = [0, 0, 1, 1]
        token = torch.tensor([[float(idx)]], dtype=torch.float32)  # [L=1, C=1]
        return {
            "raw_seq": token,
            "pinn_input": torch.zeros(1, 5, dtype=torch.float32),
            "target_x": torch.zeros(1, 1, dtype=torch.float32),
            "target_f": torch.zeros(1, 1, dtype=torch.float32),
            "fault_label": torch.tensor(labels[idx], dtype=torch.long),
            "condition_label": torch.tensor(0, dtype=torch.long),
            "file_id": file_ids[idx],
        }


class ToyModel(nn.Module):
    def forward(self, batch):
        idx = batch["raw_seq"][:, 0, 0].long()
        logits = torch.zeros(idx.shape[0], 2, device=idx.device)
        logits[idx == 0] = torch.tensor([3.0, 1.0], device=idx.device)
        logits[idx == 1] = torch.tensor([2.0, 1.0], device=idx.device)
        logits[idx == 2] = torch.tensor([1.0, 3.0], device=idx.device)
        logits[idx == 3] = torch.tensor([2.5, 2.6], device=idx.device)
        return logits


def test_evaluate_file_level_aggregation():
    ds = ToyEvalDataset()
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = ToyModel()

    metrics = evaluate(model, loader, device_str="cpu", class_labels=[0, 1], return_window_metrics=True)
    assert metrics["granularity"] == "file"
    assert metrics["num_files"] == 2
    assert metrics["accuracy"] == 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert "window_metrics" in metrics
