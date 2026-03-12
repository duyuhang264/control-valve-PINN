from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from pgm3f.config import ModelConfig, TrainConfig
from pgm3f.model import PGM3F
from pgm3f.train import train_one_epoch


class DummyDataset(Dataset):
    def __init__(self, n: int = 8, seq_len: int = 64):
        self.n = n
        self.seq_len = seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "pinn_input": torch.randn(self.seq_len, 5),
            "raw_seq": torch.randn(self.seq_len, 6),
            "target_x": torch.randn(self.seq_len, 1),
            "target_f": torch.randn(self.seq_len, 1),
            "fault_label": torch.tensor(idx % 4, dtype=torch.long),
            "condition_label": torch.tensor(idx % 2, dtype=torch.long),
            "file_id": f"f_{idx}",
        }


def test_train_one_epoch_loss_finite():
    model_cfg = ModelConfig(
        num_fault_classes=4,
        num_condition_classes=2,
        d_model=32,
        n_heads=4,
        n_layers=1,
        conv_channels=16,
        stft_n_fft=16,
        stft_hop_length=8,
        stft_win_length=16,
        pinn_hidden=32,
        pinn_layers=2,
    )
    train_cfg = TrainConfig(
        batch_size=2,
        epochs=1,
        device="cpu",
        amp=False,
        num_workers=0,
    )

    model = PGM3F(model_cfg)
    loader = DataLoader(DummyDataset(), batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = train_one_epoch(model, loader, optimizer, train_cfg, epoch=0)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
