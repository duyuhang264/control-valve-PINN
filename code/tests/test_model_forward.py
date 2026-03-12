from __future__ import annotations

import torch

from pgm3f.config import ModelConfig
from pgm3f.model import PINNModule, PGM3F


def test_pinn_forward_shapes():
    cfg = ModelConfig(
        num_fault_classes=20,
        num_condition_classes=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        conv_channels=16,
        pinn_hidden=32,
        pinn_layers=2,
    )
    model = PINNModule(cfg)
    x = torch.randn(2, 64, 5)
    x_hat, f_hat, res = model(x)

    assert x_hat.shape == (2, 64, 1)
    assert f_hat.shape == (2, 64, 1)
    assert res["r_flow"].shape == (2, 64, 1)
    assert res["r_dyn"].shape == (2, 64, 1)
    assert res["r_pos"].shape == (2, 64, 1)


def test_pgm3f_forward_shapes():
    cfg = ModelConfig(
        num_fault_classes=20,
        num_condition_classes=10,
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
    model = PGM3F(cfg)
    batch = {
        "pinn_input": torch.randn(2, 64, 5),
        "raw_seq": torch.randn(2, 64, 6),
        "target_x": torch.randn(2, 64, 1),
        "target_f": torch.randn(2, 64, 1),
        "fault_label": torch.randint(0, 20, (2,)),
        "condition_label": torch.randint(0, 10, (2,)),
    }

    logits = model(batch)
    assert logits.shape == (2, 20)

    logits2, aux = model.forward_with_aux(batch)
    assert logits2.shape == (2, 20)
    assert aux["condition_logits"].shape == (2, 10)
    assert aux["residual_seq"].shape == (2, 64, 5)
