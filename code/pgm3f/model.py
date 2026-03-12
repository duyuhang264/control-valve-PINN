from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReverseFunction.apply(x, lambd)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float = 0.0):
        super().__init__()
        seq = []
        d = in_dim
        for _ in range(max(1, layers)):
            seq.append(nn.Linear(d, hidden))
            seq.append(nn.GELU())
            if dropout > 0:
                seq.append(nn.Dropout(dropout))
            d = hidden
        seq.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PINNModule(nn.Module):
    """
    Input: [t, CV, P1_prime, P2_prime, T_prime]
    Output: X_hat, F_hat and physical residuals.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.state_net = MLP(
            in_dim=5,
            out_dim=2,
            hidden=cfg.pinn_hidden,
            layers=cfg.pinn_layers,
            dropout=cfg.dropout,
        )
        self.kv_net = MLP(in_dim=1, out_dim=1, hidden=64, layers=2, dropout=cfg.dropout)

        self.m_raw = nn.Parameter(torch.tensor(1.0))
        self.c_raw = nn.Parameter(torch.tensor(0.1))
        self.k_raw = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def _derivatives(self, y: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # y,t shape [B, L, 1]
        dy = torch.zeros_like(y)
        ddy = torch.zeros_like(y)

        dt = t[:, 1:, :] - t[:, :-1, :]
        dt = torch.clamp(dt, min=1e-6)

        dy[:, 1:, :] = (y[:, 1:, :] - y[:, :-1, :]) / dt
        ddt = dt[:, 1:, :]
        ddy[:, 2:, :] = (dy[:, 2:, :] - dy[:, 1:-1, :]) / torch.clamp(ddt, min=1e-6)
        return dy, ddy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # x: [B, L, 5] -> [t, cv, p1, p2, temp]
        b, l, _ = x.shape
        out = self.state_net(x.reshape(-1, x.shape[-1])).reshape(b, l, 2)

        x_hat = out[..., 0:1]
        f_hat = out[..., 1:2]

        t = x[..., 0:1]
        cv = x[..., 1:2]
        p1 = x[..., 2:3]
        p2 = x[..., 3:4]

        dx, ddx = self._derivatives(x_hat, t)

        m = F.softplus(self.m_raw)
        c = F.softplus(self.c_raw)
        k = F.softplus(self.k_raw)

        kv = F.softplus(self.kv_net(x_hat.reshape(-1, 1))).reshape_as(x_hat)
        dp = torch.relu(p1 - p2) + 1e-6

        flow_theory = kv * torch.sqrt(dp)
        r_flow = f_hat - flow_theory
        r_dyn = m * ddx + c * dx + k * x_hat - self.alpha * cv
        r_pos = x_hat - self.beta * cv

        residuals = {
            "r_flow": r_flow,
            "r_dyn": r_dyn,
            "r_pos": r_pos,
            "flow_theory": flow_theory,
            "dx": dx,
            "ddx": ddx,
        }
        return x_hat, f_hat, residuals


class RawSequenceEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, in_channels: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, cfg.conv_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(cfg.conv_channels, cfg.d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,L,C] -> [B,L2,D]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.encoder(x)


class ResidualEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, in_dim: int = 5):
        super().__init__()
        self.proj = nn.Linear(in_dim, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, cfg.n_layers - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.proj(x))


class TimeFreqEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_fft = cfg.stft_n_fft
        self.hop = cfg.stft_hop_length
        self.win = cfg.stft_win_length
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, cfg.d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def _stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        b, c, l = x.shape
        flat = x.reshape(b * c, l)
        min_len = max(self.n_fft, self.win)
        if l < min_len:
            flat = F.pad(flat, (0, min_len - l))
        window = torch.hann_window(self.win, device=flat.device)
        spec = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=window,
            return_complex=True,
            center=False,
        )
        mag = torch.abs(spec)
        return mag.reshape(b, c, mag.shape[-2], mag.shape[-1])

    def forward(self, raw_seq: torch.Tensor) -> torch.Tensor:
        # raw_seq [B,L,6] = [cv,f,x,p1,p2,t]
        f = raw_seq[..., 1]
        x = raw_seq[..., 2]
        dp = raw_seq[..., 3] - raw_seq[..., 4]
        sig = torch.stack([f, x, dp], dim=1)
        spec = self._stft_mag(sig)
        feat = self.cnn(spec)
        b, d, h, w = feat.shape
        return feat.reshape(b, d, h * w).transpose(1, 2)


class PGM3F(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        use_residual_branch: bool = True,
        use_tf_branch: bool = True,
        use_cross_attention: bool = True,
        use_domain_adversarial: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.use_residual_branch = use_residual_branch
        self.use_tf_branch = use_tf_branch
        self.use_cross_attention = use_cross_attention
        self.use_domain_adversarial = use_domain_adversarial

        self.pinn = PINNModule(cfg)
        self.raw_encoder = RawSequenceEncoder(cfg, in_channels=6)
        self.res_encoder = ResidualEncoder(cfg, in_dim=5)
        self.tf_encoder = TimeFreqEncoder(cfg)

        self.cross_res = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.cross_tf = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.fuse_norm = nn.LayerNorm(cfg.d_model)

        self.gate = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 3),
        )
        self.concat_proj = nn.Linear(cfg.d_model * 3, cfg.d_model)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_fault_classes),
        )

        self.condition_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.num_condition_classes),
        )

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    def forward_with_aux(
        self,
        batch: Dict[str, torch.Tensor],
        grl_alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pinn_in = batch["pinn_input"]
        raw_seq = batch["raw_seq"]
        target_x = batch.get("target_x")
        target_f = batch.get("target_f")

        x_hat, f_hat, residuals = self.pinn(pinn_in)
        e_x = x_hat - target_x if target_x is not None else torch.zeros_like(x_hat)
        e_f = f_hat - target_f if target_f is not None else torch.zeros_like(f_hat)
        residual_seq = torch.cat([residuals["r_flow"], residuals["r_dyn"], residuals["r_pos"], e_x, e_f], dim=-1)

        raw_tokens = self.raw_encoder(raw_seq)
        raw_pool = self._pool(raw_tokens)

        if self.use_residual_branch:
            res_tokens = self.res_encoder(residual_seq)
            res_pool = self._pool(res_tokens)
        else:
            res_tokens = torch.zeros_like(raw_tokens)
            res_pool = torch.zeros_like(raw_pool)

        if self.use_tf_branch:
            tf_tokens = self.tf_encoder(raw_seq)
            tf_pool = self._pool(tf_tokens)
        else:
            tf_tokens = torch.zeros_like(raw_tokens)
            tf_pool = torch.zeros_like(raw_pool)

        if self.use_cross_attention:
            fused = raw_tokens
            if self.use_residual_branch:
                attn_res, _ = self.cross_res(raw_tokens, res_tokens, res_tokens)
                fused = fused + attn_res
            if self.use_tf_branch:
                attn_tf, _ = self.cross_tf(raw_tokens, tf_tokens, tf_tokens)
                fused = fused + attn_tf
            fused = self.fuse_norm(fused)
            fused_pool = self._pool(fused)

            gate_logits = self.gate(torch.cat([raw_pool, res_pool, tf_pool], dim=-1))
            gate = torch.softmax(gate_logits, dim=-1)
            stacked = torch.stack([fused_pool, res_pool, tf_pool], dim=1)
            fused_repr = torch.sum(stacked * gate.unsqueeze(-1), dim=1)
        else:
            fused_repr = self.concat_proj(torch.cat([raw_pool, res_pool, tf_pool], dim=-1))
            gate = torch.zeros(raw_pool.shape[0], 3, device=raw_pool.device)

        logits = self.classifier(fused_repr)

        if self.use_domain_adversarial:
            cond_logits = self.condition_head(grad_reverse(fused_repr, grl_alpha))
        else:
            cond_logits = torch.zeros(
                fused_repr.shape[0],
                self.cfg.num_condition_classes,
                device=fused_repr.device,
            )

        aux = {
            "x_hat": x_hat,
            "f_hat": f_hat,
            "residuals": residuals,
            "residual_seq": residual_seq,
            "raw_pool": raw_pool,
            "res_pool": res_pool,
            "tf_pool": tf_pool,
            "fused_repr": fused_repr,
            "condition_logits": cond_logits,
            "gate": gate,
        }
        return logits, aux

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits, _ = self.forward_with_aux(batch, grl_alpha=1.0)
        return logits


def alignment_loss(raw_pool: torch.Tensor, res_pool: torch.Tensor, tf_pool: torch.Tensor) -> torch.Tensor:
    raw_norm = F.normalize(raw_pool, dim=-1)
    res_norm = F.normalize(res_pool, dim=-1)
    tf_norm = F.normalize(tf_pool, dim=-1)

    l1 = F.mse_loss(raw_norm, res_norm)
    l2 = F.mse_loss(raw_norm, tf_norm)
    l3 = F.mse_loss(res_norm, tf_norm)
    return (l1 + l2 + l3) / 3.0

