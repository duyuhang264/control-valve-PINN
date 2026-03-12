from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from .config import ModelConfig
from .model import PINNModule, TimeFreqEncoder


class CNNBaseline(nn.Module):
    def __init__(self, cfg: ModelConfig, in_channels: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, cfg.num_fault_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["raw_seq"].transpose(1, 2)
        feat = self.net(x).squeeze(-1)
        return self.classifier(feat)


class TransformerBaseline(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(6, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_fault_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.encoder(self.proj(batch["raw_seq"]))
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


class RawTFBaseline(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.raw = TransformerBaseline(cfg)
        self.tf = TimeFreqEncoder(cfg)
        self.tf_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.head = nn.Linear(cfg.d_model * 2, cfg.num_fault_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raw_tokens = self.raw.encoder(self.raw.proj(batch["raw_seq"]))
        raw_pool = raw_tokens.mean(dim=1)

        tf_tokens = self.tf(batch["raw_seq"])
        tf_pool = self.tf_proj(tf_tokens.mean(dim=1))

        return self.head(torch.cat([raw_pool, tf_pool], dim=-1))


def build_baseline(name: str, cfg: ModelConfig) -> nn.Module:
    name = name.lower().strip()
    if name == "cnn":
        return CNNBaseline(cfg)
    if name == "transformer":
        return TransformerBaseline(cfg)
    if name in {"raw_tf", "raw+tf"}:
        return RawTFBaseline(cfg)
    raise ValueError(f"Unsupported baseline name: {name}")


def _window_stats(arr: np.ndarray) -> np.ndarray:
    feats = []
    for i in range(arr.shape[1]):
        v = arr[:, i]
        feats.extend(
            [
                float(np.mean(v)),
                float(np.std(v)),
                float(np.min(v)),
                float(np.max(v)),
                float(np.median(v)),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def _collect_pinn_features(
    pinn: PINNModule,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    pinn.eval()
    xs: List[np.ndarray] = []
    ys: List[int] = []

    with torch.no_grad():
        for batch in loader:
            pinn_in = batch["pinn_input"].to(device)
            x_hat, f_hat, residuals = pinn(pinn_in)
            e_x = x_hat - batch["target_x"].to(device)
            e_f = f_hat - batch["target_f"].to(device)
            res = torch.cat([residuals["r_flow"], residuals["r_dyn"], residuals["r_pos"], e_x, e_f], dim=-1)

            res_np = res.detach().cpu().numpy()
            y_np = batch["fault_label"].detach().cpu().numpy()
            for i in range(res_np.shape[0]):
                xs.append(_window_stats(res_np[i]))
                ys.append(int(y_np[i]))

    return np.asarray(xs), np.asarray(ys)


def _pretrain_pinn_epoch(
    pinn: PINNModule,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    w_flow: float = 1.0,
    w_dyn: float = 1.0,
    w_pos: float = 1.0,
) -> Dict[str, float]:
    pinn.train()
    total_loss = 0.0
    total_data = 0.0
    total_phys = 0.0
    n_batches = 0

    for batch in train_loader:
        pinn_in = batch["pinn_input"].to(device)
        target_x = batch["target_x"].to(device)
        target_f = batch["target_f"].to(device)

        x_hat, f_hat, residuals = pinn(pinn_in)
        loss_data = F.mse_loss(x_hat, target_x) + F.mse_loss(f_hat, target_f)
        loss_phys = (
            w_flow * torch.mean(residuals["r_flow"] ** 2)
            + w_dyn * torch.mean(residuals["r_dyn"] ** 2)
            + w_pos * torch.mean(residuals["r_pos"] ** 2)
        )
        loss = loss_data + loss_phys

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())
        total_data += float(loss_data.detach().item())
        total_phys += float(loss_phys.detach().item())
        n_batches += 1

    if n_batches == 0:
        return {"loss": float("nan"), "loss_data": float("nan"), "loss_phys": float("nan")}
    return {
        "loss": total_loss / n_batches,
        "loss_data": total_data / n_batches,
        "loss_phys": total_phys / n_batches,
    }


def run_pinn_xgboost_baseline(
    pinn: PINNModule,
    train_loader,
    test_loader,
    device: torch.device,
    pretrain_epochs: int = 1,
    pretrain_lr: float = 1e-3,
    w_flow: float = 1.0,
    w_dyn: float = 1.0,
    w_pos: float = 1.0,
) -> Dict[str, object]:
    pretrain_history: List[Dict[str, float]] = []
    if pretrain_epochs > 0:
        optimizer = torch.optim.AdamW(pinn.parameters(), lr=pretrain_lr, weight_decay=1e-4)
        for _ in range(pretrain_epochs):
            metrics = _pretrain_pinn_epoch(
                pinn=pinn,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                w_flow=w_flow,
                w_dyn=w_dyn,
                w_pos=w_pos,
            )
            pretrain_history.append(metrics)

    x_train, y_train = _collect_pinn_features(pinn, train_loader, device)
    x_test, y_test = _collect_pinn_features(pinn, test_loader, device)

    if x_train.size == 0 or x_test.size == 0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "per_class_recall": [],
            "model": "none",
            "pinn_pretrain": pretrain_history,
        }

    unique_train = sorted(set(y_train.tolist()))
    if len(unique_train) < 2:
        majority = int(unique_train[0]) if unique_train else 0
        pred = np.full(shape=y_test.shape, fill_value=majority, dtype=y_test.dtype)
        classes = sorted(set(y_test.tolist()) | {majority})
        return {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "per_class_recall": recall_score(y_test, pred, labels=classes, average=None, zero_division=0).tolist(),
            "classes": classes,
            "model": "majority_single_class_train",
            "pinn_pretrain": pretrain_history,
        }

    if XGBClassifier is not None:
        num_class = int(len(sorted(set(y_train.tolist()) | set(y_test.tolist()))))
        clf = XGBClassifier(
            objective="multi:softmax",
            num_class=max(2, num_class),
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
        )
        model_name = "xgboost"
    else:
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
        model_name = "random_forest"

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    pred = np.asarray(pred)
    if pred.ndim > 1:
        pred = np.argmax(pred, axis=1)
    pred = pred.astype(np.int64)

    classes = sorted(set(y_test.tolist()) | set(pred.tolist()))
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "per_class_recall": recall_score(y_test, pred, labels=classes, average=None, zero_division=0).tolist(),
        "classes": classes,
        "model": model_name,
        "pinn_pretrain": pretrain_history,
    }
