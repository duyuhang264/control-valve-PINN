from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import TrainConfig
from .model import alignment_loss
from .utils import choose_device


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _compute_pgm3f_losses(
    logits: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    use_physics_loss: bool = True,
    use_domain_adversarial: bool = True,
) -> Dict[str, torch.Tensor]:
    target_fault = batch["fault_label"]
    target_cond = batch["condition_label"]

    smoothing = float(max(0.0, cfg.label_smoothing))
    loss_cls = F.cross_entropy(logits, target_fault, label_smoothing=smoothing)
    loss_data = F.mse_loss(aux["x_hat"], batch["target_x"]) + F.mse_loss(aux["f_hat"], batch["target_f"])

    r = aux["residuals"]
    loss_phys = (
        cfg.w_flow * torch.mean(r["r_flow"] ** 2)
        + cfg.w_dyn * torch.mean(r["r_dyn"] ** 2)
        + cfg.w_pos * torch.mean(r["r_pos"] ** 2)
    )
    if not use_physics_loss:
        loss_phys = torch.zeros_like(loss_phys)

    loss_adv = F.cross_entropy(aux["condition_logits"], target_cond)
    if not use_domain_adversarial:
        loss_adv = torch.zeros_like(loss_adv)

    loss_align = alignment_loss(aux["raw_pool"], aux["res_pool"], aux["tf_pool"])

    total = (
        loss_cls
        + cfg.lambda_data * loss_data
        + cfg.lambda_phys * loss_phys
        + cfg.lambda_adv * loss_adv
        + cfg.lambda_align * loss_align
    )

    return {
        "total": total,
        "loss_cls": loss_cls,
        "loss_data": loss_data,
        "loss_phys": loss_phys,
        "loss_adv": loss_adv,
        "loss_align": loss_align,
    }


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int = 0,
    use_physics_loss: bool = True,
    use_domain_adversarial: bool = True,
) -> Dict[str, float]:
    model.train()
    device = choose_device(cfg.device)
    scaler = GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)

    metrics = defaultdict(float)
    n_batches = 0

    for i, batch in enumerate(loader):
        batch = _to_device(batch, device)
        progress = (epoch + i / max(1, len(loader))) / max(1, cfg.epochs)
        grl_alpha = float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(cfg.amp and device.type == "cuda")):
            if hasattr(model, "forward_with_aux"):
                logits, aux = model.forward_with_aux(batch, grl_alpha=grl_alpha)
                losses = _compute_pgm3f_losses(
                    logits=logits,
                    aux=aux,
                    batch=batch,
                    cfg=cfg,
                    use_physics_loss=use_physics_loss,
                    use_domain_adversarial=use_domain_adversarial,
                )
            else:
                logits = model(batch)
                smoothing = float(max(0.0, cfg.label_smoothing))
                loss_cls = F.cross_entropy(logits, batch["fault_label"], label_smoothing=smoothing)
                z = torch.zeros_like(loss_cls)
                losses = {
                    "total": loss_cls,
                    "loss_cls": loss_cls,
                    "loss_data": z,
                    "loss_phys": z,
                    "loss_adv": z,
                    "loss_align": z,
                }

        scaler.scale(losses["total"]).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for k, v in losses.items():
            metrics[k] += float(v.detach().item())
        n_batches += 1

    if n_batches == 0:
        return {"total": float("nan")}

    return {k: v / n_batches for k, v in metrics.items()}

