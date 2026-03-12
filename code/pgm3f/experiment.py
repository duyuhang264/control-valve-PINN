from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .baselines import build_baseline, run_pinn_xgboost_baseline
from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from .data import (
    PGM3FWindowDataset,
    build_manifest,
    fit_scaler,
    resolve_effective_source_fs,
    summarize_manifest,
)
from .evaluate import evaluate
from .model import PGM3F
from .residual_tree import (
    build_file_features,
    evaluate_residual_tree,
    fit_physical_proxy,
    train_residual_tree,
)
from .reporting import export_paper_materials
from .splits import iter_folds, make_splits, validate_split_leakage
from .train import train_one_epoch
from .utils import choose_device, ensure_dir, save_json, set_seed


def _resolve_protocols(protocol: str) -> List[str]:
    p = protocol.lower().strip()
    if p in {"both", "all", "random+loco"}:
        return ["random_grouped", "loco"]
    if p in {"random", "random_grouped"}:
        return ["random_grouped"]
    if p in {"loco", "leave_one_condition_out"}:
        return ["loco"]
    raise ValueError(f"Unsupported protocol: {protocol}")


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def _select_best_eval(history: List[Dict[str, object]], metric: str = "macro_f1") -> Tuple[int, Dict[str, object]]:
    if not history:
        return 0, {}
    best_idx = 0
    best_val = float("-inf")
    for i, rec in enumerate(history):
        eval_metrics = rec.get("eval", {})
        try:
            val = float(eval_metrics.get(metric, float("-inf")))
        except Exception:
            val = float("-inf")
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx + 1, history[best_idx].get("eval", {})


def _build_loaders(
    manifest,
    train_ids: List[str],
    test_ids: List[str],
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    scaler_path: Path,
    effective_source_fs_hz: int,
    preprocess_cache_dir: str,
) -> Tuple[DataLoader, DataLoader, Dict[str, object], Dict[int, int], Dict[int, int]]:
    train_manifest = manifest[manifest["file_id"].isin(train_ids)].reset_index(drop=True)
    scaler = fit_scaler(
        train_manifest,
        cfg=data_cfg,
        out_json_path=str(scaler_path),
        effective_source_fs_hz=effective_source_fs_hz,
        preprocess_cache_dir=preprocess_cache_dir,
    )

    fault_values = sorted(manifest["fault_id"].unique().tolist())
    cond_values = sorted(manifest["condition_id"].unique().tolist())
    fault_to_idx = {v: i for i, v in enumerate(fault_values)}
    cond_to_idx = {v: i for i, v in enumerate(cond_values)}

    train_ds = PGM3FWindowDataset(
        manifest=manifest,
        file_ids=train_ids,
        cfg=data_cfg,
        scaler=scaler,
        fault_to_idx=fault_to_idx,
        condition_to_idx=cond_to_idx,
        train_mode=True,
        train_windows_per_file=data_cfg.train_windows_per_file,
        seed=train_cfg.seed,
        effective_source_fs_hz=effective_source_fs_hz,
        preprocess_cache_dir=preprocess_cache_dir,
    )
    test_ds = PGM3FWindowDataset(
        manifest=manifest,
        file_ids=test_ids,
        cfg=data_cfg,
        scaler=scaler,
        fault_to_idx=fault_to_idx,
        condition_to_idx=cond_to_idx,
        train_mode=False,
        effective_source_fs_hz=effective_source_fs_hz,
        preprocess_cache_dir=preprocess_cache_dir,
    )

    pin_memory = choose_device(train_cfg.device).type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, scaler, fault_to_idx, cond_to_idx


def _train_pgm3f(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    epochs: int,
    class_labels: List[int],
    use_residual_branch: bool = True,
    use_tf_branch: bool = True,
    use_cross_attention: bool = True,
    use_domain_adversarial: bool = True,
    use_physics_loss: bool = True,
) -> Dict[str, object]:
    device = choose_device(train_cfg.device)
    model = PGM3F(
        cfg=model_cfg,
        use_residual_branch=use_residual_branch,
        use_tf_branch=use_tf_branch,
        use_cross_attention=use_cross_attention,
        use_domain_adversarial=use_domain_adversarial,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    epoch_cfg = deepcopy(train_cfg)
    epoch_cfg.epochs = max(1, int(epochs))

    history = []
    for epoch in range(epoch_cfg.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            cfg=epoch_cfg,
            epoch=epoch,
            use_physics_loss=use_physics_loss,
            use_domain_adversarial=use_domain_adversarial,
        )
        eval_metrics = evaluate(
            model,
            test_loader,
            device_str=str(device),
            class_labels=class_labels,
            return_window_metrics=True,
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "eval": eval_metrics})

    best_epoch, best_eval = _select_best_eval(history, metric=train_cfg.select_best_metric)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_eval": best_eval,
        "last_eval": history[-1]["eval"] if history else {},
        "final_eval": best_eval if history else {},
    }


def _train_baseline(
    baseline_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    epochs: int,
    class_labels: List[int],
) -> Dict[str, object]:
    device = choose_device(train_cfg.device)
    model = build_baseline(baseline_name, model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    epoch_cfg = deepcopy(train_cfg)
    epoch_cfg.epochs = max(1, int(epochs))

    history = []
    for epoch in range(epoch_cfg.epochs):
        train_metrics = train_one_epoch(
            model=model,  # type: ignore[arg-type]
            loader=train_loader,
            optimizer=optimizer,
            cfg=epoch_cfg,
            epoch=epoch,
            use_physics_loss=False,
            use_domain_adversarial=False,
        )
        eval_metrics = evaluate(
            model,  # type: ignore[arg-type]
            test_loader,
            device_str=str(device),
            class_labels=class_labels,
            return_window_metrics=True,
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "eval": eval_metrics})

    best_epoch, best_eval = _select_best_eval(history, metric=train_cfg.select_best_metric)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_eval": best_eval,
        "last_eval": history[-1]["eval"] if history else {},
        "final_eval": best_eval if history else {},
    }


def _run_fold(
    manifest,
    train_ids: List[str],
    test_ids: List[str],
    protocol_name: str,
    fold_dir: Path,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    exp_cfg: ExperimentConfig,
    effective_source_fs_hz: int,
    preprocess_cache_dir: str,
) -> Dict[str, object]:
    scaler_path = fold_dir / "scaler.json"
    class_labels = list(range(model_cfg.num_fault_classes))

    candidate_batch_sizes = [int(train_cfg.batch_size)]
    if int(train_cfg.fallback_batch_size) != int(train_cfg.batch_size):
        candidate_batch_sizes.append(int(train_cfg.fallback_batch_size))

    last_error: Exception | None = None
    for batch_size in candidate_batch_sizes:
        local_train_cfg = deepcopy(train_cfg)
        local_train_cfg.batch_size = int(batch_size)
        try:
            train_loader, test_loader, _, _, _ = _build_loaders(
                manifest=manifest,
                train_ids=train_ids,
                test_ids=test_ids,
                data_cfg=data_cfg,
                train_cfg=local_train_cfg,
                scaler_path=scaler_path,
                effective_source_fs_hz=effective_source_fs_hz,
                preprocess_cache_dir=preprocess_cache_dir,
            )

            fold_result: Dict[str, object] = {"used_batch_size": int(batch_size)}
            random_protocol = protocol_name == "random_grouped"

            pgm3f_epochs = exp_cfg.random_main_epochs if random_protocol else exp_cfg.loco_main_epochs
            fold_result["pgm3f"] = _train_pgm3f(
                train_loader=train_loader,
                test_loader=test_loader,
                model_cfg=model_cfg,
                train_cfg=local_train_cfg,
                epochs=pgm3f_epochs,
                class_labels=class_labels,
            )

            if random_protocol and exp_cfg.run_ablations:
                ablations: Dict[str, object] = {}
                for name, flags in exp_cfg.ablations.items():
                    kwargs = {
                        "use_residual_branch": flags.get("use_residual_branch", True),
                        "use_tf_branch": flags.get("use_tf_branch", True),
                        "use_cross_attention": flags.get("use_cross_attention", True),
                        "use_domain_adversarial": flags.get("use_domain_adversarial", True),
                        "use_physics_loss": flags.get("use_physics_loss", True),
                    }
                    ablations[name] = _train_pgm3f(
                        train_loader=train_loader,
                        test_loader=test_loader,
                        model_cfg=model_cfg,
                        train_cfg=local_train_cfg,
                        epochs=exp_cfg.random_ablation_epochs,
                        class_labels=class_labels,
                        **kwargs,
                    )
                fold_result["ablations"] = ablations

            if random_protocol and exp_cfg.run_baselines:
                baselines: Dict[str, object] = {}
                for name in ["cnn", "transformer", "raw_tf"]:
                    baselines[name] = _train_baseline(
                        baseline_name=name,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        model_cfg=model_cfg,
                        train_cfg=local_train_cfg,
                        epochs=exp_cfg.random_baseline_epochs,
                        class_labels=class_labels,
                    )

                device = choose_device(local_train_cfg.device)
                pinn_for_baseline = PGM3F(model_cfg).to(device).pinn
                baselines["pinn_xgboost"] = run_pinn_xgboost_baseline(
                    pinn=pinn_for_baseline,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    pretrain_epochs=local_train_cfg.pinn_pretrain_epochs,
                    pretrain_lr=local_train_cfg.pinn_pretrain_lr,
                    w_flow=local_train_cfg.w_flow,
                    w_dyn=local_train_cfg.w_dyn,
                    w_pos=local_train_cfg.w_pos,
                )
                fold_result["baselines"] = baselines

            return fold_result
        except MemoryError as exc:
            last_error = exc
            if batch_size == candidate_batch_sizes[-1]:
                raise
        except RuntimeError as exc:
            last_error = exc
            if not _is_oom_error(exc) or batch_size == candidate_batch_sizes[-1]:
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to run fold with all candidate batch sizes")


def _run_experiment_pgm3f(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    exp_cfg: ExperimentConfig,
) -> Dict[str, object]:
    set_seed(train_cfg.seed)
    out_dir = ensure_dir(exp_cfg.output_dir)

    if exp_cfg.use_cpu_small_preset and choose_device(train_cfg.device).type == "cpu":
        model_cfg = ModelConfig.cpu_small(
            num_fault_classes=model_cfg.num_fault_classes,
            num_condition_classes=model_cfg.num_condition_classes,
        )

    quality_path = out_dir / "data_quality_report.csv"
    manifest = build_manifest(
        data_root=data_cfg.data_root,
        report_path=str(quality_path),
        expected_columns=data_cfg.expected_columns,
        default_duration_s=data_cfg.default_duration_s,
        expected_faults=data_cfg.expected_faults,
        expected_conditions=data_cfg.expected_conditions,
        expected_openings=data_cfg.expected_openings,
    )
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")

    effective_source_fs_hz = resolve_effective_source_fs(manifest, data_cfg)
    manifest_summary = summarize_manifest(
        manifest,
        expected_faults=data_cfg.expected_faults,
        expected_conditions=data_cfg.expected_conditions,
        expected_openings=data_cfg.expected_openings,
    )
    manifest_summary["effective_source_fs_hz"] = int(effective_source_fs_hz)

    preprocess_cache_dir = ""
    if data_cfg.preprocessed_cache_enabled:
        preprocess_cache_dir = (
            data_cfg.preprocessed_cache_dir.strip()
            if str(data_cfg.preprocessed_cache_dir).strip()
            else str(out_dir / "preprocessed_cache")
        )
        ensure_dir(preprocess_cache_dir)

    results: Dict[str, object] = {
        "method": "pgm3f",
        "manifest_summary": manifest_summary,
        "protocols": {},
        "runtime_seconds": 0.0,
        "preprocess_cache_dir": preprocess_cache_dir,
        "config_snapshot": {
            "data_cfg": data_cfg.__dict__,
            "model_cfg": model_cfg.__dict__,
            "train_cfg": train_cfg.__dict__,
            "exp_cfg": exp_cfg.__dict__,
        },
    }

    total_start = time.time()
    protocols = _resolve_protocols(exp_cfg.protocol)
    for protocol in protocols:
        protocol_dir = ensure_dir(out_dir / protocol)
        split_map = make_splits(
            manifest=manifest,
            protocol=protocol,
            test_size=exp_cfg.test_size,
            random_state=exp_cfg.random_state,
        )
        leakage_report = validate_split_leakage(split_map)

        protocol_result: Dict[str, object] = {
            "leakage_report": leakage_report,
            "folds": {},
        }

        for fold_name, train_ids, test_ids in iter_folds(split_map):
            fold_start = time.time()
            fold_dir = ensure_dir(protocol_dir / fold_name)
            fold_result = _run_fold(
                manifest=manifest,
                train_ids=train_ids,
                test_ids=test_ids,
                protocol_name=protocol,
                fold_dir=fold_dir,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                exp_cfg=exp_cfg,
                effective_source_fs_hz=effective_source_fs_hz,
                preprocess_cache_dir=preprocess_cache_dir,
            )
            fold_result["runtime_seconds"] = time.time() - fold_start
            protocol_result["folds"][fold_name] = fold_result
            save_json(fold_result, fold_dir / "results.json")

        results["protocols"][protocol] = protocol_result
        save_json(protocol_result, protocol_dir / "summary.json")

    results["runtime_seconds"] = time.time() - total_start

    if exp_cfg.export_paper_materials:
        results["paper_materials"] = export_paper_materials(results, out_dir)

    save_json(results, out_dir / "summary.json")
    return results


def _prepare_common(
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    exp_cfg: ExperimentConfig,
) -> Tuple[Path, object, int, Dict[str, object], str]:
    set_seed(train_cfg.seed)
    out_dir = ensure_dir(exp_cfg.output_dir)

    quality_path = out_dir / "data_quality_report.csv"
    manifest = build_manifest(
        data_root=data_cfg.data_root,
        report_path=str(quality_path),
        expected_columns=data_cfg.expected_columns,
        default_duration_s=data_cfg.default_duration_s,
        expected_faults=data_cfg.expected_faults,
        expected_conditions=data_cfg.expected_conditions,
        expected_openings=data_cfg.expected_openings,
    )
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")

    effective_source_fs_hz = resolve_effective_source_fs(manifest, data_cfg)
    manifest_summary = summarize_manifest(
        manifest,
        expected_faults=data_cfg.expected_faults,
        expected_conditions=data_cfg.expected_conditions,
        expected_openings=data_cfg.expected_openings,
    )
    manifest_summary["effective_source_fs_hz"] = int(effective_source_fs_hz)

    preprocess_cache_dir = ""
    if data_cfg.preprocessed_cache_enabled:
        preprocess_cache_dir = (
            data_cfg.preprocessed_cache_dir.strip()
            if str(data_cfg.preprocessed_cache_dir).strip()
            else str(out_dir / "preprocessed_cache")
        )
        ensure_dir(preprocess_cache_dir)

    return out_dir, manifest, effective_source_fs_hz, manifest_summary, preprocess_cache_dir


def _run_fold_residual_tree(
    manifest,
    base_feature_df,
    train_ids: List[str],
    test_ids: List[str],
    fold_name: str,
    fold_dir: Path,
    data_cfg: DataConfig,
    exp_cfg: ExperimentConfig,
    effective_source_fs_hz: int,
    preprocess_cache_dir: str,
    feature_cache_dir: str,
) -> Dict[str, object]:
    feature_t0 = time.time()
    train_manifest = manifest[manifest["file_id"].isin(train_ids)].reset_index(drop=True)
    proxy_params = fit_physical_proxy(
        train_manifest=train_manifest,
        data_cfg=data_cfg,
        effective_source_fs_hz=effective_source_fs_hz,
        preprocess_cache_dir=preprocess_cache_dir,
    )
    residual_df = build_file_features(
        manifest=manifest,
        cfg={
            "data_cfg": data_cfg,
            "effective_source_fs_hz": effective_source_fs_hz,
            "preprocess_cache_dir": preprocess_cache_dir,
            "feature_cache_dir": feature_cache_dir,
            "num_workers": exp_cfg.num_workers_feat,
            "segment_count": 5,
            "feature_set": "residual",
            "proxy_params": proxy_params,
        },
    )
    feat_df = base_feature_df.merge(
        residual_df.drop(columns=["path", "fault_id", "condition_id", "opening_id"], errors="ignore"),
        on="file_id",
        how="inner",
    )
    feature_seconds = time.time() - feature_t0

    train_df = feat_df[feat_df["file_id"].isin(train_ids)].reset_index(drop=True)
    test_df = feat_df[feat_df["file_id"].isin(test_ids)].reset_index(drop=True)

    train_t0 = time.time()
    model_bundle = train_residual_tree(
        train_df=train_df,
        cfg={
            "exp_cfg": exp_cfg,
            "hard_classes": exp_cfg.hard_classes,
            "tau_main": exp_cfg.tau_main,
            "disable_experts": exp_cfg.disable_experts,
            "expert_val_size": exp_cfg.expert_val_size,
            "expert_beta": exp_cfg.expert_beta,
        },
    )
    train_seconds = time.time() - train_t0

    eval_t0 = time.time()
    eval_out = evaluate_residual_tree(
        model_bundle=model_bundle,
        test_df=test_df,
        cfg={
            "tau_main": exp_cfg.tau_main,
            "hard_classes": exp_cfg.hard_classes,
            "disable_experts": exp_cfg.disable_experts,
        },
    )
    eval_seconds = time.time() - eval_t0

    fold_result: Dict[str, object] = {
        "method": "residual_tree",
        "fold_name": fold_name,
        "proxy_params": proxy_params,
        "residual_tree": eval_out,
        "runtime_breakdown": {
            "feature_seconds": feature_seconds,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
        },
    }

    # Random protocol ablation: disable experts.
    if fold_name == "random" and not exp_cfg.disable_experts:
        fold_result["ablations"] = {
            "disable_experts": {
                "final_eval": eval_out.get("main_eval", {}),
            }
        }

    return fold_result


def _run_experiment_residual_tree(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    exp_cfg: ExperimentConfig,
) -> Dict[str, object]:
    del model_cfg
    out_dir, manifest, effective_source_fs_hz, manifest_summary, preprocess_cache_dir = _prepare_common(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        exp_cfg=exp_cfg,
    )

    feature_cache_dir = ""
    if exp_cfg.feature_cache_enabled:
        feature_cache_dir = (
            exp_cfg.feature_cache_dir.strip()
            if str(exp_cfg.feature_cache_dir).strip()
            else str(out_dir / "feature_cache")
        )
        ensure_dir(feature_cache_dir)

    base_feature_t0 = time.time()
    base_feature_df = build_file_features(
        manifest=manifest,
        cfg={
            "data_cfg": data_cfg,
            "effective_source_fs_hz": effective_source_fs_hz,
            "preprocess_cache_dir": preprocess_cache_dir,
            "feature_cache_dir": feature_cache_dir,
            "num_workers": exp_cfg.num_workers_feat,
            "segment_count": 5,
            "feature_set": "static",
        },
    )
    base_feature_seconds = time.time() - base_feature_t0

    results: Dict[str, object] = {
        "method": "residual_tree",
        "manifest_summary": manifest_summary,
        "protocols": {},
        "runtime_seconds": 0.0,
        "preprocess_cache_dir": preprocess_cache_dir,
        "feature_cache_dir": feature_cache_dir,
        "base_feature_seconds": base_feature_seconds,
        "config_snapshot": {
            "data_cfg": data_cfg.__dict__,
            "exp_cfg": exp_cfg.__dict__,
        },
    }

    total_start = time.time()
    protocols = _resolve_protocols(exp_cfg.protocol)
    for protocol in protocols:
        protocol_dir = ensure_dir(out_dir / protocol)
        split_map = make_splits(
            manifest=manifest,
            protocol=protocol,
            test_size=exp_cfg.test_size,
            random_state=exp_cfg.random_state,
        )
        leakage_report = validate_split_leakage(split_map)
        protocol_result: Dict[str, object] = {
            "leakage_report": leakage_report,
            "folds": {},
        }

        for fold_name, train_ids, test_ids in iter_folds(split_map):
            fold_start = time.time()
            fold_dir = ensure_dir(protocol_dir / fold_name)
            fold_result = _run_fold_residual_tree(
                manifest=manifest,
                base_feature_df=base_feature_df,
                train_ids=train_ids,
                test_ids=test_ids,
                fold_name=fold_name,
                fold_dir=fold_dir,
                data_cfg=data_cfg,
                exp_cfg=exp_cfg,
                effective_source_fs_hz=effective_source_fs_hz,
                preprocess_cache_dir=preprocess_cache_dir,
                feature_cache_dir=feature_cache_dir,
            )
            fold_result["runtime_seconds"] = time.time() - fold_start
            protocol_result["folds"][fold_name] = fold_result
            save_json(fold_result, fold_dir / "results.json")

        results["protocols"][protocol] = protocol_result
        save_json(protocol_result, protocol_dir / "summary.json")

    results["runtime_seconds"] = time.time() - total_start
    if exp_cfg.export_paper_materials:
        results["paper_materials"] = export_paper_materials(results, out_dir)
    save_json(results, out_dir / "summary.json")
    return results


def run_experiment(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    exp_cfg: ExperimentConfig,
) -> Dict[str, object]:
    method = str(getattr(exp_cfg, "method", "pgm3f")).strip().lower()
    if method == "residual_tree":
        return _run_experiment_residual_tree(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            exp_cfg=exp_cfg,
        )
    return _run_experiment_pgm3f(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        exp_cfg=exp_cfg,
    )
