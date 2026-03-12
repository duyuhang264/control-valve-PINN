from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pgm3f.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from pgm3f.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PG-M3F experiments")
    p.add_argument("--data-root", required=True, help="Path to DAMADICS CSV root")
    p.add_argument("--output-dir", default="outputs", help="Output directory")
    p.add_argument("--protocol", default="both", choices=["both", "random_grouped", "loco"])
    p.add_argument("--method", default="residual_tree", choices=["residual_tree", "pgm3f"])
    p.add_argument("--epochs", type=int, default=3, help="Main model epochs for random protocol")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--fallback-batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-fs", type=int, default=10000)
    p.add_argument("--target-fs", type=int, default=500)
    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--stride-sec", type=float, default=2.0)
    p.add_argument("--lowpass-cutoff", type=float, default=200.0)
    p.add_argument("--default-duration-s", type=float, default=200.0)
    p.add_argument("--num-fault-classes", type=int, default=20)
    p.add_argument("--num-condition-classes", type=int, default=10)
    p.add_argument("--train-windows-per-file", type=int, default=12)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--best-metric", default="macro_f1", choices=["macro_f1", "accuracy", "balanced_accuracy"])
    p.add_argument("--disable-preprocess-cache", action="store_true")
    p.add_argument("--preprocess-cache-dir", default="")
    p.add_argument("--disable-auto-fs-detect", action="store_true")
    p.add_argument("--disable-cpu-small-preset", action="store_true")
    p.add_argument("--num-workers-feat", type=int, default=0)
    p.add_argument("--tau-main", type=float, default=0.55)
    p.add_argument("--hard-classes", default="0,8,14,19")
    p.add_argument("--disable-experts", action="store_true")
    p.add_argument("--disable-feature-cache", action="store_true")
    p.add_argument("--feature-cache-dir", default="")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--no-ablations", action="store_true")
    p.add_argument("--no-paper-materials", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hard_classes = [int(v.strip()) for v in str(args.hard_classes).split(",") if str(v).strip()]

    data_cfg = DataConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        auto_detect_source_fs=not args.disable_auto_fs_detect,
        source_fs_hz=args.source_fs,
        target_fs_hz=args.target_fs,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        lowpass_cutoff_hz=args.lowpass_cutoff,
        default_duration_s=args.default_duration_s,
        train_windows_per_file=args.train_windows_per_file,
        preprocessed_cache_enabled=not args.disable_preprocess_cache,
        preprocessed_cache_dir=args.preprocess_cache_dir,
    )
    model_cfg = ModelConfig(
        num_fault_classes=args.num_fault_classes,
        num_condition_classes=args.num_condition_classes,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        fallback_batch_size=args.fallback_batch_size,
        device=args.device,
        label_smoothing=args.label_smoothing,
        select_best_metric=args.best_metric,
    )
    exp_cfg = ExperimentConfig(
        protocol=args.protocol,
        output_dir=args.output_dir,
        method=args.method,
        use_cpu_small_preset=not args.disable_cpu_small_preset,
        run_baselines=not args.no_baselines,
        run_ablations=not args.no_ablations,
        export_paper_materials=not args.no_paper_materials,
        random_main_epochs=args.epochs,
        num_workers_feat=args.num_workers_feat,
        tau_main=args.tau_main,
        hard_classes=hard_classes,
        disable_experts=args.disable_experts,
        feature_cache_enabled=not args.disable_feature_cache,
        feature_cache_dir=args.feature_cache_dir,
    )

    results = run_experiment(data_cfg=data_cfg, model_cfg=model_cfg, train_cfg=train_cfg, exp_cfg=exp_cfg)
    print("Finished experiment")
    print(f"Output: {Path(args.output_dir).resolve()}")
    print("Summary keys:", list(results.keys()))


if __name__ == "__main__":
    main()
