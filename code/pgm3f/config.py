from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    data_root: str
    output_dir: str = "outputs"
    auto_detect_source_fs: bool = True
    expected_columns: List[str] = field(
        default_factory=lambda: [
            "time",
            "CV",
            "F",
            "X_prime",
            "P1_prime",
            "P2_prime",
            "T_prime",
        ]
    )
    expected_faults: int = 20
    expected_conditions: int = 10
    expected_openings: int = 20
    default_duration_s: float = 200.0
    target_fs_hz: int = 500
    source_fs_hz: int = 10000
    lowpass_cutoff_hz: float = 200.0
    window_sec: float = 4.0
    stride_sec: float = 2.0
    max_file_cache: int = 16
    train_windows_per_file: int = 12
    train_window_jitter: int = 1
    shuffle_files_each_epoch: bool = True
    preprocessed_cache_enabled: bool = True
    preprocessed_cache_dir: str = ""
    preprocessed_cache_dtype: str = "float32"


@dataclass
class ModelConfig:
    num_fault_classes: int = 20
    num_condition_classes: int = 10
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 2
    conv_channels: int = 64
    stft_n_fft: int = 64
    stft_hop_length: int = 32
    stft_win_length: int = 64
    dropout: float = 0.1
    pinn_hidden: int = 128
    pinn_layers: int = 4

    @classmethod
    def cpu_small(
        cls,
        num_fault_classes: int = 20,
        num_condition_classes: int = 10,
    ) -> "ModelConfig":
        return cls(
            num_fault_classes=num_fault_classes,
            num_condition_classes=num_condition_classes,
            d_model=64,
            n_heads=4,
            n_layers=1,
            conv_channels=32,
            pinn_hidden=64,
            pinn_layers=3,
        )


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    fallback_batch_size: int = 16
    num_workers: int = 0
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    device: str = "cuda"
    grad_clip: float = 1.0
    log_every: int = 20
    lambda_data: float = 1.0
    lambda_phys: float = 1.0
    lambda_adv: float = 0.1
    lambda_align: float = 0.1
    w_flow: float = 1.0
    w_dyn: float = 1.0
    w_pos: float = 1.0
    label_smoothing: float = 0.05
    select_best_metric: str = "macro_f1"
    pinn_pretrain_epochs: int = 1
    pinn_pretrain_lr: float = 1e-3


@dataclass
class ExperimentConfig:
    protocol: str = "both"
    test_size: float = 0.2
    random_state: int = 42
    output_dir: str = "outputs"
    method: str = "pgm3f"
    model_variant: str = "pgm3f"
    use_cpu_small_preset: bool = True
    run_baselines: bool = True
    run_ablations: bool = True
    export_paper_materials: bool = True
    random_main_epochs: int = 3
    random_baseline_epochs: int = 1
    random_ablation_epochs: int = 1
    loco_main_epochs: int = 1
    num_workers_feat: int = 0
    tau_main: float = 0.55
    hard_classes: List[int] = field(default_factory=lambda: [0, 8, 14, 19])
    disable_experts: bool = False
    feature_cache_enabled: bool = True
    feature_cache_dir: str = ""
    expert_val_size: float = 0.15
    expert_beta: float = 2.0
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_min_child_weight: float = 1.0
    ablations: Dict[str, Dict[str, bool]] = field(
        default_factory=lambda: {
            "no_residual_branch": {"use_residual_branch": False},
            "no_tf_branch": {"use_tf_branch": False},
            "concat_fusion": {"use_cross_attention": False},
            "no_domain_adversarial": {"use_domain_adversarial": False},
            "no_physics_loss": {"use_physics_loss": False},
        }
    )

    def ensure_output_dir(self) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out
