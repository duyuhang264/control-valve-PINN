from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pgm3f.config import DataConfig
from pgm3f.data import build_manifest
from pgm3f.residual_tree import build_file_features, predict_with_experts


def _write_csv(path: Path, n: int = 401):
    t = np.arange(n) * 0.01
    df = pd.DataFrame(
        {
            "time": t,
            "CV": 0.5 + 0.1 * np.sin(2 * np.pi * 0.2 * t),
            "F": 1.0 + 0.2 * np.sin(2 * np.pi * 0.3 * t),
            "X_prime": 0.3 + 0.15 * np.cos(2 * np.pi * 0.1 * t),
            "P1_prime": 0.8 + 0.05 * np.sin(2 * np.pi * 0.05 * t),
            "P2_prime": 0.2 + 0.03 * np.cos(2 * np.pi * 0.05 * t),
            "T_prime": 0.6 + 0.02 * np.sin(2 * np.pi * 0.07 * t),
        }
    )
    df.to_csv(path, index=False)


def test_feature_cache_consistency(tmp_path: Path):
    root = tmp_path / "data"
    (root / "fault0-condition1").mkdir(parents=True)
    (root / "fault1-condition1").mkdir(parents=True)
    _write_csv(root / "fault0-condition1" / "fault0-condition1-1.csv")
    _write_csv(root / "fault1-condition1" / "fault1-condition1-1.csv")

    manifest = build_manifest(str(root))
    data_cfg = DataConfig(
        data_root=str(root),
        source_fs_hz=100,
        target_fs_hz=100,
    )
    cache_dir = tmp_path / "feature_cache"
    f1 = build_file_features(
        manifest=manifest,
        cfg={
            "data_cfg": data_cfg,
            "effective_source_fs_hz": 100,
            "preprocess_cache_dir": str(tmp_path / "pre_cache"),
            "feature_cache_dir": str(cache_dir),
            "feature_set": "static",
            "num_workers": 0,
        },
    )
    f2 = build_file_features(
        manifest=manifest,
        cfg={
            "data_cfg": data_cfg,
            "effective_source_fs_hz": 100,
            "preprocess_cache_dir": str(tmp_path / "pre_cache"),
            "feature_cache_dir": str(cache_dir),
            "feature_set": "static",
            "num_workers": 0,
        },
    )
    pd.testing.assert_frame_equal(
        f1.sort_values("file_id").reset_index(drop=True),
        f2.sort_values("file_id").reset_index(drop=True),
        check_exact=False,
        atol=1e-6,
        rtol=1e-6,
    )


class _DummyMain:
    classes_ = np.array([0, 1], dtype=np.int64)

    def predict_proba(self, x):
        out = np.zeros((x.shape[0], 2), dtype=np.float64)
        # sample 0 high confidence class 1, sample 1 low confidence class 1.
        out[0] = [0.1, 0.9]
        out[1] = [0.45, 0.55]
        return out


class _DummyExpert:
    classes_ = np.array([0, 1], dtype=np.int64)

    def predict_proba(self, x):
        out = np.zeros((x.shape[0], 2), dtype=np.float64)
        # Expert strongly supports class 0 for both samples.
        out[:, 0] = 0.15
        out[:, 1] = 0.85
        return out


def test_expert_gate_applies_only_to_low_confidence_samples():
    test_df = pd.DataFrame(
        {
            "file_id": ["a", "b"],
            "fault_id": [1, 1],
            "condition_id": [1, 1],
            "opening_id": [1, 1],
            "f1": [0.0, 1.0],
        }
    )
    model_bundle = {
        "main_model": _DummyMain(),
        "experts": {0: {"model": _DummyExpert(), "threshold": 0.8}},
        "classes": [0, 1],
        "tau_main": 0.8,
        "disable_experts": False,
        "base_feature_columns": ["f1"],
        "feature_columns": ["f1", "dlt__f1"],
        "condition_template": pd.DataFrame({"condition_id": [1], "f1": [0.0]}),
        "global_template": pd.Series({"f1": 0.0}),
    }
    pred = predict_with_experts(model_bundle, test_df, cfg={"tau_main": 0.8, "disable_experts": False})
    assert pred.tolist() == [1, 0]
