from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pgm3f.config import DataConfig
from pgm3f.data import PGM3FWindowDataset, build_manifest, preprocess_file


def _write_csv(path: Path, n: int = 1001):
    t = np.arange(n) * 0.01
    df = pd.DataFrame(
        {
            "time": t,
            "CV": np.ones(n) * 0.5,
            "F": np.sin(t) + 1.0,
            "X_prime": np.cos(t),
            "P1_prime": np.ones(n) * 0.8,
            "P2_prime": np.ones(n) * 0.2,
            "T_prime": np.ones(n) * 0.6,
        }
    )
    df.to_csv(path, index=False)


def test_train_sampling_fixed_windows_per_file(tmp_path: Path):
    root = tmp_path / "data"
    (root / "fault0-condition1").mkdir(parents=True)
    (root / "fault1-condition1").mkdir(parents=True)
    _write_csv(root / "fault0-condition1" / "fault0-condition1-1.csv")
    _write_csv(root / "fault1-condition1" / "fault1-condition1-1.csv")

    manifest = build_manifest(str(root))
    cfg = DataConfig(
        data_root=str(root),
        source_fs_hz=100,
        target_fs_hz=100,
        window_sec=2.0,
        stride_sec=1.0,
        train_windows_per_file=3,
    )
    file_ids = manifest["file_id"].tolist()
    ds = PGM3FWindowDataset(
        manifest=manifest,
        file_ids=file_ids,
        cfg=cfg,
        train_mode=True,
        train_windows_per_file=3,
    )
    assert len(ds) == 6
    ds.set_epoch(1)
    assert len(ds) == 6


def test_eval_uses_all_windows(tmp_path: Path):
    root = tmp_path / "data"
    (root / "fault0-condition1").mkdir(parents=True)
    _write_csv(root / "fault0-condition1" / "fault0-condition1-1.csv")

    manifest = build_manifest(str(root))
    cfg = DataConfig(
        data_root=str(root),
        source_fs_hz=100,
        target_fs_hz=100,
        window_sec=2.0,
        stride_sec=1.0,
    )
    file_ids = manifest["file_id"].tolist()
    ds = PGM3FWindowDataset(
        manifest=manifest,
        file_ids=file_ids,
        cfg=cfg,
        train_mode=False,
    )
    # 1001 samples, window=200, stride=100 => 9 windows
    assert len(ds) == 9


def test_preprocess_cache_written(tmp_path: Path):
    root = tmp_path / "data"
    (root / "fault0-condition1").mkdir(parents=True)
    csv_path = root / "fault0-condition1" / "fault0-condition1-1.csv"
    _write_csv(csv_path)

    cache_dir = tmp_path / "pre_cache"
    cfg = DataConfig(
        data_root=str(root),
        source_fs_hz=100,
        target_fs_hz=100,
        preprocessed_cache_enabled=True,
        preprocessed_cache_dir=str(cache_dir),
    )
    _ = preprocess_file(str(csv_path), cfg, source_fs_hz=100, cache_dir=str(cache_dir))
    files = list(cache_dir.glob("*.npy"))
    assert len(files) == 1
