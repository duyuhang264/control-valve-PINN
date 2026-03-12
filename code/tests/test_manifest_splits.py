from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pgm3f.config import DataConfig
from pgm3f.data import build_manifest, resolve_effective_source_fs
from pgm3f.splits import make_splits, validate_split_leakage


def _write_csv(path: Path, n: int = 200):
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


def test_manifest_and_splits(tmp_path: Path):
    root = tmp_path / "data"
    (root / "fault0-condition1").mkdir(parents=True)
    (root / "fault1-condition1").mkdir(parents=True)
    (root / "fault0-condition2").mkdir(parents=True)
    (root / "fault1-condition2").mkdir(parents=True)

    _write_csv(root / "fault0-condition1" / "fault0-condition1-1.csv")
    _write_csv(root / "fault1-condition1" / "fault1-condition1-1.csv")
    _write_csv(root / "fault0-condition2" / "fault0-condition2-1.csv")
    _write_csv(root / "fault1-condition2" / "fault1-condition2-1.csv")

    quality_path = tmp_path / "quality.csv"
    manifest = build_manifest(str(root), report_path=str(quality_path), default_duration_s=200.0)

    assert len(manifest) == 4
    assert quality_path.exists()
    assert manifest["fault_id"].nunique() == 2
    assert manifest["condition_id"].nunique() == 2

    split = make_splits(manifest, protocol="random_grouped", test_size=0.5, random_state=0)
    report = validate_split_leakage(split)
    assert report["train_test_disjoint"] is True

    loco = make_splits(manifest, protocol="loco")
    report_loco = validate_split_leakage(loco)
    assert all(report_loco.values())


def test_manifest_reports_missing_expected_combinations(tmp_path: Path):
    root = tmp_path / "data"
    (root / "fault0-condition1").mkdir(parents=True)
    (root / "fault1-condition1").mkdir(parents=True)

    _write_csv(root / "fault0-condition1" / "fault0-condition1-1.csv")
    _write_csv(root / "fault1-condition1" / "fault1-condition1-1.csv")

    quality_path = tmp_path / "quality.csv"
    manifest = build_manifest(
        str(root),
        report_path=str(quality_path),
        default_duration_s=200.0,
        expected_faults=2,
        expected_conditions=2,
        expected_openings=2,
    )

    assert len(manifest) == 2
    quality = pd.read_csv(quality_path)
    missing_rows = quality[quality["issue_type"] == "missing_expected_sample"]
    assert len(missing_rows) == 6
    assert "Fault0-condition1-opening2" in missing_rows["missing_combo"].tolist()


def test_resolve_effective_source_fs(tmp_path: Path):
    root = tmp_path / "data"
    (root / "fault0-condition1").mkdir(parents=True)
    _write_csv(root / "fault0-condition1" / "fault0-condition1-1.csv")

    manifest = build_manifest(str(root))
    cfg = DataConfig(data_root=str(root), source_fs_hz=10000, auto_detect_source_fs=True)
    effective_fs = resolve_effective_source_fs(manifest, cfg)
    assert effective_fs == 100
