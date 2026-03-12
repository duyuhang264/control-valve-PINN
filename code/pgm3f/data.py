from __future__ import annotations

import math
import re
from hashlib import sha1
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[override]
        pass

from .config import DataConfig
from .utils import save_json

try:
    from scipy.signal import butter, filtfilt
except Exception:  # pragma: no cover
    butter = None
    filtfilt = None


FOLDER_PATTERN = re.compile(r"fault(\d+)-condition(\d+)$", re.IGNORECASE)
FILE_PATTERN = re.compile(r"fault(\d+)(?:-condition(\d+))?-(\d+)\.csv$", re.IGNORECASE)

CANONICAL_COLUMNS = ["time", "CV", "F", "X_prime", "P1_prime", "P2_prime", "T_prime"]
COLUMN_ALIASES = {
    "time": "time",
    "t": "time",
    "cv": "CV",
    "f": "F",
    "x_prime": "X_prime",
    "xprime": "X_prime",
    "x": "X_prime",
    "p1_prime": "P1_prime",
    "p1prime": "P1_prime",
    "p1": "P1_prime",
    "p2_prime": "P2_prime",
    "p2prime": "P2_prime",
    "p2": "P2_prime",
    "t_prime": "T_prime",
    "tprime": "T_prime",
    "temp_prime": "T_prime",
}


@dataclass
class SampleMeta:
    path: str
    file_id: str
    fault_id: int
    condition_id: int
    opening_id: int
    duration_s: float
    fs_hz: float
    n_samples: int


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().replace(" ", "").lower()
        mapped = COLUMN_ALIASES.get(key)
        if mapped is not None:
            rename_map[col] = mapped
    return df.rename(columns=rename_map)


def _parse_ids(path: Path) -> Tuple[int, int, int]:
    fault_id = -1
    condition_id = -1
    opening_id = -1

    folder_match = FOLDER_PATTERN.search(path.parent.name)
    if folder_match:
        fault_id = int(folder_match.group(1))
        condition_id = int(folder_match.group(2))

    file_match = FILE_PATTERN.search(path.name)
    if file_match:
        fault_from_file = int(file_match.group(1))
        cond_from_file = file_match.group(2)
        opening_id = int(file_match.group(3))
        if fault_id < 0:
            fault_id = fault_from_file
        if condition_id < 0 and cond_from_file is not None:
            condition_id = int(cond_from_file)

    return fault_id, condition_id, opening_id


def _estimate_fs_from_preview(preview: pd.DataFrame) -> float:
    if "time" not in preview.columns or len(preview) < 3:
        return float("nan")
    t = preview["time"].to_numpy(dtype=np.float64)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def _count_data_rows(path: Path) -> int:
    with path.open("rb") as f:
        newline_count = 0
        while True:
            buf = f.read(1024 * 1024)
            if not buf:
                break
            newline_count += buf.count(b"\n")
    if newline_count <= 0:
        return 0
    return max(0, int(newline_count - 1))


def _estimate_duration(default_duration_s: float, n_samples: int, fs_hz: float) -> float:
    if n_samples > 1 and np.isfinite(fs_hz) and fs_hz > 0:
        return float((n_samples - 1) / fs_hz)
    return float(default_duration_s)


def _expected_missing_combinations(
    quality_df: pd.DataFrame,
    expected_faults: int,
    expected_conditions: int,
    expected_openings: int,
) -> List[Tuple[int, int, int]]:
    observed = {
        (int(row.fault_id), int(row.condition_id), int(row.opening_id))
        for row in quality_df.itertuples(index=False)
        if int(row.fault_id) >= 0 and int(row.condition_id) >= 0 and int(row.opening_id) >= 0
    }
    missing: List[Tuple[int, int, int]] = []
    for f in range(expected_faults):
        for c in range(1, expected_conditions + 1):
            for o in range(1, expected_openings + 1):
                if (f, c, o) not in observed:
                    missing.append((f, c, o))
    return missing


def build_manifest(
    data_root: str,
    report_path: Optional[str] = None,
    expected_columns: Optional[Iterable[str]] = None,
    default_duration_s: float = 200.0,
    expected_faults: Optional[int] = None,
    expected_conditions: Optional[int] = None,
    expected_openings: Optional[int] = None,
) -> pd.DataFrame:
    expected_columns = list(expected_columns or CANONICAL_COLUMNS)
    root = Path(data_root)
    rows: List[Dict[str, object]] = []
    quality_rows: List[Dict[str, object]] = []

    for path in sorted(root.rglob("*.csv")):
        fault_id, condition_id, opening_id = _parse_ids(path)
        file_id = str(path.relative_to(root)).replace("\\", "/")
        qrow: Dict[str, object] = {
            "path": str(path),
            "file_id": file_id,
            "fault_id": fault_id,
            "condition_id": condition_id,
            "opening_id": opening_id,
            "parse_ok": False,
            "issue_type": "read_error",
            "missing_columns": "",
            "read_error": "",
            "duration_s": default_duration_s,
            "fs_hz": np.nan,
            "n_samples": 0,
            "preview_rows": 0,
            "missing_combo": "",
        }

        try:
            preview = pd.read_csv(path, nrows=128)
            preview = _normalize_columns(preview)
            qrow["preview_rows"] = len(preview)
            qrow["n_samples"] = _count_data_rows(path)

            missing = [c for c in expected_columns if c not in preview.columns]
            if missing:
                qrow["issue_type"] = "missing_columns"
                qrow["missing_columns"] = ";".join(missing)
            else:
                fs_hz = _estimate_fs_from_preview(preview)
                qrow["fs_hz"] = fs_hz
                qrow["duration_s"] = _estimate_duration(default_duration_s, int(qrow["n_samples"]), fs_hz)
                qrow["parse_ok"] = True
                qrow["issue_type"] = "ok"
                sample = SampleMeta(
                    path=str(path),
                    file_id=file_id,
                    fault_id=fault_id,
                    condition_id=condition_id,
                    opening_id=opening_id,
                    duration_s=float(qrow["duration_s"]),
                    fs_hz=float(fs_hz) if np.isfinite(fs_hz) else float("nan"),
                    n_samples=int(qrow["n_samples"]),
                )
                rows.append(asdict(sample))
        except Exception as exc:
            qrow["read_error"] = str(exc)

        quality_rows.append(qrow)

    manifest = pd.DataFrame(rows)
    quality = pd.DataFrame(quality_rows)

    ef = int(expected_faults) if expected_faults is not None else None
    ec = int(expected_conditions) if expected_conditions is not None else None
    eo = int(expected_openings) if expected_openings is not None else None
    if ef is not None and ec is not None and eo is not None and not quality.empty:
        missing_combos = _expected_missing_combinations(quality, ef, ec, eo)
        if missing_combos:
            missing_rows = []
            for f, c, o in missing_combos:
                rel_path = f"Fault{f}/Fault{f}-condition{c}/Fault{f}-condition{c}-{o}.csv"
                missing_rows.append(
                    {
                        "path": "",
                        "file_id": rel_path,
                        "fault_id": f,
                        "condition_id": c,
                        "opening_id": o,
                        "parse_ok": False,
                        "issue_type": "missing_expected_sample",
                        "missing_columns": "",
                        "read_error": "missing_expected_sample",
                        "duration_s": default_duration_s,
                        "fs_hz": np.nan,
                        "n_samples": 0,
                        "preview_rows": 0,
                        "missing_combo": f"Fault{f}-condition{c}-opening{o}",
                    }
                )
            quality = pd.concat([quality, pd.DataFrame(missing_rows)], ignore_index=True)

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        quality.to_csv(report_path, index=False, encoding="utf-8")

    return manifest


def resolve_effective_source_fs(manifest: pd.DataFrame, cfg: DataConfig) -> int:
    if cfg.auto_detect_source_fs and not manifest.empty and "fs_hz" in manifest.columns:
        fs = manifest["fs_hz"].to_numpy(dtype=np.float64)
        fs = fs[np.isfinite(fs) & (fs > 0)]
        if len(fs) > 0:
            return int(round(float(np.median(fs))))
    return int(cfg.source_fs_hz)


def load_csv_standardized(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[CANONICAL_COLUMNS].copy()


def anti_alias_downsample(
    arr: np.ndarray,
    source_fs_hz: int,
    target_fs_hz: int,
    cutoff_hz: float,
) -> np.ndarray:
    if source_fs_hz <= target_fs_hz:
        return arr

    ratio = source_fs_hz / target_fs_hz
    step = int(round(ratio))
    if abs(ratio - step) > 1e-6:
        raise ValueError(f"source_fs/target_fs must be integer, got {ratio}")

    out = arr.copy()
    if butter is not None and filtfilt is not None:
        nyq = 0.5 * source_fs_hz
        wn = min(0.99, cutoff_hz / nyq)
        b, a = butter(4, wn, btype="low")
        for i in range(1, out.shape[1]):
            out[:, i] = filtfilt(b, a, out[:, i], method="pad")

    return out[::step]


def _resolve_cache_dir(cache_dir: Optional[str], cfg: DataConfig) -> Optional[Path]:
    if cache_dir:
        p = Path(cache_dir)
    elif cfg.preprocessed_cache_enabled and cfg.preprocessed_cache_dir:
        p = Path(cfg.preprocessed_cache_dir)
    else:
        return None
    p.mkdir(parents=True, exist_ok=True)
    return p


def _preprocess_cache_path(path: str, cache_dir: Path, source_fs_hz: int, cfg: DataConfig) -> Path:
    key = f"{Path(path).resolve()}|src={source_fs_hz}|tar={cfg.target_fs_hz}|cut={cfg.lowpass_cutoff_hz}"
    digest = sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


def preprocess_file(
    path: str,
    cfg: DataConfig,
    source_fs_hz: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    src_fs = int(source_fs_hz if source_fs_hz is not None else cfg.source_fs_hz)

    cache_root = _resolve_cache_dir(cache_dir, cfg)
    npy_path: Optional[Path] = None
    if cache_root is not None:
        npy_path = _preprocess_cache_path(path, cache_root, src_fs, cfg)
        if npy_path.exists():
            arr_cached = np.load(npy_path, allow_pickle=False)
            return pd.DataFrame(arr_cached, columns=CANONICAL_COLUMNS)

    df = load_csv_standardized(path)
    arr = df.to_numpy(dtype=np.float64)
    arr_ds = anti_alias_downsample(
        arr=arr,
        source_fs_hz=src_fs,
        target_fs_hz=cfg.target_fs_hz,
        cutoff_hz=cfg.lowpass_cutoff_hz,
    )
    if cache_root is not None and npy_path is not None:
        dtype = np.float32 if str(cfg.preprocessed_cache_dtype).lower() == "float32" else np.float64
        np.save(npy_path, arr_ds.astype(dtype, copy=False), allow_pickle=False)
    return pd.DataFrame(arr_ds, columns=CANONICAL_COLUMNS)


def _resolve_file_source_fs(row_fs_hz: float, effective_source_fs_hz: Optional[int], fallback_source_fs_hz: int) -> int:
    if np.isfinite(row_fs_hz) and row_fs_hz > 0:
        return int(round(float(row_fs_hz)))
    if effective_source_fs_hz is not None and effective_source_fs_hz > 0:
        return int(effective_source_fs_hz)
    return int(fallback_source_fs_hz)


def _downsampled_len(n_samples: int, source_fs_hz: int, target_fs_hz: int) -> int:
    if n_samples <= 0:
        return 0
    if source_fs_hz <= target_fs_hz:
        return int(n_samples)
    ratio = source_fs_hz / target_fs_hz
    step = int(round(ratio))
    if abs(ratio - step) > 1e-6:
        return int(n_samples)
    return int(math.ceil(n_samples / step))


def fit_scaler(
    manifest: pd.DataFrame,
    cfg: DataConfig,
    out_json_path: Optional[str] = None,
    effective_source_fs_hz: Optional[int] = None,
    preprocess_cache_dir: Optional[str] = None,
) -> Dict[str, object]:
    feats = ["CV", "F", "X_prime", "P1_prime", "P2_prime", "T_prime"]
    total_n = 0
    total_sum = np.zeros(len(feats), dtype=np.float64)
    total_sum_sq = np.zeros(len(feats), dtype=np.float64)

    for row in manifest.itertuples(index=False):
        row_fs = float(getattr(row, "fs_hz", float("nan")))
        source_fs = _resolve_file_source_fs(row_fs, effective_source_fs_hz, cfg.source_fs_hz)
        df = preprocess_file(
            row.path,
            cfg,
            source_fs_hz=source_fs,
            cache_dir=preprocess_cache_dir,
        )
        x = df[feats].to_numpy(dtype=np.float64)
        total_n += x.shape[0]
        total_sum += np.sum(x, axis=0)
        total_sum_sq += np.sum(x * x, axis=0)

    if total_n == 0:
        raise ValueError("No samples found when fitting scaler")

    mean = total_sum / total_n
    var = np.maximum(total_sum_sq / total_n - mean * mean, 1e-12)
    std = np.sqrt(var)
    scaler = {
        "features": feats,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_samples": int(total_n),
        "effective_source_fs_hz": int(effective_source_fs_hz if effective_source_fs_hz is not None else cfg.source_fs_hz),
    }

    if out_json_path:
        save_json(scaler, out_json_path)

    return scaler


def apply_scaler(df: pd.DataFrame, scaler: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    feats = scaler["features"]
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    x = out[feats].to_numpy(dtype=np.float64)
    out[feats] = (x - mean) / std
    return out


class PGM3FWindowDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        file_ids: List[str],
        cfg: DataConfig,
        scaler: Optional[Dict[str, object]] = None,
        fault_to_idx: Optional[Dict[int, int]] = None,
        condition_to_idx: Optional[Dict[int, int]] = None,
        train_mode: bool = False,
        train_windows_per_file: Optional[int] = None,
        seed: int = 42,
        effective_source_fs_hz: Optional[int] = None,
        preprocess_cache_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.scaler = scaler
        self.train_mode = train_mode
        self.train_windows_per_file = (
            int(train_windows_per_file)
            if train_windows_per_file is not None
            else int(cfg.train_windows_per_file)
        )
        self.seed = int(seed)
        self.effective_source_fs_hz = effective_source_fs_hz
        self.preprocess_cache_dir = preprocess_cache_dir
        source_for_windows = (
            int(effective_source_fs_hz)
            if effective_source_fs_hz is not None and effective_source_fs_hz > 0
            else int(cfg.source_fs_hz)
        )
        self.window_fs_hz = min(source_for_windows, int(cfg.target_fs_hz))
        self.window_len = max(1, int(cfg.window_sec * self.window_fs_hz))
        self.stride_len = max(1, int(cfg.stride_sec * self.window_fs_hz))
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

        self.file_df = (
            manifest[manifest["file_id"].isin(file_ids)]
            .drop_duplicates(subset=["file_id"])
            .reset_index(drop=True)
        )
        if self.file_df.empty:
            raise ValueError("No files selected for dataset")

        fault_values = sorted(self.file_df["fault_id"].unique().tolist())
        cond_values = sorted(self.file_df["condition_id"].unique().tolist())
        self.fault_to_idx = fault_to_idx or {f: i for i, f in enumerate(fault_values)}
        self.condition_to_idx = condition_to_idx or {c: i for i, c in enumerate(cond_values)}

        self.file_source_fs: List[int] = []
        self.file_window_starts: List[List[int]] = []
        for row in self.file_df.itertuples(index=False):
            row_fs = float(getattr(row, "fs_hz", float("nan")))
            source_fs = _resolve_file_source_fs(row_fs, self.effective_source_fs_hz, cfg.source_fs_hz)
            self.file_source_fs.append(source_fs)

            raw_n_samples = int(getattr(row, "n_samples", 0))
            if raw_n_samples <= 0:
                duration_s = float(getattr(row, "duration_s", cfg.default_duration_s))
                raw_n_samples = max(1, int(round(duration_s * source_fs)) + 1)

            n_ds = _downsampled_len(raw_n_samples, source_fs_hz=source_fs, target_fs_hz=cfg.target_fs_hz)
            max_start = max(0, n_ds - self.window_len)
            starts = list(range(0, max_start + 1, self.stride_len))
            if not starts:
                starts = [0]
            self.file_window_starts.append(starts)

        self.index_map: List[Tuple[int, int]] = []
        if self.train_mode:
            self.set_epoch(0)
        else:
            self.index_map = self._build_eval_index_map()

    def _build_eval_index_map(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for file_idx, starts in enumerate(self.file_window_starts):
            for start in starts:
                out.append((file_idx, int(start)))
        return out

    def _sample_file_starts(self, starts: List[int], rng: np.random.Generator) -> List[int]:
        k = int(self.train_windows_per_file)
        if k <= 0:
            return [int(v) for v in starts]

        if len(starts) == 1:
            return [int(starts[0])] * k

        if len(starts) >= k:
            base = np.linspace(0, len(starts) - 1, num=k)
            idx = np.rint(base).astype(np.int64)
            jitter = max(0, int(self.cfg.train_window_jitter))
            if jitter > 0:
                idx = idx + rng.integers(-jitter, jitter + 1, size=k)
            idx = np.clip(idx, 0, len(starts) - 1)
            return [int(starts[int(i)]) for i in idx]

        out = [int(v) for v in starts]
        extra = rng.integers(0, len(starts), size=k - len(starts))
        out.extend(int(starts[int(i)]) for i in extra)
        rng.shuffle(out)
        return out

    def set_epoch(self, epoch: int) -> None:
        if not self.train_mode:
            return
        rng = np.random.default_rng(self.seed + int(epoch))
        new_index: List[Tuple[int, int]] = []
        file_order = np.arange(len(self.file_window_starts), dtype=np.int64)
        if self.cfg.shuffle_files_each_epoch:
            rng.shuffle(file_order)
        for file_idx in file_order.tolist():
            starts = self.file_window_starts[int(file_idx)]
            sampled = self._sample_file_starts(starts, rng)
            for start in sampled:
                new_index.append((int(file_idx), int(start)))
        self.index_map = new_index

    def __len__(self) -> int:
        return len(self.index_map)

    def _load_cached(self, file_idx: int, file_id: str, path: str) -> pd.DataFrame:
        if file_id in self.cache:
            self.cache.move_to_end(file_id)
            return self.cache[file_id]

        source_fs = self.file_source_fs[file_idx]
        df = preprocess_file(
            path,
            self.cfg,
            source_fs_hz=source_fs,
            cache_dir=self.preprocess_cache_dir,
        )
        if self.scaler is not None:
            df = apply_scaler(df, self.scaler)

        self.cache[file_id] = df
        if len(self.cache) > self.cfg.max_file_cache:
            self.cache.popitem(last=False)
        return df

    def _slice_or_pad(self, arr: np.ndarray, start: int, length: int) -> np.ndarray:
        end = start + length
        if end <= arr.shape[0]:
            return arr[start:end]

        out = np.zeros((length, arr.shape[1]), dtype=arr.dtype)
        valid = max(0, arr.shape[0] - start)
        if valid > 0:
            out[:valid] = arr[start : start + valid]
            out[valid:] = arr[arr.shape[0] - 1]
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if torch is None:
            raise RuntimeError("torch is required for PGM3FWindowDataset")
        file_idx, start = self.index_map[idx]
        row = self.file_df.iloc[file_idx]
        df = self._load_cached(file_idx=file_idx, file_id=row.file_id, path=row.path)
        arr = df.to_numpy(dtype=np.float32)
        win = self._slice_or_pad(arr, start, self.window_len)

        t = win[:, 0:1]
        cv = win[:, 1:2]
        f = win[:, 2:3]
        x = win[:, 3:4]
        p1 = win[:, 4:5]
        p2 = win[:, 5:6]
        temp = win[:, 6:7]

        raw_seq = np.concatenate([cv, f, x, p1, p2, temp], axis=1)
        pinn_input = np.concatenate([t, cv, p1, p2, temp], axis=1)

        return {
            "pinn_input": torch.from_numpy(pinn_input),
            "raw_seq": torch.from_numpy(raw_seq),
            "target_x": torch.from_numpy(x),
            "target_f": torch.from_numpy(f),
            "fault_label": torch.tensor(self.fault_to_idx[int(row.fault_id)], dtype=torch.long),
            "condition_label": torch.tensor(self.condition_to_idx[int(row.condition_id)], dtype=torch.long),
            "file_id": row.file_id,
        }


def summarize_manifest(
    manifest: pd.DataFrame,
    expected_faults: Optional[int] = None,
    expected_conditions: Optional[int] = None,
    expected_openings: Optional[int] = None,
) -> Dict[str, object]:
    if manifest.empty:
        out: Dict[str, object] = {
            "num_files": 0,
            "num_faults": 0,
            "num_conditions": 0,
            "num_openings": 0,
            "median_fs_hz": float("nan"),
            "effective_source_fs_hz": float("nan"),
        }
    else:
        fs = manifest["fs_hz"].to_numpy(dtype=np.float64)
        fs = fs[np.isfinite(fs) & (fs > 0)]
        median_fs = float(np.median(fs)) if len(fs) > 0 else float("nan")
        out = {
            "num_files": int(len(manifest)),
            "num_faults": int(manifest["fault_id"].nunique()),
            "num_conditions": int(manifest["condition_id"].nunique()),
            "num_openings": int(manifest["opening_id"].nunique()),
            "median_fs_hz": median_fs,
            "effective_source_fs_hz": int(round(median_fs)) if np.isfinite(median_fs) else float("nan"),
        }

    if (
        expected_faults is not None
        and expected_conditions is not None
        and expected_openings is not None
    ):
        expected_total = int(expected_faults) * int(expected_conditions) * int(expected_openings)
        out["expected_total_files"] = expected_total
        out["missing_expected_files"] = max(0, expected_total - int(out["num_files"]))
    return out
