from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from .config import DataConfig, ExperimentConfig
from .data import preprocess_file
from .utils import ensure_dir


META_COLUMNS = {
    "path",
    "file_id",
    "fault_id",
    "condition_id",
    "opening_id",
    "duration_s",
    "fs_hz",
    "n_samples",
}

STATIC_CHANNELS = [
    "CV",
    "F",
    "X_prime",
    "P1_prime",
    "P2_prime",
    "T_prime",
    "dp",
    "dx",
    "ddx",
]


def _hash_cfg(payload: Dict[str, Any]) -> str:
    dumped = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()[:16]


def _resolve_source_fs(row_fs_hz: float, effective_source_fs_hz: int, fallback_source_fs_hz: int) -> int:
    if np.isfinite(row_fs_hz) and row_fs_hz > 0:
        return int(round(float(row_fs_hz)))
    if effective_source_fs_hz > 0:
        return int(effective_source_fs_hz)
    return int(fallback_source_fs_hz)


def _safe_skew(v: np.ndarray) -> float:
    if v.size < 3:
        return 0.0
    mu = float(np.mean(v))
    sd = float(np.std(v))
    if sd < 1e-12:
        return 0.0
    centered = v - mu
    m3 = float(np.mean(centered ** 3))
    return m3 / (sd ** 3 + 1e-12)


def _safe_kurtosis(v: np.ndarray) -> float:
    if v.size < 4:
        return 0.0
    mu = float(np.mean(v))
    sd = float(np.std(v))
    if sd < 1e-12:
        return 0.0
    centered = v - mu
    m4 = float(np.mean(centered ** 4))
    return m4 / (sd ** 4 + 1e-12) - 3.0


def _spectral_triplet(v: np.ndarray, fs_hz: float) -> Tuple[float, float, float]:
    if v.size < 8 or fs_hz <= 0:
        return 0.0, 0.0, 0.0
    x = v.astype(np.float64, copy=False)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    if spec.size <= 1:
        return 0.0, 0.0, 0.0
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    spec_nz = spec.copy()
    spec_nz[0] = 0.0
    peak_idx = int(np.argmax(spec_nz))
    peak_freq = float(freqs[peak_idx]) if peak_idx < freqs.size else 0.0
    band_mask = (freqs >= 0.1) & (freqs <= min(10.0, fs_hz * 0.5))
    band_power = float(np.sum(spec[band_mask]))
    total_power = float(np.sum(spec)) + 1e-12
    p = spec / total_power
    denom = math.log(float(p.size) + 1e-12)
    spec_entropy = float(-np.sum(p * np.log(p + 1e-12)) / denom) if denom > 0 else 0.0
    return peak_freq, band_power, spec_entropy


def _feature_stats(v: np.ndarray, fs_hz: float, with_spectral: bool = True) -> Dict[str, float]:
    x = v.astype(np.float64, copy=False)
    out: Dict[str, float] = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "skew": _safe_skew(x),
        "kurtosis": _safe_kurtosis(x),
        "energy": float(np.mean(x * x)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }
    if with_spectral:
        peak_freq, band_power, spec_entropy = _spectral_triplet(x, fs_hz=fs_hz)
        out["peak_freq"] = peak_freq
        out["band_power"] = band_power
        out["spec_entropy"] = spec_entropy
    else:
        out["peak_freq"] = 0.0
        out["band_power"] = 0.0
        out["spec_entropy"] = 0.0
    return out


def _segment_features(v: np.ndarray, segment_count: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    splits = np.array_split(v, max(1, int(segment_count)))
    for idx, seg in enumerate(splits):
        if seg.size == 0:
            seg = np.zeros(1, dtype=np.float64)
        out[f"seg{idx+1}_mean"] = float(np.mean(seg))
        out[f"seg{idx+1}_std"] = float(np.std(seg))
        out[f"seg{idx+1}_min"] = float(np.min(seg))
        out[f"seg{idx+1}_max"] = float(np.max(seg))
        out[f"seg{idx+1}_energy"] = float(np.mean(seg * seg))
    return out


def _derive_signals(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    t = df["time"].to_numpy(dtype=np.float64)
    cv = df["CV"].to_numpy(dtype=np.float64)
    f = df["F"].to_numpy(dtype=np.float64)
    x = df["X_prime"].to_numpy(dtype=np.float64)
    p1 = df["P1_prime"].to_numpy(dtype=np.float64)
    p2 = df["P2_prime"].to_numpy(dtype=np.float64)
    temp = df["T_prime"].to_numpy(dtype=np.float64)

    dt_arr = np.diff(t)
    dt_arr = dt_arr[np.isfinite(dt_arr) & (dt_arr > 0)]
    dt = float(np.median(dt_arr)) if dt_arr.size > 0 else 1.0
    dp = p1 - p2
    dx = np.gradient(x, dt)
    ddx = np.gradient(dx, dt)
    u_flow = cv * np.sign(dp) * np.sqrt(np.abs(dp) + 1e-8)
    dyn_drive = cv - x
    return {
        "CV": cv,
        "F": f,
        "X_prime": x,
        "P1_prime": p1,
        "P2_prime": p2,
        "T_prime": temp,
        "dp": dp,
        "dx": dx,
        "ddx": ddx,
        "u_flow": u_flow,
        "dyn_drive": dyn_drive,
    }


def _residual_channels(signals: Dict[str, np.ndarray], proxy: Dict[str, float]) -> Dict[str, np.ndarray]:
    f_hat = proxy["flow_a"] * signals["u_flow"] + proxy["flow_b"]
    x_hat = proxy["pos_a"] * signals["CV"] + proxy["pos_b"]
    dx_hat = proxy["dyn_k"] * signals["dyn_drive"] + proxy["dyn_b"]
    return {
        "r_flow": signals["F"] - f_hat,
        "r_pos": signals["X_prime"] - x_hat,
        "r_dyn": signals["dx"] - dx_hat,
    }


def _feature_block(
    channels: Dict[str, np.ndarray],
    fs_hz: float,
    segment_count: int,
    with_spectral: bool,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, values in channels.items():
        stats = _feature_stats(values, fs_hz=fs_hz, with_spectral=with_spectral)
        seg_stats = _segment_features(values, segment_count=segment_count)
        for k, v in stats.items():
            out[f"{name}__{k}"] = float(v)
        for k, v in seg_stats.items():
            out[f"{name}__{k}"] = float(v)
    return out


def _worker_extract_features(task: Dict[str, Any]) -> Dict[str, Any]:
    row = task["row"]
    feature_set = task["feature_set"]
    proxy = task.get("proxy_params")
    segment_count = int(task.get("segment_count", 5))
    data_cfg = DataConfig(
        data_root=task.get("data_root", "."),
        source_fs_hz=int(task["source_fs_hz"]),
        target_fs_hz=int(task["target_fs_hz"]),
        lowpass_cutoff_hz=float(task["lowpass_cutoff_hz"]),
        preprocessed_cache_enabled=bool(task.get("preprocess_cache_dir")),
        preprocessed_cache_dir=str(task.get("preprocess_cache_dir") or ""),
    )
    df = preprocess_file(
        str(row["path"]),
        cfg=data_cfg,
        source_fs_hz=int(task["source_fs_hz"]),
        cache_dir=str(task.get("preprocess_cache_dir") or ""),
    )
    signals = _derive_signals(df)
    fs_hz = float(task.get("effective_fs_hz", 0))
    if fs_hz <= 0 and len(df) >= 2:
        dt = float(np.median(np.diff(df["time"].to_numpy(dtype=np.float64))))
        fs_hz = float(1.0 / dt) if dt > 0 else 1.0
    if fs_hz <= 0:
        fs_hz = 1.0

    features: Dict[str, float] = {}
    if feature_set in {"static", "all"}:
        static_channels = {k: signals[k] for k in STATIC_CHANNELS}
        features.update(
            _feature_block(
                static_channels,
                fs_hz=fs_hz,
                segment_count=segment_count,
                with_spectral=True,
            )
        )

    if feature_set in {"residual", "all"}:
        if proxy is None:
            raise ValueError("proxy_params is required for residual feature extraction")
        residuals = _residual_channels(signals, proxy=proxy)
        # Residual channels use lighter stats to keep LOCO runtime bounded.
        for name, values in residuals.items():
            stats = _feature_stats(values, fs_hz=fs_hz, with_spectral=False)
            seg_stats = _segment_features(values, segment_count=segment_count)
            for k, v in stats.items():
                features[f"{name}__{k}"] = float(v)
            for k, v in seg_stats.items():
                features[f"{name}__{k}"] = float(v)

    out: Dict[str, Any] = {
        "file_id": row["file_id"],
        "path": row["path"],
        "fault_id": int(row["fault_id"]),
        "condition_id": int(row["condition_id"]),
        "opening_id": int(row["opening_id"]),
    }
    out.update(features)
    return out


def _normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(np.float32)
    return out


def _cache_payload(
    manifest: pd.DataFrame,
    feature_set: str,
    segment_count: int,
    effective_source_fs_hz: int,
    data_cfg: DataConfig,
    proxy_params: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "feature_set": feature_set,
        "segment_count": int(segment_count),
        "effective_source_fs_hz": int(effective_source_fs_hz),
        "target_fs_hz": int(data_cfg.target_fs_hz),
        "lowpass_cutoff_hz": float(data_cfg.lowpass_cutoff_hz),
        "num_files": int(len(manifest)),
    }
    if proxy_params is not None:
        payload["proxy"] = {k: round(float(v), 8) for k, v in proxy_params.items()}
    return payload


def build_file_features(manifest: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()

    data_cfg: DataConfig = cfg["data_cfg"]
    effective_source_fs_hz = int(cfg.get("effective_source_fs_hz", data_cfg.source_fs_hz))
    preprocess_cache_dir = str(cfg.get("preprocess_cache_dir", "") or "")
    feature_cache_dir = str(cfg.get("feature_cache_dir", "") or "")
    num_workers = int(cfg.get("num_workers", 0))
    segment_count = int(cfg.get("segment_count", 5))
    feature_set = str(cfg.get("feature_set", "static")).strip().lower()
    proxy_params = cfg.get("proxy_params")

    cache_key = _hash_cfg(
        _cache_payload(
            manifest=manifest,
            feature_set=feature_set,
            segment_count=segment_count,
            effective_source_fs_hz=effective_source_fs_hz,
            data_cfg=data_cfg,
            proxy_params=proxy_params,
        )
    )
    cache_path: Optional[Path] = None
    if feature_cache_dir:
        cache_root = ensure_dir(feature_cache_dir)
        cache_path = cache_root / f"features_{feature_set}_{cache_key}.pkl"
        if cache_path.exists():
            cached = pd.read_pickle(cache_path)
            return cached.sort_values("file_id").reset_index(drop=True)

    subset = manifest[
        ["path", "file_id", "fault_id", "condition_id", "opening_id", "fs_hz"]
    ].drop_duplicates(subset=["file_id"])
    rows = subset.to_dict(orient="records")
    tasks: List[Dict[str, Any]] = []
    for row in rows:
        source_fs = _resolve_source_fs(
            row_fs_hz=float(row.get("fs_hz", float("nan"))),
            effective_source_fs_hz=effective_source_fs_hz,
            fallback_source_fs_hz=int(data_cfg.source_fs_hz),
        )
        tasks.append(
            {
                "row": row,
                "feature_set": feature_set,
                "proxy_params": proxy_params,
                "segment_count": segment_count,
                "source_fs_hz": source_fs,
                "target_fs_hz": int(data_cfg.target_fs_hz),
                "lowpass_cutoff_hz": float(data_cfg.lowpass_cutoff_hz),
                "effective_fs_hz": float(min(source_fs, int(data_cfg.target_fs_hz))),
                "preprocess_cache_dir": preprocess_cache_dir,
                "data_root": data_cfg.data_root,
            }
        )

    if num_workers > 1 and len(tasks) > 4:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            records = list(executor.map(_worker_extract_features, tasks))
    else:
        records = [_worker_extract_features(task) for task in tasks]

    feat_df = pd.DataFrame(records).sort_values("file_id").reset_index(drop=True)
    feat_df = _normalize_numeric_columns(feat_df)
    if cache_path is not None:
        feat_df.to_pickle(cache_path)
    return feat_df


def _fit_linear(sum_z: float, sum_y: float, sum_zz: float, sum_zy: float, n: int) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    denom = n * sum_zz - sum_z * sum_z
    if abs(denom) < 1e-12:
        a = sum_zy / (sum_zz + 1e-12)
        b = (sum_y - a * sum_z) / max(1, n)
        return float(a), float(b)
    a = (n * sum_zy - sum_z * sum_y) / denom
    b = (sum_y - a * sum_z) / n
    return float(a), float(b)


def fit_physical_proxy(
    train_manifest: pd.DataFrame,
    data_cfg: DataConfig,
    effective_source_fs_hz: int,
    preprocess_cache_dir: str = "",
) -> Dict[str, float]:
    work_df = train_manifest.copy()
    healthy = work_df[work_df["fault_id"] == 0]
    if healthy.empty:
        healthy = work_df

    flow_sum_z = flow_sum_y = flow_sum_zz = flow_sum_zy = 0.0
    pos_sum_z = pos_sum_y = pos_sum_zz = pos_sum_zy = 0.0
    dyn_sum_z = dyn_sum_y = dyn_sum_zz = dyn_sum_zy = 0.0
    n_flow = n_pos = n_dyn = 0

    for row in healthy.itertuples(index=False):
        source_fs = _resolve_source_fs(
            row_fs_hz=float(getattr(row, "fs_hz", float("nan"))),
            effective_source_fs_hz=effective_source_fs_hz,
            fallback_source_fs_hz=int(data_cfg.source_fs_hz),
        )
        df = preprocess_file(
            str(row.path),
            cfg=data_cfg,
            source_fs_hz=source_fs,
            cache_dir=preprocess_cache_dir,
        )
        sig = _derive_signals(df)
        cv = sig["CV"]
        f = sig["F"]
        x = sig["X_prime"]
        u_flow = sig["u_flow"]
        dyn_drive = sig["dyn_drive"]
        dx = sig["dx"]

        flow_sum_z += float(np.sum(u_flow))
        flow_sum_y += float(np.sum(f))
        flow_sum_zz += float(np.sum(u_flow * u_flow))
        flow_sum_zy += float(np.sum(u_flow * f))
        n_flow += int(u_flow.size)

        pos_sum_z += float(np.sum(cv))
        pos_sum_y += float(np.sum(x))
        pos_sum_zz += float(np.sum(cv * cv))
        pos_sum_zy += float(np.sum(cv * x))
        n_pos += int(cv.size)

        dyn_sum_z += float(np.sum(dyn_drive))
        dyn_sum_y += float(np.sum(dx))
        dyn_sum_zz += float(np.sum(dyn_drive * dyn_drive))
        dyn_sum_zy += float(np.sum(dyn_drive * dx))
        n_dyn += int(dyn_drive.size)

    flow_a, flow_b = _fit_linear(flow_sum_z, flow_sum_y, flow_sum_zz, flow_sum_zy, n_flow)
    pos_a, pos_b = _fit_linear(pos_sum_z, pos_sum_y, pos_sum_zz, pos_sum_zy, n_pos)
    dyn_k, dyn_b = _fit_linear(dyn_sum_z, dyn_sum_y, dyn_sum_zz, dyn_sum_zy, n_dyn)

    return {
        "flow_a": flow_a,
        "flow_b": flow_b,
        "pos_a": pos_a,
        "pos_b": pos_b,
        "dyn_k": dyn_k,
        "dyn_b": dyn_b,
        "healthy_files_used": int(healthy["file_id"].nunique()),
    }


def _compute_metrics(y_true: Iterable[int], y_pred: Iterable[int], classes: List[int]) -> Dict[str, Any]:
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int64)
    if y_true_arr.size == 0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "per_class_recall": [0.0 for _ in classes],
            "classes": classes,
            "confusion_matrix": [[0 for _ in classes] for _ in classes],
            "num_samples": 0,
            "num_files": 0,
            "granularity": "file",
        }
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "per_class_recall": recall_score(y_true_arr, y_pred_arr, labels=classes, average=None, zero_division=0).tolist(),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(y_true_arr.size),
        "num_files": int(y_true_arr.size),
        "granularity": "file",
    }


def _base_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLUMNS and not c.startswith("dlt__")]


def _build_condition_templates(train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    healthy = train_df[train_df["fault_id"] == 0]
    if healthy.empty:
        healthy = train_df
    cond_template = healthy.groupby("condition_id", as_index=False)[feature_cols].mean()
    global_template = healthy[feature_cols].mean()
    return cond_template, global_template


def _augment_delta_to_healthy(
    df: pd.DataFrame,
    feature_cols: List[str],
    cond_template: pd.DataFrame,
    global_template: pd.Series,
) -> Tuple[pd.DataFrame, List[str]]:
    merged = df.copy()
    template_cols = [f"tmpl__{c}" for c in feature_cols]
    template_renamed = cond_template.rename(columns={c: f"tmpl__{c}" for c in feature_cols})
    merged = merged.merge(template_renamed, on="condition_id", how="left")
    base_mat = merged[feature_cols].to_numpy(dtype=np.float32, copy=False)
    tmpl_mat = merged[template_cols].to_numpy(dtype=np.float32, copy=False)
    for j, c in enumerate(feature_cols):
        fill_val = float(global_template.get(c, 0.0))
        col = tmpl_mat[:, j]
        if np.isnan(col).any():
            col[np.isnan(col)] = fill_val
    delta_mat = base_mat - tmpl_mat
    delta_cols = [f"dlt__{c}" for c in feature_cols]
    delta_df = pd.DataFrame(delta_mat.astype(np.float32, copy=False), columns=delta_cols, index=merged.index)
    merged = pd.concat([merged.drop(columns=template_cols), delta_df], axis=1)
    return merged, delta_cols


def _build_multiclass_model(cfg: ExperimentConfig, num_classes: int):
    if XGBClassifier is not None:
        return XGBClassifier(
            objective="multi:softprob",
            num_class=max(2, int(num_classes)),
            eval_metric="mlogloss",
            n_estimators=int(cfg.xgb_n_estimators),
            max_depth=int(cfg.xgb_max_depth),
            learning_rate=float(cfg.xgb_learning_rate),
            subsample=float(cfg.xgb_subsample),
            colsample_bytree=float(cfg.xgb_colsample_bytree),
            reg_lambda=float(cfg.xgb_reg_lambda),
            min_child_weight=float(cfg.xgb_min_child_weight),
            tree_method="hist",
            n_jobs=-1,
            random_state=int(cfg.random_state),
        )
    return RandomForestClassifier(
        n_estimators=700,
        random_state=int(cfg.random_state),
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def _build_binary_model(cfg: ExperimentConfig):
    if XGBClassifier is not None:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=max(200, int(cfg.xgb_n_estimators // 2)),
            max_depth=max(3, int(cfg.xgb_max_depth) - 2),
            learning_rate=float(cfg.xgb_learning_rate),
            subsample=float(cfg.xgb_subsample),
            colsample_bytree=float(cfg.xgb_colsample_bytree),
            reg_lambda=float(cfg.xgb_reg_lambda),
            min_child_weight=float(cfg.xgb_min_child_weight),
            tree_method="hist",
            n_jobs=-1,
            random_state=int(cfg.random_state),
        )
    return RandomForestClassifier(
        n_estimators=500,
        random_state=int(cfg.random_state),
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def _predict_proba_aligned(model: Any, x: np.ndarray, classes: List[int]) -> np.ndarray:
    proba = model.predict_proba(x)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = np.stack([1.0 - proba, proba], axis=1)

    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        if proba.shape[1] == len(classes):
            return proba
        out = np.zeros((proba.shape[0], len(classes)), dtype=np.float64)
        m = min(out.shape[1], proba.shape[1])
        out[:, :m] = proba[:, :m]
        return out

    out = np.zeros((proba.shape[0], len(classes)), dtype=np.float64)
    model_classes = [int(c) for c in np.asarray(model_classes).tolist()]
    cls_to_col = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model_classes):
        if c in cls_to_col:
            out[:, cls_to_col[c]] = proba[:, j]
    row_sum = np.sum(out, axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return out / row_sum


def _class_weight_vector(y: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(y, return_counts=True)
    total = float(y.size)
    n_cls = float(unique.size)
    cls_w = {int(c): total / (n_cls * float(cnt)) for c, cnt in zip(unique, counts)}
    return np.asarray([cls_w.get(int(v), 1.0) for v in y], dtype=np.float32)


def _tune_expert_threshold(
    y_true_bin: np.ndarray,
    prob: np.ndarray,
    beta: float = 2.0,
) -> float:
    if y_true_bin.size == 0:
        return 0.5
    candidates = np.linspace(0.05, 0.95, 37)
    best_th = 0.5
    best_score = -1.0
    for th in candidates:
        pred = (prob >= th).astype(np.int64)
        score = float(fbeta_score(y_true_bin, pred, beta=beta, zero_division=0))
        if score > best_score:
            best_score = score
            best_th = float(th)
    return best_th


def _apply_expert_subset(
    y_main: np.ndarray,
    max_prob: np.ndarray,
    tau_main: float,
    expert_defs: Dict[int, Dict[str, Any]],
    expert_probs: Dict[int, np.ndarray],
    selected_hc: List[int],
) -> np.ndarray:
    if len(selected_hc) == 0:
        return y_main.copy()
    low_mask = max_prob < tau_main
    if not np.any(low_mask):
        return y_main.copy()

    pred = y_main.copy()
    scores = np.full((y_main.size, len(selected_hc)), fill_value=-np.inf, dtype=np.float64)
    for j, hc in enumerate(selected_hc):
        th = float(expert_defs[hc]["threshold"])
        p = expert_probs[hc]
        scores[:, j] = np.where(p >= th, p, -np.inf)
    best_j = np.argmax(scores, axis=1)
    best_score = scores[np.arange(scores.shape[0]), best_j]
    apply_mask = low_mask & np.isfinite(best_score)
    if np.any(apply_mask):
        pred[apply_mask] = np.asarray(selected_hc, dtype=np.int64)[best_j[apply_mask]]
    return pred


def train_residual_tree(train_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    exp_cfg: ExperimentConfig = cfg["exp_cfg"]
    hard_classes = [int(c) for c in cfg.get("hard_classes", exp_cfg.hard_classes)]
    tau_main = float(cfg.get("tau_main", exp_cfg.tau_main))
    disable_experts = bool(cfg.get("disable_experts", exp_cfg.disable_experts))
    val_size = float(cfg.get("expert_val_size", exp_cfg.expert_val_size))
    beta = float(cfg.get("expert_beta", exp_cfg.expert_beta))

    all_classes = sorted(int(v) for v in train_df["fault_id"].unique().tolist())
    base_cols = _base_feature_columns(train_df)
    if len(base_cols) == 0:
        raise ValueError("No feature columns found for residual_tree")

    stratify = train_df["fault_id"] if train_df["fault_id"].nunique() > 1 else None
    min_per_class = int(train_df["fault_id"].value_counts().min()) if not train_df.empty else 0
    val_count = int(round(float(val_size) * float(len(train_df))))
    n_classes = int(train_df["fault_id"].nunique())
    can_stratify_split = val_size > 0 and min_per_class >= 2 and val_count >= n_classes and (len(train_df) - val_count) >= n_classes
    if can_stratify_split:
        main_train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=int(exp_cfg.random_state),
            stratify=stratify,
        )
    else:
        main_train_df = train_df
        val_df = train_df.iloc[:0].copy()

    cond_template_sub, global_template_sub = _build_condition_templates(main_train_df, base_cols)
    main_train_aug, delta_cols = _augment_delta_to_healthy(
        main_train_df,
        feature_cols=base_cols,
        cond_template=cond_template_sub,
        global_template=global_template_sub,
    )
    feature_cols = base_cols + delta_cols

    x_main_train = main_train_aug[feature_cols].to_numpy(dtype=np.float32)
    y_main_train = main_train_aug["fault_id"].to_numpy(dtype=np.int64)

    main_model = _build_multiclass_model(exp_cfg, num_classes=max(2, len(all_classes)))
    if XGBClassifier is not None and isinstance(main_model, XGBClassifier):
        sw = _class_weight_vector(y_main_train)
        main_model.fit(x_main_train, y_main_train, sample_weight=sw)
    else:
        main_model.fit(x_main_train, y_main_train)

    experts: Dict[int, Dict[str, Any]] = {}
    tuning_info: Dict[int, Dict[str, float]] = {}
    if not disable_experts and not val_df.empty:
        val_aug, _ = _augment_delta_to_healthy(
            val_df,
            feature_cols=base_cols,
            cond_template=cond_template_sub,
            global_template=global_template_sub,
        )
        x_val = val_aug[feature_cols].to_numpy(dtype=np.float32)
        y_val = val_aug["fault_id"].to_numpy(dtype=np.int64)
        val_proba = _predict_proba_aligned(main_model, x_val, classes=all_classes)
        max_prob = np.max(val_proba, axis=1)
        low_mask = max_prob < tau_main

        y_main_val = np.asarray(all_classes, dtype=np.int64)[np.argmax(val_proba, axis=1)]
        candidate_probs: Dict[int, np.ndarray] = {}
        for hc in hard_classes:
            y_bin_train = (y_main_train == hc).astype(np.int64)
            if int(np.sum(y_bin_train)) < 8:
                continue
            expert_model = _build_binary_model(exp_cfg)
            if XGBClassifier is not None and isinstance(expert_model, XGBClassifier):
                sw = _class_weight_vector(y_bin_train)
                expert_model.fit(x_main_train, y_bin_train, sample_weight=sw)
            else:
                expert_model.fit(x_main_train, y_bin_train)

            if np.any(low_mask):
                p_val = np.asarray(expert_model.predict_proba(x_val), dtype=np.float64)
                p_pos = p_val[:, 1] if p_val.ndim == 2 and p_val.shape[1] >= 2 else p_val.reshape(-1)
                th = _tune_expert_threshold(
                    y_true_bin=(y_val[low_mask] == hc).astype(np.int64),
                    prob=p_pos[low_mask],
                    beta=beta,
                )
            else:
                p_val = np.asarray(expert_model.predict_proba(x_val), dtype=np.float64)
                p_pos = p_val[:, 1] if p_val.ndim == 2 and p_val.shape[1] >= 2 else p_val.reshape(-1)
                th = 0.5
            experts[int(hc)] = {"model": expert_model, "threshold": float(th)}
            tuning_info[int(hc)] = {"threshold": float(th)}
            candidate_probs[int(hc)] = np.asarray(p_pos, dtype=np.float64)

        if experts:
            selected: List[int] = []
            baseline_macro = float(f1_score(y_val, y_main_val, average="macro", zero_division=0))
            remaining = list(experts.keys())
            while remaining:
                best_gain = 0.0
                best_hc: Optional[int] = None
                best_macro = baseline_macro
                for hc in remaining:
                    trial = selected + [hc]
                    pred_trial = _apply_expert_subset(
                        y_main=y_main_val,
                        max_prob=max_prob,
                        tau_main=tau_main,
                        expert_defs=experts,
                        expert_probs=candidate_probs,
                        selected_hc=trial,
                    )
                    macro = float(f1_score(y_val, pred_trial, average="macro", zero_division=0))
                    gain = macro - baseline_macro
                    if gain > best_gain + 1e-10:
                        best_gain = gain
                        best_hc = int(hc)
                        best_macro = macro
                if best_hc is None:
                    break
                selected.append(best_hc)
                remaining = [hc for hc in remaining if int(hc) != int(best_hc)]
                baseline_macro = best_macro

            experts = {hc: experts[hc] for hc in selected}
            for hc in list(tuning_info.keys()):
                tuning_info[hc]["selected"] = 1.0 if hc in experts else 0.0

    # Refit on full training set.
    cond_template_full, global_template_full = _build_condition_templates(train_df, base_cols)
    full_aug, full_delta_cols = _augment_delta_to_healthy(
        train_df,
        feature_cols=base_cols,
        cond_template=cond_template_full,
        global_template=global_template_full,
    )
    final_feature_cols = base_cols + full_delta_cols
    x_full = full_aug[final_feature_cols].to_numpy(dtype=np.float32)
    y_full = full_aug["fault_id"].to_numpy(dtype=np.int64)
    final_main = _build_multiclass_model(exp_cfg, num_classes=max(2, len(all_classes)))
    if XGBClassifier is not None and isinstance(final_main, XGBClassifier):
        sw = _class_weight_vector(y_full)
        final_main.fit(x_full, y_full, sample_weight=sw)
    else:
        final_main.fit(x_full, y_full)

    final_experts: Dict[int, Dict[str, Any]] = {}
    if experts:
        for hc in experts.keys():
            y_bin = (y_full == int(hc)).astype(np.int64)
            if int(np.sum(y_bin)) < 8:
                continue
            exp_model = _build_binary_model(exp_cfg)
            if XGBClassifier is not None and isinstance(exp_model, XGBClassifier):
                sw = _class_weight_vector(y_bin)
                exp_model.fit(x_full, y_bin, sample_weight=sw)
            else:
                exp_model.fit(x_full, y_bin)
            final_experts[int(hc)] = {
                "model": exp_model,
                "threshold": float(experts[int(hc)]["threshold"]),
            }

    return {
        "main_model": final_main,
        "experts": final_experts,
        "hard_classes": [int(v) for v in hard_classes],
        "tau_main": float(tau_main),
        "disable_experts": bool(disable_experts),
        "base_feature_columns": base_cols,
        "feature_columns": final_feature_cols,
        "condition_template": cond_template_full,
        "global_template": global_template_full,
        "classes": all_classes,
        "expert_tuning": tuning_info,
    }


def _prepare_inference_features(model_bundle: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    base_cols = model_bundle["base_feature_columns"]
    cond_template = model_bundle["condition_template"]
    global_template = model_bundle["global_template"]
    aug, _ = _augment_delta_to_healthy(df, base_cols, cond_template, global_template)
    return aug


def predict_with_experts(model_bundle: Dict[str, Any], test_df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    aug = _prepare_inference_features(model_bundle, test_df)
    feature_cols = model_bundle["feature_columns"]
    x = aug[feature_cols].to_numpy(dtype=np.float32)

    classes = [int(c) for c in model_bundle["classes"]]
    proba_main = _predict_proba_aligned(model_bundle["main_model"], x, classes=classes)
    y_main = np.asarray(classes, dtype=np.int64)[np.argmax(proba_main, axis=1)]

    tau_main = float(cfg.get("tau_main", model_bundle.get("tau_main", 0.65)))
    disable_experts = bool(cfg.get("disable_experts", model_bundle.get("disable_experts", False)))
    experts = model_bundle.get("experts", {})
    if disable_experts or not experts:
        return y_main

    max_prob = np.max(proba_main, axis=1)
    low_mask = max_prob < tau_main
    if not np.any(low_mask):
        return y_main

    y_final = y_main.copy()
    expert_scores = np.full((x.shape[0], len(experts)), fill_value=-np.inf, dtype=np.float64)
    expert_classes = list(experts.keys())
    for j, hc in enumerate(expert_classes):
        expert_info = experts[hc]
        model = expert_info["model"]
        th = float(expert_info["threshold"])
        p = np.asarray(model.predict_proba(x), dtype=np.float64)
        p_pos = p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.reshape(-1)
        valid = p_pos >= th
        expert_scores[:, j] = np.where(valid, p_pos, -np.inf)

    best_j = np.argmax(expert_scores, axis=1)
    best_score = expert_scores[np.arange(expert_scores.shape[0]), best_j]
    apply_mask = low_mask & np.isfinite(best_score)
    if np.any(apply_mask):
        chosen = np.asarray(expert_classes, dtype=np.int64)[best_j[apply_mask]]
        y_final[apply_mask] = chosen
    return y_final


def evaluate_residual_tree(
    model_bundle: Dict[str, Any],
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    aug = _prepare_inference_features(model_bundle, test_df)
    feature_cols = model_bundle["feature_columns"]
    classes = [int(c) for c in model_bundle["classes"]]
    x = aug[feature_cols].to_numpy(dtype=np.float32)
    y_true = test_df["fault_id"].to_numpy(dtype=np.int64)

    proba_main = _predict_proba_aligned(model_bundle["main_model"], x, classes=classes)
    y_pred_main = np.asarray(classes, dtype=np.int64)[np.argmax(proba_main, axis=1)]
    y_pred_final = predict_with_experts(model_bundle, test_df=test_df, cfg=cfg)

    main_eval = _compute_metrics(y_true=y_true, y_pred=y_pred_main, classes=classes)
    final_eval = _compute_metrics(y_true=y_true, y_pred=y_pred_final, classes=classes)

    tau_main = float(cfg.get("tau_main", model_bundle.get("tau_main", 0.65)))
    max_prob = np.max(proba_main, axis=1)
    low_mask = max_prob < tau_main
    changed = y_pred_main != y_pred_final
    hard_classes = [int(c) for c in cfg.get("hard_classes", model_bundle.get("hard_classes", []))]
    recall_before = (
        recall_score(
            y_true,
            y_pred_main,
            labels=hard_classes,
            average=None,
            zero_division=0,
        ).tolist()
        if hard_classes
        else []
    )
    recall_after = (
        recall_score(
            y_true,
            y_pred_final,
            labels=hard_classes,
            average=None,
            zero_division=0,
        ).tolist()
        if hard_classes
        else []
    )

    return {
        "main_eval": main_eval,
        "final_eval": final_eval,
        "expert": {
            "tau_main": tau_main,
            "trigger_rate": float(np.mean(low_mask)) if low_mask.size else 0.0,
            "trigger_count": int(np.sum(low_mask)),
            "reassigned_count": int(np.sum(changed)),
            "hard_classes": hard_classes,
            "hard_recall_before": recall_before,
            "hard_recall_after": recall_after,
        },
    }
