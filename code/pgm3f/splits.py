from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    manifest: pd.DataFrame,
    protocol: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, List[str]]:
    if manifest.empty:
        raise ValueError("Manifest is empty")

    protocol = protocol.lower().strip()
    file_df = manifest[["file_id", "fault_id", "condition_id"]].drop_duplicates().reset_index(drop=True)

    if protocol in {"random", "random_grouped"}:
        stratify = file_df["fault_id"] if file_df["fault_id"].nunique() > 1 else None
        try:
            train_df, test_df = train_test_split(
                file_df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            train_df, test_df = train_test_split(
                file_df,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
                stratify=None,
            )

        return {
            "train": train_df["file_id"].tolist(),
            "test": test_df["file_id"].tolist(),
        }

    if protocol in {"loco", "leave_one_condition_out"}:
        out: Dict[str, List[str]] = {}
        for cond in sorted(file_df["condition_id"].unique().tolist()):
            test_df = file_df[file_df["condition_id"] == cond]
            train_df = file_df[file_df["condition_id"] != cond]
            out[f"condition_{cond}_train"] = train_df["file_id"].tolist()
            out[f"condition_{cond}_test"] = test_df["file_id"].tolist()
        return out

    raise ValueError(f"Unsupported protocol: {protocol}")


def validate_split_leakage(split_map: Dict[str, List[str]]) -> Dict[str, bool]:
    report: Dict[str, bool] = {}

    if "train" in split_map and "test" in split_map:
        report["train_test_disjoint"] = _is_disjoint(split_map["train"], split_map["test"])

    test_keys = sorted([k for k in split_map if k.endswith("_test")])
    for key in test_keys:
        train_key = key.replace("_test", "_train")
        if train_key in split_map:
            report[f"{train_key}_disjoint_{key}"] = _is_disjoint(split_map[train_key], split_map[key])

    return report


def _is_disjoint(a: List[str], b: List[str]) -> bool:
    sa: Set[str] = set(a)
    sb: Set[str] = set(b)
    return len(sa.intersection(sb)) == 0


def iter_folds(split_map: Dict[str, List[str]]):
    if "train" in split_map and "test" in split_map:
        yield "random", split_map["train"], split_map["test"]
        return

    cond_ids = sorted(
        {
            int(k.split("_")[1])
            for k in split_map.keys()
            if k.startswith("condition_") and k.endswith("_train")
        }
    )

    for cond in cond_ids:
        train_key = f"condition_{cond}_train"
        test_key = f"condition_{cond}_test"
        if train_key in split_map and test_key in split_map:
            yield f"condition_{cond}", split_map[train_key], split_map[test_key]

