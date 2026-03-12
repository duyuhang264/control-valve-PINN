from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _safe_metric(d: Dict[str, object], key: str) -> float:
    v = d.get(key, float("nan"))
    try:
        return float(v)
    except Exception:
        return float("nan")


def _pick_primary_eval(fold_data: Dict[str, object]) -> Tuple[str, Dict[str, object], Dict[str, object]]:
    if "residual_tree" in fold_data:
        block = fold_data.get("residual_tree", {})
        if isinstance(block, dict):
            return (
                "residual_tree",
                block.get("final_eval", {}) if isinstance(block.get("final_eval", {}), dict) else {},
                block,
            )
    if "pgm3f" in fold_data:
        block = fold_data.get("pgm3f", {})
        if isinstance(block, dict):
            return (
                "pgm3f",
                block.get("final_eval", {}) if isinstance(block.get("final_eval", {}), dict) else {},
                block,
            )
    return "unknown", {}, {}


def export_paper_materials(results: Dict[str, object], out_dir: str | Path) -> Dict[str, str]:
    out = Path(out_dir) / "paper_materials"
    out.mkdir(parents=True, exist_ok=True)

    main_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []
    loco_rows: List[Dict[str, object]] = []
    hard_rows: List[Dict[str, object]] = []
    runtime_rows: List[Dict[str, object]] = []

    protocols = results.get("protocols", {})
    random_cm_written = False

    for protocol_name, protocol_data in protocols.items():
        if not isinstance(protocol_data, dict):
            continue
        folds = protocol_data.get("folds", {})
        if not isinstance(folds, dict):
            continue

        for fold_name, fold_data in folds.items():
            if not isinstance(fold_data, dict):
                continue

            primary_model, primary_eval, primary_block = _pick_primary_eval(fold_data)
            if primary_eval:
                main_rows.append(
                    {
                        "protocol": protocol_name,
                        "fold": fold_name,
                        "model": primary_model,
                        "accuracy": _safe_metric(primary_eval, "accuracy"),
                        "macro_f1": _safe_metric(primary_eval, "macro_f1"),
                        "balanced_accuracy": _safe_metric(primary_eval, "balanced_accuracy"),
                        "num_files": int(primary_eval.get("num_files", primary_eval.get("num_samples", 0))),
                    }
                )

            if protocol_name == "random_grouped" and not random_cm_written and primary_eval.get("confusion_matrix"):
                cm = primary_eval["confusion_matrix"]
                pd.DataFrame(cm).to_csv(out / f"confusion_matrix_random_{primary_model}.csv", index=False)
                random_cm_written = True

            if protocol_name == "loco" and primary_eval:
                cond_id = fold_name.replace("condition_", "")
                loco_rows.append(
                    {
                        "condition": cond_id,
                        "model": primary_model,
                        "accuracy": _safe_metric(primary_eval, "accuracy"),
                        "macro_f1": _safe_metric(primary_eval, "macro_f1"),
                        "balanced_accuracy": _safe_metric(primary_eval, "balanced_accuracy"),
                    }
                )

            baselines = fold_data.get("baselines", {})
            if protocol_name == "random_grouped" and isinstance(baselines, dict):
                for model_name, model_result in baselines.items():
                    final_eval = model_result.get("final_eval", model_result) if isinstance(model_result, dict) else {}
                    if not isinstance(final_eval, dict):
                        continue
                    main_rows.append(
                        {
                            "protocol": protocol_name,
                            "fold": fold_name,
                            "model": model_name,
                            "accuracy": _safe_metric(final_eval, "accuracy"),
                            "macro_f1": _safe_metric(final_eval, "macro_f1"),
                            "balanced_accuracy": _safe_metric(final_eval, "balanced_accuracy"),
                            "num_files": int(final_eval.get("num_files", final_eval.get("num_samples", 0))),
                        }
                    )

            ablations = fold_data.get("ablations", {})
            if protocol_name == "random_grouped" and isinstance(ablations, dict):
                for ab_name, ab_result in ablations.items():
                    if not isinstance(ab_result, dict):
                        continue
                    final_eval = ab_result.get("final_eval", {})
                    if not isinstance(final_eval, dict):
                        continue
                    ablation_rows.append(
                        {
                            "fold": fold_name,
                            "ablation": ab_name,
                            "accuracy": _safe_metric(final_eval, "accuracy"),
                            "macro_f1": _safe_metric(final_eval, "macro_f1"),
                            "balanced_accuracy": _safe_metric(final_eval, "balanced_accuracy"),
                        }
                    )

            if primary_model == "residual_tree":
                expert = primary_block.get("expert", {}) if isinstance(primary_block, dict) else {}
                if isinstance(expert, dict):
                    hard_classes = expert.get("hard_classes", [])
                    before = expert.get("hard_recall_before", [])
                    after = expert.get("hard_recall_after", [])
                    for i, hc in enumerate(hard_classes):
                        hard_rows.append(
                            {
                                "protocol": protocol_name,
                                "fold": fold_name,
                                "hard_class": int(hc),
                                "recall_before": float(before[i]) if i < len(before) else float("nan"),
                                "recall_after": float(after[i]) if i < len(after) else float("nan"),
                                "trigger_rate": _safe_metric(expert, "trigger_rate"),
                            }
                        )

                runtime = fold_data.get("runtime_breakdown", {})
                if isinstance(runtime, dict):
                    runtime_rows.append(
                        {
                            "protocol": protocol_name,
                            "fold": fold_name,
                            "feature_seconds": _safe_metric(runtime, "feature_seconds"),
                            "train_seconds": _safe_metric(runtime, "train_seconds"),
                            "eval_seconds": _safe_metric(runtime, "eval_seconds"),
                            "total_seconds": _safe_metric(fold_data, "runtime_seconds"),
                        }
                    )

    pd.DataFrame(main_rows).to_csv(out / "main_results.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out / "ablation_random.csv", index=False)
    pd.DataFrame(loco_rows).to_csv(out / "loco_per_condition.csv", index=False)
    pd.DataFrame(hard_rows).to_csv(out / "hard_class_expert_compare.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(out / "runtime_breakdown.csv", index=False)

    caption_text = """# IEEE Figure/Table Caption Templates

## Tab.1 Main comparison
File-level diagnosis performance under grouped random split and LOCO protocols on DAMADICS (20 classes).

## Fig.1 Confusion matrix
File-level 20-class confusion matrix of the primary method under grouped random split.

## Tab.2 Ablation study
Ablation results under grouped random split.

## Tab.3 Hard-class expert analysis
Recall changes for hard classes (Fault0/Fault8/Fault14/Fault19) before and after expert reranking.

## Tab.4 Runtime breakdown
CPU runtime decomposition across feature extraction, training, and evaluation.
"""
    (out / "caption_templates.md").write_text(caption_text, encoding="utf-8")

    return {
        "paper_materials_dir": str(out),
        "main_results_csv": str(out / "main_results.csv"),
        "ablation_csv": str(out / "ablation_random.csv"),
        "loco_csv": str(out / "loco_per_condition.csv"),
        "hard_class_csv": str(out / "hard_class_expert_compare.csv"),
        "runtime_csv": str(out / "runtime_breakdown.csv"),
        "caption_templates": str(out / "caption_templates.md"),
    }
