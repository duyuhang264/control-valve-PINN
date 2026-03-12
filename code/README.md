# PG-M3F (Physics-Guided Multi-source Fusion)

This repository implements the `PG-M3F` framework for DAMADICS control-valve fault diagnosis with:

- Multi-source fusion: raw sequence + PINN residual + time-frequency branch
- Cross-attention fusion and gated aggregation
- Domain-adversarial condition invariance (GRL)
- Two protocols: grouped random split and LOCO (leave-one-condition-out)
- File-level evaluation via mean-softmax aggregation from window predictions

## Core Interfaces

- `build_manifest(data_root) -> pd.DataFrame[SampleMeta]`
- `make_splits(manifest, protocol) -> dict[str, list[file_id]]`
- `PINNModule.forward(x) -> (x_hat, f_hat, residual_dict)`
- `PGM3F.forward(batch) -> logits`
- `train_one_epoch(model, loader, cfg) -> dict(loss_items)`
- `evaluate(model, loader) -> dict(metrics)`

## Data Format

Expected CSV columns:

`time,CV,F,X_prime,P1_prime,P2_prime,T_prime`

Expected folder/file pattern:

- Folder: `faultX-conditionY`
- File: `faultX-conditionY-Z.csv` or `faultX-Z.csv`

## Quick Start

```bash
python scripts/build_manifest.py --data-root "D:/path/to/data" --output-dir outputs
python scripts/run_experiment.py --data-root "D:/path/to/data" --output-dir outputs --protocol both
```

## Outputs

- `manifest.csv`
- `data_quality_report.csv`
- `summary.json`
- per-fold metrics and scaler files
- `paper_materials/` with main tables, LOCO per-condition results, confusion matrix CSV, caption templates

## Notes

- Split unit is always file-level (`file_id`), preventing train/test leakage across windows.
- Source sampling rate is auto-detected from manifest median `fs_hz` by default.
- Sliding windows default to `4s` with `2s` stride; training uses `12 windows/file/epoch` sampling on CPU.
- Training iterates file-grouped samples (dataset-level epoch shuffle), reducing disk cache thrashing on CPU.
- Preprocessed arrays are cached in `output_dir/preprocessed_cache` by default and reused across folds.
- Best-epoch checkpoint is selected by `macro_f1` (configurable), instead of always using the last epoch.
