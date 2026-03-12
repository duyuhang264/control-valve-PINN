# Experiment Protocol

## Primary Protocols
- Grouped random split (file-level, stratified by fault)
- LOCO (leave-one-condition-out)

## Primary Metrics
- Accuracy
- Macro-F1
- Balanced Accuracy
- Per-class Recall
- 20-class confusion matrix
- Reported at file-level (window probabilities averaged by `file_id`)

## Baselines
- CNN
- Transformer
- Raw+TF (no PINN)
- PINN+XGBoost

## Ablations
- no_residual_branch
- no_tf_branch
- concat_fusion (no cross-attention)
- no_domain_adversarial
- no_physics_loss

## Leakage Rule
All splits are file-level. Windows inherit `file_id`, and no `file_id` can appear in both train and test.
