# Model artifacts (canonical)

**Active headline runs** (everything else lives under `artifacts/archive/models/`):

| Directory | Role |
|-----------|------|
| `clinical_baseline_u_ignore_temporal_strong_v2` | Triage-only logistic regression |
| `clinical_xgb_u_ignore_temporal_strong_v2` | Triage-only XGBoost |
| `image_multilabel_pretrain_densenet121_strong_v2` | **Upstream** multilabel DenseNet checkpoint (`checkpoints/best.pt`) used to initialize the two runs below |
| `image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3` | Image-only pneumonia fine-tune |
| `multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3` | Multimodal (triage + image) |

**Evaluation companions** (repo root, not under `models/`):

- Bootstrap: `artifacts/evaluation/bootstrap_*_stronger_lr_v3.json`
- Calibration: `artifacts/evaluation/calibration_stronger_lr_v3/`

**Archive:** Previous phase1 runs, ImageNet-only, labs baselines, `image_multilabel_pretrain_densenet121_main`, etc. → `artifacts/archive/models/from_models_root_2026_03/`.

Neural `.pt` checkpoints may be gitignored locally; keep `config.json`, `summary.json`, `history.json`, and `tabular_preprocessor.joblib` (multimodal) with the run.
