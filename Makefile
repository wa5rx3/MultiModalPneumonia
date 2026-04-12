# Makefile — Multimodal Pneumonia Detection Pipeline
# Requires: .venv activated or use PYTHON variable below
PYTHON ?= python
SEED   := 42

# ─── Data pipeline ───────────────────────────────────────────────────────────
preprocess:
	bash scripts/run_data_pipeline.sh

preprocess_labs:
	bash scripts/run_lab_pipeline.sh

# ─── Pretraining ─────────────────────────────────────────────────────────────
pretrain:
	$(PYTHON) -m src.training.train_image_multilabel_pretrain

# ─── Image fine-tuning ───────────────────────────────────────────────────────
finetune_image:
	$(PYTHON) -m src.training.train_image_pneumonia_finetune

# ─── Multimodal training (canonical run) ─────────────────────────────────────
finetune_multimodal:
	$(PYTHON) -m src.training.train_multimodal_pneumonia

# ─── Clinical baselines ──────────────────────────────────────────────────────
train_clinical_lr:
	$(PYTHON) -m src.training.train_clinical_baseline --seed $(SEED)

train_clinical_xgb:
	$(PYTHON) -m src.training.train_clinical_xgb --seed $(SEED)

train_clinical: train_clinical_lr train_clinical_xgb

# ─── Evaluation ──────────────────────────────────────────────────────────────
bootstrap_delta:
	$(PYTHON) -m src.evaluation.bootstrap_eval \
	  --model-a artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
	  --model-b artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
	  --output-json artifacts/evaluation/bootstrap_multimodal_vs_image.json \
	  --n-bootstrap 2000 --seed $(SEED)

calibration:
	$(PYTHON) -m src.evaluation.calibration_analysis \
	  --output-dir artifacts/evaluation/calibration_final \
	  --n-bins 10 --bootstrap --n-bootstrap 2000 \
	  --model "Image" artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
	  --model "Multimodal" artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv

feature_ablation:
	$(PYTHON) scripts/collect_feature_ablation_results.py

evaluate: bootstrap_delta calibration feature_ablation

# ─── SHAP ────────────────────────────────────────────────────────────────────
shap:
	$(PYTHON) scripts/generate_shap_clinical.py

# ─── Publication report ───────────────────────────────────────────────────────
report:
	$(PYTHON) scripts/generate_publication_report.py

# ─── Testing ─────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ─── Full pipeline ────────────────────────────────────────────────────────────
all: pretrain finetune_image finetune_multimodal train_clinical evaluate shap report

.PHONY: preprocess preprocess_labs pretrain finetune_image finetune_multimodal \
        train_clinical_lr train_clinical_xgb train_clinical bootstrap_delta \
        calibration feature_ablation evaluate shap report all test
