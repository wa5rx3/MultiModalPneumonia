# Multimodal Pneumonia Detection — MIMIC-CXR + MIMIC-IV/ED

BSc thesis codebase. Investigates whether combining chest X-rays with structured ED triage data improves pneumonia detection over image-alone, using a time-safe cohort anchored at the moment of CXR acquisition (t₀).

**Data:** MIMIC-CXR-JPG v2.1.0 + MIMIC-IV-ED (PhysioNet, credentialed access required)
**Backbone:** DenseNet121, pretrained on multilabel CheXpert labels, fine-tuned for binary pneumonia
**Cohort:** 9,137 ED-anchored studies · 80/10/10 patient-level temporal split · test *n* = 1,075

---

## Key Results

All metrics on the held-out test set (temporal split, `u_ignore` label policy). Bootstrap CIs are patient-level paired, *n* = 2,000 replicates.

| Model | AUROC | AUPRC | ECE |
|---|---|---|---|
| Clinical Logistic Regression | 0.606 | 0.548 | 0.037 |
| Clinical XGBoost | 0.611 | 0.567 | 0.046 |
| Image-only (DenseNet121) | **0.746** | **0.724** | 0.067 |
| Multimodal — Concat MLP | 0.736 | 0.714 | **0.040** |
| Multimodal — Attention Fusion | 0.737 | 0.710 | 0.132 |

**Primary finding:** The multimodal model does not significantly outperform image-only (ΔAUROC = −0.009, 95% CI [−0.023, +0.005], P(Δ>0) = 0.10). Its advantage is **calibration**: ECE 0.040 vs 0.067. Both deep learning models substantially and significantly outperform clinical baselines (P(Δ>0) = 1.0 for both comparisons).

**Non-ED generalisation (internal check only):** Image model on non-ED MIMIC-CXR (n = 9,589): AUROC 0.534, 95% CI [0.520, 0.548]. The DenseNet backbone was pretrained on this population — this is not an independent external validation.

Full metrics, bootstrap CIs, and calibration statistics: [`artifacts/evaluation/final_publication_report.json`](artifacts/evaluation/final_publication_report.json)

---

## Repository Structure

```
├── src/
│   ├── data/           # Cohort construction & feature engineering (19 scripts)
│   ├── datasets/       # PyTorch Dataset classes (binary, multilabel, multimodal)
│   ├── models/         # Model architectures (DenseNet121, TabularMLP, fusion)
│   ├── training/       # Training CLIs for all model types
│   ├── evaluation/     # Bootstrap, calibration, decision curve analysis
│   ├── interpretability/ # SHAP, Grad-CAM
│   └── qc/             # Data quality checks
│
├── scripts/            # Figure generation, SHAP, publication report
├── configs/
│   ├── experiments/    # Hyperparameter configs for headline runs
│   └── paths.local.example.yaml  # Copy → paths.local.yaml, set MIMIC roots
│
├── artifacts/
│   ├── evaluation/     # Bootstrap JSONs, calibration plots, SHAP figures,
│   │                   # ablation table, final_publication_report.json
│   ├── interpretability/ # Grad-CAM heatmaps (true positives, false negatives)
│   └── models/         # config.json + metrics.json per model run
│                       # (weights not committed — see Reproducibility below)
│
├── Dockerfile          # pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime base
├── Makefile            # Full pipeline: make all
└── streamlit_app.py    # Lightweight dashboard for inspecting run metrics
```

---

## Evaluation Artifacts

Everything needed to verify the thesis numbers is in `artifacts/evaluation/`:

| File / Directory | Contents |
|---|---|
| `final_publication_report.json` | All locked metrics: AUROC, AUPRC, ECE, Brier, bootstrap CIs, pairwise ΔAUROCs |
| `bootstrap_*.json` | Paired bootstrap outputs for each model comparison |
| `calibration_final/` | Reliability diagrams + ECE/Brier/H-L statistics |
| `feature_ablation_results.csv` | AUROC across 16 feature ablation configurations |
| `shap/` | SHAP beeswarm + bar charts; top feature: O₂ saturation (mean |SHAP| = 0.148) |
| `interpretability/gradcam_val_fn/` | Grad-CAM overlays for false-negative cases |

---

## Design Decisions

**t₀ anchor:** CXR acquisition datetime (DICOM StudyDate + StudyTime). All clinical features are ED triage intake measurements — structurally preceding imaging. No post-t₀ features.

**Label policy:** CheXpert uncertain labels treated as negative (`u_ignore` = exclude uncertain studies from training). Sensitivity to this choice tested via `u_zero` and `u_one` variants.

**Split:** Patient-level ordinal rank by first t₀. Each patient belongs to exactly one split. No patient-level leakage.

**Fusion:** Concat MLP (DenseNet image embedding + TabularMLP embedding → joint head). Attention fusion tested as ablation — similar AUROC but severely miscalibrated (ECE 0.132).

---

## Reproducing the Pipeline

### Requirements

```bash
pip install -r requirements.txt
# or: pip install -e .
```

MIMIC data requires PhysioNet credentialed access. Once downloaded, copy and fill in the paths config:

```bash
cp configs/paths.local.example.yaml configs/paths.local.yaml
# edit paths.local.yaml with your local MIMIC roots
```

### Pipeline

```bash
make pretrain           # DenseNet121 multilabel pretraining on MIMIC-CXR
make finetune_image     # Binary pneumonia fine-tuning (image-only)
make finetune_multimodal  # Multimodal (image + triage vitals)
make train_clinical     # Logistic regression + XGBoost baselines
make evaluate           # Bootstrap, calibration, feature ablation
make shap               # SHAP values for clinical XGBoost
make report             # Aggregate final_publication_report.json
```

Or run the full pipeline end-to-end:

```bash
make all
```

### Docker

```bash
docker build -t multimodal-pneumonia .
docker run --gpus all \
  -v /path/to/mimic_cxr_jpg:/workspace/mimic_cxr_jpg:ro \
  -v /path/to/mimic_iv_ed:/workspace/mimic_iv_ed:ro \
  -v /path/to/artifacts:/workspace/artifacts \
  multimodal-pneumonia make all
```

### Model Weights

Trained model weights (`.pt` checkpoints, ~800 MB–1.2 GB each) are not stored in this repository. To inspect or run inference with the trained models, contact the author.

---

## Compliance

This project uses MIMIC-CXR-JPG v2.1.0, MIMIC-IV, and MIMIC-IV-ED under the PhysioNet Credentialed Health Data License. No patient-level data is included in this repository. Researchers wishing to reproduce results must complete PhysioNet credentialing at [physionet.org](https://physionet.org).
