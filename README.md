# Multimodal Pneumonia Detection

BSc thesis project. The question: does adding structured ED triage data (vitals, acuity, missingness flags) to a chest X-ray model actually improve pneumonia detection, or does the image already capture everything useful?

Short answer: calibration improves, discrimination does not.

**Data:** MIMIC-CXR-JPG v2.1.0 + MIMIC-IV-ED (PhysioNet credentialed access required)  
**Cohort:** 9,137 ED-anchored studies, 80/10/10 patient-level temporal split, test set n = 1,075  
**Backbone:** DenseNet-121 pretrained on 14-label CheXpert task, fine-tuned for binary pneumonia

---

## Results

Test set metrics (`u_ignore` label policy, 2,000-replicate patient-level paired bootstrap).

| Model | AUROC | AUPRC | ECE |
|---|---|---|---|
| Logistic Regression (triage only) | 0.606 | 0.548 | 0.037 |
| XGBoost (triage only) | 0.611 | 0.567 | 0.046 |
| Image-only DenseNet-121 | **0.746** | **0.724** | 0.067 |
| Multimodal — Concat MLP | 0.736 | 0.714 | **0.040** |
| Multimodal — Attention Fusion | 0.737 | 0.710 | 0.132 |

The multimodal concat model does not significantly outperform image-only on discrimination (ΔAUROC = −0.009, 95% CI [−0.023, +0.005], P(Δ>0) = 0.10). The meaningful difference is calibration: ECE drops from 0.067 to 0.040, a 40% reduction. Paired bootstrap on ΔECE gives 95% CI [−0.041, +0.003] with P(ΔECE<0) = 0.961 — strong directional evidence, borderline by a zero-exclusion CI criterion. Post-hoc temperature scaling on the image model (T = 1.21) reduces image ECE only to 0.065, so the calibration gap is not recoverable via a single-scalar correction. Both deep-learning models beat the triage-only baselines by a large margin (P(Δ>0) = 1.0).

Attention fusion matched image-only on AUROC but was severely miscalibrated (ECE 0.132), making it unsuitable at this feature-set size.

Internal note on non-ED generalization: running the image model on non-ED MIMIC-CXR (n = 9,589) gives AUROC 0.534 [0.520, 0.548]. The backbone was pretrained on this population so this is not an independent test.

All numbers, CIs, and pairwise comparisons: [`artifacts/evaluation/final_publication_report.json`](artifacts/evaluation/final_publication_report.json)

---

## Structure

```
src/
  data/             cohort construction and feature engineering (19 scripts)
  datasets/         PyTorch Dataset classes for binary, multilabel, multimodal
  models/           DenseNet-121 backbone, TabularMLP, concat and attention fusion
  training/         training scripts for all model types
  evaluation/       bootstrap, calibration analysis, decision curve analysis
  interpretability/ Grad-CAM
  qc/               data quality checks

scripts/            figure generation, SHAP, temperature scaling, delta-ECE bootstrap, publication report aggregation
configs/
  experiments/      YAML configs for the main training runs
  paths.local.example.yaml

artifacts/
  evaluation/       bootstrap JSONs, calibration plots, ablation CSV, SHAP figures
  interpretability/ Grad-CAM overlays (TP, FP, FN cases)
  models/           config.json and metrics.json per run (weights not committed)

tests/              77 unit tests covering models, feature engineering, bootstrap eval
Dockerfile
Makefile
streamlit_app.py    dashboard for browsing run metrics and bootstrap results
```

---

## Cohort and methodology notes

**t₀ anchor:** CXR acquisition timestamp (DICOM StudyDate + StudyTime). Every clinical feature comes from the ED triage intake, which structurally precedes imaging. There are no post-t₀ features.

**Label policy:** CheXpert uncertain labels are excluded from training (`u_ignore`). Results under `u_zero` and `u_one` policies are included in the ablation.

**Split:** Patients ranked by their first t₀, then assigned 80/10/10. No patient appears in more than one split.

---

## Setup

```bash
pip install -r requirements_dev.txt
pip install -e . --no-deps
```

MIMIC data requires PhysioNet credentialed access. After downloading, set your local paths:

```bash
cp configs/paths.local.example.yaml configs/paths.local.yaml
# fill in MIMIC roots in paths.local.yaml
```

Build the cohort manifests from raw MIMIC data first (required before any training step):

```bash
bash scripts/run_data_pipeline.sh
```

Then run the training and evaluation pipeline:

```bash
make pretrain             # multilabel CheXpert pretraining
make finetune_image       # image-only binary pneumonia
make finetune_multimodal  # image + triage fusion
make train_clinical       # logistic regression and XGBoost baselines
make evaluate             # bootstrap, calibration, ablation
make shap                 # SHAP for XGBoost
make report               # write final_publication_report.json
make all                  # full pipeline (after run_data_pipeline.sh)
```

Docker (requires GPU):

```bash
docker build -t multimodal-pneumonia .
docker run --gpus all \
  -v /path/to/mimic_cxr_jpg:/workspace/mimic_cxr_jpg:ro \
  -v /path/to/mimic_iv_ed:/workspace/mimic_iv_ed:ro \
  -v /path/to/artifacts:/workspace/artifacts \
  multimodal-pneumonia make all
```

Trained weights (~800 MB–1.2 GB per checkpoint) are not in the repository. Contact the author for access.

---

## Tests

```bash
python -m pytest tests/ -v
```

77 tests, 1 skipped (requires the temporal constraint columns). Tests cover model forward passes, feature clipping bounds, bootstrap metric computation, and pipeline QC invariants. No MIMIC data required.

---

## Data compliance

Uses MIMIC-CXR-JPG v2.1.0, MIMIC-IV, and MIMIC-IV-ED under the PhysioNet Credentialed Health Data License. No patient-level data is committed to this repository. Reproducing results requires completing PhysioNet credentialing at [physionet.org](https://physionet.org).
 
