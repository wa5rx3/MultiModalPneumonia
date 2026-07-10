# Multimodal Pneumonia Detection

BSc thesis project, extended into a journal manuscript. The question: does adding structured ED triage data (vitals, acuity, missingness flags) to a chest X-ray model actually improve pneumonia detection, or does the image already capture everything useful?

Short answer: under multi-seed evaluation, discrimination does not improve, and the one apparent calibration gain is fragile across seeds and is recovered by a single-parameter recalibration of the image model alone.

**Data:** MIMIC-CXR-JPG v2.1.0 + MIMIC-IV v2.2 + MIMIC-IV-ED v2.2 (PhysioNet credentialed access required)  
**Cohort:** 9,154 ED-anchored studies, patient-level temporal 80/10/10 split (train 7,144 / val 930 / test 1,080); evaluated test set n = 1,075 (prevalence 45.3%)  
**Backbone:** DenseNet-121 pretrained on the multilabel CheXpert-style task using 182,637 non-ED MIMIC-CXR-JPG studies, then fine-tuned for binary pneumonia

---

## Results

Test-set performance, mean ± SD over five random seeds (`u_ignore` label policy; patient-level paired bootstrap for within-checkpoint intervals).

| Model | AUROC | AUPRC | ECE |
|---|---|---|---|
| Logistic Regression (triage only) | 0.606 | 0.548 | 0.037 |
| XGBoost (triage only) | 0.611 | 0.567 | 0.046 |
| Image-only DenseNet-121 | 0.737 ± 0.003 | 0.719 ± 0.004 | 0.053 ± 0.008 |
| Multimodal — concat (triage) | 0.741 ± 0.006 | 0.715 ± 0.005 | 0.040 ± 0.014 |
| Multimodal — attention (triage) | 0.738 ± 0.006 | 0.713 ± 0.012 | 0.076 ± 0.042 |
| Multimodal — concat (+ labs) | 0.747 ± 0.004 | 0.724 ± 0.004 | 0.043 ± 0.011 |

Adding triage vitals does not change discrimination: the paired concat − image ΔAUROC is +0.004 ± 0.009 across seeds, and a two-one-sided test (TOST) against a ±0.05 margin confirms equivalence (90% bootstrap CI [+0.000, +0.012]). Both deep models beat the triage-only baselines by more than 12 AUROC points.

A calibration advantage that looks large from a single checkpoint (image ECE 0.067 vs concat 0.040) shrinks to ΔECE −0.013 ± 0.016 across five seeds, and temperature scaling of the image model alone (T = 1.37) lowers its ECE from 0.060 to 0.042, matching concat. The calibration gap therefore does not motivate fusion. Attention fusion matches image AUROC but is badly and consistently miscalibrated (single-checkpoint ECE 0.132).

The small `+labs` gain (+0.009 ± 0.005) is reproduced entirely by laboratory missingness indicators, not the measured values (flags-only ablation), and is not leakage. The image model externally validates on NIH ChestX-ray14 (AUROC 0.722 ± 0.005, 112,120 radiographs).

Multi-seed metrics, bootstrap deltas and equivalence tests: `artifacts/evaluation/multiseed/` and `artifacts/evaluation/equivalence/`.

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

Trained weights (~800 MB–1.2 GB per checkpoint) are not redistributed under the PhysioNet Data Use Agreement. To obtain weights, complete PhysioNet credentialing and re-run `make all` after configuring `configs/paths.local.yaml`.

---

## Tests

```bash
python -m pytest tests/ -v
```

77 tests, 1 skipped (requires the temporal constraint columns). Tests cover model forward passes, feature clipping bounds, bootstrap metric computation, and pipeline QC invariants. No MIMIC data required.

---

## Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard reads pre-computed metrics from `artifacts/evaluation/` and runs without MIMIC access for browsing results, calibration plots, bootstrap comparisons, and Grad-CAM overlays.

---

## Data compliance

Uses MIMIC-CXR-JPG v2.1.0, MIMIC-IV v2.2, and MIMIC-IV-ED v2.2 under the PhysioNet Credentialed Health Data License. No patient-level data is committed to this repository. Reproducing results requires completing PhysioNet credentialing at [physionet.org](https://physionet.org).
 
