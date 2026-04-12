#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PATHS_FILE="configs/paths.local.yaml"
if [ ! -f "$PATHS_FILE" ]; then
    echo "ERROR: $PATHS_FILE not found. Copy configs/paths.local.example.yaml and fill in your paths." >&2
    exit 1
fi

_yaml_val() {
    grep -E "^${1}:" "$PATHS_FILE" | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'\''
}

CXR_ROOT="$(_yaml_val mimic_cxr_root)"
IV_ROOT="$(_yaml_val mimic_iv_root)"
ED_ROOT="$(_yaml_val mimic_iv_ed_root)"
LAB_DIR="$(_yaml_val labevents_dir)"

if [ -z "$CXR_ROOT" ] || [ -z "$ED_ROOT" ]; then
    echo "ERROR: mimic_cxr_root and mimic_iv_ed_root must be set in $PATHS_FILE" >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"
MANIFESTS="artifacts/manifests"
TABLES="artifacts/tables"
LOGS="artifacts/logs"

mkdir -p "$MANIFESTS" "$TABLES" "$LOGS/qc" artifacts/models artifacts/evaluation

echo "[1/19] build_cohort"
$PYTHON -m src.data.build_cohort \
    --base-root "$CXR_ROOT" \
    --metadata-root "$CXR_ROOT"

echo "[2/19] build_primary_imaging_cohort"
$PYTHON -m src.data.build_primary_imaging_cohort

echo "[3/19] link_cxr_to_edstays"
$PYTHON -m src.data.link_cxr_to_edstays \
    --edstays "$ED_ROOT/edstays.csv.gz"

echo "[4/19] build_final_ed_cohort"
$PYTHON -m src.data.build_final_ed_cohort

echo "[5/19] build_temporal_patient_split"
$PYTHON -m src.data.build_temporal_patient_split

echo "[6/19] link_cxr_to_triage"
$PYTHON -m src.data.link_cxr_to_triage \
    --triage "$ED_ROOT/triage.csv.gz"

echo "[7/19] build_triage_features"
$PYTHON -m src.data.build_triage_features

echo "[8/19] build_triage_model_table"
$PYTHON -m src.data.build_triage_model_table

echo "[9/19] build_pneumonia_labels_from_chexpert"
$PYTHON -m src.data.build_pneumonia_labels_from_chexpert \
    --metadata-root "$CXR_ROOT"

echo "[10/19] build_pneumonia_training_table (u_ignore)"
$PYTHON -m src.data.build_pneumonia_training_table \
    --policy u_ignore \
    --output "$MANIFESTS/cxr_pneumonia_training_table_u_ignore.parquet" \
    --report "$MANIFESTS/cxr_pneumonia_training_table_u_ignore_report.json"

echo "[11/19] build_pneumonia_training_table (u_one)"
$PYTHON -m src.data.build_pneumonia_training_table \
    --policy u_one \
    --output "$MANIFESTS/cxr_pneumonia_training_table_u_one.parquet" \
    --report "$MANIFESTS/cxr_pneumonia_training_table_u_one_report.json"

echo "[12/19] build_clinical_pneumonia_training_table (u_ignore)"
$PYTHON -m src.data.build_clinical_pneumonia_training_table \
    --label-table "$MANIFESTS/cxr_pneumonia_training_table_u_ignore.parquet" \
    --output "$MANIFESTS/cxr_clinical_pneumonia_training_table_u_ignore.parquet" \
    --report "$MANIFESTS/cxr_clinical_pneumonia_training_table_u_ignore_report.json"

echo "[13/19] build_clinical_pneumonia_training_table (u_one)"
$PYTHON -m src.data.build_clinical_pneumonia_training_table \
    --label-table "$MANIFESTS/cxr_pneumonia_training_table_u_one.parquet" \
    --output "$MANIFESTS/cxr_clinical_pneumonia_training_table_u_one.parquet" \
    --report "$MANIFESTS/cxr_clinical_pneumonia_training_table_u_one_report.json"

echo "[14/19] apply_temporal_split (u_ignore)"
$PYTHON -m src.data.apply_temporal_split \
    --base-cohort "$MANIFESTS/cxr_final_ed_cohort_with_temporal_split.parquet" \
    --input-table "$MANIFESTS/cxr_clinical_pneumonia_training_table_u_ignore.parquet" \
    --output-table "$MANIFESTS/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet" \
    --report "$MANIFESTS/cxr_clinical_pneumonia_training_table_u_ignore_temporal_report.json"

echo "[15/19] apply_temporal_split (u_one)"
$PYTHON -m src.data.apply_temporal_split \
    --base-cohort "$MANIFESTS/cxr_final_ed_cohort_with_temporal_split.parquet" \
    --input-table "$MANIFESTS/cxr_clinical_pneumonia_training_table_u_one.parquet" \
    --output-table "$MANIFESTS/cxr_clinical_pneumonia_training_table_u_one_temporal.parquet" \
    --report "$MANIFESTS/cxr_clinical_pneumonia_training_table_u_one_temporal_report.json"

echo "[16/19] build_image_pneumonia_finetune_table"
$PYTHON -m src.data.build_image_pneumonia_finetune_table \
    --label-table "$MANIFESTS/cxr_pneumonia_training_table_u_ignore.parquet"

echo "[17/19] build_image_pretraining_split"
$PYTHON -m src.data.build_image_pretraining_split

echo "[18/19] build_image_multilabel_pretrain_table"
$PYTHON -m src.data.build_image_multilabel_pretrain_table \
    --metadata-root "$CXR_ROOT"

echo "[19/19] build_nonED_image_eval_table"
$PYTHON -m src.data.build_nonED_image_eval_table \
    --chexpert-labels "$CXR_ROOT/mimic-cxr-2.0.0-chexpert.csv.gz"

echo ""
echo "Main pipeline complete. Manifests written to $MANIFESTS/"
echo ""
echo "To run the lab ablation pipeline (optional, needed for +labs models):"
echo "  See scripts/run_lab_pipeline.sh"
