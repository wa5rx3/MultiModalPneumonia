#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PATHS_FILE="configs/paths.local.yaml"
if [ ! -f "$PATHS_FILE" ]; then
    echo "ERROR: $PATHS_FILE not found." >&2
    exit 1
fi

_yaml_val() {
    grep -E "^${1}:" "$PATHS_FILE" | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'\''
}

IV_ROOT="$(_yaml_val mimic_iv_root)"
LAB_DIR="$(_yaml_val labevents_dir)"

if [ -z "$IV_ROOT" ] || [ -z "$LAB_DIR" ]; then
    echo "ERROR: mimic_iv_root and labevents_dir must be set in $PATHS_FILE" >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"
MANIFESTS="artifacts/manifests"
TABLES="artifacts/tables"
LOGS="artifacts/logs"

mkdir -p "$TABLES" "$LOGS"

FEATURE_MAP="$TABLES/lab_feature_map.json"
if [ ! -f "$FEATURE_MAP" ]; then
    echo "ERROR: $FEATURE_MAP not found." >&2
    echo "Run build_lab_feature_candidates.py, review the output CSV, then manually" >&2
    echo "create $FEATURE_MAP mapping concept names to lists of MIMIC itemids." >&2
    exit 1
fi

echo "[1/5] link_cxr_to_admissions"
$PYTHON -m src.data.link_cxr_to_admissions \
    --admissions "$IV_ROOT/admissions.csv.gz"

echo "[2/5] extract_labevents_for_cohort"
$PYTHON -m src.data.extract_labevents_for_cohort \
    --labevents-dir "$LAB_DIR" \
    --cohort "$MANIFESTS/cxr_final_ed_cohort_with_temporal_split.parquet" \
    --feature-map "$FEATURE_MAP" \
    --output "$TABLES/cohort_labevents.parquet" \
    --report "$LOGS/cohort_labevents_report.json"

echo "[3/5] build_lab_features_from_labevents"
$PYTHON -m src.data.build_lab_features_from_labevents

echo "[4/5] build_clinical_labs_pneumonia_training_table"
$PYTHON -m src.data.build_clinical_labs_pneumonia_training_table

echo "[5/5] filter_to_lab_overlap"
$PYTHON -m src.data.filter_to_lab_overlap

echo ""
echo "Lab pipeline complete."
