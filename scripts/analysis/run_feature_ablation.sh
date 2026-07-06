#!/usr/bin/env bash
set -e
PY=.venv/Scripts/python.exe
TAB=artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet
BB=artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt
for grp in vitals_only vitals_plus_acuity no_missing_flags; do
  $PY -m src.training.train_multimodal_pneumonia \
    --input-table "$TAB" --image-backbone-checkpoint "$BB" \
    --output-dir "artifacts/models/multiseed/ablate_${grp}_seed42" \
    --lr-head 5e-5 --lr-backbone 1e-5 --epochs 30 --patience 8 --batch-size 16 \
    --image-size 224 --num-workers 4 --seed 42 --fusion-type concat \
    --tabular-feature-groups "$grp"
  rm -f "artifacts/models/multiseed/ablate_${grp}_seed42/checkpoints/epoch_"*.pt
done
echo "FEATURE ABLATION DONE"
