#!/usr/bin/env bash
# Label-policy sensitivity: train image-only and concat-multimodal under the
# u_zero and u_one CheXpert-uncertainty policies (u_ignore is the main result).
# Seed 42 (representative), using the per-policy clinical tables.
set -e
PY=.venv/Scripts/python.exe
M=artifacts/manifests
BB=artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt
OUT=artifacts/models/multiseed

for pol in u_zero u_one; do
  TAB=$M/cxr_clinical_pneumonia_training_table_${pol}_temporal.parquet
  # image-only (uses image + target from the same table)
  $PY -m src.training.train_image_pneumonia_finetune \
    --input-table "$TAB" --pretrained-checkpoint "$BB" \
    --output-dir "$OUT/image_${pol}_seed42" \
    --lr-head 5e-5 --lr-backbone 1e-5 --epochs 40 --patience 10 --batch-size 16 \
    --image-size 224 --num-workers 4 --seed 42
  rm -f "$OUT/image_${pol}_seed42/checkpoints/epoch_"*.pt
  # concat multimodal
  $PY -m src.training.train_multimodal_pneumonia \
    --input-table "$TAB" --image-backbone-checkpoint "$BB" \
    --output-dir "$OUT/concat_${pol}_seed42" \
    --lr-head 5e-5 --lr-backbone 1e-5 --epochs 30 --patience 8 --batch-size 16 \
    --image-size 224 --num-workers 4 --seed 42 --fusion-type concat \
    --tabular-feature-groups all
  rm -f "$OUT/concat_${pol}_seed42/checkpoints/epoch_"*.pt
done
echo "LABEL SENSITIVITY DONE"
