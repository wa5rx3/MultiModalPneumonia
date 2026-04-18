"""Run image-only inference on non-ED MIMIC-CXR studies and compute bootstrap metrics.

This is an internal generalization check (not external validation): the DenseNet
backbone was pretrained on non-ED MIMIC-CXR studies, so predictions reflect
in-distribution generalization across ED vs. non-ED imaging contexts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cxr_binary_dataset import CXRBinaryDataset
from src.evaluation.bootstrap_eval import (
    bootstrap_patient_level,
    compute_metrics,
    summarize_bootstrap,
)
from src.training.train_image_pneumonia_finetune import build_model, build_transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-table",
        type=str,
        default="artifacts/manifests/nonED_image_eval_table.parquet",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "artifacts/models/"
            "image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/"
            "checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/evaluation/nonED_generalization_image_predictions.csv",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/evaluation/nonED_generalization_image.json",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(pretrained_ckpt_path=args.checkpoint)
    model = model.to(device)
    model.eval()

    _, eval_transform = build_transforms(args.image_size)

    df_meta = pd.read_parquet(args.eval_table)
    print(f"Eval table: {len(df_meta):,} rows, {df_meta['subject_id'].nunique():,} subjects")

    dataset = CXRBinaryDataset(table_path=args.eval_table, split="eval", transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_subject_ids: list[int] = []
    all_study_ids: list[int] = []
    all_targets: list[float] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            all_subject_ids.extend(batch["subject_id"].tolist())
            all_study_ids.extend(batch["study_id"].tolist())
            all_targets.extend(batch["target"].detach().cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    pred_df = pd.DataFrame({
        "subject_id": all_subject_ids,
        "study_id": all_study_ids,
        "target": [int(t) for t in all_targets],
        "pred_prob": all_probs,
    })

    pred_df["prob"] = pred_df["pred_prob"]

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pred_df.drop(columns=["prob"]).to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

    point = compute_metrics(pred_df)
    print(f"Point estimate — AUROC: {point['auroc']:.4f}, AUPRC: {point['auprc']:.4f}")

    boot_df, skipped = bootstrap_patient_level(
        pred_df, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed
    )
    boot_summary = summarize_bootstrap(boot_df)

    n_pos = int((pred_df["target"] == 1).sum())
    n_neg = int((pred_df["target"] == 0).sum())
    n_subjects = int(pred_df["subject_id"].nunique())

    result = {
        "checkpoint": args.checkpoint,
        "eval_table": args.eval_table,
        "n_total": int(len(pred_df)),
        "n_subjects": n_subjects,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": float(pred_df["target"].mean()),
        "point_estimate": point,
        "bootstrap": {
            "n_replicates": args.n_bootstrap,
            "n_skipped": int(skipped),
            "seed": args.bootstrap_seed,
            "auroc": boot_summary["auroc"],
            "auprc": boot_summary["auprc"],
        },
        "note": (
            "Non-ED MIMIC-CXR internal generalization. "
            "DenseNet backbone was pretrained on this population during multilabel pretraining. "
            "Label as internal generalization check, not external validation."
        ),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved results to {args.output_json}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
