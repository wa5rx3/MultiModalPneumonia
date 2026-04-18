from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

from src.interpretability.gradcam import run_gradcam


def build_model(checkpoint_path: str) -> torch.nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, 1)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def resolve_target_layer(model: torch.nn.Module, layer_path: str) -> torch.nn.Module:
    current: torch.nn.Module = model
    for part in layer_path.split("."):
        if not hasattr(current, part):
            raise ValueError(f"Invalid --target-layer '{layer_path}': missing '{part}' in path.")
        current = getattr(current, part)
    return current


def validate_predictions_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in predictions CSV: {missing}")


def select_examples(df: pd.DataFrame, mode: str, threshold: float, top_k: int) -> pd.DataFrame:
    pred_pos = df["pred_prob"] >= threshold
    target_pos = df["target"] == 1

    if mode == "fp":
        subset = df[(pred_pos) & (~target_pos)]
        return subset.sort_values("pred_prob", ascending=False).head(top_k)
    if mode == "tp":
        subset = df[(pred_pos) & (target_pos)]
        return subset.sort_values("pred_prob", ascending=False).head(top_k)


    subset = df[(~pred_pos) & (target_pos)].copy()
    subset["distance_to_threshold"] = threshold - subset["pred_prob"]
    subset = subset.sort_values(["distance_to_threshold", "pred_prob"], ascending=[True, False]).head(top_k)
    return subset.drop(columns=["distance_to_threshold"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--mode", type=str, choices=["fp", "tp", "fn"], default="fp")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-layer", type=str, default="features.norm5")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions_csv)
    validate_predictions_columns(pred, required=["image_path", "target", "pred_prob"])
    pred["target"] = pred["target"].astype(int)
    pred["pred_prob"] = pred["pred_prob"].astype(float)
    pred = pred[pred["pred_prob"].between(0.0, 1.0, inclusive="both")].copy()

    df = select_examples(pred, mode=args.mode, threshold=args.threshold, top_k=args.top_k)
    if df.empty:
        raise RuntimeError(
            f"No rows match mode={args.mode} at threshold={args.threshold}. "
            "Try a different mode/threshold or verify predictions CSV."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.checkpoint).to(device)
    target_layer = resolve_target_layer(model, args.target_layer)
    transform = build_transform(image_size=args.image_size)

    selection_summary = {
        "mode": args.mode,
        "threshold": args.threshold,
        "top_k_requested": args.top_k,
        "top_k_selected": int(len(df)),
        "target_layer": args.target_layer,
        "image_size": args.image_size,
        "alpha": args.alpha,
        "checkpoint": args.checkpoint,
        "predictions_csv": args.predictions_csv,
        "device": str(device),
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")

    for i, row in enumerate(df.itertuples(index=False), start=1):
        image_path = Path(row.image_path)
        if not image_path.is_file():
            print(f"Skipping missing image: {image_path}")
            continue

        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        result = run_gradcam(
            model=model,
            target_layer=target_layer,
            input_tensor=tensor,
            original_tensor=tensor.detach().cpu(),
            class_idx=None,
            alpha=args.alpha,
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(result.image_rgb)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(result.heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(result.overlay_rgb)
        pred_label = int(row.pred_prob >= args.threshold)
        axes[2].set_title(
            f"Overlay\npred={row.pred_prob:.3f} pred_y={pred_label} target={row.target}"
        )
        axes[2].axis("off")

        save_path = out_dir / f"{args.mode}_{i:02d}_study_{getattr(row, 'study_id', 'na')}.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        np.save(
            heatmap_dir / f"{args.mode}_{i:02d}_study_{getattr(row, 'study_id', 'na')}_heatmap.npy",
            result.heatmap.astype(np.float32),
        )

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()