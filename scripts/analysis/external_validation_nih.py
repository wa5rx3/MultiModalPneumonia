"""External validation on NIH ChestX-ray14 (P4c).

Runs OUR fine-tuned image-only DenseNet-121 models (the 5 multi-seed checkpoints)
on the NIH ChestX-ray14 dataset -- a different institution, scanner population, and
label pipeline -- to test cross-dataset generalization. This is true external
validation (our model -> outside data), addressing the single biggest reviewer
objection to a single-institution MIMIC study.

Label: NIH 'Finding Labels' contains 'Pneumonia' -> positive. Preprocessing matches
our MIMIC eval transform exactly (Resize 224, ImageNet normalize, 3-channel) so the
only thing that changes is the data distribution.

Usage:
  python -m scripts.analysis.external_validation_nih \
    --images-dir  D:/nih_cxr/images \
    --labels-csv  D:/nih_cxr/Data_Entry_2017.csv \
    [--limit N] [--batch-size 32] [--device cuda]

Output: artifacts/evaluation/external_nih/*.{json,csv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torchvision import models, transforms

from src.evaluation.calibration_analysis import compute_ece_mce

OUT = Path("artifacts/evaluation/external_nih")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_RUNS = [f"artifacts/models/multiseed/image_seed{s}" for s in (42, 123, 456, 789, 1000)]


def eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_image_model(ckpt_path: Path, device) -> nn.Module:
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 1)
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval()


def index_images(images_dir: Path) -> dict[str, Path]:
    """Map NIH filename (e.g. 00000001_000.png) -> full path, recursively."""
    idx = {}
    for p in images_dir.rglob("*.png"):
        idx.setdefault(p.name, p)
    return idx


def metrics(y, p) -> dict:
    ece, _, _ = compute_ece_mce(np.asarray(y), np.asarray(p), n_bins=10)
    return {"n": int(len(y)), "positive_rate": float(np.mean(y)),
            "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "ece": float(ece), "brier": float(brier_score_loss(y, p))}


@torch.no_grad()
def infer(model, paths, tfm, device, batch_size) -> np.ndarray:
    probs = []
    buf = []
    def flush():
        if not buf:
            return
        t = torch.stack(buf).to(device)
        probs.append(torch.sigmoid(model(t)).squeeze(1).cpu().numpy())
        buf.clear()
    for p in paths:
        img = Image.open(p).convert("RGB")
        buf.append(tfm(img))
        if len(buf) >= batch_size:
            flush()
    flush()
    return np.concatenate(probs)


def summarize(vals) -> dict:
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "values": [float(x) for x in a]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("Indexing images ...", flush=True)
    idx = index_images(Path(args.images_dir))
    print(f"  found {len(idx)} png files on disk", flush=True)

    lab = pd.read_csv(args.labels_csv)
    # NIH columns: 'Image Index', 'Finding Labels'
    col_img = "Image Index" if "Image Index" in lab.columns else lab.columns[0]
    col_lbl = "Finding Labels" if "Finding Labels" in lab.columns else lab.columns[1]
    lab = lab[[col_img, col_lbl]].copy()
    lab["path"] = lab[col_img].map(idx)
    lab = lab.dropna(subset=["path"])
    lab["target"] = lab[col_lbl].astype(str).str.contains("Pneumonia").astype(int)
    if args.limit:
        lab = lab.head(args.limit)
    paths = lab["path"].tolist()
    y = lab["target"].to_numpy()
    print(f"Evaluating on {len(paths)} NIH images | pneumonia prevalence {y.mean():.4f}", flush=True)
    if len(np.unique(y)) < 2:
        raise SystemExit("Only one class present; cannot compute AUROC.")

    tfm = eval_transform()
    per_seed = {}
    prob_accum = np.zeros(len(paths), dtype=np.float64)
    for run in args.runs:
        ckpt = Path(run) / "checkpoints" / "best.pt"
        if not ckpt.is_file():
            print(f"  [skip] no checkpoint at {ckpt}", flush=True)
            continue
        seed = Path(run).name
        print(f"  inferring {seed} ...", flush=True)
        model = build_image_model(ckpt, device)
        p = infer(model, paths, tfm, device, args.batch_size)
        per_seed[seed] = metrics(y, p)
        prob_accum += p
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not per_seed:
        raise SystemExit("No image checkpoints found.")

    # across-seed summary of each metric + the seed-averaged ensemble
    ens = metrics(y, prob_accum / len(per_seed))
    summary = {
        "dataset": "NIH ChestX-ray14",
        "n_images": len(paths),
        "pneumonia_prevalence": float(y.mean()),
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "across_seed": {m: summarize([per_seed[s][m] for s in per_seed])
                        for m in ["auroc", "auprc", "ece", "brier"]},
        "seed_ensemble": ens,
        "internal_reference": {"image_only_auroc_mean": 0.7373, "note": "our MIMIC test AUROC (5 seeds)"},
    }
    with open(OUT / "external_nih_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([{"seed": s, **per_seed[s]} for s in per_seed]).to_csv(OUT / "external_nih_per_seed.csv", index=False)

    print("\n=== NIH ChestX-ray14 external validation (our image model) ===")
    a = summary["across_seed"]
    print(f"  across 5 seeds: AUROC {a['auroc']['mean']:.4f}+/-{a['auroc']['std']:.4f}  "
          f"AUPRC {a['auprc']['mean']:.4f}  ECE {a['ece']['mean']:.4f}")
    print(f"  seed-ensemble : AUROC {ens['auroc']:.4f}  AUPRC {ens['auprc']:.4f}")
    print(f"  (internal MIMIC test AUROC ~0.737) -> external drop = "
          f"{0.7373 - a['auroc']['mean']:+.4f}")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
