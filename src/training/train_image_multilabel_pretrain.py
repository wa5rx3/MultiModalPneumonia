from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from src.datasets.cxr_multilabel_dataset import CHEXPERT_LABEL_COLS, CXRMultilabelDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MaskedBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        mask = mask.float()

        loss = self.loss_fn(logits, targets)
        loss = loss * mask

        denom = mask.sum()
        if denom.item() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        return loss.sum() / denom


def build_model(num_labels: int) -> nn.Module:
    model = models.densenet121(weights="IMAGENET1K_V1")
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_labels)
    return model


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
                p=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, eval_transform


def compute_label_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in CHEXPERT_LABEL_COLS:
        vals = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "non_missing": int(vals.notna().sum()),
            "positive": int((vals == 1).sum()),
            "uncertain": int((vals == -1).sum()),
            "negative": int((vals == 0).sum()),
        }
    return stats


def compute_micro_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    if y_true.size == 0:
        return {"micro_auroc": None, "micro_auprc": None, "valid_targets": 0}

    unique = np.unique(y_true)
    micro_auroc = float(roc_auc_score(y_true, y_prob)) if len(unique) > 1 else None
    micro_auprc = float(average_precision_score(y_true, y_prob)) if len(unique) > 1 else None

    return {
        "micro_auroc": micro_auroc,
        "micro_auprc": micro_auprc,
        "valid_targets": int(y_true.size),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_batches = 0
    total_masked_labels = 0.0
    use_autocast = use_amp and device.type == "cuda"

    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()

        if use_autocast:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets, mask)
        else:
            logits = model(images)
            loss = criterion(logits, targets, mask)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)

        if mask_np.any():
            all_targets.append(targets_np[mask_np].astype(np.float32))
            all_probs.append(probs[mask_np].astype(np.float32))

        total_loss += float(loss.item())
        total_batches += 1
        total_masked_labels += float(mask.sum().item())

    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "masked_labels_used": int(total_masked_labels),
    }

    if all_targets and all_probs:
        y_true = np.concatenate(all_targets)
        y_prob = np.concatenate(all_probs)
        metrics.update(compute_micro_metrics(y_true, y_prob))
    else:
        metrics.update({"micro_auroc": None, "micro_auprc": None, "valid_targets": 0})

    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    max_grad_norm: float | None = 1.0,
) -> dict:
    model.train()

    total_loss = 0.0
    total_batches = 0
    total_masked_labels = 0.0
    use_amp = scaler is not None and device.type == "cuda"

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets, mask)

            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets, mask)

            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        total_masked_labels += float(mask.sum().item())

    return {
        "loss": total_loss / max(total_batches, 1),
        "masked_labels_used": int(total_masked_labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-table",
        type=str,
        default="artifacts/manifests/mimic_cxr_multilabel_pretrain_table.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/image_multilabel_pretrain_densenet121_strong_v2",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="If set, use for backbone and head (overrides --lr-head / --lr-backbone).",
    )
    parser.add_argument("--lr-head", type=float, default=1e-4, dest="lr_head")
    parser.add_argument("--lr-backbone", type=float, default=3e-5, dest="lr_backbone")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_micro_auprc"],
    )
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_parquet(args.input_table).copy()
    if "pretrain_split" not in df.columns:
        raise ValueError(f"Expected 'pretrain_split' in {args.input_table}")

    df_train = df[df["pretrain_split"] == "pretrain_train"].copy()
    df_val = df[df["pretrain_split"] == "pretrain_internal_val"].copy()

    if len(df_train) == 0 or len(df_val) == 0:
        raise ValueError(
            "One or more pretraining splits are empty. "
            f"pretrain_train={len(df_train)}, pretrain_internal_val={len(df_val)}"
        )

    train_transform, eval_transform = build_transforms(args.image_size)

    train_dataset = CXRMultilabelDataset(
        table_path=args.input_table,
        split="pretrain_train",
        transform=train_transform,
    )
    val_dataset = CXRMultilabelDataset(
        table_path=args.input_table,
        split="pretrain_internal_val",
        transform=eval_transform,
    )

    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )

    model = build_model(num_labels=len(CHEXPERT_LABEL_COLS)).to(device)
    criterion = MaskedBCELoss()

    lr_head = float(args.lr) if args.lr is not None else float(args.lr_head)
    lr_backbone = float(args.lr) if args.lr is not None else float(args.lr_backbone)

    optimizer = AdamW(
        [
            {"params": model.features.parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    if args.selection_metric == "val_loss":
        scheduler_mode = "min"
        scheduler_patience = 2
    else:
        scheduler_mode = "max"
        scheduler_patience = 2

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=0.5,
        patience=scheduler_patience,
        min_lr=1e-6,
    )

    use_amp = torch.cuda.is_available() and not args.disable_amp
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    config = {
        "input_table": args.input_table,
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_head": lr_head,
        "lr_backbone": lr_backbone,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_grad_norm": args.max_grad_norm,
        "use_amp": bool(use_amp),
        "device": str(device),
        "model_name": "densenet121",
        "selection_metric": args.selection_metric,
        "num_labels": len(CHEXPERT_LABEL_COLS),
        "label_cols": CHEXPERT_LABEL_COLS,
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "train_label_stats": compute_label_stats(df_train),
        "val_label_stats": compute_label_stats(df_val),
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if args.selection_metric == "val_loss":
        best_value = math.inf
    else:
        best_value = -math.inf

    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        if args.selection_metric == "val_loss":
            current_value = float(val_metrics["loss"])
        else:
            current_value = (
                float(val_metrics["micro_auprc"])
                if val_metrics["micro_auprc"] is not None
                else 0.0
            )

        scheduler.step(current_value)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_masked_labels_used": train_metrics["masked_labels_used"],
            "val_loss": val_metrics["loss"],
            "val_masked_labels_used": val_metrics["masked_labels_used"],
            "val_micro_auroc": val_metrics["micro_auroc"],
            "val_micro_auprc": val_metrics["micro_auprc"],
            "val_valid_targets": val_metrics["valid_targets"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(epoch_record)

        print(json.dumps(epoch_record, indent=2))

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_micro_auroc": val_metrics["micro_auroc"],
                "val_micro_auprc": val_metrics["micro_auprc"],
                "config": config,
            },
            checkpoints_dir / f"epoch_{epoch}.pt",
        )

        improved = (
            current_value < best_value
            if args.selection_metric == "val_loss"
            else current_value > best_value
        )

        if improved:
            best_value = current_value
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_micro_auroc": val_metrics["micro_auroc"],
                    "val_micro_auprc": val_metrics["micro_auprc"],
                    "config": config,
                },
                checkpoints_dir / "best.pt",
            )
            print(
                f"New best checkpoint saved at epoch {epoch} "
                f"with {args.selection_metric}={best_value:.6f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No {args.selection_metric} improvement for {epochs_without_improvement} epoch(s). "
                f"Best so far: epoch {best_epoch}, value={best_value:.6f}"
            )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping triggered after epoch {epoch}. "
                f"Best epoch was {best_epoch} with {args.selection_metric}={best_value:.6f}"
            )
            break

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_ckpt_path = checkpoints_dir / "best.pt"
    if not best_ckpt_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")

    summary = {
        "best_selection_metric": args.selection_metric,
        "best_value": float(best_value),
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPretraining complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()