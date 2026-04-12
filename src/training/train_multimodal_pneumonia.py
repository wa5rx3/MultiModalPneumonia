from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets.cxr_multimodal_dataset import CXRMultimodalDataset
from src.models.clinical_baseline import FEATURE_GROUP_COLUMNS, MISSING_INDICATOR_COLS
from src.models.multimodal_model import MultimodalPneumoniaModel


TABULAR_PREPROCESSOR_FILENAME = "tabular_preprocessor.joblib"

TRIAGE_NUMERIC_COLS = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "acuity",
    "temperature_missing",
    "heartrate_missing",
    "resprate_missing",
    "o2sat_missing",
    "sbp_missing",
    "dbp_missing",
    "pain_missing",
    "acuity_missing",
    "is_pa",
    "is_ap",
]

TRIAGE_CATEGORICAL_COLS = [
    "gender",
    "race",
    "arrival_transport",
]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))],
                p=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_tabular_preprocessor(
    numeric_cols: list | None = None,
    categorical_cols: list | None = None,
) -> ColumnTransformer:
    if numeric_cols is None:
        numeric_cols = TRIAGE_NUMERIC_COLS
    if categorical_cols is None:
        categorical_cols = TRIAGE_CATEGORICAL_COLS

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [("num", numeric_transformer, numeric_cols)]

    if categorical_cols:
        try:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                ("onehot", onehot),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers)


def prepare_tabular_df(
    df: pd.DataFrame,
    numeric_cols: list | None = None,
    categorical_cols: list | None = None,
) -> pd.DataFrame:
    if numeric_cols is None:
        numeric_cols = TRIAGE_NUMERIC_COLS
    if categorical_cols is None:
        categorical_cols = TRIAGE_CATEGORICAL_COLS

    needed = numeric_cols + categorical_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected triage columns: {missing}")

    X = df[needed].copy()

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in MISSING_INDICATOR_COLS:
        if col in numeric_cols:
            X[col] = X[col].fillna(1).astype(int)

    for col in ["is_pa", "is_ap"]:
        if col in numeric_cols:
            X[col] = X[col].fillna(0).astype(int)

    for col in categorical_cols:
        X[col] = X[col].astype("string").fillna("UNKNOWN").str.strip()
        X[col] = X[col].replace({"": "UNKNOWN"})

    return X


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    unique_classes = np.unique(y_true)

    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(unique_classes) > 1 else None,
        "auprc": float(average_precision_score(y_true, y_prob)) if len(unique_classes) > 1 else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _to_python_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _extract_id_records_from_batch(batch: dict) -> list[dict]:
    subject_ids = _to_python_list(batch.get("subject_id"))
    study_ids = _to_python_list(batch.get("study_id"))
    dicom_ids = _to_python_list(batch.get("dicom_id"))
    image_paths = _to_python_list(batch.get("image_path"))

    batch_size = len(subject_ids) if subject_ids else 0
    if batch_size == 0:
        return []

    optional_fields = {
        "study_id": study_ids,
        "dicom_id": dicom_ids,
        "image_path": image_paths,
    }

    for field_name, field_values in optional_fields.items():
        if field_values and len(field_values) != batch_size:
            raise ValueError(
                f"Batch field '{field_name}' length {len(field_values)} "
                f"does not match subject_id length {batch_size}."
            )

    records = []
    for i in range(batch_size):
        record = {"subject_id": int(subject_ids[i])}
        if study_ids:
            record["study_id"] = int(study_ids[i]) if study_ids[i] is not None else None
        if dicom_ids:
            record["dicom_id"] = str(dicom_ids[i]) if dicom_ids[i] is not None else None
        if image_paths:
            record["image_path"] = str(image_paths[i]) if image_paths[i] is not None else None
        records.append(record)

    return records


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()

    total_loss = 0.0
    total_batches = 0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_id_records: list[dict] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).float().unsqueeze(1)

        logits = model(images, tabular)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        targs = targets.squeeze(1).detach().cpu().numpy()

        total_loss += loss.item()
        total_batches += 1
        all_targets.append(targs)
        all_probs.append(probs)

        id_records = _extract_id_records_from_batch(batch)
        if not id_records:
            raise ValueError(
                "Evaluation batch does not contain subject_id. "
                "Update CXRMultimodalDataset.__getitem__ to return IDs."
            )
        if len(id_records) != len(targs):
            raise ValueError(
                f"ID record count {len(id_records)} does not match target count {len(targs)} in batch."
            )
        all_id_records.extend(id_records)

    if not all_targets or not all_probs:
        raise ValueError("Evaluation loader produced no batches.")

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    if len(all_id_records) != len(y_true):
        raise ValueError(
            f"Collected {len(all_id_records)} ID rows but {len(y_true)} predictions."
        )

    pred_df = pd.DataFrame(all_id_records)
    pred_df["target"] = y_true.astype(int)
    pred_df["pred_prob"] = y_prob.astype(float)

    metrics = evaluate_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(total_batches, 1)

    return metrics, y_true, y_prob, pred_df


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
    use_amp = scaler is not None and device.type == "cuda"

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images, tabular)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, tabular)
            loss = criterion(logits, targets)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return {"loss": total_loss / max(total_batches, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-table",
        type=str,
        default="artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet",
    )
    parser.add_argument(
        "--image-backbone-checkpoint",
        type=str,
        default="artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="If set, use this LR for backbone and tabular/fusion (overrides --lr-head / --lr-backbone).",
    )
    parser.add_argument("--lr-head", type=float, default=1e-4, dest="lr_head")
    parser.add_argument("--lr-backbone", type=float, default=3e-5, dest="lr_backbone")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tabular-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--freeze-image-backbone", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--fusion-type",
        type=str,
        default="concat",
        choices=["concat", "attention"],
        dest="fusion_type",
    )
    parser.add_argument(
        "--tabular-feature-groups",
        type=str,
        default="all",
        choices=["all", "vitals_only", "vitals_plus_acuity", "no_missing_flags"],
        dest="tabular_feature_groups",
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

    required_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "image_path",
        "target",
        "temporal_split",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input table missing required columns: {missing}")

    train_df = df[df["temporal_split"] == "train"].reset_index(drop=True).copy()
    val_df = df[df["temporal_split"] == "validate"].reset_index(drop=True).copy()
    test_df = df[df["temporal_split"] == "test"].reset_index(drop=True).copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "One or more splits are empty. "
            f"train={len(train_df)}, validate={len(val_df)}, test={len(test_df)}"
        )

    tab_group = FEATURE_GROUP_COLUMNS[args.tabular_feature_groups]
    active_numeric_cols = tab_group["numeric"]
    active_categorical_cols = tab_group["categorical"]

    preprocessor = build_tabular_preprocessor(active_numeric_cols, active_categorical_cols)
    X_train_tab = prepare_tabular_df(train_df, active_numeric_cols, active_categorical_cols)
    X_val_tab = prepare_tabular_df(val_df, active_numeric_cols, active_categorical_cols)
    X_test_tab = prepare_tabular_df(test_df, active_numeric_cols, active_categorical_cols)

    train_tabular = preprocessor.fit_transform(X_train_tab).astype(np.float32)
    val_tabular = preprocessor.transform(X_val_tab).astype(np.float32)
    test_tabular = preprocessor.transform(X_test_tab).astype(np.float32)

    preprocessor_path = output_dir / TABULAR_PREPROCESSOR_FILENAME
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved train-fitted tabular preprocessor to {preprocessor_path}")

    tabular_input_dim = int(train_tabular.shape[1])

    train_transform, eval_transform = build_transforms(args.image_size)

    train_dataset = CXRMultimodalDataset(
        df=train_df,
        tabular_array=train_tabular,
        transform=train_transform,
    )
    val_dataset = CXRMultimodalDataset(
        df=val_df,
        tabular_array=val_tabular,
        transform=eval_transform,
    )
    test_dataset = CXRMultimodalDataset(
        df=test_df,
        tabular_array=test_tabular,
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )

    if args.fusion_type == "attention":
        from src.models.multimodal_model_attn import MultimodalPneumoniaModelAttn
        model = MultimodalPneumoniaModelAttn(
            tabular_input_dim=tabular_input_dim,
            tabular_hidden_dim=args.tabular_hidden_dim,
            d_model=args.fusion_hidden_dim,
            dropout=args.dropout,
        )
    else:
        model = MultimodalPneumoniaModel(
            tabular_input_dim=tabular_input_dim,
            tabular_hidden_dim=args.tabular_hidden_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            dropout=args.dropout,
        )

    if args.image_backbone_checkpoint:
        model.load_image_backbone_from_checkpoint(args.image_backbone_checkpoint)

    if args.freeze_image_backbone:
        model.freeze_image_backbone()
        print("Image backbone frozen.")

    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    train_targets = pd.to_numeric(train_df["target"], errors="raise").astype(int)
    pos = int((train_targets == 1).sum())
    neg = int((train_targets == 0).sum())
    pos_weight_value = (neg / pos) if pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lr_head = float(args.lr) if args.lr is not None else float(args.lr_head)
    lr_backbone = float(args.lr) if args.lr is not None else float(args.lr_backbone)

    backbone_params = [p for p in model.image_backbone.parameters() if p.requires_grad]
    rest_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("image_backbone.")
    ]

    param_groups: list[dict] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if rest_params:
        param_groups.append({"params": rest_params, "lr": lr_head})
    if not param_groups:
        raise ValueError("No trainable parameters (check --freeze-image-backbone).")

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    use_amp = torch.cuda.is_available() and not args.disable_amp
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    config = {
        "input_table": args.input_table,
        "image_backbone_checkpoint": args.image_backbone_checkpoint,
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
        "tabular_hidden_dim": args.tabular_hidden_dim,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "dropout": args.dropout,
        "freeze_image_backbone": args.freeze_image_backbone,
        "device": str(device),
        "model_name": "multimodal_densenet121_triage",
        "fusion_type": args.fusion_type,
        "tabular_feature_groups": args.tabular_feature_groups,
        "selection_metric": "val_auprc",
        "tabular_input_dim": tabular_input_dim,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_positives": pos,
        "train_negatives": neg,
        "pos_weight": float(pos_weight_value),
        "triage_numeric_cols": active_numeric_cols,
        "triage_categorical_cols": active_categorical_cols,
        "tabular_preprocessor_file": TABULAR_PREPROCESSOR_FILENAME,
        "sklearn_version": sklearn_version,
        "max_grad_norm": args.max_grad_norm,
        "use_amp": bool(use_amp),
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    best_val_auprc = -math.inf
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

        val_metrics, _, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_val_auprc = (
            float(val_metrics["auprc"]) if val_metrics["auprc"] is not None else 0.0
        )
        scheduler.step(current_val_auprc)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "lr_groups": [float(g["lr"]) for g in optimizer.param_groups],
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
                "val_auprc": val_metrics["auprc"],
                "config": config,
            },
            checkpoints_dir / f"epoch_{epoch}.pt",
        )

        if current_val_auprc > best_val_auprc:
            best_val_auprc = current_val_auprc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_auprc": val_metrics["auprc"],
                    "config": config,
                },
                checkpoints_dir / "best.pt",
            )
            print(
                f"New best checkpoint saved at epoch {epoch} "
                f"with val AUPRC={best_val_auprc:.6f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No val AUPRC improvement for {epochs_without_improvement} epoch(s). "
                f"Best so far: epoch {best_epoch}, AUPRC={best_val_auprc:.6f}"
            )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping triggered after epoch {epoch}. "
                f"Best epoch was {best_epoch} with val AUPRC={best_val_auprc:.6f}"
            )
            break

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_ckpt_path = checkpoints_dir / "best.pt"
    if not best_ckpt_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")

    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_metrics, _, _, val_pred = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )
    test_metrics, _, _, test_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    val_pred.to_csv(output_dir / "val_predictions.csv", index=False)
    test_pred.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "best_val_auprc": float(best_val_auprc),
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "output_dir": str(output_dir),
        "tabular_preprocessor_file": str(preprocessor_path),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()