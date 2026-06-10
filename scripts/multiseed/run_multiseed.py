"""Multi-seed replication orchestrator.

Retrains the three deep models across several seeds, sharing the existing
multilabel-pretrained DenseNet-121 backbone so the analysis isolates
fine-tuning stochasticity (data order, dropout masks, head/fusion init),
not pretraining stochasticity. Each run replicates the canonical recipe
exactly (learning rates, epochs, patience, backbone source); only --seed
and --output-dir change.

Canonical recipes (from saved config.json):
  image   : lr_head 5e-5 / lr_backbone 1e-5, epochs 40, patience 10,
            backbone = multilabel pretrain best.pt
  concat  : lr_head 5e-5 / lr_backbone 1e-5, epochs 30, patience 8,
            backbone = multilabel pretrain best.pt
  attn    : lr_head 1e-4 / lr_backbone 3e-5, epochs 30, patience 8,
            backbone = the SAME SEED's fine-tuned image model best.pt

Resumable: a run whose output_dir already has summary.json is skipped.
Disk hygiene: per-epoch checkpoints are pruned after each run, keeping best.pt.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PRETRAIN_BACKBONE = "artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt"
IMAGE_TABLE = "artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet"
MM_TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT_ROOT = Path("artifacts/models/multiseed")

DEFAULT_SEEDS = [42, 123, 456, 789, 1000]


def image_dir(seed: int) -> Path:
    return OUT_ROOT / f"image_seed{seed}"


def concat_dir(seed: int) -> Path:
    return OUT_ROOT / f"concat_seed{seed}"


def attn_dir(seed: int) -> Path:
    return OUT_ROOT / f"attn_seed{seed}"


def image_cmd(seed: int) -> list[str]:
    return [
        sys.executable, "-m", "src.training.train_image_pneumonia_finetune",
        "--input-table", IMAGE_TABLE,
        "--pretrained-checkpoint", PRETRAIN_BACKBONE,
        "--output-dir", str(image_dir(seed)),
        "--lr-head", "5e-5", "--lr-backbone", "1e-5",
        "--epochs", "40", "--patience", "10", "--batch-size", "16",
        "--image-size", "224", "--num-workers", "4", "--seed", str(seed),
    ]


def concat_cmd(seed: int) -> list[str]:
    return [
        sys.executable, "-m", "src.training.train_multimodal_pneumonia",
        "--input-table", MM_TABLE,
        "--image-backbone-checkpoint", PRETRAIN_BACKBONE,
        "--output-dir", str(concat_dir(seed)),
        "--lr-head", "5e-5", "--lr-backbone", "1e-5",
        "--epochs", "30", "--patience", "8", "--batch-size", "16",
        "--image-size", "224", "--num-workers", "4", "--seed", str(seed),
        "--fusion-type", "concat", "--tabular-feature-groups", "all",
    ]


def attn_cmd(seed: int) -> list[str]:
    # attention fusion is initialised from this seed's fine-tuned image model
    backbone = str(image_dir(seed) / "checkpoints" / "best.pt")
    return [
        sys.executable, "-m", "src.training.train_multimodal_pneumonia",
        "--input-table", MM_TABLE,
        "--image-backbone-checkpoint", backbone,
        "--output-dir", str(attn_dir(seed)),
        "--lr-head", "1e-4", "--lr-backbone", "3e-5",
        "--epochs", "30", "--patience", "8", "--batch-size", "16",
        "--image-size", "224", "--num-workers", "4", "--seed", str(seed),
        "--fusion-type", "attention", "--tabular-feature-groups", "all",
    ]


def prune_epoch_ckpts(model_dir: Path) -> int:
    ck = model_dir / "checkpoints"
    removed = 0
    if ck.is_dir():
        for f in ck.glob("epoch_*.pt"):
            f.unlink()
            removed += 1
    return removed


def run_one(arch: str, seed: int, cmd: list[str], out_dir: Path) -> None:
    summary = out_dir / "summary.json"
    if summary.is_file():
        print(f"[skip] {arch} seed={seed} already done ({summary})", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    print(f"[run ] {arch} seed={seed} -> {out_dir}", flush=True)
    print(f"       cmd: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(
            f"[FAIL] {arch} seed={seed} returncode={proc.returncode} after {dt:.0f}s; see {log_path}"
        )
    n = prune_epoch_ckpts(out_dir)
    print(f"[done] {arch} seed={seed} in {dt/60:.1f} min; pruned {n} epoch ckpts", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["image", "concat"],
                    choices=["image", "concat", "attn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan: list[tuple[str, int, list[str], Path]] = []
    # image first (attn depends on it), then concat, then attn
    for arch in ["image", "concat", "attn"]:
        if arch not in args.archs:
            continue
        for seed in args.seeds:
            if arch == "image":
                plan.append((arch, seed, image_cmd(seed), image_dir(seed)))
            elif arch == "concat":
                plan.append((arch, seed, concat_cmd(seed), concat_dir(seed)))
            else:
                plan.append((arch, seed, attn_cmd(seed), attn_dir(seed)))

    print(f"Planned {len(plan)} runs: archs={args.archs} seeds={args.seeds}", flush=True)
    for arch, seed, cmd, out_dir in plan:
        if args.dry_run:
            print(f"[dry ] {arch} seed={seed}: {' '.join(cmd)}")
            continue
        run_one(arch, seed, cmd, out_dir)
    print("All planned runs complete.", flush=True)


if __name__ == "__main__":
    main()
