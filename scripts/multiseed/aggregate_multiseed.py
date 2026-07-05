"""Aggregate multi-seed replication results.

For each architecture and seed, loads test_predictions.csv, computes
AUROC / AUPRC / ECE (10-bin uniform, same estimator as the thesis) / Brier,
then reports across-seed mean +/- std. Computes the paired per-seed deltas
(concat - image) for AUROC and ECE on aligned test rows, and summarises the
across-seed distribution of those deltas. This is the analysis that decides
whether the calibration advantage (H3) is seed-invariant or a single-seed
artifact.

Outputs:
  artifacts/evaluation/multiseed/per_seed_metrics.csv
  artifacts/evaluation/multiseed/multiseed_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.calibration_analysis import compute_ece_mce

OUT_ROOT = Path("artifacts/models/multiseed")
EVAL_OUT = Path("artifacts/evaluation/multiseed")
ALIGN_KEYS = ["subject_id", "study_id", "dicom_id"]


def load_pred(arch: str, seed: int) -> pd.DataFrame | None:
    p = OUT_ROOT / f"{arch}_seed{seed}" / "test_predictions.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    df["target"] = pd.to_numeric(df["target"]).astype(int)
    df["pred_prob"] = pd.to_numeric(df["pred_prob"]).astype(float)
    return df


def point_metrics(df: pd.DataFrame, n_bins: int = 10) -> dict:
    y = df["target"].to_numpy()
    p = df["pred_prob"].to_numpy()
    ece, mce, _ = compute_ece_mce(y, p, n_bins=n_bins)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier_score_loss(y, p)),
        "mean_pred_prob": float(p.mean()),
    }


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n_seeds": int(len(a)),
        "values": [float(x) for x in a],
    }


def paired_delta(concat: pd.DataFrame, image: pd.DataFrame, n_bins: int = 10) -> dict:
    m = concat.merge(image, on=ALIGN_KEYS, suffixes=("_c", "_i"))
    if len(m) != len(image) or len(m) != len(concat):
        raise ValueError(
            f"Alignment mismatch: concat={len(concat)} image={len(image)} merged={len(m)}"
        )
    if not (m["target_c"] == m["target_i"]).all():
        raise ValueError("Target mismatch after alignment.")
    y = m["target_c"].to_numpy()
    pc = m["pred_prob_c"].to_numpy()
    pi = m["pred_prob_i"].to_numpy()
    ece_c, _, _ = compute_ece_mce(y, pc, n_bins=n_bins)
    ece_i, _, _ = compute_ece_mce(y, pi, n_bins=n_bins)
    return {
        "delta_auroc": float(roc_auc_score(y, pc) - roc_auc_score(y, pi)),
        "delta_auprc": float(average_precision_score(y, pc) - average_precision_score(y, pi)),
        "delta_ece": float(ece_c - ece_i),
        "delta_brier": float(brier_score_loss(y, pc) - brier_score_loss(y, pi)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["image", "concat", "attn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1000])
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    preds: dict[tuple[str, int], pd.DataFrame] = {}
    for arch in args.archs:
        for seed in args.seeds:
            df = load_pred(arch, seed)
            if df is None:
                continue
            preds[(arch, seed)] = df
            m = point_metrics(df, n_bins=args.n_bins)
            m.update({"arch": arch, "seed": seed})
            rows.append(m)

    if not rows:
        raise SystemExit("No multi-seed predictions found yet.")

    per_seed = pd.DataFrame(rows)[
        ["arch", "seed", "n", "positive_rate", "auroc", "auprc", "ece", "mce", "brier", "mean_pred_prob"]
    ].sort_values(["arch", "seed"]).reset_index(drop=True)
    per_seed.to_csv(EVAL_OUT / "per_seed_metrics.csv", index=False)

    summary: dict = {"n_bins": args.n_bins, "seeds": args.seeds, "per_arch": {}, "paired_vs_image": {}}
    for arch in args.archs:
        sub = per_seed[per_seed["arch"] == arch]
        if sub.empty:
            continue
        summary["per_arch"][arch] = {
            metric: summarize(sub[metric].tolist())
            for metric in ["auroc", "auprc", "ece", "mce", "brier", "mean_pred_prob"]
        }

    # paired deltas per seed: every fusion arch vs image-only, on identical test rows
    fusion_archs = [a for a in args.archs if a != "image"]
    all_delta_rows = []
    for arch in fusion_archs:
        deltas = {k: [] for k in ["delta_auroc", "delta_auprc", "delta_ece", "delta_brier"]}
        for seed in args.seeds:
            m = preds.get((arch, seed))
            i = preds.get(("image", seed))
            if m is None or i is None:
                continue
            d = paired_delta(m, i, n_bins=args.n_bins)
            all_delta_rows.append({"arch": arch, "seed": seed, **d})
            for k in deltas:
                deltas[k].append(d[k])
        if any(deltas.values()):
            summary["paired_vs_image"][arch] = {}
            for k, vals in deltas.items():
                s = summarize(vals)
                arr = np.asarray(vals)
                # lower is better for ece/brier, higher for auroc/auprc
                s["frac_seeds_favoring_fusion"] = float((arr < 0).mean() if k in ("delta_ece", "delta_brier")
                                                        else (arr > 0).mean())
                summary["paired_vs_image"][arch][k] = s

    if all_delta_rows:
        pd.DataFrame(all_delta_rows).to_csv(EVAL_OUT / "per_seed_paired_delta.csv", index=False)

    with open(EVAL_OUT / "multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # console report
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Per-seed metrics ===")
    print(per_seed.to_string(index=False))
    print("\n=== Across-seed mean +/- std ===")
    for arch, md in summary["per_arch"].items():
        line = ", ".join(f"{m}={md[m]['mean']:.4f}+/-{md[m]['std']:.4f}" for m in ["auroc", "auprc", "ece", "brier"])
        print(f"{arch:8s}: {line}")
    for arch, dd in summary["paired_vs_image"].items():
        print(f"\n=== Paired ({arch} - image), across seeds ===")
        for k, s in dd.items():
            print(f"{k:12s}: mean={s['mean']:+.4f} std={s['std']:.4f} range=[{s['min']:+.4f},{s['max']:+.4f}] "
                  f"frac_favoring_fusion={s['frac_seeds_favoring_fusion']:.2f}")
    print(f"\nWrote {EVAL_OUT}/per_seed_metrics.csv and multiseed_summary.json")


if __name__ == "__main__":
    main()
