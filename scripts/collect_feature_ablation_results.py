"""Collect feature ablation results across clinical and multimodal model dirs.

Scans output dirs matching known naming patterns, extracts test AUROC/AUPRC
from metrics.json or summary.json, and writes a single CSV summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FEATURE_GROUPS = [
    "all",
    "vitals_only",
    "demographics_only",
    "acuity_only",
    "vitals_plus_acuity",
    "no_missing_flags",
]


MODEL_PATTERNS = [
    (
        "clinical_logistic",
        "clinical_baseline_u_ignore_{fg}_temporal_strong_v2",
        "metrics.json",
        ["test_metrics", "auroc"],
        ["test_metrics", "auprc"],
    ),
    (
        "clinical_xgb",
        "clinical_xgb_u_ignore_{fg}_temporal_strong_v2",
        "metrics.json",
        ["test_metrics", "auroc"],
        ["test_metrics", "auprc"],
    ),
    (
        "multimodal",
        "multimodal_pneumonia_{fg}_u_ignore_temporal_stronger_lr_v3",
        "summary.json",
        ["test_metrics", "auroc"],
        ["test_metrics", "auprc"],
    ),
]


CANONICAL_ALL_PATTERNS = [
    (
        "clinical_logistic",
        "clinical_baseline_u_ignore_temporal_strong_v2",
        "metrics.json",
        ["test_metrics", "auroc"],
        ["test_metrics", "auprc"],
        "all",
    ),
    (
        "clinical_xgb",
        "clinical_xgb_u_ignore_temporal_strong_v2",
        "metrics.json",
        ["test_metrics", "auroc"],
        ["test_metrics", "auprc"],
        "all",
    ),
    (
        "multimodal",
        "multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3",
        "summary.json",
        ["test_metrics", "auroc"],
        ["test_metrics", "auprc"],
        "all",
    ),
]


def get_nested(d: dict, keys: list[str]):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def load_metric(model_dir: Path, metrics_file: str, auroc_path: list, auprc_path: list):
    f = model_dir / metrics_file
    if not f.is_file():
        return None, None
    with open(f, encoding="utf-8") as fh:
        data = json.load(fh)
    return get_nested(data, auroc_path), get_nested(data, auprc_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        type=str,
        default="artifacts/models",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/evaluation/feature_ablation_results.csv",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    rows: list[dict] = []


    for model_type, dir_name, metrics_file, auroc_path, auprc_path, fg in CANONICAL_ALL_PATTERNS:
        model_dir = models_dir / dir_name
        if model_dir.is_dir():
            auroc, auprc = load_metric(model_dir, metrics_file, auroc_path, auprc_path)
            rows.append({
                "model_type": model_type,
                "feature_groups": fg,
                "model_dir": str(model_dir),
                "test_auroc": auroc,
                "test_auprc": auprc,
                "found": auroc is not None,
            })
        else:
            rows.append({
                "model_type": model_type,
                "feature_groups": fg,
                "model_dir": str(model_dir),
                "test_auroc": None,
                "test_auprc": None,
                "found": False,
            })


    for model_type, dir_pattern, metrics_file, auroc_path, auprc_path in MODEL_PATTERNS:
        for fg in FEATURE_GROUPS:
            if fg == "all":
                continue
            dir_name = dir_pattern.format(fg=fg)
            model_dir = models_dir / dir_name
            if model_dir.is_dir():
                auroc, auprc = load_metric(model_dir, metrics_file, auroc_path, auprc_path)
                rows.append({
                    "model_type": model_type,
                    "feature_groups": fg,
                    "model_dir": str(model_dir),
                    "test_auroc": auroc,
                    "test_auprc": auprc,
                    "found": auroc is not None,
                })
            else:
                rows.append({
                    "model_type": model_type,
                    "feature_groups": fg,
                    "model_dir": str(model_dir),
                    "test_auroc": None,
                    "test_auprc": None,
                    "found": False,
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["model_type", "feature_groups"]).reset_index(drop=True)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")
    print(df.to_string(index=False))

    missing = df[~df["found"]]
    if not missing.empty:
        print(f"\n{len(missing)} model dir(s) not yet found (training pending):")
        for _, r in missing.iterrows():
            print(f"  {r['model_type']} / {r['feature_groups']}: {r['model_dir']}")


if __name__ == "__main__":
    main()
