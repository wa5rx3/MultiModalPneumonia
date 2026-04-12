#!/usr/bin/env python3
"""
regenerate_all_results.py — Verify that all result artefacts used in thesis_v2
exist and are non-trivially sized.

Does NOT re-run experiments (requires MIMIC data + GPU).
Verifies artefact existence and basic integrity against expected sizes.

Run from project root:
    python scripts/regenerate_all_results.py

Exit code 0 if all artefacts pass, 1 if any fail.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_JSON_BYTES = 1000   # 1 KB for JSON files
MIN_PNG_BYTES = 10_000  # 10 KB for image files
MIN_CSV_BYTES = 500     # 500 B for CSV files

REQUIRED_ARTEFACTS = [
    # Source of truth — all numerical results
    ("artifacts/evaluation/final_publication_report.json", MIN_JSON_BYTES, "json"),

    # Feature ablation
    ("artifacts/evaluation/feature_ablation_results.csv", MIN_CSV_BYTES, "csv"),

    # Calibration outputs
    (
        "artifacts/evaluation/calibration_stronger_lr_v3/"
        "reliability_diagram_all_models.png",
        MIN_PNG_BYTES, "png"
    ),
    (
        "artifacts/evaluation/calibration_stronger_lr_v3/"
        "calibration_metrics.json",
        MIN_JSON_BYTES, "json"
    ),

    # SHAP figures
    ("artifacts/evaluation/shap/shap_summary_beeswarm.png", MIN_PNG_BYTES, "png"),
    ("artifacts/evaluation/shap/shap_summary_bar.png", MIN_PNG_BYTES, "png"),

    # DCA
    ("artifacts/evaluation/dca/decision_curve_standardized.png", MIN_PNG_BYTES, "png"),

    # Generated thesis figures (from scripts/generate_thesis_figures.py)
    (
        "thesis_new_docs/figures/generated_results/fig_a1_pr_curves.png",
        MIN_PNG_BYTES, "png"
    ),
    (
        "thesis_new_docs/figures/generated_results/fig_a3_ablation_bars.png",
        MIN_PNG_BYTES, "png"
    ),
    (
        "thesis_new_docs/figures/generated_results/fig_a4_label_sensitivity.png",
        MIN_PNG_BYTES, "png"
    ),
    (
        "thesis_new_docs/figures/generated_results/fig_a5_training_curves.png",
        MIN_PNG_BYTES, "png"
    ),
    (
        "thesis_new_docs/figures/generated_results/fig_a6_missing_heatmap.png",
        MIN_PNG_BYTES, "png"
    ),
    (
        "thesis_new_docs/figures/generated_results/fig_a7_vital_distributions.png",
        MIN_PNG_BYTES, "png"
    ),
    (
        "thesis_new_docs/figures/generated_results/fig_a8_label_distribution.png",
        MIN_PNG_BYTES, "png"
    ),

    # Original result figures
    (
        "thesis_new_docs/figures/original_results/"
        "roc_curve_all_models.png",
        MIN_PNG_BYTES, "png"
    ),
    (
        "thesis_new_docs/figures/original_results/"
        "calibration_stronger_lr_v3_reliability_diagram_all_models.png",
        MIN_PNG_BYTES, "png"
    ),
    (
        "thesis_new_docs/figures/original_results/"
        "shap_summary_beeswarm.png",
        MIN_PNG_BYTES, "png"
    ),
    (
        "thesis_new_docs/figures/original_results/"
        "decision_curve_standardized.png",
        MIN_PNG_BYTES, "png"
    ),
]


def check_json_loads(path):
    """Returns True if the file parses as valid JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def verify():
    passed = []
    failed = []

    for rel_path, min_bytes, ftype in REQUIRED_ARTEFACTS:
        abs_path = os.path.join(PROJECT_ROOT, rel_path.replace("/", os.sep))

        if not os.path.isfile(abs_path):
            failed.append((rel_path, f"MISSING — file not found at {abs_path}"))
            continue

        size = os.path.getsize(abs_path)
        if size < min_bytes:
            failed.append((rel_path, f"TOO SMALL ({size} bytes < {min_bytes})"))
            continue

        if ftype == "json" and not check_json_loads(abs_path):
            failed.append((rel_path, "INVALID JSON — parse failed"))
            continue

        passed.append((rel_path, size))

    # Print results
    print("ARTEFACT VERIFICATION REPORT")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}\n")

    print("PASSED:")
    for rel_path, size in passed:
        print(f"  ✓  {os.path.basename(rel_path):<55}  {size / 1024:>7.1f} KB")

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for rel_path, reason in failed:
            print(f"  ✗  {os.path.basename(rel_path):<55}  {reason}")
        print(f"\nResult: {len(passed)}/{len(passed) + len(failed)} artefacts present.")
        return False
    else:
        print(f"\nResult: All {len(passed)} artefacts VERIFIED.")

        # Extra: validate key numbers in final_publication_report.json
        report_path = os.path.join(
            PROJECT_ROOT,
            "artifacts", "evaluation", "final_publication_report.json"
        )
        print("\nSpot-checking locked results from final_publication_report.json:")
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        checks = [
            ("image_u_ignore AUROC", report["models"]["image_u_ignore"]["auroc"],
             0.745, 0.747),
            ("multimodal_u_ignore AUROC", report["models"]["multimodal_u_ignore"]["auroc"],
             0.735, 0.737),
            ("multimodal_vs_image delta_auroc mean",
             report["pairwise_comparisons"]["multimodal_vs_image"]["delta_auroc"]["mean"],
             -0.011, -0.008),
        ]

        all_checks_ok = True
        for label, value, lo, hi in checks:
            ok = lo <= value <= hi
            status = "✓" if ok else "✗"
            print(f"  {status}  {label}: {value:.4f} (expected [{lo}, {hi}])")
            if not ok:
                all_checks_ok = False

        if not all_checks_ok:
            print("\nWARNING: Some spot-checks failed. Report may have changed.")
            return False

        return True


if __name__ == "__main__":
    ok = verify()
    sys.exit(0 if ok else 1)
