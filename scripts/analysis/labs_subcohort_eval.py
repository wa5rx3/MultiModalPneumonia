"""Sensitivity analysis: evaluate on the lab-present test subcohort.

Only ~25% of ED-anchored CXR studies have any resulted lab at or before t0, so the
full-cohort labs comparison is diluted by imputation. This restricts evaluation to
the test studies that actually have >=1 lab and compares image-only, triage-concat,
and triage+labs there — the fair test of whether labs help WHEN AVAILABLE. Reported
across seeds; note the subcohort is small (n~270), so CIs are wide.

Output: artifacts/evaluation/multiseed/labs_subcohort_eval.{json,csv}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.analysis._pred_utils import available_seeds, load_pred, summarize
from src.evaluation.calibration_analysis import compute_ece_mce

LABS_TABLE = Path("artifacts/manifests/cxr_clinical_labs_training_table_u_ignore_temporal.parquet")
FEATURE_MAP = Path("artifacts/tables/lab_feature_map.json")
EVAL_OUT = Path("artifacts/evaluation/multiseed")
SEEDS = [42, 123, 456, 789, 1000]
ARCHS = ["image", "concat", "labs"]


def lab_present_test_studies() -> set[int]:
    concepts = list(json.load(open(FEATURE_MAP)).keys())
    df = pd.read_parquet(LABS_TABLE)
    test = df[df["temporal_split"] == "test"]
    value_cols = [c for c in concepts if c in test.columns]
    present = test[test[value_cols].notna().any(axis=1)]
    return set(present["study_id"].astype("int64").tolist())


def metrics(y, p) -> dict:
    ece, _, _ = compute_ece_mce(y, p, n_bins=10)
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "ece": float(ece), "brier": float(brier_score_loss(y, p))}


def main() -> None:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    keep = lab_present_test_studies()
    print(f"Lab-present test studies: {len(keep)}")

    rows = []
    store: dict = {}
    for arch in ARCHS:
        for seed in available_seeds(arch, SEEDS):
            df = load_pred(arch, seed)
            sub = df[df["study_id"].astype("int64").isin(keep)]
            if sub["target"].nunique() < 2:
                continue
            m = metrics(sub["target"].to_numpy(), sub["pred_prob"].to_numpy())
            m.update({"arch": arch, "seed": seed, "n": int(len(sub)), "prevalence": float(sub["target"].mean())})
            rows.append(m)
            for k in ("auroc", "auprc", "ece", "brier"):
                store.setdefault((arch, k), []).append(m[k])

    if not rows:
        raise SystemExit("No predictions available yet for subcohort eval.")

    per = pd.DataFrame(rows)
    per.to_csv(EVAL_OUT / "labs_subcohort_eval.csv", index=False)

    summary = {"n_lab_present_test": len(keep), "subcohort_n": int(per["n"].iloc[0]),
               "subcohort_prevalence": round(float(per["prevalence"].iloc[0]), 4), "per_arch": {}}
    for arch in ARCHS:
        md = {k: summarize(store.get((arch, k), [])) for k in ("auroc", "auprc", "ece", "brier")}
        if md["auroc"]["n"]:
            summary["per_arch"][arch] = md

    with open(EVAL_OUT / "labs_subcohort_eval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Lab-present subcohort (n~{summary.get('subcohort_n')}, "
          f"prev {summary.get('subcohort_prevalence')}), mean +/- std across seeds ===")
    for arch, md in summary["per_arch"].items():
        print(f"{arch:8s}: AUROC {md['auroc']['mean']:.4f}+/-{md['auroc']['std']:.4f}  "
              f"AUPRC {md['auprc']['mean']:.4f}  ECE {md['ece']['mean']:.4f}  Brier {md['brier']['mean']:.4f}")
    print(f"\nWrote {EVAL_OUT}/labs_subcohort_eval.json/.csv")


if __name__ == "__main__":
    main()
