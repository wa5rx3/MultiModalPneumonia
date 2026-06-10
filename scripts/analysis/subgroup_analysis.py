"""Subgroup / fairness audit across seeds.

Stratifies test-set discrimination (AUROC, AUPRC) and calibration (ECE, Brier)
by sex, age band, broad race group, ED acuity, and radiograph view, for the
image-only and concat-multimodal models. Reports per-subgroup metrics as
mean +/- std across seeds, and the concat - image deltas per subgroup, to check
whether the (dis)advantage of adding triage context is uniform across clinically
relevant groups. Prior work documents systematic underdiagnosis of CXR findings
in underrepresented groups (Seyyed-Kalantari et al. 2021), so this is a
prerequisite for any deployment claim.

Output: artifacts/evaluation/multiseed/subgroup_metrics.{json,csv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.analysis._pred_utils import available_seeds, load_pred, summarize
from src.evaluation.calibration_analysis import compute_ece_mce

EVAL_OUT = Path("artifacts/evaluation/multiseed")
CLINICAL_TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
PATIENTS = "D:/mimic_iv/patients.csv.gz"
ALIGN_KEYS = ["subject_id", "study_id", "dicom_id"]
MIN_SUBGROUP_N = 30  # do not report metrics for tiny strata


def race_group(s: str) -> str:
    if not isinstance(s, str):
        return "Other/Unknown"
    u = s.upper()
    if u.startswith("WHITE"):
        return "White"
    if "BLACK" in u or "AFRICAN" in u:
        return "Black"
    if "HISPANIC" in u or "LATINO" in u:
        return "Hispanic"
    if "ASIAN" in u:
        return "Asian"
    return "Other/Unknown"


def age_band(a: float) -> str:
    if pd.isna(a):
        return "Unknown"
    if a < 40:
        return "<40"
    if a < 60:
        return "40-59"
    if a < 80:
        return "60-79"
    return "80+"


def build_metadata() -> pd.DataFrame:
    df = pd.read_parquet(CLINICAL_TABLE)
    df = df[df["temporal_split"] == "test"].copy()
    pat = pd.read_csv(PATIENTS, usecols=["subject_id", "anchor_age"])
    df = df.merge(pat, on="subject_id", how="left")
    df["sex"] = df["gender"].map({"M": "Male", "F": "Female"}).fillna("Unknown")
    df["age_band"] = df["anchor_age"].map(age_band)
    df["race_group"] = df["race"].map(race_group)
    df["acuity_group"] = np.where(df["acuity"] <= 2, "High acuity (ESI 1-2)",
                                  np.where(df["acuity"].notna(), "Lower acuity (ESI 3-5)", "Unknown"))
    df["view"] = np.where(df["is_pa"] == 1, "PA", np.where(df["is_ap"] == 1, "AP", "Other"))
    return df[ALIGN_KEYS + ["sex", "age_band", "race_group", "acuity_group", "view"]]


def subgroup_metrics(y: np.ndarray, p: np.ndarray) -> dict | None:
    if len(np.unique(y)) < 2 or len(y) < MIN_SUBGROUP_N:
        return None
    ece, _, _ = compute_ece_mce(y, p, n_bins=10)
    return {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "ece": float(ece), "brier": float(brier_score_loss(y, p)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1000])
    args = ap.parse_args()
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    meta = build_metadata()
    factors = ["sex", "age_band", "race_group", "acuity_group", "view"]

    # collect per (factor,level,arch,metric) -> {seed: value}
    store: dict = {}
    for arch in ["image", "concat"]:
        for seed in available_seeds(arch, args.seeds):
            pred = load_pred(arch, seed).merge(meta, on=ALIGN_KEYS, how="inner")
            for factor in factors:
                for level, grp in pred.groupby(factor):
                    m = subgroup_metrics(grp["target"].to_numpy(), grp["pred_prob"].to_numpy())
                    if m is None:
                        continue
                    for metric in ["auroc", "auprc", "ece", "brier", "n", "prevalence"]:
                        store.setdefault((factor, str(level), arch, metric), {})[seed] = m[metric]

    rows = []
    for (factor, level, arch, metric), seedvals in store.items():
        if metric in ("n", "prevalence"):
            continue
        s = summarize(seedvals.values())
        rows.append({"factor": factor, "level": level, "arch": arch, "metric": metric,
                     "mean": s["mean"], "std": s["std"], "n_seeds": s["n"]})
    df_rows = pd.DataFrame(rows).sort_values(["factor", "level", "metric", "arch"]).reset_index(drop=True)
    df_rows.to_csv(EVAL_OUT / "subgroup_metrics.csv", index=False)

    # paired concat - image per subgroup (common seeds), for auroc and ece
    paired = []
    for factor in factors:
        levels = sorted({lv for (f, lv, a, mt) in store if f == factor})
        for level in levels:
            for metric in ["auroc", "ece"]:
                di = store.get((factor, level, "image", metric), {})
                dc = store.get((factor, level, "concat", metric), {})
                common = sorted(set(di) & set(dc))
                if len(common) < 2:
                    continue
                deltas = [dc[s] - di[s] for s in common]
                nrep = store.get((factor, level, "image", "n"), {})
                paired.append({"factor": factor, "level": level, "metric": metric,
                               "delta_mean": float(np.mean(deltas)), "delta_std": float(np.std(deltas, ddof=1)),
                               "n_subgroup": int(np.mean(list(nrep.values()))) if nrep else None})
    pd.DataFrame(paired).to_csv(EVAL_OUT / "subgroup_paired_delta.csv", index=False)

    summary = {"factors": factors, "min_subgroup_n": MIN_SUBGROUP_N,
               "n_seeds_requested": len(args.seeds),
               "per_subgroup_csv": "subgroup_metrics.csv",
               "paired_delta_csv": "subgroup_paired_delta.csv"}
    with open(EVAL_OUT / "subgroup_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Subgroup AUROC / ECE (image vs concat, mean across seeds) ===")
    piv = df_rows[df_rows["metric"].isin(["auroc", "ece"])]
    for factor in factors:
        print(f"\n[{factor}]")
        sub = piv[piv["factor"] == factor]
        for level in sorted(sub["level"].unique()):
            cells = {}
            for _, r in sub[sub["level"] == level].iterrows():
                cells[(r["arch"], r["metric"])] = r["mean"]
            print(f"  {level:24s} AUROC img={cells.get(('image','auroc'),float('nan')):.3f} "
                  f"con={cells.get(('concat','auroc'),float('nan')):.3f} | "
                  f"ECE img={cells.get(('image','ece'),float('nan')):.3f} "
                  f"con={cells.get(('concat','ece'),float('nan')):.3f}")
    print(f"\nWrote {EVAL_OUT}/subgroup_metrics.csv, subgroup_paired_delta.csv")


if __name__ == "__main__":
    main()
