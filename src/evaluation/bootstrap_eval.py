from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()


    if "prob" not in df.columns:
        if "pred_prob" in df.columns:
            df["prob"] = df["pred_prob"]
        elif "prediction" in df.columns:
            df["prob"] = df["prediction"]
        elif "logit" in df.columns:
            df["prob"] = 1.0 / (1.0 + np.exp(-df["logit"]))
        else:
            raise ValueError(
                f"No probability-like column found in {path}. "
                f"Expected one of: prob, pred_prob, prediction, logit. "
                f"Got columns: {list(df.columns)}"
            )

    required = {"subject_id", "target", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="raise").astype(int)
    df["target"] = pd.to_numeric(df["target"], errors="raise").astype(int)
    df["prob"] = pd.to_numeric(df["prob"], errors="raise").astype(float)

    if "study_id" in df.columns:
        df["study_id"] = pd.to_numeric(df["study_id"], errors="raise").astype(int)

    return df


def get_alignment_keys(df: pd.DataFrame) -> list[str]:
    keys = ["subject_id"]
    if "study_id" in df.columns:
        keys.append("study_id")
    return keys


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["target"].to_numpy()
    y_prob = df["prob"].to_numpy()

    if len(np.unique(y_true)) < 2:
        raise ValueError("Cannot compute AUROC/AUPRC when only one class is present.")

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "median": float(np.median(values)),
    }


def pregroup_by_patient(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(pid): g.copy() for pid, g in df.groupby("subject_id", sort=False)}


def sample_grouped_rows(
    grouped: dict[int, pd.DataFrame],
    sampled_patients: Iterable[int],
) -> pd.DataFrame:
    parts = [grouped[int(pid)] for pid in sampled_patients]
    return pd.concat(parts, ignore_index=True)


def bootstrap_patient_level(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed)

    grouped = pregroup_by_patient(df)
    patients = np.array(sorted(grouped.keys()))
    results: list[dict[str, float]] = []
    skipped = 0

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        sampled_df = sample_grouped_rows(grouped, sampled_patients)

        try:
            results.append(compute_metrics(sampled_df))
        except ValueError:
            skipped += 1

    return pd.DataFrame(results), skipped


def assert_aligned_for_delta(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    keys_a = get_alignment_keys(df_a)
    keys_b = get_alignment_keys(df_b)
    if keys_a != keys_b:
        raise ValueError(
            f"Model A and B do not share the same alignment keys. "
            f"A keys={keys_a}, B keys={keys_b}"
        )

    keys = keys_a

    cols_a = keys + ["target", "prob"]
    cols_b = keys + ["target", "prob"]

    a = df_a[cols_a].copy()
    b = df_b[cols_b].copy()

    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)

    a_key_set = set(map(tuple, a[keys].to_numpy()))
    b_key_set = set(map(tuple, b[keys].to_numpy()))
    if a_key_set != b_key_set:
        only_a = len(a_key_set - b_key_set)
        only_b = len(b_key_set - a_key_set)
        raise ValueError(
            f"Model A and B are not aligned on the same evaluation rows. "
            f"Rows only in A: {only_a}, rows only in B: {only_b}"
        )

    merged = a.merge(
        b,
        on=keys,
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )

    if not np.array_equal(merged["target_a"].to_numpy(), merged["target_b"].to_numpy()):
        raise ValueError("Targets differ between aligned Model A and Model B rows.")

    aligned_a = merged[keys + ["target_a", "prob_a"]].rename(
        columns={"target_a": "target", "prob_a": "prob"}
    )
    aligned_b = merged[keys + ["target_b", "prob_b"]].rename(
        columns={"target_b": "target", "prob_b": "prob"}
    )

    return aligned_a, aligned_b, keys


def bootstrap_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    df_a, df_b, _ = assert_aligned_for_delta(df_a, df_b)

    rng = np.random.default_rng(seed)
    grouped_a = pregroup_by_patient(df_a)
    grouped_b = pregroup_by_patient(df_b)
    patients = np.array(sorted(grouped_a.keys()))

    deltas: list[dict[str, float]] = []
    skipped = 0

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap delta", leave=False):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)

        sampled_a = sample_grouped_rows(grouped_a, sampled_patients)
        sampled_b = sample_grouped_rows(grouped_b, sampled_patients)

        try:
            m_a = compute_metrics(sampled_a)
            m_b = compute_metrics(sampled_b)
            deltas.append(
                {
                    "delta_auroc": float(m_a["auroc"] - m_b["auroc"]),
                    "delta_auprc": float(m_a["auprc"] - m_b["auprc"]),
                }
            )
        except ValueError:
            skipped += 1

    return pd.DataFrame(deltas), skipped


def summarize_bootstrap(df_boot: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        "auroc": summarize_distribution(df_boot["auroc"].to_numpy()),
        "auprc": summarize_distribution(df_boot["auprc"].to_numpy()),
    }


def summarize_delta(df_delta: pd.DataFrame) -> dict[str, dict[str, float]]:
    auroc_vals = df_delta["delta_auroc"].to_numpy()
    auprc_vals = df_delta["delta_auprc"].to_numpy()

    out = {
        "delta_auroc": summarize_distribution(auroc_vals),
        "delta_auprc": summarize_distribution(auprc_vals),
    }
    out["delta_auroc"]["p_positive"] = float(np.mean(auroc_vals > 0))
    out["delta_auprc"]["p_positive"] = float(np.mean(auprc_vals > 0))
    return out


def pretty_print_summary(title: str, point_metrics: dict, bootstrap_summary: dict, skipped: int, n_bootstrap: int) -> None:
    print(f"\n=== {title} ===")
    print(
        f"Point estimate: "
        f"AUROC={point_metrics['auroc']:.4f}, "
        f"AUPRC={point_metrics['auprc']:.4f}"
    )
    print(
        f"Bootstrap AUROC mean={bootstrap_summary['auroc']['mean']:.4f} "
        f"[{bootstrap_summary['auroc']['ci_low']:.4f}, {bootstrap_summary['auroc']['ci_high']:.4f}]"
    )
    print(
        f"Bootstrap AUPRC mean={bootstrap_summary['auprc']['mean']:.4f} "
        f"[{bootstrap_summary['auprc']['ci_low']:.4f}, {bootstrap_summary['auprc']['ci_high']:.4f}]"
    )
    print(f"Skipped replicates: {skipped}/{n_bootstrap}")


def pretty_print_delta(delta_summary: dict, skipped: int, n_bootstrap: int) -> None:
    print("\n=== Delta (Model A - Model B) ===")
    print(
        f"Delta AUROC mean={delta_summary['delta_auroc']['mean']:.4f} "
        f"[{delta_summary['delta_auroc']['ci_low']:.4f}, {delta_summary['delta_auroc']['ci_high']:.4f}], "
        f"p(delta>0)={delta_summary['delta_auroc']['p_positive']:.3f}"
    )
    print(
        f"Delta AUPRC mean={delta_summary['delta_auprc']['mean']:.4f} "
        f"[{delta_summary['delta_auprc']['ci_low']:.4f}, {delta_summary['delta_auprc']['ci_high']:.4f}], "
        f"p(delta>0)={delta_summary['delta_auprc']['p_positive']:.3f}"
    )
    print(f"Skipped delta replicates: {skipped}/{n_bootstrap}")


def main(args: argparse.Namespace) -> None:
    df_a = load_predictions(args.model_a)
    df_b = load_predictions(args.model_b) if args.model_b else None

    point_a = compute_metrics(df_a)
    boot_a, skipped_a = bootstrap_patient_level(
        df_a, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    summary_a = summarize_bootstrap(boot_a)
    pretty_print_summary("Model A", point_a, summary_a, skipped_a, args.n_bootstrap)

    output: dict[str, object] = {
        "model_a_path": args.model_a,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "model_a": {
            "point_metrics": point_a,
            "bootstrap_summary": summary_a,
            "skipped_bootstraps": skipped_a,
            "n_rows": int(len(df_a)),
            "n_subjects": int(df_a["subject_id"].nunique()),
        },
    }

    if args.save_bootstrap_csv:
        save_dir = Path(args.output_json).parent if args.output_json else Path(".")
        save_dir.mkdir(parents=True, exist_ok=True)
        boot_a.to_csv(save_dir / "bootstrap_model_a.csv", index=False)

    if df_b is not None:
        point_b = compute_metrics(df_b)
        boot_b, skipped_b = bootstrap_patient_level(
            df_b, n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        summary_b = summarize_bootstrap(boot_b)
        pretty_print_summary("Model B", point_b, summary_b, skipped_b, args.n_bootstrap)

        delta_df, skipped_delta = bootstrap_delta(
            df_a, df_b, n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        delta_summary = summarize_delta(delta_df)
        pretty_print_delta(delta_summary, skipped_delta, args.n_bootstrap)

        output["model_b_path"] = args.model_b
        output["model_b"] = {
            "point_metrics": point_b,
            "bootstrap_summary": summary_b,
            "skipped_bootstraps": skipped_b,
            "n_rows": int(len(df_b)),
            "n_subjects": int(df_b["subject_id"].nunique()),
        }
        output["delta_a_minus_b"] = {
            "bootstrap_summary": delta_summary,
            "skipped_bootstraps": skipped_delta,
        }

        if args.save_bootstrap_csv:
            save_dir = Path(args.output_json).parent if args.output_json else Path(".")
            boot_b.to_csv(save_dir / "bootstrap_model_b.csv", index=False)
            delta_df.to_csv(save_dir / "bootstrap_delta_a_minus_b.csv", index=False)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved summary JSON to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True, help="Path to Model A prediction CSV")
    parser.add_argument("--model-b", default=None, help="Optional path to Model B prediction CSV")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save summary JSON",
    )
    parser.add_argument(
        "--save-bootstrap-csv",
        action="store_true",
        help="If set, save bootstrap replicate CSVs next to output JSON",
    )

    args = parser.parse_args()
    main(args)