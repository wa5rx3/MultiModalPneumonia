from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CHEXPERT_LABEL_COLS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def find_single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return matches[0]


def load_chexpert(metadata_root: Path) -> pd.DataFrame:
    chexpert_path = find_single_file(metadata_root, "*chexpert.csv.gz")
    return pd.read_csv(chexpert_path).copy()


def choose_merge_keys(manifest: pd.DataFrame, chex: pd.DataFrame, allow_fallback: bool) -> list[str]:
    preferred = ["subject_id", "study_id", "dicom_id"]
    fallback = ["subject_id", "study_id"]

    preferred_ok = all(c in manifest.columns for c in preferred) and all(c in chex.columns for c in preferred)
    fallback_ok = all(c in manifest.columns for c in fallback) and all(c in chex.columns for c in fallback)

    if preferred_ok:
        return preferred
    if allow_fallback and fallback_ok:
        return fallback

    raise ValueError(
        "Could not find acceptable merge keys.\n"
        f"Preferred keys: {preferred}\n"
        f"Allow fallback: {allow_fallback}\n"
        f"Manifest cols: {list(manifest.columns)}\n"
        f"CheXpert cols: {list(chex.columns)}"
    )


def build_conflict_report(chex: pd.DataFrame, group_keys: list[str], label_cols: list[str]) -> dict:
    sub = chex[group_keys + label_cols].copy()

    per_label_conflicts = {}
    for label in label_cols:
        tmp = sub.dropna(subset=[label]).copy()
        if tmp.empty:
            per_label_conflicts[label] = 0
            continue

        grp = (
            tmp.groupby(group_keys, dropna=False)[label]
            .nunique()
            .reset_index(name="n_unique")
        )
        per_label_conflicts[label] = int((grp["n_unique"] > 1).sum())

    total_conflict_groups_any_label = 0
    if not sub.empty:
        keys = sub[group_keys].drop_duplicates().copy()
        keys["_conflict"] = False

        for label in label_cols:
            tmp = sub.dropna(subset=[label]).copy()
            if tmp.empty:
                continue

            grp = (
                tmp.groupby(group_keys, dropna=False)[label]
                .nunique()
                .reset_index(name="n_unique")
            )
            grp["_conflict_this_label"] = grp["n_unique"] > 1

            keys = keys.merge(
                grp[group_keys + ["_conflict_this_label"]],
                on=group_keys,
                how="left",
            )
            keys["_conflict_this_label"] = keys["_conflict_this_label"].fillna(False)
            keys["_conflict"] = keys["_conflict"] | keys["_conflict_this_label"]
            keys = keys.drop(columns=["_conflict_this_label"])

        total_conflict_groups_any_label = int(keys["_conflict"].sum())

    return {
        "group_keys": group_keys,
        "groups_total": int(len(sub[group_keys].drop_duplicates())),
        "groups_with_any_label_conflict": int(total_conflict_groups_any_label),
        "per_label_conflict_groups": per_label_conflicts,
    }


def collapse_chexpert(chex: pd.DataFrame, merge_keys: list[str], label_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    needed = [c for c in merge_keys + label_cols if c in chex.columns]
    sub = chex[needed].copy()

    qc = build_conflict_report(sub, merge_keys, label_cols)

    if qc["groups_with_any_label_conflict"] > 0:
        raise ValueError(
            "Conflicting CheXpert labels found at merge key level. "
            f"QC summary: {qc}"
        )

    sub = sub.drop_duplicates()

    key_counts = sub.groupby(merge_keys, dropna=False).size()
    multi_key_rows = int((key_counts > 1).sum())
    if multi_key_rows > 0:
        raise ValueError(
            "Multiple rows remain after drop_duplicates for selected merge keys. "
            f"multi_key_rows={multi_key_rows}"
        )

    post_qc = {
        "rows_after_drop_duplicates": int(len(sub)),
        "unique_merge_key_rows": int(len(key_counts)),
    }

    return sub, {"pre_qc": qc, "post_qc": post_qc}


def validate_label_values(df: pd.DataFrame, label_cols: list[str]) -> dict:
    allowed = {-1.0, 0.0, 1.0}
    bad_counts = {}

    for col in label_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = vals.dropna()[~vals.dropna().isin(allowed)]
        bad_counts[col] = int(len(bad))

    return {
        "bad_value_counts": bad_counts,
        "columns_with_bad_values": [k for k, v in bad_counts.items() if v > 0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain-manifest",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_with_pretrain_split.parquet",
    )
    parser.add_argument(
        "--metadata-root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/mimic_cxr_multilabel_pretrain_table.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/mimic_cxr_multilabel_pretrain_table_report.json",
    )
    parser.add_argument(
        "--allow-fallback-study-merge",
        action="store_true",
        help="Allow fallback from (subject_id, study_id, dicom_id) to (subject_id, study_id).",
    )
    args = parser.parse_args()

    print("Loading pretraining manifest...")
    manifest = pd.read_parquet(args.pretrain_manifest).copy()

    required_manifest_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "image_path",
        "pretrain_split",
    ]
    missing_manifest = [c for c in required_manifest_cols if c not in manifest.columns]
    if missing_manifest:
        raise ValueError(f"Pretraining manifest missing columns: {missing_manifest}")

    allowed_splits = {"pretrain_train", "pretrain_internal_val"}
    manifest = manifest[manifest["pretrain_split"].isin(allowed_splits)].copy()

    print("Loading CheXpert labels...")
    chex = load_chexpert(Path(args.metadata_root)).copy()

    label_cols_present = [c for c in CHEXPERT_LABEL_COLS if c in chex.columns]
    if len(label_cols_present) != len(CHEXPERT_LABEL_COLS):
        missing_labels = sorted(set(CHEXPERT_LABEL_COLS) - set(label_cols_present))
        raise ValueError(f"CheXpert file missing expected label columns: {missing_labels}")

    merge_keys = choose_merge_keys(
        manifest=manifest,
        chex=chex,
        allow_fallback=args.allow_fallback_study_merge,
    )

    label_value_qc = validate_label_values(chex, CHEXPERT_LABEL_COLS)
    if label_value_qc["columns_with_bad_values"]:
        raise ValueError(f"Unexpected label values found: {label_value_qc}")

    chex_collapsed, chex_qc = collapse_chexpert(chex, merge_keys, CHEXPERT_LABEL_COLS)

    merged = manifest.merge(
        chex_collapsed,
        on=merge_keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    merged["chexpert_row_found"] = merged["_merge"].eq("both")
    merged = merged.drop(columns=["_merge"])

    for col in CHEXPERT_LABEL_COLS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")



        merged[f"{col}_mask"] = merged[col].isin([0.0, 1.0])

    output_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "image_path",
        "pretrain_split",
        "chexpert_row_found",
    ] + CHEXPERT_LABEL_COLS + [f"{c}_mask" for c in CHEXPERT_LABEL_COLS]

    out = merged[output_cols].copy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    report = {
        "rows": int(len(out)),
        "subjects": int(out["subject_id"].nunique()),
        "studies": int(out["study_id"].nunique()),
        "dicoms": int(out["dicom_id"].nunique()),
        "merge_keys_used": merge_keys,
        "fallback_study_merge_allowed": bool(args.allow_fallback_study_merge),
        "pretrain_split_row_counts": {
            k: int(v) for k, v in out["pretrain_split"].value_counts().to_dict().items()
        },
        "chexpert_row_found": int(out["chexpert_row_found"].sum()),
        "chexpert_row_missing": int((~out["chexpert_row_found"]).sum()),
        "label_non_missing_counts": {
            col: int(out[col].notna().sum()) for col in CHEXPERT_LABEL_COLS
        },
        "label_positive_counts": {
            col: int((out[col] == 1).sum()) for col in CHEXPERT_LABEL_COLS
        },
        "label_uncertain_counts": {
            col: int((out[col] == -1).sum()) for col in CHEXPERT_LABEL_COLS
        },
        "label_supervised_counts": {
            col: int(out[f"{col}_mask"].sum()) for col in CHEXPERT_LABEL_COLS
        },
        "label_value_qc": label_value_qc,
        "chexpert_qc": chex_qc,
        "notes": [
            "Pretraining table uses rows selected from the upstream pretraining manifest.",
            "CheXpert labels are merged with explicit preferred/fallback key logic.",
            "Raw labels preserve 1 / 0 / -1 / NaN.",
            "Supervised masks are TRUE only for 0/1 labels; uncertain (-1) and NaN are masked out.",
        ],
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()