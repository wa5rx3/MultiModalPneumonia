from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.models.multimodal_model import MultimodalPneumoniaModel

try:
    from src.interpretability.gradcam import run_gradcam
except ImportError:  # pragma: no cover
    run_gradcam = None  # type: ignore[misc, assignment]


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EVAL_DIR = ARTIFACTS_DIR / "evaluation"
INTERPRET_DIR = ARTIFACTS_DIR / "interpretability"
RUNS_REGISTRY = ARTIFACTS_DIR / "runs" / "registry.json"

MAIN_RUN_HINTS = [
    "clinical_baseline",
    "clinical_xgb",
    "image_pneumonia_finetune",
    "multimodal_pneumonia",
]

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

MISSING_INDICATOR_COLS = [
    "temperature_missing",
    "heartrate_missing",
    "resprate_missing",
    "o2sat_missing",
    "sbp_missing",
    "dbp_missing",
    "pain_missing",
    "acuity_missing",
]

METRIC_HELP = {
    "test_auroc": "Higher is better. Global ranking ability.",
    "test_auprc": "Higher is better. More informative than AUROC for positive-class retrieval.",
    "test_accuracy": "Thresholded at 0.5. Less informative than AUROC/AUPRC here.",
    "test_f1": "Thresholded at 0.5. Harmonic mean of precision and recall.",
}


@dataclass
class InferenceAssets:
    run_dir: Path
    display_name: str
    ckpt_path: Path
    config_path: Path
    preprocessor_path: Path


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _fmt_metric(v: Any, digits: int = 3) -> str:
    fv = _safe_float(v)
    if fv is None or np.isnan(fv):
        return "—"
    return f"{fv:.{digits}f}"


def _load_json_direct(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_json(path_str: str) -> dict[str, Any] | list[Any] | None:
    return _load_json_direct(Path(path_str))


@st.cache_data(show_spinner=False)
def read_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def read_registry(path_str: str) -> list[dict[str, Any]]:
    data = _load_json_direct(Path(path_str))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def pretty_name_from_dir(name: str) -> str:
    lowered = name.lower()
    if "clinical_baseline" in lowered:
        return "Clinical Logistic"
    if "clinical_xgb" in lowered:
        return "Clinical XGBoost"
    if "image_pneumonia_finetune" in lowered:
        return "Image-only DenseNet121"
    if "multimodal_pneumonia" in lowered:
        return "Multimodal DenseNet121 + Triage"
    if "pretrain" in lowered:
        return "Image Multilabel Pretrain"
    return name


def infer_stage_from_dir(name: str) -> str:
    lowered = name.lower()
    if "clinical_baseline" in lowered:
        return "clinical_logistic"
    if "clinical_xgb" in lowered:
        return "clinical_xgb"
    if "image_pneumonia_finetune" in lowered:
        return "image_finetune"
    if "multimodal_pneumonia" in lowered:
        return "multimodal"
    if "pretrain" in lowered:
        return "pretrain"
    return "other"


def _parse_run_suffix(name: str) -> str:
    for token in ["stronger_lr_v", "strong_v", "phase", "main"]:
        idx = name.find(token)
        if idx != -1:
            return name[idx:]
    return name


def extract_metrics_payload(run_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    candidates = [run_dir / "summary.json", run_dir / "metrics.json"]
    for path in candidates:
        payload = read_json(str(path))
        if isinstance(payload, dict):
            return payload, rel(path)
    return None, None


def extract_run_record(run_dir: Path) -> dict[str, Any]:
    payload, source_file = extract_metrics_payload(run_dir)
    config_path = run_dir / "config.json"
    config_payload = read_json(str(config_path))
    stage = infer_stage_from_dir(run_dir.name)

    record: dict[str, Any] = {
        "run_dir": rel(run_dir),
        "run_name": run_dir.name,
        "display_name": pretty_name_from_dir(run_dir.name),
        "stage": stage,
        "variant": _parse_run_suffix(run_dir.name),
        "source_file": source_file,
        "available": payload is not None,
        "config_exists": config_path.exists(),
        "best_epoch": None,
        "best_val_auprc": None,
        "best_val_loss": None,
        "test_auroc": None,
        "test_auprc": None,
        "test_accuracy": None,
        "test_f1": None,
        "test_loss": None,
        "val_auroc": None,
        "val_auprc": None,
        "val_loss": None,
        "train_rows": None,
        "val_rows": None,
        "test_rows": None,
        "selection_metric": None,
        "has_test_predictions": (run_dir / "test_predictions.csv").exists(),
        "has_val_predictions": (run_dir / "val_predictions.csv").exists(),
        "has_train_predictions": (run_dir / "train_predictions.csv").exists(),
        "has_preprocessor": (run_dir / "tabular_preprocessor.joblib").exists(),
        "has_checkpoints": (run_dir / "checkpoints").exists(),
        "has_best_checkpoint": (run_dir / "checkpoints" / "best.pt").exists(),
        "modified_time": run_dir.stat().st_mtime if run_dir.exists() else 0.0,
    }

    if isinstance(config_payload, dict):
        record["selection_metric"] = config_payload.get("selection_metric")
        for k in ["train_rows", "val_rows", "test_rows", "validate_rows"]:
            if k in config_payload and record.get(k) is None:
                record[k] = _safe_int(config_payload.get(k))

    if payload is None:
        return record

    record["best_epoch"] = _safe_int(payload.get("best_epoch"))
    record["best_val_auprc"] = _safe_float(payload.get("best_val_auprc"))
    record["best_val_loss"] = _safe_float(payload.get("best_val_loss"))

    val_metrics = payload.get("val_metrics", {}) if isinstance(payload.get("val_metrics"), dict) else {}
    test_metrics = payload.get("test_metrics", {}) if isinstance(payload.get("test_metrics"), dict) else {}

    record["val_auroc"] = _safe_float(val_metrics.get("auroc"))
    record["val_auprc"] = _safe_float(val_metrics.get("auprc"))
    record["val_loss"] = _safe_float(val_metrics.get("loss"))
    record["test_auroc"] = _safe_float(test_metrics.get("auroc"))
    record["test_auprc"] = _safe_float(test_metrics.get("auprc"))
    record["test_accuracy"] = _safe_float(test_metrics.get("accuracy"))
    record["test_f1"] = _safe_float(test_metrics.get("f1"))
    record["test_loss"] = _safe_float(test_metrics.get("loss"))

    for k in ["train_rows", "val_rows", "test_rows", "validate_rows"]:
        if k in payload:
            record[k] = record.get(k) or _safe_int(payload.get(k))

    return record


@st.cache_data(show_spinner=False)
def discover_runs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if MODELS_DIR.exists():
        for child in sorted(MODELS_DIR.iterdir()):
            if child.is_dir():
                rows.append(extract_run_record(child))
    return pd.DataFrame(rows)


def _rank_run_name(name: str) -> tuple[int, int, str]:
    lower = name.lower()
    if "stronger_lr_v" in lower:
        import re
        m = re.search(r"stronger_lr_v(\d+)", lower)
        return (0, -int(m.group(1)) if m else 0, lower)
    if "strong_v" in lower:
        import re
        m = re.search(r"strong_v(\d+)", lower)
        return (1, -int(m.group(1)) if m else 0, lower)
    if "main" in lower:
        return (2, 0, lower)
    if "phase" in lower:
        return (3, 0, lower)
    return (4, 0, lower)


def best_run_for_stage(runs_df: pd.DataFrame, stage: str) -> pd.Series | None:
    if runs_df.empty or "stage" not in runs_df.columns:
        return None
    sub = runs_df[runs_df["stage"] == stage].copy()
    if sub.empty:
        return None

    sort_cols = []
    ascending = []
    if "test_auprc" in sub.columns:
        sub["_test_auprc"] = pd.to_numeric(sub["test_auprc"], errors="coerce").fillna(-1.0)
        sort_cols.append("_test_auprc")
        ascending.append(False)
    if "test_auroc" in sub.columns:
        sub["_test_auroc"] = pd.to_numeric(sub["test_auroc"], errors="coerce").fillna(-1.0)
        sort_cols.append("_test_auroc")
        ascending.append(False)
    sub["_name_rank"] = sub["run_name"].map(_rank_run_name)
    sub = sub.sort_values(sort_cols, ascending=ascending) if sort_cols else sub
    sub = sub.sort_values(by=["_name_rank", "modified_time"], ascending=[True, False], kind="stable")
    if sort_cols:
        sub = sub.sort_values(by=sort_cols + ["modified_time"], ascending=ascending + [False], kind="stable")
    return sub.iloc[0]


def summarize_best_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    stage_order = [
        "clinical_logistic",
        "clinical_xgb",
        "image_finetune",
        "multimodal",
        "pretrain",
    ]
    rows = []
    for stage in stage_order:
        row = best_run_for_stage(runs_df, stage)
        if row is not None:
            rows.append(row.to_dict())
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_bootstrap_df() -> pd.DataFrame:
    rows = []
    if EVAL_DIR.exists():
        for path in sorted(EVAL_DIR.glob("bootstrap*.json")):
            data = read_json(str(path))
            if not isinstance(data, dict):
                rows.append({"file": rel(path), "valid": False})
                continue
            delta = data.get("delta_a_minus_b", {}) if isinstance(data.get("delta_a_minus_b"), dict) else {}
            bs = delta.get("bootstrap_summary", {}) if isinstance(delta.get("bootstrap_summary"), dict) else {}
            dauroc = bs.get("delta_auroc", {}) if isinstance(bs.get("delta_auroc"), dict) else {}
            dauprc = bs.get("delta_auprc", {}) if isinstance(bs.get("delta_auprc"), dict) else {}
            model_a_path = data.get("model_a_path")
            model_b_path = data.get("model_b_path")
            rows.append(
                {
                    "file": rel(path),
                    "valid": True,
                    "model_a_path": model_a_path,
                    "model_b_path": model_b_path,
                    "model_a_name": pretty_name_from_dir(Path(model_a_path).parent.name) if model_a_path else None,
                    "model_b_name": pretty_name_from_dir(Path(model_b_path).parent.name) if model_b_path else None,
                    "delta_auroc_mean": _safe_float(dauroc.get("mean")),
                    "delta_auroc_ci_low": _safe_float(dauroc.get("ci_low")),
                    "delta_auroc_ci_high": _safe_float(dauroc.get("ci_high")),
                    "delta_auprc_mean": _safe_float(dauprc.get("mean")),
                    "delta_auprc_ci_low": _safe_float(dauprc.get("ci_low")),
                    "delta_auprc_ci_high": _safe_float(dauprc.get("ci_high")),
                    "p_delta_auroc_positive": _safe_float(dauroc.get("p_positive")),
                    "p_delta_auprc_positive": _safe_float(dauprc.get("p_positive")),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_calibration_summary() -> pd.DataFrame:
    candidates = [
        EVAL_DIR / "calibration_stronger_lr_v3" / "calibration_summary.csv",
        EVAL_DIR / "calibration_strong_v2" / "calibration_summary.csv",
        EVAL_DIR / "calibration_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            df = read_csv(str(path)).copy()
            df["source_file"] = rel(path)
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_calibration_metrics() -> dict[str, Any] | None:
    candidates = [
        EVAL_DIR / "calibration_stronger_lr_v3" / "calibration_metrics.json",
        EVAL_DIR / "calibration_strong_v2" / "calibration_metrics.json",
        EVAL_DIR / "calibration_metrics.json",
    ]
    for path in candidates:
        data = read_json(str(path))
        if isinstance(data, dict):
            return data
    return None


@st.cache_data(show_spinner=False)
def load_decision_curve_summary() -> pd.DataFrame:
    candidates = [
        EVAL_DIR / "dca" / "decision_curve_all_models.csv",
        EVAL_DIR / "decision_curve_all_models.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df["source_file"] = rel(path)
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def collect_case_predictions() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    runs_df = discover_runs()
    if runs_df.empty:
        return pd.DataFrame()

    for _, row in runs_df.iterrows():
        run_dir = ROOT / str(row["run_dir"])
        pred_path = run_dir / "test_predictions.csv"
        if not pred_path.exists():
            continue
        try:
            pred_df = read_csv(str(pred_path)).copy()
        except Exception:
            continue

        pred_df["model_name"] = row["display_name"]
        pred_df["run_name"] = row["run_name"]
        pred_df["predictions_file"] = rel(pred_path)
        rows.append(pred_df)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    if "study_id" in df.columns:
        df["case_key"] = df["subject_id"].astype(str) + " / " + df["study_id"].astype(str)
    else:
        df["case_key"] = df["subject_id"].astype(str)
    return df


def find_inference_candidates(runs_df: pd.DataFrame) -> list[InferenceAssets]:
    candidates: list[InferenceAssets] = []
    if runs_df.empty:
        return candidates

    sub = runs_df[runs_df["stage"] == "multimodal"].copy()
    if sub.empty:
        return candidates

    sub = sub[sub["has_preprocessor"] & sub["has_best_checkpoint"] & sub["config_exists"]].copy()
    if sub.empty:
        return candidates

    sub["_test_auprc"] = pd.to_numeric(sub["test_auprc"], errors="coerce").fillna(-1.0)
    sub = sub.sort_values(["_test_auprc", "modified_time"], ascending=[False, False])

    for _, row in sub.iterrows():
        run_dir = ROOT / str(row["run_dir"])
        candidates.append(
            InferenceAssets(
                run_dir=run_dir,
                display_name=f"{row['display_name']} · {row['run_name']}",
                ckpt_path=run_dir / "checkpoints" / "best.pt",
                config_path=run_dir / "config.json",
                preprocessor_path=run_dir / "tabular_preprocessor.joblib",
            )
        )
    return candidates


def find_image_gradcam_candidates(runs_df: pd.DataFrame) -> list[Path]:
    """Image-only fine-tune runs with a best checkpoint (for Grad-CAM)."""
    out: list[Path] = []
    if runs_df.empty or "stage" not in runs_df.columns:
        return out
    sub = runs_df[(runs_df["stage"] == "image_finetune") & runs_df["has_best_checkpoint"]].copy()
    for _, row in sub.iterrows():
        out.append(ROOT / str(row["run_dir"]))
    return out


@st.cache_data(show_spinner=False)
def load_multimodal_input_table_parquet(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def resolve_study_tabular_row(table: pd.DataFrame, subject_id: int, study_id: int) -> pd.DataFrame:
    if table.empty or "subject_id" not in table.columns or "study_id" not in table.columns:
        raise ValueError("Training table is missing subject_id / study_id columns.")
    mask = (table["subject_id"].astype(int) == int(subject_id)) & (
        table["study_id"].astype(int) == int(study_id)
    )
    sub = table.loc[mask]
    if sub.empty:
        raise ValueError(f"No training-table row for subject_id={subject_id}, study_id={study_id}.")
    return sub.iloc[[0]].copy()


def resolve_image_path_on_disk(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_file():
        return p
    if not p.is_absolute():
        cand = ROOT / p
        if cand.is_file():
            return cand
    return p


def resolve_gradcam_target_layer(model: nn.Module, layer_path: str) -> nn.Module:
    current: nn.Module = model
    for part in layer_path.split("."):
        if not hasattr(current, part):
            raise ValueError(f"Invalid target layer path '{layer_path}': missing '{part}'.")
        current = getattr(current, part)
    return current


class _MultimodalImageTabFixed(nn.Module):
    """Wrap multimodal model so Grad-CAM sees f(image) -> logit with fixed tabular batch."""

    def __init__(self, multimodal: MultimodalPneumoniaModel, tab_batch: torch.Tensor) -> None:
        super().__init__()
        self._m = multimodal
        self.register_buffer("_tab", tab_batch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._m(x, self._tab)


@st.cache_resource(show_spinner=False)
def load_image_only_gradcam_bundle(run_dir_str: str) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    cfg_path = run_dir / "config.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    config = read_json(str(cfg_path))
    if not isinstance(config, dict):
        raise ValueError("Invalid config.json for image run.")
    image_size = _safe_int(config.get("image_size")) or 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchvision import models

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing model_state_dict")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    transform = build_inference_transform(image_size)
    return {"model": model, "device": device, "image_size": image_size, "transform": transform}


def list_saved_gradcam_pngs() -> list[Path]:
    if not INTERPRET_DIR.exists():
        return []
    return sorted(p for p in INTERPRET_DIR.rglob("*.png") if p.is_file())


def render_gradcam_tab(runs_df: pd.DataFrame) -> None:
    st.subheader("Grad-CAM")
    st.caption(
        "Spatial heatmaps for DenseNet121 (image-only or multimodal image branch with triage fixed per study). "
        "Not for clinical use."
    )

    if run_gradcam is None:
        st.error(
            "Could not import `run_gradcam` (install deps: `pip install opencv-python-headless` and ensure `src` is on PYTHONPATH)."
        )
        return

    gal_tab, live_tab = st.tabs(["Saved PNG gallery", "Live from checkpoint"])

    with gal_tab:
        pngs = list_saved_gradcam_pngs()
        if not pngs:
            st.info(f"No PNGs found under `{rel(INTERPRET_DIR)}`. Generate examples with `python -m scripts.generate_gradcam_examples`.")
        else:
            st.caption(f"{len(pngs)} image(s) under `{rel(INTERPRET_DIR)}`.")
            cols = st.columns(2)
            for i, p in enumerate(pngs):
                with cols[i % 2]:
                    try:
                        st.image(str(p), caption=rel(p), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not load {rel(p)}: {e}")

    with live_tab:
        model_kind = st.radio("Model type", ["Image-only fine-tune", "Multimodal (image branch)"], horizontal=True)

        if model_kind == "Image-only fine-tune":
            cand_dirs = find_image_gradcam_candidates(runs_df)
            if not cand_dirs:
                st.info("No image fine-tune runs with `checkpoints/best.pt` found.")
                return
            labels = [rel(d) for d in cand_dirs]
            run_options = list(zip(labels, cand_dirs))
            choice = st.selectbox("Image run", run_options, format_func=lambda x: x[0])
            run_dir = choice[1]
        else:
            candidates = find_inference_candidates(runs_df)
            if not candidates:
                st.info("No multimodal runs with checkpoint + tabular preprocessor found.")
                return
            cmap = {c.display_name: c for c in candidates}
            selected_label = st.selectbox("Multimodal run", list(cmap.keys()))
            run_dir = cmap[selected_label].run_dir

        split = st.selectbox("Predictions split", ["test", "validate"], index=0)
        pred_name = "test_predictions.csv" if split == "test" else "val_predictions.csv"
        pred_path = run_dir / pred_name
        if not pred_path.exists():
            st.warning(f"Missing {pred_name} under `{rel(run_dir)}`.")
            return

        try:
            pred_df = read_csv(str(pred_path)).copy()
        except Exception as e:
            st.error(f"Could not read predictions: {e}")
            return

        required = {"image_path", "target", "pred_prob"}
        if not required.issubset(pred_df.columns):
            st.error(f"Predictions CSV must contain columns: {sorted(required)}.")
            return

        search = st.text_input("Filter by subject_id / study_id / path substring", value="")
        view = pred_df.copy()
        if search.strip():
            needle = search.strip().lower()
            mask = view["image_path"].astype(str).str.lower().str.contains(needle)
            for col in ("subject_id", "study_id"):
                if col in view.columns:
                    mask |= view[col].astype(str).str.contains(needle)
            view = view[mask]
        if view.empty:
            st.warning("No prediction rows match the filter.")
            return

        display_keys: list[str] = []
        for idx, r in view.iterrows():
            sid = r.get("subject_id", "")
            stid = r.get("study_id", "")
            display_keys.append(f"{sid} / {stid}  |  pred={float(r['pred_prob']):.3f}  target={int(r['target'])}")

        pick = st.selectbox("Select case", range(len(view)), format_func=lambda i: display_keys[i])
        row = view.iloc[int(pick)]
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])
        img_path = resolve_image_path_on_disk(str(row["image_path"]))
        if not img_path.is_file():
            st.error(f"Image not found on disk: {img_path}")
            return

        layer_default = (
            "image_backbone.features.norm5"
            if model_kind == "Multimodal (image branch)"
            else "features.norm5"
        )
        target_layer_path = st.text_input(
            "Target layer",
            value=layer_default,
            help="Usually `features.norm5` (image-only) or `image_backbone.features.norm5` (multimodal).",
            key=f"gradcam_layer_{model_kind}",
        )
        alpha = st.slider("Overlay alpha", min_value=0.1, max_value=0.9, value=0.35, step=0.05)

        if st.button("Generate Grad-CAM", type="primary", key="gradcam_go"):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                pil = Image.open(img_path).convert("RGB")

                tl = target_layer_path.strip()
                if model_kind == "Image-only fine-tune":
                    bundle = load_image_only_gradcam_bundle(str(run_dir))
                    model = bundle["model"]
                    transform = bundle["transform"]
                    cam_model = model
                    target_layer = resolve_gradcam_target_layer(model, tl)
                else:
                    bundle_inf = load_inference_bundle(str(run_dir))
                    full_model: MultimodalPneumoniaModel = bundle_inf["model"]
                    preprocessor = bundle_inf["preprocessor"]
                    transform = bundle_inf["image_transform"]
                    config = bundle_inf["config"]
                    input_table = config.get("input_table")
                    if not input_table:
                        st.error("multimodal config.json missing input_table.")
                        return
                    table = load_multimodal_input_table_parquet(str(input_table))
                    tab_row = resolve_study_tabular_row(table, subject_id, study_id)
                    tab_arr = preprocessor.transform(tab_row).astype(np.float32)
                    tab_tensor = torch.tensor(tab_arr, dtype=torch.float32, device=device)
                    cam_model = _MultimodalImageTabFixed(full_model, tab_tensor)
                    if tl.startswith("features.") and not tl.startswith("image_backbone."):
                        tl = "image_backbone." + tl
                    target_layer = resolve_gradcam_target_layer(full_model, tl)

                tensor = transform(pil).unsqueeze(0).to(device)
                cam_model.train(False)

                with st.spinner("Running backward pass…"):
                    result = run_gradcam(
                        model=cam_model,
                        target_layer=target_layer,
                        input_tensor=tensor,
                        original_tensor=tensor.detach().cpu(),
                        class_idx=None,
                        alpha=float(alpha),
                    )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Input (denorm.)**")
                    st.image(result.image_rgb, use_container_width=True)
                with c2:
                    st.markdown("**Heatmap**")
                    st.image(result.heatmap, clamp=True, use_container_width=True)
                with c3:
                    st.markdown("**Overlay**")
                    st.image(result.overlay_rgb, use_container_width=True)

                st.caption(
                    f"Case subject_id={subject_id}, study_id={study_id} · "
                    f"row target={int(row['target'])} · pred_prob={float(row['pred_prob']):.4f} · "
                    f"image `{rel(img_path) if str(img_path).startswith(str(ROOT)) else img_path}`"
                )
            except Exception as e:
                st.exception(e)


def render_metric_card(title: str, value: Any, help_text: str | None = None) -> None:
    st.metric(title, _fmt_metric(value))
    if help_text:
        st.caption(help_text)


def render_key_findings(best_df: pd.DataFrame, bootstrap_df: pd.DataFrame) -> None:
    st.markdown("### Key Findings")
    if best_df.empty:
        st.info("No successful runs discovered yet.")
        return

    image = best_run_for_stage(best_df, "image_finetune")
    multi = best_run_for_stage(best_df, "multimodal")
    clin = best_run_for_stage(best_df, "clinical_xgb")
    if clin is None:
        clin = best_run_for_stage(best_df, "clinical_logistic")

    msgs = []
    if image is not None:
        msgs.append(
            f"**Best image-only run**: `{image['run_name']}` with test AUROC **{_fmt_metric(image['test_auroc'])}** and AUPRC **{_fmt_metric(image['test_auprc'])}**."
        )
    if multi is not None:
        msgs.append(
            f"**Best multimodal run**: `{multi['run_name']}` with test AUROC **{_fmt_metric(multi['test_auroc'])}** and AUPRC **{_fmt_metric(multi['test_auprc'])}**."
        )
    if clin is not None:
        msgs.append(
            f"**Best clinical run**: `{clin['run_name']}` with test AUROC **{_fmt_metric(clin['test_auroc'])}** and AUPRC **{_fmt_metric(clin['test_auprc'])}**."
        )

    for msg in msgs:
        st.markdown(f"- {msg}")

    if not bootstrap_df.empty:
        sub = bootstrap_df.copy()
        pair = sub[
            sub["file"].astype(str).str.contains("multimodal", case=False)
            & sub["file"].astype(str).str.contains("image", case=False)
        ]
        if not pair.empty:
            row = pair.iloc[0]
            st.info(
                "Bootstrap comparison indicates that multimodal does not clearly outperform image-only: "
                f"ΔAUROC={_fmt_metric(row.get('delta_auroc_mean'))} "
                f"[{_fmt_metric(row.get('delta_auroc_ci_low'))}, {_fmt_metric(row.get('delta_auroc_ci_high'))}], "
                f"ΔAUPRC={_fmt_metric(row.get('delta_auprc_mean'))} "
                f"[{_fmt_metric(row.get('delta_auprc_ci_low'))}, {_fmt_metric(row.get('delta_auprc_ci_high'))}]."
            )


def render_overview_tab(runs_df: pd.DataFrame, bootstrap_df: pd.DataFrame) -> None:
    st.subheader("Overview")
    if runs_df.empty:
        st.warning("No runs found under artifacts/models.")
        return

    best_df = summarize_best_runs(runs_df)
    render_key_findings(best_df, bootstrap_df)

    st.markdown("### Best Run per Stage")
    display_cols = [
        "display_name",
        "run_name",
        "test_auroc",
        "test_auprc",
        "test_accuracy",
        "test_f1",
        "best_epoch",
        "selection_metric",
        "run_dir",
    ]
    st.dataframe(best_df[[c for c in display_cols if c in best_df.columns]], use_container_width=True, hide_index=True)

    metrics_plot = best_df[["display_name", "test_auroc", "test_auprc"]].dropna(how="all", subset=["test_auroc", "test_auprc"])
    if not metrics_plot.empty:
        plot_df = metrics_plot.set_index("display_name")
        st.markdown("### Test AUROC / AUPRC")
        st.bar_chart(plot_df)

    cal_df = load_calibration_summary()
    if not cal_df.empty:
        st.markdown("### Calibration Summary")
        st.dataframe(cal_df, use_container_width=True, hide_index=True)


def render_runs_tab(runs_df: pd.DataFrame) -> None:
    st.subheader("Run Explorer")
    if runs_df.empty:
        st.warning("No runs found.")
        return

    stage_options = ["all"] + sorted(runs_df["stage"].dropna().unique().tolist())
    selected_stage = st.selectbox("Filter by stage", stage_options)
    only_with_test = st.checkbox("Only show runs with test predictions", value=False)

    view_df = runs_df.copy()
    if selected_stage != "all":
        view_df = view_df[view_df["stage"] == selected_stage]
    if only_with_test:
        view_df = view_df[view_df["has_test_predictions"]]

    sort_col = st.selectbox(
        "Sort by",
        ["test_auprc", "test_auroc", "best_val_auprc", "run_name", "modified_time"],
        index=0,
    )
    ascending = st.checkbox("Ascending", value=False)
    if sort_col in view_df.columns:
        view_df = view_df.sort_values(sort_col, ascending=ascending, na_position="last")

    st.dataframe(view_df, use_container_width=True, hide_index=True)

    if view_df.empty:
        return

    run_names = view_df["run_name"].tolist()
    selected = st.selectbox("Inspect run", run_names)
    row = view_df[view_df["run_name"] == selected].iloc[0]
    run_dir = ROOT / str(row["run_dir"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Test AUROC", row.get("test_auroc"), METRIC_HELP["test_auroc"])
    with c2:
        render_metric_card("Test AUPRC", row.get("test_auprc"), METRIC_HELP["test_auprc"])
    with c3:
        render_metric_card("Test Accuracy", row.get("test_accuracy"), METRIC_HELP["test_accuracy"])
    with c4:
        render_metric_card("Test F1", row.get("test_f1"), METRIC_HELP["test_f1"])

    st.markdown("### Files")
    files = []
    for fname in ["config.json", "summary.json", "metrics.json", "history.json", "val_predictions.csv", "test_predictions.csv"]:
        path = run_dir / fname
        if path.exists():
            files.append({"file": fname, "path": rel(path)})
    if (run_dir / "checkpoints" / "best.pt").exists():
        files.append({"file": "checkpoints/best.pt", "path": rel(run_dir / "checkpoints" / "best.pt")})
    if files:
        st.dataframe(pd.DataFrame(files), use_container_width=True, hide_index=True)

    cfg = read_json(str(run_dir / "config.json"))
    if isinstance(cfg, dict):
        with st.expander("config.json"):
            st.json(cfg)

    hist_path = run_dir / "history.json"
    hist = read_json(str(hist_path))
    if isinstance(hist, list) and hist:
        hist_df = pd.DataFrame(hist)
        st.markdown("### Training History")
        chart_cols = [c for c in ["train_loss", "val_loss", "val_auprc", "val_auroc"] if c in hist_df.columns]
        if chart_cols:
            st.line_chart(hist_df.set_index("epoch")[chart_cols])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


def render_bootstrap_tab(bootstrap_df: pd.DataFrame) -> None:
    st.subheader("Bootstrap Comparisons")
    if bootstrap_df.empty:
        st.info("No bootstrap outputs found under artifacts/evaluation.")
        return

    st.dataframe(bootstrap_df, use_container_width=True, hide_index=True)

    valid_df = bootstrap_df[bootstrap_df.get("valid", False)].copy() if "valid" in bootstrap_df.columns else bootstrap_df.copy()
    if valid_df.empty:
        return

    selected_file = st.selectbox("Inspect bootstrap result", valid_df["file"].tolist())
    row = valid_df[valid_df["file"] == selected_file].iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("ΔAUROC mean", _fmt_metric(row.get("delta_auroc_mean")))
        st.caption(f"CI: {_fmt_metric(row.get('delta_auroc_ci_low'))} to {_fmt_metric(row.get('delta_auroc_ci_high'))}")
    with c2:
        st.metric("ΔAUPRC mean", _fmt_metric(row.get("delta_auprc_mean")))
        st.caption(f"CI: {_fmt_metric(row.get('delta_auprc_ci_low'))} to {_fmt_metric(row.get('delta_auprc_ci_high'))}")

    if _safe_float(row.get("delta_auprc_ci_low")) is not None and _safe_float(row.get("delta_auprc_ci_high")) is not None:
        low = float(row["delta_auprc_ci_low"])
        high = float(row["delta_auprc_ci_high"])
        if low <= 0 <= high:
            st.warning("AUPRC difference CI crosses 0. There is no clear evidence of superiority.")
        else:
            st.success("AUPRC difference CI does not cross 0.")


def render_case_explorer_tab() -> None:
    st.subheader("Case Explorer")
    case_df = collect_case_predictions()
    if case_df.empty:
        st.info("No test prediction files found.")
        return

    search = st.text_input("Filter case key / subject / study", value="")
    working = case_df.copy()
    if search.strip():
        needle = search.strip().lower()
        mask = working["case_key"].astype(str).str.lower().str.contains(needle)
        if "subject_id" in working.columns:
            mask |= working["subject_id"].astype(str).str.contains(needle)
        if "study_id" in working.columns:
            mask |= working["study_id"].astype(str).str.contains(needle)
        working = working[mask]

    keys = working["case_key"].drop_duplicates().sort_values().tolist()
    if not keys:
        st.warning("No cases match the current filter.")
        return

    selected_key = st.selectbox("Select case", keys)
    case_rows = working[working["case_key"] == selected_key].copy()

    meta_cols = [c for c in ["subject_id", "study_id", "dicom_id", "image_path", "target"] if c in case_rows.columns]
    if meta_cols:
        st.markdown("### Case Metadata")
        meta_row = case_rows[meta_cols].iloc[0]
        meta_display = meta_row.to_frame(name="value")
        st.dataframe(meta_display, width="stretch")

        image_path = None
        if "image_path" in case_rows.columns:
            non_null = case_rows["image_path"].dropna()
            if not non_null.empty:
                image_path = non_null.iloc[0]
        if image_path:
            img_path = Path(str(image_path))
            if not img_path.is_absolute():
                img_path = ROOT / img_path
            if img_path.exists():
                caption = rel(img_path) if str(img_path).startswith(str(ROOT)) else str(img_path)
                st.image(str(img_path), caption=caption, width=380)

    st.markdown("### Model Predictions")
    compare_cols = [c for c in ["model_name", "run_name", "pred_prob", "target", "predictions_file"] if c in case_rows.columns]
    compare_df = case_rows[compare_cols].sort_values("pred_prob", ascending=False)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    if {"model_name", "pred_prob"}.issubset(compare_df.columns):
        st.bar_chart(compare_df[["model_name", "pred_prob"]].drop_duplicates().set_index("model_name"))


def build_inference_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_tabular_input_df_from_form(values: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for field in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]:
        row[field] = values.get(field)
        row[f"{field}_missing"] = int(values.get(field) is None)

    row["is_pa"] = int(values.get("view_position") == "PA")
    row["is_ap"] = int(values.get("view_position") == "AP")
    row["gender"] = values.get("gender", "UNKNOWN")
    row["race"] = values.get("race", "UNKNOWN")
    row["arrival_transport"] = values.get("arrival_transport", "UNKNOWN")

    df = pd.DataFrame([row])
    for col in TRIAGE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in MISSING_INDICATOR_COLS:
        df[col] = df[col].fillna(1).astype(int)
    for col in ["is_pa", "is_ap"]:
        df[col] = df[col].fillna(0).astype(int)
    for col in TRIAGE_CATEGORICAL_COLS:
        df[col] = df[col].astype("string").fillna("UNKNOWN").str.strip().replace({"": "UNKNOWN"})
    return df


@st.cache_resource(show_spinner=False)
def load_inference_bundle(run_dir_str: str) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    preprocessor_path = run_dir / "tabular_preprocessor.joblib"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    config = read_json(str(config_path))
    if not isinstance(config, dict):
        raise ValueError("Could not read multimodal config.json")

    tabular_input_dim = _safe_int(config.get("tabular_input_dim"))
    tabular_hidden_dim = _safe_int(config.get("tabular_hidden_dim")) or 128
    fusion_hidden_dim = _safe_int(config.get("fusion_hidden_dim")) or 256
    dropout = _safe_float(config.get("dropout")) or 0.2
    image_size = _safe_int(config.get("image_size")) or 224
    if tabular_input_dim is None:
        raise ValueError("tabular_input_dim missing from config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalPneumoniaModel(
        tabular_input_dim=tabular_input_dim,
        tabular_hidden_dim=tabular_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=dropout,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain model_state_dict")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    preprocessor = joblib.load(preprocessor_path)

    return {
        "device": device,
        "model": model,
        "preprocessor": preprocessor,
        "image_transform": build_inference_transform(image_size),
        "config": config,
        "checkpoint_path": str(ckpt_path),
        "preprocessor_path": str(preprocessor_path),
    }


def run_multimodal_inference(run_dir: Path, image: Image.Image, triage_form_values: dict[str, Any]) -> dict[str, Any]:
    bundle = load_inference_bundle(str(run_dir))
    model: MultimodalPneumoniaModel = bundle["model"]
    preprocessor = bundle["preprocessor"]
    transform = bundle["image_transform"]
    device: torch.device = bundle["device"]

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    tab_df = build_tabular_input_df_from_form(triage_form_values)
    tab_array = preprocessor.transform(tab_df).astype(np.float32)
    tab_tensor = torch.tensor(tab_array, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(image_tensor, tab_tensor)
        prob = torch.sigmoid(logits).squeeze().item()

    return {"probability": float(prob), "tabular_df": tab_df, "bundle": bundle}


def _numeric_input_with_missing(label: str, key_prefix: str, min_value: float, max_value: float, default_value: float, step: float = 1.0, fmt: str | None = None) -> float | None:
    c1, c2 = st.columns([3, 2])
    with c2:
        missing = st.checkbox(f"Missing {label}", value=False, key=f"{key_prefix}_missing")
    with c1:
        value = st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=step,
            format=fmt,
            disabled=missing,
            key=f"{key_prefix}_value",
        )
    return None if missing else value


def render_inference_tab(runs_df: pd.DataFrame) -> None:
    st.subheader("Inference")
    st.caption("Research prototype only. Not for clinical use.")

    candidates = find_inference_candidates(runs_df)
    if not candidates:
        st.info("No multimodal runs with config, checkpoint, and preprocessor were found.")
        return

    candidate_map = {c.display_name: c for c in candidates}
    default_index = 0
    selected_label = st.selectbox("Inference model", list(candidate_map.keys()), index=default_index)
    selected_assets = candidate_map[selected_label]

    try:
        bundle = load_inference_bundle(str(selected_assets.run_dir))
        st.success("Inference assets loaded.")
        with st.expander("Loaded resources"):
            st.write("Run directory:", rel(selected_assets.run_dir))
            st.write("Checkpoint:", bundle["checkpoint_path"])
            st.write("Preprocessor:", bundle["preprocessor_path"])
            st.write("Selection metric:", bundle["config"].get("selection_metric", "—"))
            st.write("Image size:", bundle["config"].get("image_size", "—"))
    except Exception as e:
        st.error(f"Could not load inference resources: {e}")
        return

    uploaded_file = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])

    st.markdown("### Triage Inputs")
    st.caption("Unchecked numeric fields will be passed as observed values. Checked fields are explicitly marked missing for the model.")

    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = _numeric_input_with_missing("Temperature (F)", "temperature", 80.0, 110.0, 98.6, 0.1, "%.1f")
        heartrate = _numeric_input_with_missing("Heart rate", "heartrate", 0.0, 250.0, 90.0, 1.0, "%.0f")
        resprate = _numeric_input_with_missing("Respiratory rate", "resprate", 0.0, 80.0, 18.0, 1.0, "%.0f")
    with c2:
        o2sat = _numeric_input_with_missing("O2 saturation", "o2sat", 0.0, 100.0, 97.0, 1.0, "%.0f")
        sbp = _numeric_input_with_missing("Systolic BP", "sbp", 0.0, 300.0, 120.0, 1.0, "%.0f")
        dbp = _numeric_input_with_missing("Diastolic BP", "dbp", 0.0, 200.0, 80.0, 1.0, "%.0f")
    with c3:
        pain = _numeric_input_with_missing("Pain", "pain", 0.0, 10.0, 0.0, 1.0, "%.0f")
        acuity = _numeric_input_with_missing("Acuity", "acuity", 1.0, 5.0, 3.0, 1.0, "%.0f")
        gender = st.selectbox("Gender", options=["M", "F", "UNKNOWN"])
        race = st.text_input("Race", value="UNKNOWN")
        arrival_transport = st.text_input("Arrival transport", value="UNKNOWN")

    view_position = st.radio("View position", options=["AP", "PA", "Unknown"], horizontal=True)

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded chest X-ray", width=380)

    if st.button("Run inference", type="primary"):
        if image is None:
            st.error("Please upload a chest X-ray image first.")
            return

        triage_values = {
            "temperature": temperature,
            "heartrate": heartrate,
            "resprate": resprate,
            "o2sat": o2sat,
            "sbp": sbp,
            "dbp": dbp,
            "pain": pain,
            "acuity": acuity,
            "gender": gender,
            "race": race,
            "arrival_transport": arrival_transport,
            "view_position": view_position if view_position in {"AP", "PA"} else "UNKNOWN",
        }

        try:
            result = run_multimodal_inference(selected_assets.run_dir, image, triage_values)
            prob = result["probability"]
            st.markdown("### Predicted Pneumonia Probability")
            st.metric("Probability", f"{prob:.3f}")
            if prob >= 0.8:
                st.error("High predicted probability")
            elif prob >= 0.5:
                st.warning("Moderate predicted probability")
            else:
                st.success("Lower predicted probability")

            with st.expander("Tabular inputs used by the model"):
                st.dataframe(result["tabular_df"], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Inference failed: {e}")


def render_artifact_gallery() -> None:
    st.subheader("Evaluation Artifacts")
    image_candidates = [
        EVAL_DIR / "calibration_stronger_lr_v3" / "reliability_diagram_all_models.png",
        EVAL_DIR / "calibration_strong_v2" / "reliability_diagram_all_models.png",
        EVAL_DIR / "reliability_diagram_all_models.png",
        EVAL_DIR / "dca" / "decision_curve.png",
        EVAL_DIR / "decision_curve.png",
    ]
    shown = 0
    cols = st.columns(2)
    for idx, path in enumerate(image_candidates):
        if path.exists():
            with cols[shown % 2]:
                st.image(str(path), caption=rel(path), use_container_width=True)
            shown += 1
    if shown == 0:
        st.caption("No evaluation images found yet.")


def main() -> None:
    st.set_page_config(page_title="Multimodal Pneumonia Dashboard", layout="wide")
    st.title("Multimodal Pneumonia Dashboard")
    st.caption("Experiment dashboard for clinical, image-only, and multimodal pneumonia models.")

    runs_df = discover_runs()
    bootstrap_df = load_bootstrap_df()
    registry = read_registry(str(RUNS_REGISTRY))

    if runs_df.empty and bootstrap_df.empty:
        st.error("No runs or evaluation outputs were found under artifacts/.")
        st.stop()

    with st.sidebar:
        st.markdown("### Project Snapshot")
        st.write(f"Model runs discovered: **{len(runs_df)}**")
        st.write(f"Bootstrap comparisons: **{len(bootstrap_df)}**")
        st.write(f"Registry entries: **{len(registry)}**")
        if not runs_df.empty:
            best_image = best_run_for_stage(runs_df, "image_finetune")
            best_multi = best_run_for_stage(runs_df, "multimodal")
            if best_image is not None:
                st.write(f"Best image AUPRC: **{_fmt_metric(best_image.get('test_auprc'))}**")
            if best_multi is not None:
                st.write(f"Best multimodal AUPRC: **{_fmt_metric(best_multi.get('test_auprc'))}**")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Overview",
            "Run Explorer",
            "Bootstrap",
            "Case Explorer",
            "Inference",
            "Grad-CAM",
            "Artifacts",
        ]
    )

    with tab1:
        render_overview_tab(runs_df, bootstrap_df)
    with tab2:
        render_runs_tab(runs_df)
    with tab3:
        render_bootstrap_tab(bootstrap_df)
    with tab4:
        render_case_explorer_tab()
    with tab5:
        render_inference_tab(runs_df)
    with tab6:
        render_gradcam_tab(runs_df)
    with tab7:
        render_artifact_gallery()


if __name__ == "__main__":
    main()
