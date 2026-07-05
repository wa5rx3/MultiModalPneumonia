"""Modern CXR foundation-model baseline (reviewer-defense for P4b).

Uses torchxrayvision's CheXpert-trained DenseNet-121 (NOT trained on MIMIC, so no
test contamination on our cohort) two ways on the identical held-out split:
  (1) zero-shot: the model's own 'Pneumonia' output probability -> metrics. Since
      the backbone was trained on external (CheXpert) data, this doubles as an
      external-model -> our-data generalization check.
  (2) linear probe: logistic regression on frozen 1024-d pooled features, fit on
      our TRAIN only, evaluated on our TEST.
Compares both to our from-scratch fine-tuned DenseNet-121 (image-only ~0.737).
If the modern backbone is not clearly better, it confirms the backbone choice does
not change the paper's conclusions.

Output: artifacts/evaluation/modern_backbone/*.{json,csv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import torchxrayvision as xrv
from torchxrayvision.datasets import XRayResizer
from src.data.build_cohort import make_expected_image_path
from src.evaluation.calibration_analysis import compute_ece_mce

TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
CXR_ROOT = "D:/mimic_data"
OUT = Path("artifacts/evaluation/modern_backbone")
WEIGHTS = "densenet121-res224-chex"


def load_img_tensor(subject_id, study_id, dicom_id, resizer) -> np.ndarray | None:
    p = make_expected_image_path(Path(CXR_ROOT), int(subject_id), int(study_id), str(dicom_id))
    if not p.is_file():
        return None
    img = np.array(Image.open(p).convert("L"), dtype=np.float32)
    img = xrv.datasets.normalize(img, 255)[None, ...]
    return resizer(img)  # (1, 224, 224)


def metrics(y, p) -> dict:
    ece, _, _ = compute_ece_mce(np.asarray(y), np.asarray(p), n_bins=10)
    return {"n": int(len(y)), "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "ece": float(ece), "brier": float(brier_score_loss(y, p))}


@torch.no_grad()
def extract(df, model, device, resizer, pneumo_idx, batch_size=32):
    feats, zshot, targets, keep = [], [], [], []
    buf, meta = [], []

    def flush():
        if not buf:
            return
        t = torch.from_numpy(np.stack(buf)).to(device)
        f = model.features2(t).cpu().numpy()
        o = torch.sigmoid(model(t))[:, pneumo_idx].cpu().numpy()
        feats.append(f)
        zshot.append(o)
        for m_ in meta:
            targets.append(m_)
        buf.clear(); meta.clear()

    for _, r in df.iterrows():
        img = load_img_tensor(r["subject_id"], r["study_id"], r["dicom_id"], resizer)
        if img is None:
            continue
        buf.append(img); meta.append(int(r["target"])); keep.append(True)
        if len(buf) >= batch_size:
            flush()
    flush()
    return np.concatenate(feats), np.concatenate(zshot), np.asarray(targets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: limit rows per split")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = xrv.models.DenseNet(weights=WEIGHTS).to(device).eval()
    pneumo_idx = model.pathologies.index("Pneumonia")
    resizer = XRayResizer(224)

    df = pd.read_parquet(TABLE)
    data = {}
    for split in ["train", "test"]:
        sub = df[df["temporal_split"] == split]
        if args.limit:
            sub = sub.head(args.limit)
        print(f"Extracting {split}: {len(sub)} rows ...", flush=True)
        data[split] = extract(sub, model, device, resizer, pneumo_idx, args.batch_size)

    Xtr, ztr, ytr = data["train"]
    Xte, zte, yte = data["test"]
    print(f"train feats {Xtr.shape}, test feats {Xte.shape}", flush=True)

    results = {"weights": WEIGHTS, "note": "CheXpert-trained (non-MIMIC) backbone; no MIMIC test contamination"}
    # (1) zero-shot
    results["zeroshot_test"] = metrics(yte, zte)
    # (2) linear probe (fit on train, eval on test)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    probe_te = clf.predict_proba(Xte)[:, 1]
    results["linear_probe_test"] = metrics(yte, probe_te)

    pd.DataFrame({"target": yte, "zeroshot_prob": zte, "probe_prob": probe_te}).to_csv(
        OUT / "modern_backbone_test_predictions.csv", index=False)
    with open(OUT / "modern_backbone_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Modern backbone (CheXpert DenseNet-121) on our test set ===")
    z, lp = results["zeroshot_test"], results["linear_probe_test"]
    print(f"  zero-shot Pneumonia : AUROC {z['auroc']:.4f}  AUPRC {z['auprc']:.4f}  ECE {z['ece']:.4f}  Brier {z['brier']:.4f}")
    print(f"  linear probe        : AUROC {lp['auroc']:.4f}  AUPRC {lp['auprc']:.4f}  ECE {lp['ece']:.4f}  Brier {lp['brier']:.4f}")
    print(f"  (our finetuned DenseNet image-only reference: AUROC ~0.737)")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
