"""Run the multilabel pretrained DenseNet on the cohort to get per-finding image scores.

Needed for the multi-condition dissociation: the Edema score is the image signal for heart
failure (as the pneumonia-specific model is the signal for pneumonia). Outputs sigmoid
scores for all 14 CheXpert findings on the cohort val + test splits.
Output: artifacts/evaluation/multilabel_scores/{val,test}_finding_scores.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

CKPT = "artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/multilabel_scores")
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
            "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
KEYS = ["subject_id", "study_id", "dicom_id"]
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run(split_df):
    rows, out = [], []
    batch, meta = [], []
    for _, r in split_df.iterrows():
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except Exception:
            continue
        batch.append(TF(img)); meta.append(r)
        if len(batch) == 32:
            rows.append((torch.stack(batch), meta)); batch, meta = [], []
    if batch:
        rows.append((torch.stack(batch), meta))
    return rows


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = models.densenet121(weights=None)
    m.classifier = torch.nn.Linear(m.classifier.in_features, 14)
    m.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state_dict"])
    m.to(DEVICE).eval()

    d = pd.read_parquet(TABLE)
    for split in ["validate", "test"]:
        sub = d[d.temporal_split == split].reset_index(drop=True)
        recs = []
        for i in range(0, len(sub), 32):
            chunk = sub.iloc[i:i + 32]
            imgs, keep = [], []
            for _, r in chunk.iterrows():
                try:
                    imgs.append(TF(Image.open(r["image_path"]).convert("RGB"))); keep.append(r)
                except Exception:
                    continue
            if not imgs:
                continue
            p = torch.sigmoid(m(torch.stack(imgs).to(DEVICE))).cpu().numpy()
            for r, row in zip(keep, p):
                recs.append({**{k: r[k] for k in KEYS}, **{f: float(v) for f, v in zip(FINDINGS, row)}})
        df = pd.DataFrame(recs)
        name = "test" if split == "test" else "val"
        df.to_csv(OUT / f"{name}_finding_scores.csv", index=False)
        print(f"{name}: {len(df)} studies scored")


if __name__ == "__main__":
    main()
