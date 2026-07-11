"""Score the 14-finding multilabel model over the expanded ED-CXR cohort (train+val+test).

Gives every study a full radiographic finding profile (pneumonia, cardiomegaly, edema,
effusion, consolidation, support devices, ...) so the fusion ladder can use a fair image
baseline for outcome targets rather than a single pneumonia probability. The multilabel model
was pretrained on non-ED studies with zero patient overlap with these splits (verified).

Output: artifacts/evaluation/multilabel_scores/expanded_{train,val,test}_finding_scores.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

CKPT = "artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt"
TABLE = "artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet"
OUT = Path("artifacts/evaluation/multilabel_scores")
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
            "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
KEYS = ["subject_id", "study_id", "dicom_id"]
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 14)
    m.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state_dict"])
    m.to(DEVICE).eval()
    d = pd.read_parquet(TABLE)
    if "exists" in d.columns:
        d = d[d.exists.fillna(True)]
    for split in ["train", "validate", "test"]:
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
        name = {"validate": "val"}.get(split, split)
        pd.DataFrame(recs).to_csv(OUT / f"expanded_{name}_finding_scores.csv", index=False)
        print(f"{name}: {len(recs)}/{len(sub)} scored")


if __name__ == "__main__":
    main()
