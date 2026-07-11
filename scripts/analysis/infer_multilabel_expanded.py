"""Score the 14-finding multilabel model over the expanded ED-CXR cohort (train+val+test).

Uses a multi-worker DataLoader so image loading off disk runs in parallel with GPU compute
(the disk was the bottleneck at ~13 img/s single-threaded). Gives every study a full
radiographic finding profile so the fusion ladder can use a fair image baseline for outcome
targets. The multilabel model was pretrained on non-ED studies with zero patient overlap with
these splits (verified).

Output: artifacts/evaluation/multilabel_scores/expanded_{train,val,test}_finding_scores.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
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


class CXR(Dataset):
    def __init__(self, df):
        self.rows = df[["image_path"] + KEYS].to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            img = TF(Image.open(r["image_path"]).convert("RGB"))
            ok = 1
        except Exception:
            img = torch.zeros(3, 224, 224)
            ok = 0
        return img, int(r["subject_id"]), int(r["study_id"]), str(r["dicom_id"]), ok


@torch.no_grad()
def score(model, df):
    loader = DataLoader(CXR(df), batch_size=64, num_workers=8, pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)
    recs = []
    t0 = time.time()
    done = 0
    for imgs, subj, study, dicom, ok in loader:
        p = torch.sigmoid(model(imgs.to(DEVICE, non_blocking=True))).cpu().numpy()
        ok = ok.numpy()
        for j in range(len(p)):
            if ok[j]:
                recs.append({"subject_id": int(subj[j]), "study_id": int(study[j]), "dicom_id": dicom[j],
                             **{f: float(v) for f, v in zip(FINDINGS, p[j])}})
        done += len(p)
        if done % 3200 == 0:
            print(f"  {done}/{len(df)} ({done / max(time.time() - t0, 1):.0f} img/s)", flush=True)
    return pd.DataFrame(recs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 14)
    m.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state_dict"])
    m.to(DEVICE).eval()
    d = pd.read_parquet(TABLE)
    if "exists" in d.columns:
        d = d[d.exists.fillna(True)]
    for split in ["validate", "test", "train"]:
        sub = d[d.temporal_split == split].reset_index(drop=True)
        name = {"validate": "val"}.get(split, split)
        t0 = time.time()
        df = score(m, sub)
        df.to_csv(OUT / f"expanded_{name}_finding_scores.csv", index=False)
        print(f"{name}: {len(df)}/{len(sub)} scored in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
