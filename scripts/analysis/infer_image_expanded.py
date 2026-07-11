"""Run the five fine-tuned image models on the expanded ED-CXR cohort (val + test).

Produces pneumonia image scores for the full 81k-study ED cohort so the outcome rungs of the
fusion ladder (admission, ICU transfer, mortality, culture) can be evaluated with proper power.
No patient in the image models' training set appears in these splits (verified separately).

Output: artifacts/models/multiseed/image_seed{s}/expanded_{val,test}_predictions.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

SEEDS = [42, 123, 456, 789, 1000]
KEYS = ["subject_id", "study_id", "dicom_id"]
TABLE = "artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_model(seed):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 1)
    ckpt = torch.load(f"artifacts/models/multiseed/image_seed{seed}/checkpoints/best.pt", map_location="cpu")
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(DEVICE).eval()


@torch.no_grad()
def run(model, df):
    recs = []
    for i in range(0, len(df), 32):
        chunk = df.iloc[i:i + 32]
        imgs, keep = [], []
        for _, r in chunk.iterrows():
            try:
                imgs.append(TF(Image.open(r["image_path"]).convert("RGB"))); keep.append(r)
            except Exception:
                continue
        if not imgs:
            continue
        p = torch.sigmoid(model(torch.stack(imgs).to(DEVICE))).squeeze(1).cpu().numpy()
        for r, pv in zip(keep, np.atleast_1d(p)):
            recs.append({**{k: r[k] for k in KEYS}, "pred_prob": float(pv)})
    return pd.DataFrame(recs)


def main():
    d = pd.read_parquet(TABLE)
    d = d[d.exists.fillna(True)] if "exists" in d.columns else d
    for seed in SEEDS:
        model = load_model(seed)
        out = Path(f"artifacts/models/multiseed/image_seed{seed}")
        for split in ["validate", "test"]:
            sub = d[d.temporal_split == split].reset_index(drop=True)
            preds = run(model, sub)
            name = "test" if split == "test" else "val"
            preds.to_csv(out / f"expanded_{name}_predictions.csv", index=False)
            print(f"seed {seed} {name}: {len(preds)}/{len(sub)} scored")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
