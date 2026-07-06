"""Grad-CAM montage on the rebuilt image model (seed 42) for representative
TEST cases: a confident true positive, a false positive, a false negative.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.data.build_cohort import make_expected_image_path
from src.interpretability.gradcam import run_gradcam

RUN = Path("artifacts/models/multiseed/image_seed42")
OUT = Path("manuscript/figures")
CXR = "D:/mimic_data"


def load_model(device):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 1)
    ck = torch.load(RUN / "checkpoints" / "best.pt", map_location=device)
    m.load_state_dict(ck["model_state_dict"])
    return m.to(device).eval()


def tfm():
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    t = tfm()

    df = pd.read_csv(RUN / "test_predictions.csv")
    tp = df[df.target == 1].sort_values("pred_prob", ascending=False).iloc[0]
    fp = df[df.target == 0].sort_values("pred_prob", ascending=False).iloc[0]
    fn = df[df.target == 1].sort_values("pred_prob", ascending=True).iloc[0]
    cases = [("True positive", tp), ("False positive", fp), ("False negative", fn)]

    fig, axes = plt.subplots(3, 2, figsize=(6.5, 9))
    target_layer = model.features.norm5
    for i, (label, r) in enumerate(cases):
        p = make_expected_image_path(Path(CXR), int(r.subject_id), int(r.study_id), str(r.dicom_id))
        img = Image.open(p).convert("RGB")
        x = t(img).unsqueeze(0).to(device)
        res = run_gradcam(model=model, target_layer=target_layer, input_tensor=x,
                          original_tensor=x.detach().cpu(), class_idx=None, alpha=0.4)
        axes[i, 0].imshow(res.image_rgb); axes[i, 0].set_title(f"{label}\ntarget={int(r.target)}, pred={r.pred_prob:.2f}", fontsize=9)
        axes[i, 1].imshow(res.overlay_rgb); axes[i, 1].set_title("Grad-CAM overlay", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig12_gradcam.png", dpi=180); plt.close(fig)
    print("wrote fig12_gradcam.png; cases:",
          {"TP": round(float(tp.pred_prob), 3), "FP": round(float(fp.pred_prob), 3), "FN": round(float(fn.pred_prob), 3)})


if __name__ == "__main__":
    main()
