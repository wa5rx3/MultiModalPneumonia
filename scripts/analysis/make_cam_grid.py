"""CAM-variant comparison grid (fig18) for the image model.

Six class-activation-mapping variants (CAM, Grad-CAM, Grad-CAM++, Score-CAM,
Smooth Grad-CAM++, XGrad-CAM) computed on the trained DenseNet-121 image model for a
handful of confident true-positive radiographs, laid out as rows = samples, columns =
methods (plus the original image). Lets the reader compare how each method localises the
region driving the pneumonia prediction. Real maps from the committed checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image

import torch.nn.functional as F
from torchcam.methods import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, XGradCAM

IMG_DIR = "artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3"
PATHS = "artifacts/models/multiseed/image_seed42/test_predictions.csv"
OUT = Path("manuscript/figures/fig18_cam_grid.png")
N_SAMPLES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


def build_model():
    m = models.densenet121(weights=None)
    m.classifier = torch.nn.Linear(m.classifier.in_features, 1)
    sd = torch.load(Path(IMG_DIR) / "checkpoints" / "best.pt", map_location="cpu")["model_state_dict"]
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


def pick_samples():
    canon = pd.read_csv(Path(IMG_DIR) / "test_predictions.csv")
    paths = pd.read_csv(PATHS)[["dicom_id", "image_path"]]
    m = canon.merge(paths, on="dicom_id")
    tp = m[m.target == 1].sort_values("pred_prob", ascending=False).head(N_SAMPLES)
    return list(tp["image_path"]), list(tp["pred_prob"])


def main():
    model = build_model()
    paths, probs = pick_samples()
    imgs = [Image.open(p).convert("RGB") for p in paths]
    rgb224 = [to_pil_image(transforms.Resize((224, 224))(transforms.ToTensor()(im))) for im in imgs]
    tensors = [TF(im).unsqueeze(0).to(DEVICE) for im in imgs]

    target = model.features.norm5
    methods = [
        ("CAM", lambda: CAM(model, target_layer=target, fc_layer=model.classifier)),
        ("Grad-CAM", lambda: GradCAM(model, target_layer=target)),
        ("Grad-CAM++", lambda: GradCAMpp(model, target_layer=target)),
        ("Score-CAM", lambda: ScoreCAM(model, target_layer=target)),
        ("Smooth Grad-CAM++", lambda: SmoothGradCAMpp(model, target_layer=target)),
        ("XGrad-CAM", lambda: XGradCAM(model, target_layer=target)),
    ]

    gray = [np.asarray(b, dtype=np.float32) / 255.0 for b in rgb224]  # 224x224x3
    cmap = plt.cm.jet

    def blend(cam_lr, base_rgb, alpha=0.45):
        # bilinear upsample the low-res CAM to 224 (no bicubic ringing), then colour-blend
        up = F.interpolate(cam_lr[None, None], size=(224, 224), mode="bilinear",
                           align_corners=False)[0, 0].numpy()
        up = (up - up.min()) / (up.max() - up.min() + 1e-8)
        heat = cmap(up)[..., :3]
        return np.clip(alpha * heat + (1 - alpha) * base_rgb, 0, 1)

    # overlays[method][sample] -> HxWx3 float
    overlays = {name: [] for name, _ in methods}
    for name, make in methods:
        ext = make()
        for t, g in zip(tensors, gray):
            model.zero_grad(set_to_none=True)
            scores = model(t)
            cam = ext(0, scores)[0].squeeze(0).detach().cpu()
            overlays[name].append(blend(cam, g))
        ext.remove_hooks()
        print(f"  done {name}", flush=True)

    ncol = 1 + len(methods)
    fig, axes = plt.subplots(N_SAMPLES, ncol, figsize=(2.0 * ncol, 2.0 * N_SAMPLES))
    col_titles = ["Original"] + [n for n, _ in methods]
    for r in range(N_SAMPLES):
        axes[r, 0].imshow(rgb224[r]); axes[r, 0].set_ylabel(f"p={probs[r]:.2f}", fontsize=9)
        for c, (name, _) in enumerate(methods, start=1):
            axes[r, c].imshow(overlays[name][r])
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
